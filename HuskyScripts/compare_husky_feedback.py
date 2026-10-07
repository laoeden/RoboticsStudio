#!/usr/bin/env python3
"""Compare Husky GPS world position and motion heading with odometry."""

import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix

EARTH_RADIUS_METERS = 6378137.0


def yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class FeedbackComparison(Node):
    def __init__(self, origin_latitude: float, origin_longitude: float,
                 origin_heading: float) -> None:
        super().__init__('compare_husky_feedback')
        self.origin_latitude = math.radians(origin_latitude)
        self.origin_longitude = math.radians(origin_longitude)
        self.origin_heading = math.radians(origin_heading)
        self.gps_xy = None
        self.previous_gps_xy = None
        self.gps_heading = None
        self.gps_heading_reference = None
        self.gps_time = 0.0
        self.odom_xy = None
        self.odom_world_offset = None
        self.odom_yaw = None
        self.odom_time = 0.0
        self.goal_xy = None
        self.goal_time = 0.0

        self.create_subscription(
            NavSatFix, '/husky1/gps/fix', self.on_gps, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, '/husky1/odometry', self.on_odom, qos_profile_sensor_data
        )
        self.create_subscription(PointStamped, '/husky1/goal', self.on_goal, 10)
        self.create_timer(0.5, self.report)

    def on_gps(self, message: NavSatFix) -> None:
        if not math.isfinite(message.latitude) or not math.isfinite(message.longitude):
            return
        latitude = math.radians(message.latitude)
        longitude = math.radians(message.longitude)
        east = (longitude - self.origin_longitude) * EARTH_RADIUS_METERS * math.cos(
            self.origin_latitude
        )
        north = (latitude - self.origin_latitude) * EARTH_RADIUS_METERS
        cos_heading = math.cos(self.origin_heading)
        sin_heading = math.sin(self.origin_heading)
        gps_xy = (
            east * cos_heading + north * sin_heading,
            -east * sin_heading + north * cos_heading,
        )
        if self.gps_heading_reference is None:
            self.gps_heading_reference = gps_xy
        else:
            dx = gps_xy[0] - self.gps_heading_reference[0]
            dy = gps_xy[1] - self.gps_heading_reference[1]
            if math.hypot(dx, dy) >= 0.05:
                self.gps_heading = math.atan2(dy, dx)
                self.gps_heading_reference = gps_xy
        self.previous_gps_xy = gps_xy
        self.gps_xy = gps_xy
        self.gps_time = time.monotonic()
        self.calibrate_odom_offset()

    def on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        self.odom_xy = (pose.position.x, pose.position.y)
        self.odom_yaw = yaw_from_quaternion(pose.orientation)
        self.odom_time = time.monotonic()
        self.calibrate_odom_offset()

    def calibrate_odom_offset(self) -> None:
        if self.gps_xy is None or self.odom_xy is None:
            return
        measured_offset = (
            self.gps_xy[0] - self.odom_xy[0],
            self.gps_xy[1] - self.odom_xy[1],
        )
        if self.odom_world_offset is None:
            self.odom_world_offset = measured_offset
            self.get_logger().info(
                f'Calibrated odom-to-world offset: '
                f'({self.odom_world_offset[0]:+.3f}, '
                f'{self.odom_world_offset[1]:+.3f}) m'
            )
        else:
            alpha = 0.1
            self.odom_world_offset = (
                (1.0 - alpha) * self.odom_world_offset[0]
                + alpha * measured_offset[0],
                (1.0 - alpha) * self.odom_world_offset[1]
                + alpha * measured_offset[1],
            )

    def on_goal(self, message: PointStamped) -> None:
        self.goal_xy = (message.point.x, message.point.y)
        self.goal_time = time.monotonic()

    def report(self) -> None:
        now = time.monotonic()
        if self.gps_xy is None or self.odom_xy is None:
            self.get_logger().info(
                f'waiting: GPS={self.gps_xy is not None} '
                f'odom={self.odom_xy is not None}'
            )
            return

        aligned_odom_xy = self.odom_xy
        if self.odom_world_offset is not None:
            aligned_odom_xy = (
                self.odom_xy[0] + self.odom_world_offset[0],
                self.odom_xy[1] + self.odom_world_offset[1],
            )
        position_error_x = self.gps_xy[0] - aligned_odom_xy[0]
        position_error_y = self.gps_xy[1] - aligned_odom_xy[1]
        position_error = math.hypot(position_error_x, position_error_y)
        gps_age = now - self.gps_time
        odom_age = now - self.odom_time
        gps_heading_text = 'warming up'
        heading_error_text = 'n/a'
        if self.gps_heading is not None:
            gps_heading_text = f'{math.degrees(self.gps_heading):7.2f} deg'
            heading_error = wrap_angle(self.gps_heading - self.odom_yaw)
            heading_error_text = f'{math.degrees(heading_error):+7.2f} deg'

        goal_text = 'none'
        if self.goal_xy is not None:
            goal_distance = math.hypot(
                self.gps_xy[0] - self.goal_xy[0],
                self.gps_xy[1] - self.goal_xy[1],
            )
            goal_text = (
                f'goal=({self.goal_xy[0]:6.2f}, {self.goal_xy[1]:6.2f}) '
                f'distance={goal_distance:5.2f} m'
            )

        self.get_logger().info(
            f'GPS xy=({self.gps_xy[0]:7.3f}, {self.gps_xy[1]:7.3f}) '
            f'odom xy=({aligned_odom_xy[0]:7.3f}, {aligned_odom_xy[1]:7.3f}) '
            f'error=({position_error_x:+7.3f}, {position_error_y:+7.3f}) '
            f'|error|={position_error:6.3f} m '
            f'odom_yaw={math.degrees(self.odom_yaw):7.2f} deg '
            f'gps_heading={gps_heading_text} '
            f'heading_error={heading_error_text} '
            f'age=({gps_age:.2f}s, {odom_age:.2f}s) {goal_text}'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin-latitude', type=float, default=0.0)
    parser.add_argument('--origin-longitude', type=float, default=0.0)
    parser.add_argument('--origin-heading', type=float, default=0.0)
    args = parser.parse_args()
    rclpy.init()
    node = FeedbackComparison(
        args.origin_latitude,
        args.origin_longitude,
        args.origin_heading,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
