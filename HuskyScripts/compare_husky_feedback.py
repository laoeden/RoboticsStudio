#!/usr/bin/env python3
"""Compare Husky feedback and plot goal-travel diagnostics in real time."""

import argparse
import math
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
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
                 origin_heading: float, odom_to_world_heading: float,
                 history_seconds: float) -> None:
        super().__init__('compare_husky_feedback')
        self.lock = threading.Lock()
        self.history_seconds = history_seconds
        self.origin_latitude = math.radians(origin_latitude)
        self.origin_longitude = math.radians(origin_longitude)
        self.origin_heading = math.radians(origin_heading)
        self.odom_to_world_heading = math.radians(odom_to_world_heading)
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
        self.goals = []
        self.samples = []
        self.command_linear = 0.0
        self.command_angular = 0.0
        self.command_time = 0.0

        self.create_subscription(
            NavSatFix, '/husky1/gps/fix', self.on_gps, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, '/husky1/odometry', self.on_odom, qos_profile_sensor_data
        )
        self.create_subscription(PointStamped, '/husky1/goal', self.on_goal, 10)
        self.create_subscription(Twist, '/husky1/cmd_vel', self.on_command, 10)
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
        with self.lock:
            self.previous_gps_xy = gps_xy
            self.gps_xy = gps_xy
            self.gps_time = time.monotonic()
            self.calibrate_odom_offset()

    def on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        with self.lock:
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
        if not all(math.isfinite(value) for value in (
            message.point.x, message.point.y
        )):
            return
        with self.lock:
            now = time.monotonic()
            self.goal_xy = (message.point.x, message.point.y)
            self.goal_time = now
            self.goals.append((message.point.x, message.point.y, now))

    def on_command(self, message: Twist) -> None:
        if not all(math.isfinite(value) for value in (
            message.linear.x, message.angular.z
        )):
            return
        with self.lock:
            self.command_linear = message.linear.x
            self.command_angular = message.angular.z
            self.command_time = time.monotonic()

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            self._trim_history(now)
            return {
                'gps': self.gps_xy,
                'odom': self.aligned_odom_xy(),
                'yaw': self.odom_yaw,
                'goal': self.goal_xy,
                'goals': list(self.goals),
                'samples': list(self.samples),
                'command': (self.command_linear, self.command_angular),
            }

    def aligned_odom_xy(self):
        if self.odom_xy is None:
            return None
        if self.odom_world_offset is None:
            return self.odom_xy
        return (
            self.odom_xy[0] + self.odom_world_offset[0],
            self.odom_xy[1] + self.odom_world_offset[1],
        )

    def _trim_history(self, now: float) -> None:
        cutoff = now - self.history_seconds
        self.goals = [goal for goal in self.goals if goal[2] >= cutoff]
        self.samples = [
            sample for sample in self.samples if sample[0] >= cutoff
        ]

    def record_sample(self, now: float) -> None:
        if self.gps_xy is None or self.odom_yaw is None:
            return
        goal_distance = math.nan
        desired_heading = math.nan
        if self.goal_xy is not None:
            dx = self.goal_xy[0] - self.gps_xy[0]
            dy = self.goal_xy[1] - self.gps_xy[1]
            goal_distance = math.hypot(dx, dy)
            desired_heading = math.atan2(dy, dx)
        aligned_odom = self.aligned_odom_xy()
        position_error = math.nan
        if aligned_odom is not None:
            position_error = math.hypot(
                self.gps_xy[0] - aligned_odom[0],
                self.gps_xy[1] - aligned_odom[1],
            )
        self.samples.append((
            now,
            self.gps_xy[0],
            self.gps_xy[1],
            aligned_odom[0] if aligned_odom is not None else math.nan,
            aligned_odom[1] if aligned_odom is not None else math.nan,
            wrap_angle(self.odom_yaw + self.odom_to_world_heading),
            desired_heading,
            self.command_linear,
            self.command_angular,
            goal_distance,
            position_error,
        ))
        self._trim_history(now)

    def report(self) -> None:
        now = time.monotonic()
        with self.lock:
            self.record_sample(now)
        if self.gps_xy is None or self.odom_xy is None:
            self.get_logger().info(
                f'waiting: GPS={self.gps_xy is not None} '
                f'odom={self.odom_xy is not None}'
            )
            return

        aligned_odom_xy = self.aligned_odom_xy()
        position_error_x = self.gps_xy[0] - aligned_odom_xy[0]
        position_error_y = self.gps_xy[1] - aligned_odom_xy[1]
        position_error = math.hypot(position_error_x, position_error_y)
        gps_age = now - self.gps_time
        odom_age = now - self.odom_time
        gps_heading_text = 'warming up'
        heading_error_text = 'n/a'
        if self.gps_heading is not None:
            gps_heading_text = f'{math.degrees(self.gps_heading):7.2f} deg'
            heading_error = wrap_angle(
                self.gps_heading
                - (self.odom_yaw + self.odom_to_world_heading)
            )
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
    parser.add_argument(
        '--odom-to-world-heading',
        type=float,
        default=0.0,
        help='odometry yaw offset in degrees (default: 0)',
    )
    parser.add_argument(
        '--history-seconds',
        type=float,
        default=300.0,
        help='seconds retained by the live plots (default: 300)',
    )
    parser.add_argument(
        '--plot-period',
        type=float,
        default=0.25,
        help='live plot refresh period in seconds (default: 0.25)',
    )
    args = parser.parse_args()
    if args.history_seconds <= 0 or args.plot_period <= 0:
        parser.error('history-seconds and plot-period must be positive')
    rclpy.init()
    node = FeedbackComparison(
        args.origin_latitude,
        args.origin_longitude,
        args.origin_heading,
        args.odom_to_world_heading,
        args.history_seconds,
    )
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    map_axis, heading_axis = axes[0]
    command_axis, error_axis = axes[1]
    map_axis.set_title('Husky travel path')
    map_axis.set_xlabel('X (m)')
    map_axis.set_ylabel('Y (m)')
    map_axis.axis('equal')
    heading_axis.set_title('Heading: actual vs goal direction')
    heading_axis.set_ylabel('degrees')
    command_axis.set_title('Commanded velocity')
    command_axis.set_ylabel('m/s or rad/s')
    error_axis.set_title('Goal distance and GPS/odometry error')
    error_axis.set_ylabel('metres')
    gps_line, = map_axis.plot([], [], label='GPS position')
    odom_line, = map_axis.plot([], [], '--', label='aligned odometry')
    goal_points, = map_axis.plot([], [], 'x', markersize=8, label='received goals')
    actual_heading_line, = heading_axis.plot([], [], label='odometry yaw')
    desired_heading_line, = heading_axis.plot([], [], label='direction to goal')
    linear_line, = command_axis.plot([], [], label='linear.x')
    angular_line, = command_axis.plot([], [], label='angular.z')
    distance_line, = error_axis.plot([], [], label='goal distance')
    position_error_line, = error_axis.plot([], [], label='GPS - odometry')
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
        axis.legend(loc='best')
    figure.tight_layout()

    def update(_frame):
        snapshot = node.snapshot()
        samples = np.asarray(snapshot['samples'], dtype=float)
        if samples.ndim != 2 or not len(samples):
            return
        relative_time = samples[:, 0] - samples[-1, 0]
        gps_line.set_data(samples[:, 1], samples[:, 2])
        odom_line.set_data(samples[:, 3], samples[:, 4])
        goals = np.asarray(snapshot['goals'], dtype=float)
        if goals.ndim == 2 and len(goals):
            goal_points.set_data(goals[:, 0], goals[:, 1])
        actual_heading_line.set_data(
            relative_time, np.degrees(np.unwrap(samples[:, 5]))
        )
        desired_heading_line.set_data(
            relative_time, np.degrees(np.unwrap(samples[:, 6]))
        )
        linear_line.set_data(relative_time, samples[:, 7])
        angular_line.set_data(relative_time, samples[:, 8])
        distance_line.set_data(relative_time, samples[:, 9])
        position_error_line.set_data(relative_time, samples[:, 10])
        for axis in axes.flat:
            axis.relim()
            axis.autoscale_view()
        map_axis.axis('equal')
        figure.canvas.draw_idle()

    animation = __import__('matplotlib.animation', fromlist=['FuncAnimation']).FuncAnimation(
        figure,
        update,
        interval=args.plot_period * 1000.0,
        cache_frame_data=False,
    )
    try:
        plt.show()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        del animation
        rclpy.shutdown()
        ros_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
