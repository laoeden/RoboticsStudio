#!/usr/bin/env python3
"""Move the simulated Husky to absolute XY coordinates in its odometry frame."""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, NavSatFix
from std_msgs.msg import Bool
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

try:
    from grid_path_planner import plan_path
except ImportError:
    from HuskyScripts.grid_path_planner import plan_path

def quaternion_yaw(quaternion) -> float:
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def matching_frame(first: str, second: str) -> bool:
    if first == second:
        return True
    return {first, second} <= {'husky1/odom', 'husky1_odom'}


EARTH_RADIUS_METERS = 6378137.0


class HuskyMotion(Node):
    def __init__(self) -> None:
        super().__init__('husky_motion', namespace='husky1')
        defaults = {
            'linear_gain': 1.5,
            'max_speed': 2.0,
            'angular_gain': 1.5,
            'max_angular_speed': 1.2,
            'heading_tolerance': 0.15,
            'position_tolerance': 0.2,
            'odom_timeout': 5.0,
            'gps_timeout': 5.0,
            'gps_origin_latitude': 0.0,
            'gps_origin_longitude': 0.0,
            'gps_origin_heading': 0.0,
            'initial_heading': 0.0,
            'odom_calibration_alpha': 0.1,
            'goal_timeout': 120.0,
            'progress_timeout': 5.0,
            'minimum_progress': 0.05,
            'obstacle_stopping_enabled': False,
            'progress_stopping_enabled': False,
            'obstacle_distance': 1.0,
            'front_angle': math.pi / 4.0,
            'planning_range': 5.0,
            'path_resolution': 0.2,
            'robot_clearance': 0.45,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            value = self.get_parameter(name).value
            if name in ('obstacle_stopping_enabled', 'progress_stopping_enabled'):
                value = bool(value)
            else:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f'{name} must be finite')
                if name == 'gps_origin_latitude':
                    if not -90.0 <= value <= 90.0:
                        raise ValueError(f'{name} must be between -90 and 90')
                elif name == 'gps_origin_longitude':
                    if not -180.0 <= value <= 180.0:
                        raise ValueError(f'{name} must be between -180 and 180')
                elif name in ('gps_origin_heading', 'initial_heading'):
                    value = math.radians(value)
                elif name == 'odom_calibration_alpha':
                    if not 0.0 < value <= 1.0:
                        raise ValueError(f'{name} must be in (0, 1]')
                elif value <= 0:
                    raise ValueError(f'{name} must be positive')
            setattr(self, name, value)

        self.odom = None
        self.odom_received = 0.0
        self.odom_heading = self.initial_heading
        self.odom_world_offset = None
        self.gps_position = None
        self.previous_gps_position = None
        self.gps_heading_reference = None
        self.gps_heading = self.initial_heading
        self.gps_heading_received = False
        self.gps_received = 0.0
        self.goal = None
        self.goal_started = 0.0
        self.best_goal_distance = math.inf
        self.last_progress_time = 0.0
        self.obstacle_detected = False
        self.estop_active = False
        self.scan_obstacle_points = np.empty((0, 2), dtype=np.float32)
        self.scan_received = False
        self.path = np.empty((0, 2), dtype=np.float32)
        self.path_index = 0
        self.path_replan_requested = False
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.obstacle_stop_publisher = self.create_publisher(
            Bool,
            'obstacle_stop',
            10,
        )
        self.create_subscription(
            Odometry,
            'odometry',
            self.on_odom,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            NavSatFix,
            'gps/fix',
            self.on_gps,
            qos_profile_sensor_data,
        )
        self.create_subscription(PointStamped, 'goal', self.on_goal, 10)
        self.create_subscription(
            LaserScan,
            'scan',
            self.on_scan,
            qos_profile_sensor_data,
        )
        self.create_timer(
            0.05,
            self.tick,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        self.get_logger().info('Ready: waiting for GPS, odometry, and a goal.')

    def on_scan(self, message: LaserScan) -> None:
        ranges = np.asarray(message.ranges, dtype=np.float32)
        angles = message.angle_min + np.arange(
            len(message.ranges), dtype=np.float32
        ) * message.angle_increment
        valid = np.isfinite(ranges)
        valid &= ranges >= message.range_min
        valid &= ranges <= min(message.range_max, self.planning_range)
        if self.odom is not None and self.gps_position is not None and np.any(valid):
            pose = self.odom.pose.pose
            yaw = self.odom_heading
            local_x = ranges[valid] * np.cos(angles[valid])
            local_y = ranges[valid] * np.sin(angles[valid])
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            self.scan_obstacle_points = np.column_stack((
                self.gps_position[0] + cos_yaw * local_x - sin_yaw * local_y,
                self.gps_position[1] + sin_yaw * local_x + cos_yaw * local_y,
            )).astype(np.float32)
            self.scan_received = True

        if self.goal is None:
            return
        front_ranges = []
        for index, distance in enumerate(message.ranges):
            angle = (
                message.angle_min + index * message.angle_increment + math.pi
            ) % (2.0 * math.pi) - math.pi
            if (
                abs(angle) <= self.front_angle
                and math.isfinite(distance)
                and message.range_min <= distance <= message.range_max
            ):
                front_ranges.append(distance)

        nearest_front_range = min(front_ranges, default=math.inf)
        if nearest_front_range > self.obstacle_distance or self.obstacle_detected:
            return
        pose = self.odom.pose.pose if self.odom is not None else None
        if pose is not None and self.gps_position is not None and self.goal is not None:
            candidate_path = plan_path(
                self.gps_position[0],
                self.gps_position[1],
                np.array([self.goal.point.x, self.goal.point.y], dtype=np.float32),
                self.scan_obstacle_points,
                resolution=self.path_resolution,
                robot_clearance=self.robot_clearance,
            )
            if candidate_path.size:
                candidate_target = candidate_path[min(1, len(candidate_path) - 1)]
                current_target = (
                    self.path[self.path_index]
                    if self.path.size and self.path_index < len(self.path)
                    else None
                )
                needs_new_path = (
                    current_target is None
                    or np.linalg.norm(candidate_target - current_target) > 0.4
                )
                if needs_new_path:
                    self.path = candidate_path
                    self.path_index = 0
                    self.path_replan_requested = False
                    self.get_logger().warn(
                        f'Obstacle at {nearest_front_range:.2f} m; following '
                        'replanned path.'
                    )
                return
        if not self.obstacle_stopping_enabled:
            self.get_logger().warn(
                f'Obstacle stopping disabled; continuing past lidar return '
                f'at {nearest_front_range:.2f} m.',
                throttle_duration_sec=2.0,
            )
            return
        self.obstacle_detected = True
        self.goal = None
        self.stop()
        event = Bool()
        event.data = True
        self.obstacle_stop_publisher.publish(event)
        self.get_logger().error(
            f'Collision point reached: nearest front lidar return is '
            f'{nearest_front_range:.2f} m (limit {self.obstacle_distance:.2f} m); '
            'movement cancelled.'
        )

    def on_odom(self, message: Odometry) -> None:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        values = (
            position.x,
            position.y,
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or sum(value * value for value in values[2:]) < 1e-12
        ):
            self.odom = None
            return
        self.odom = message
        self.odom_heading = quaternion_yaw(orientation)
        self.odom_received = time.monotonic()
        self.calibrate_odom_world_offset()

    def calibrate_odom_world_offset(self) -> None:
        if self.gps_position is None or self.odom is None:
            return
        position = self.odom.pose.pose.position
        measured_offset = np.array((
            self.gps_position[0] - position.x,
            self.gps_position[1] - position.y,
        ), dtype=np.float32)
        if self.odom_world_offset is None:
            self.odom_world_offset = measured_offset
            self.get_logger().info(
                'Calibrated odometry to world offset: '
                f'({self.odom_world_offset[0]:.3f}, '
                f'{self.odom_world_offset[1]:.3f}) m'
            )
        else:
            alpha = self.odom_calibration_alpha
            self.odom_world_offset = (
                (1.0 - alpha) * self.odom_world_offset
                + alpha * measured_offset
            )

    def odom_world_position(self) -> np.ndarray | None:
        if self.odom is None or self.odom_world_offset is None:
            return None
        position = self.odom.pose.pose.position
        return self.odom_world_offset + np.array(
            [position.x, position.y],
            dtype=np.float32,
        )

    def on_gps(self, message: NavSatFix) -> None:
        if not all(math.isfinite(value) for value in (
            message.latitude,
            message.longitude,
        )):
            return
        latitude = math.radians(message.latitude)
        longitude = math.radians(message.longitude)
        origin_latitude = math.radians(self.gps_origin_latitude)
        origin_longitude = math.radians(self.gps_origin_longitude)
        east = (longitude - origin_longitude) * EARTH_RADIUS_METERS \
            * math.cos(origin_latitude)
        north = (latitude - origin_latitude) * EARTH_RADIUS_METERS
        cos_heading = math.cos(self.gps_origin_heading)
        sin_heading = math.sin(self.gps_origin_heading)
        gps_position = np.array((
            east * cos_heading + north * sin_heading,
            -east * sin_heading + north * cos_heading,
        ), dtype=np.float32)
        if self.gps_heading_reference is None:
            self.gps_heading_reference = gps_position.copy()
        else:
            displacement = gps_position - self.gps_heading_reference
            if np.linalg.norm(displacement) >= 0.05:
                self.gps_heading = math.atan2(
                    float(displacement[1]),
                    float(displacement[0]),
                )
                self.gps_heading_received = True
                self.gps_heading_reference = gps_position.copy()
        self.previous_gps_position = gps_position
        self.gps_position = gps_position
        self.gps_received = time.monotonic()
        self.calibrate_odom_world_offset()

    def on_mission_command(self, message: String) -> None:
        if message.data == 'EMERGENCY_STOP':
            self.estop_active = True
            self.goal = None
            self.stop()
            self.get_logger().warn('EMERGENCY STOP ACTIVE')

        elif message.data == 'EMERGENCY_STOP_RESET':
            self.estop_active = False
            self.get_logger().info('Emergency stop reset.')


    def on_goal(self, message: PointStamped) -> None:
        if self.estop_active:
            self.get_logger().warn('Rejected goal: emergency stop is active.')
            return

        if not all(math.isfinite(value) for value in (message.point.x, message.point.y)):
            self.get_logger().error('Rejected goal: X and Y must be finite.')
            return
        if (
            self.odom is None
            or time.monotonic() - self.odom_received > self.odom_timeout
            or self.gps_position is None
            or time.monotonic() - self.gps_received > self.gps_timeout
        ):
            self.get_logger().error('Rejected goal: waiting for fresh GPS and odometry.')
            return
        if not matching_frame(message.header.frame_id, self.odom.header.frame_id):
            self.get_logger().error(
                f'Goal frame {message.header.frame_id!r} does not match '
                f'odometry frame {self.odom.header.frame_id!r}.'
            )
            return
        duplicate_goal = (
            self.goal is not None
            and math.hypot(
                self.goal.point.x - message.point.x,
                self.goal.point.y - message.point.y,
            ) <= 0.05
        )
        if duplicate_goal:
            return
        self.goal = message
        self.goal_started = time.monotonic()
        self.best_goal_distance = math.inf
        self.last_progress_time = self.goal_started
        self.obstacle_detected = False
        self.path = np.empty((0, 2), dtype=np.float32)
        self.path_index = 0
        self.path_replan_requested = True
        clear_event = Bool()
        clear_event.data = False
        self.obstacle_stop_publisher.publish(clear_event)
        self.get_logger().info(
            f'Moving to XY ({message.point.x}, {message.point.y}); Z ignored.'
        )

    def stop(self) -> None:
        self.publisher.publish(Twist())

    def tick(self) -> None:
        if self.estop_active:
            self.stop()
            return

        if self.goal is None:
            self.stop()
            return

        now = time.monotonic()

        if self.odom is None or self.gps_position is None:
            self.stop()
            return
        if not matching_frame(self.odom.header.frame_id, self.goal.header.frame_id):
            self.stop()
            return

        pose = self.odom.pose.pose
        target = self.goal.point
        gps_position = self.gps_position
        odom_position = self.odom_world_position()
        if odom_position is None:
            self.stop()
            return
        goal = np.array([target.x, target.y], dtype=np.float32)
        distance = float(np.linalg.norm(goal - gps_position))
        if distance <= self.position_tolerance:
            self.get_logger().info(f'Arrived: {distance:.3f} m from target.')
            self.goal = None
            self.stop()
            return

        if not self.scan_received:
            self.get_logger().warn(
                'Lidar unavailable; obstacle stopping is disabled, '
                'so following a direct GPS path.',
                throttle_duration_sec=2.0,
            )
            self.path = np.array([
                [gps_position[0], gps_position[1]],
                [goal[0], goal[1]],
            ], dtype=np.float32)
            self.path_index = 0
            self.path_replan_requested = False
        if self.path_replan_requested or not self.path.size:
            self.path = plan_path(
                gps_position[0],
                gps_position[1],
                goal,
                self.scan_obstacle_points,
                resolution=self.path_resolution,
                robot_clearance=self.robot_clearance,
            )
            self.path_index = 0
            self.path_replan_requested = False
            if not self.path.size:
                if not self.obstacle_stopping_enabled:
                    self.path = np.array([
                        [gps_position[0], gps_position[1]],
                        [goal[0], goal[1]],
                    ], dtype=np.float32)
                    self.path_index = 0
                    self.get_logger().warn(
                        'No collision-free path found; obstacle stopping is '
                        'disabled, so continuing directly to the goal.',
                        throttle_duration_sec=2.0,
                    )
                else:
                    self.get_logger().error('No collision-free path to goal; movement cancelled.')
                    self.goal = None
                    self.stop()
                    return

        while self.path_index < len(self.path) - 1:
            if np.linalg.norm(self.path[self.path_index] - odom_position) <= 0.35:
                self.path_index += 1
            else:
                break
        path_target = self.path[self.path_index]
        dx = float(path_target[0] - odom_position[0])
        dy = float(path_target[1] - odom_position[1])

        heading = self.odom_heading
        heading_error = wrap_angle(math.atan2(dy, dx) - heading)
        command = Twist()
        command.angular.z = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, self.angular_gain * heading_error),
        )
        forward_speed = min(self.max_speed, self.linear_gain * distance)
        command.linear.x = forward_speed * max(
            0.6,
            1.0 - abs(heading_error) / math.pi,
        )
        self.publisher.publish(command)


def main() -> None:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = HuskyMotion()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop()
            time.sleep(0.1)
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
