#!/usr/bin/env python3
"""Move the simulated Husky to absolute XY coordinates in world space."""

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
            'linear_gain': 0.8,
            'max_speed': 1.5,
            'angular_gain': 2.0,
            'max_angular_speed': 1.0,
            'min_turn_speed': 0.8,
            'heading_tolerance': 0.15,
            'position_tolerance': 0.2,
            'odom_timeout': 5.0,
            'gps_timeout': 5.0,
            'gps_origin_latitude': 0.0,
            'gps_origin_longitude': 0.0,
            'gps_origin_heading': 0.0,
            'initial_heading': 0.0,
            'odom_to_world_heading': 0.0,
            'odom_calibration_alpha': 0.1,
            'goal_timeout': 120.0,
            'progress_timeout': 5.0,
            'minimum_progress': 0.05,
            'stuck_recovery_enabled': True,
            'stuck_recovery_duration': 1.5,
            'stuck_recovery_speed': 0.4,
            'stuck_recovery_turn_speed': 1.2,
            'max_stuck_recoveries': 2,
            'obstacle_stopping_enabled': True,
            'progress_stopping_enabled': False,
            'obstacle_distance': 1.0,
            'front_angle': math.pi / 4.0,
            'planning_range': 5.0,
            'path_resolution': 0.2,
            'robot_clearance': 1.0,
            # Lidar hits this close to the charging pad are ignored: the pad and its
            # marker are part of the dock, not obstacles to avoid.
            'dock_clear_radius': 1.6,
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
                elif name in (
                    'gps_origin_heading',
                    'initial_heading',
                    'odom_to_world_heading',
                ):
                    value = math.radians(value)
                elif name == 'odom_calibration_alpha':
                    if not 0.0 < value <= 1.0:
                        raise ValueError(f'{name} must be in (0, 1]')
                elif value <= 0:
                    raise ValueError(f'{name} must be positive')
            setattr(self, name, value)
        # Charging pad position in the husky1_odom frame (zero is valid here).
        self.declare_parameter('dock_x', 0.0)
        self.declare_parameter('dock_y', 0.0)
        self.dock_x = float(self.get_parameter('dock_x').value)
        self.dock_y = float(self.get_parameter('dock_y').value)

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
        self.recovery_until = 0.0
        self.recovery_turn_direction = 1.0
        self.recovery_attempts = 0
        self.last_command_linear = 0.0
        self.obstacle_detected = False
        self.estop_active = False
        self.return_wanted = False        # drive to the charging pad until docked
        self.last_return_attempt = 0.0
        self.low_return_started = False
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
        # Absolute topic: the UI publishes mission commands for every robot.
        self.create_subscription(String, '/mission_command', self.on_mission_command, 10)
        self.create_subscription(Bool, 'battery_low', self.on_battery_low, 10)
        self.create_subscription(
            String,
            'mission_command',
            self.on_mission_command,
            10,
        )
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
        self.get_logger().info('Ready: waiting for odometry and a goal.')

    def on_scan(self, message: LaserScan) -> None:
        ranges = np.asarray(message.ranges, dtype=np.float32)
        angles = message.angle_min + np.arange(
            len(message.ranges), dtype=np.float32
        ) * message.angle_increment
        finite = np.isfinite(ranges)
        valid = finite.copy()
        valid &= ranges >= message.range_min
        valid &= ranges <= min(message.range_max, self.planning_range)

        # Rays that hit within dock_clear_radius of the charging pad are ignored.
        in_dock = np.zeros(len(ranges), dtype=bool)
        if self.odom is not None and np.any(finite):
            dock_position = self.odom.pose.pose.position
            hit_x = dock_position.x + ranges[finite] * np.cos(
                angles[finite] + self.odom_heading
            )
            hit_y = dock_position.y + ranges[finite] * np.sin(
                angles[finite] + self.odom_heading
            )
            in_dock[finite] = (
                np.hypot(hit_x - self.dock_x, hit_y - self.dock_y)
                <= self.dock_clear_radius
            )
        valid &= ~in_dock

        if self.odom is not None and np.any(valid):
            pose = self.odom.pose.pose
            position = pose.position
            yaw = self.odom_heading
            local_x = ranges[valid] * np.cos(angles[valid])
            local_y = ranges[valid] * np.sin(angles[valid])
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            self.scan_obstacle_points = np.column_stack((
                position.x + cos_yaw * local_x - sin_yaw * local_y,
                position.y + sin_yaw * local_x + cos_yaw * local_y,
            )).astype(np.float32)
            self.scan_received = True
        elif self.odom is not None and np.any(finite):
            # Everything the lidar sees is part of the dock: no obstacles at all.
            self.scan_obstacle_points = np.empty((0, 2), dtype=np.float32)
            self.scan_received = True

        if self.goal is None:
            return
        front_ranges = []
        for index, distance in enumerate(message.ranges):
            if in_dock[index]:
                continue
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
        if pose is not None and self.odom is not None and self.goal is not None:
            position = pose.position
            candidate_path = plan_path(
                position.x,
                position.y,
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
        if message.data in ('CLEAR_GOAL', 'DELETE_GOAL'):
            self.goal = None
            self.path = np.empty((0, 2), dtype=np.float32)
            self.path_index = 0
            self.recovery_until = 0.0
            self.stop()
            self.get_logger().info('Active goal cleared.')
        elif message.data == 'EMERGENCY_STOP':
            self.estop_active = True
            self.goal = None
            self.stop()
            self.get_logger().warn('EMERGENCY STOP ACTIVE')

        elif message.data == 'EMERGENCY_STOP_RESET':
            self.estop_active = False
            self.get_logger().info('Emergency stop reset.')

        elif message.data == 'RETURN_TO_BASE':
            self.return_to_base('operator command')

        elif message.data == 'STOP':
            self.return_wanted = False
            self.goal = None
            self.stop()

    def on_battery_low(self, message: Bool) -> None:
        if not message.data:
            self.low_return_started = False  # recharged: re-arm for next time
        elif not self.low_return_started and self.odom is not None:
            # Latch only once odometry exists, so an early message is retried.
            self.low_return_started = True
            self.return_to_base('low battery')

    def return_to_base(self, reason: str) -> None:
        if self.odom is None:
            self.get_logger().error('Cannot return to base: no odometry yet.')
            return
        self.return_wanted = True
        self.get_logger().warn(f'Returning to charging station ({reason}).')
        self.send_return_goal()

    def send_return_goal(self) -> None:
        """Aim at the pad. tick() repeats this if the drive is cancelled or blocked."""
        self.last_return_attempt = time.monotonic()
        if self.odom is None:
            return
        goal = PointStamped()
        goal.header.frame_id = self.odom.header.frame_id
        goal.point.x = self.dock_x
        goal.point.y = self.dock_y
        self.goal = None  # a return overrides any goal in progress
        self.on_goal(goal, from_return=True)

    def on_goal(self, message: PointStamped, from_return: bool = False) -> None:
        if self.estop_active:
            self.get_logger().warn('Rejected goal: emergency stop is active.')
            return
        if self.return_wanted and not from_return:
            self.get_logger().warn('Rejected goal: returning to charging station.')
            return

        if not all(math.isfinite(value) for value in (message.point.x, message.point.y)):
            self.get_logger().error('Rejected goal: X and Y must be finite.')
            return
        if (
            self.odom is None
            or time.monotonic() - self.odom_received > self.odom_timeout
        ):
            self.get_logger().error('Rejected goal: waiting for fresh odometry.')
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
        self.recovery_until = 0.0
        self.recovery_turn_direction = 1.0
        self.recovery_attempts = 0
        self.last_command_linear = 0.0
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
        self.last_command_linear = 0.0

    def tick(self) -> None:
        if self.estop_active:
            self.stop()
            return

        if self.return_wanted and self.goal is None and self.odom is not None:
            to_dock = math.hypot(
                self.dock_x - self.odom.pose.pose.position.x,
                self.dock_y - self.odom.pose.pose.position.y,
            )
            if to_dock <= self.position_tolerance:
                self.get_logger().info('Docked at charging station.')
                self.return_wanted = False
            elif time.monotonic() - self.last_return_attempt >= 3.0:
                self.get_logger().warn(
                    f'Return to base not finished ({to_dock:.1f} m away); retrying.',
                    throttle_duration_sec=10.0,
                )
                self.send_return_goal()

        if self.goal is None:
            self.stop()
            return

        now = time.monotonic()

        if self.odom is None:
            self.stop()
            return
        if not matching_frame(self.odom.header.frame_id, self.goal.header.frame_id):
            self.stop()
            return

        pose = self.odom.pose.pose
        target = self.goal.point
        position = self.odom.pose.pose.position
        world_position = np.array([position.x, position.y], dtype=np.float32)
        goal = np.array([target.x, target.y], dtype=np.float32)
        distance = float(np.linalg.norm(goal - world_position))
        if distance <= self.position_tolerance:
            self.get_logger().info(f'Arrived: {distance:.3f} m from target.')
            if self.return_wanted:
                self.get_logger().info('Docked at charging station.')
                self.return_wanted = False
            self.goal = None
            self.stop()
            return

        if distance < self.best_goal_distance - self.minimum_progress:
            self.best_goal_distance = distance
            self.last_progress_time = now
            self.recovery_attempts = 0

        if self.last_command_linear <= 0.05:
            self.last_progress_time = now

        if self.recovery_until > now:
            recovery = Twist()
            recovery.linear.x = -self.stuck_recovery_speed
            recovery.angular.z = (
                self.recovery_turn_direction * self.stuck_recovery_turn_speed
            )
            self.publisher.publish(recovery)
            self.last_command_linear = recovery.linear.x
            return

        if (
            self.stuck_recovery_enabled
            and now - self.last_progress_time >= self.progress_timeout
        ):
            if self.recovery_attempts >= self.max_stuck_recoveries:
                self.get_logger().error(
                    'Husky did not make progress after recovery attempts; '
                    'cancelling the active goal.'
                )
                self.goal = None
                self.stop()
                return
            self.recovery_until = now + self.stuck_recovery_duration
            self.recovery_turn_direction *= -1.0
            self.recovery_attempts += 1
            self.last_progress_time = now
            self.get_logger().warn(
                'Husky appears stuck; reversing and turning to recover.'
            )
            recovery = Twist()
            recovery.linear.x = -self.stuck_recovery_speed
            recovery.angular.z = (
                self.recovery_turn_direction * self.stuck_recovery_turn_speed
            )
            self.publisher.publish(recovery)
            self.last_command_linear = recovery.linear.x
            return

        if not self.scan_received:
            self.get_logger().warn(
                'Lidar unavailable; following a direct odometry path until '
                'obstacle data is available.',
                throttle_duration_sec=2.0,
            )
            self.path = np.array([
                [world_position[0], world_position[1]],
                [goal[0], goal[1]],
            ], dtype=np.float32)
            self.path_index = 0
            self.path_replan_requested = False
        if self.path_replan_requested or not self.path.size:
            self.path = plan_path(
                world_position[0],
                world_position[1],
                goal,
                self.scan_obstacle_points,
                resolution=self.path_resolution,
                robot_clearance=self.robot_clearance,
            )
            self.path_index = 0
            self.path_replan_requested = False
            if not self.path.size:
                self.get_logger().error(
                    'No collision-free path to goal; movement cancelled.'
                )
                self.goal = None
                self.stop()
                return

        while self.path_index < len(self.path) - 1:
            if np.linalg.norm(
                self.path[self.path_index] - world_position
            ) <= 0.35:
                self.path_index += 1
            else:
                break
        path_target = self.path[self.path_index]
        dx = float(path_target[0] - world_position[0])
        dy = float(path_target[1] - world_position[1])

        heading = wrap_angle(self.odom_heading + self.odom_to_world_heading)
        heading_error = wrap_angle(math.atan2(dy, dx) - heading)
        command = Twist()
        requested_turn = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, self.angular_gain * heading_error),
        )
        if abs(heading_error) > self.heading_tolerance:
            command.angular.z = math.copysign(
                max(abs(requested_turn), self.min_turn_speed),
                heading_error,
            )
        forward_speed = min(self.max_speed, self.linear_gain * distance)
        # Rotate in place until the vehicle is facing the path target. Driving
        # while nearly reversed makes a differential-drive Husky trace a large
        # arc and can increase its distance from the goal.
        command.linear.x = (
            forward_speed
            if abs(heading_error) <= self.heading_tolerance
            else 0.0
        )
        self.publisher.publish(command)
        self.last_command_linear = command.linear.x


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