#!/usr/bin/env python3
"""Move the simulated Husky to absolute XY coordinates in its odometry frame."""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
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
    x, y, z, w = quaternion
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class HuskyMotion(Node):
    def __init__(self) -> None:
        super().__init__('husky_motion', namespace='husky1')
        defaults = {
            'linear_gain': 0.8,
            'max_speed': 1.0,
            'angular_gain': 1.5,
            'max_angular_speed': 1.2,
            'heading_tolerance': 0.15,
            'position_tolerance': 0.2,
            'odom_timeout': 5.0,
            'goal_timeout': 120.0,
            'obstacle_distance': 1.0,
            'front_angle': math.pi / 4.0,
            'planning_range': 5.0,
            'path_resolution': 0.2,
            'robot_clearance': 0.45,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            value = float(self.get_parameter(name).value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
            setattr(self, name, value)

        self.odom = None
        self.odom_received = 0.0
        self.goal = None
        self.goal_started = 0.0
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
        self.get_logger().info('Ready: waiting for XY odometry and a goal.')

    def on_scan(self, message: LaserScan) -> None:
        ranges = np.asarray(message.ranges, dtype=np.float32)
        angles = message.angle_min + np.arange(
            len(message.ranges), dtype=np.float32
        ) * message.angle_increment
        valid = np.isfinite(ranges)
        valid &= ranges >= message.range_min
        valid &= ranges <= min(message.range_max, self.planning_range)
        if self.odom is not None and np.any(valid):
            pose = self.odom.pose.pose
            yaw = quaternion_yaw((
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ))
            local_x = ranges[valid] * np.cos(angles[valid])
            local_y = ranges[valid] * np.sin(angles[valid])
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            self.scan_obstacle_points = np.column_stack((
                pose.position.x + cos_yaw * local_x - sin_yaw * local_y,
                pose.position.y + sin_yaw * local_x + cos_yaw * local_y,
            )).astype(np.float32)
            self.scan_received = True
            self.path_replan_requested = True

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
        if pose is not None and self.goal is not None:
            candidate_path = plan_path(
                pose.position.x,
                pose.position.y,
                np.array([self.goal.point.x, self.goal.point.y], dtype=np.float32),
                self.scan_obstacle_points,
                resolution=self.path_resolution,
                robot_clearance=self.robot_clearance,
            )
            if candidate_path.size:
                self.path = candidate_path
                self.path_index = 0
                self.path_replan_requested = False
                self.get_logger().warn(
                    f'Obstacle at {nearest_front_range:.2f} m; following '
                    'replanned path.'
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
        self.odom_received = time.monotonic()

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
        ):
            self.get_logger().error('Rejected goal: waiting for fresh odometry.')
            return
        if message.header.frame_id != self.odom.header.frame_id:
            self.get_logger().error(
                f'Goal frame {message.header.frame_id!r} does not match '
                f'odometry frame {self.odom.header.frame_id!r}.'
            )
            return
        self.goal = message
        self.goal_started = time.monotonic()
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

        if (
            self.odom is None
            or now - self.odom_received > self.odom_timeout
            or now - self.goal_started > self.goal_timeout
            or self.odom.header.frame_id != self.goal.header.frame_id
        ):
            self.get_logger().error(
                'Goal cancelled: odometry unavailable/changed or goal timed out.'
            )
            self.goal = None
            self.stop()
            return

        pose = self.odom.pose.pose
        target = self.goal.point
        position = np.array([pose.position.x, pose.position.y], dtype=np.float32)
        goal = np.array([target.x, target.y], dtype=np.float32)
        distance = float(np.linalg.norm(goal - position))
        if distance <= self.position_tolerance:
            self.get_logger().info(f'Arrived: {distance:.3f} m from target.')
            self.goal = None
            self.stop()
            return

        if not self.scan_received:
            self.get_logger().warn('Waiting for lidar before moving.', throttle_duration_sec=2.0)
            self.stop()
            return
        if self.path_replan_requested or not self.path.size:
            self.path = plan_path(
                pose.position.x,
                pose.position.y,
                goal,
                self.scan_obstacle_points,
                resolution=self.path_resolution,
                robot_clearance=self.robot_clearance,
            )
            self.path_index = 0
            self.path_replan_requested = False
            if not self.path.size:
                self.get_logger().error('No collision-free path to goal; movement cancelled.')
                self.goal = None
                self.stop()
                return

        while self.path_index < len(self.path) - 1:
            if np.linalg.norm(self.path[self.path_index] - position) <= 0.35:
                self.path_index += 1
            else:
                break
        path_target = self.path[self.path_index]
        dx = float(path_target[0] - position[0])
        dy = float(path_target[1] - position[1])

        heading = quaternion_yaw((
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ))
        heading_error = wrap_angle(math.atan2(dy, dx) - heading)
        command = Twist()
        command.angular.z = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, self.angular_gain * heading_error),
        )
        if abs(heading_error) <= self.heading_tolerance:
            command.linear.x = min(self.max_speed, self.linear_gain * distance)
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
