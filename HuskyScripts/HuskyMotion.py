#!/usr/bin/env python3
"""Move the simulated Husky to absolute XY coordinates in its odometry frame."""

import math
import time

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions


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
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(
            Odometry,
            'odometry',
            self.on_odom,
            qos_profile_sensor_data,
        )
        self.create_subscription(PointStamped, 'goal', self.on_goal, 10)
        self.create_timer(
            0.05,
            self.tick,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        self.get_logger().info('Ready: waiting for XY odometry and a goal.')

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

    def on_goal(self, message: PointStamped) -> None:
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
        self.get_logger().info(
            f'Moving to XY ({message.point.x}, {message.point.y}); Z ignored.'
        )

    def stop(self) -> None:
        self.publisher.publish(Twist())

    def tick(self) -> None:
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
        dx = target.x - pose.position.x
        dy = target.y - pose.position.y
        distance = math.hypot(dx, dy)
        if distance <= self.position_tolerance:
            self.get_logger().info(f'Arrived: {distance:.3f} m from target.')
            self.goal = None
            self.stop()
            return

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
