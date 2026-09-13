#try building from the drone velocity controller which is given
#!/usr/bin/env python3
"""Move the simulated Parrot to absolute coordinates in its odometry frame."""

import math
import time

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions


def body_velocity(position, target, quaternion, gain, max_speed, tolerance):
    """Return body-frame velocity and distance; quaternion is (x, y, z, w)."""
    error = tuple(b - a for a, b in zip(position, target))
    distance = math.sqrt(sum(v * v for v in error))
    if distance <= tolerance:
        return (0.0, 0.0, 0.0), distance
    scale = min(gain, max_speed / distance)
    vx, vy, vz = (v * scale for v in error)
    norm = math.sqrt(sum(v * v for v in quaternion))
    x, y, z, w = (v / norm for v in quaternion)
    # Inverse rotation: odometry-frame velocity -> model-frame cmd_vel.
    return (
        (1 - 2*y*y - 2*z*z)*vx + 2*(x*y + z*w)*vy + 2*(x*z - y*w)*vz,
        2*(x*y - z*w)*vx + (1 - 2*x*x - 2*z*z)*vy + 2*(y*z + x*w)*vz,
        2*(x*z + y*w)*vx + 2*(y*z - x*w)*vy + (1 - 2*x*x - 2*y*y)*vz,
    ), distance


class DroneMotion(Node):
    def __init__(self):
        super().__init__('drone_motion', namespace='parrot1')
        defaults = {'gain': 0.8, 'max_speed': 1.0, 'tolerance': 0.15,
                    'odom_timeout': 1.0, 'goal_timeout': 120.0}
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
        self.create_subscription(Odometry, 'odometry', self.on_odom,
                                 qos_profile_sensor_data)
        self.create_subscription(PointStamped, 'goal', self.on_goal, 10)
        # Wall-clock watchdog still stops commands when simulation time pauses.
        self.create_timer(0.05, self.tick,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info('Ready: waiting for odometry and a goal.')

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if (not all(math.isfinite(v) for v in values)
                or sum(v*v for v in values[3:]) < 1e-12):
            self.odom = None
            return
        self.odom = msg
        self.odom_received = time.monotonic()

    def on_goal(self, msg):
        if not all(math.isfinite(v) for v in (msg.point.x, msg.point.y, msg.point.z)):
            self.get_logger().error('Rejected goal: coordinates must be finite.')
            return
        if self.odom is None or time.monotonic() - self.odom_received > self.odom_timeout:
            self.get_logger().error('Rejected goal: waiting for fresh odometry.')
            return
        if msg.header.frame_id != self.odom.header.frame_id:
            self.get_logger().error(f'Rejected goal: expected frame {self.odom.header.frame_id!r}.')
            return
        self.goal = msg
        self.goal_started = time.monotonic()
        self.get_logger().info(f'Moving to ({msg.point.x}, {msg.point.y}, {msg.point.z})')

    def stop(self):
        self.publisher.publish(Twist())

    def tick(self):
        if self.goal is None:
            self.stop()
            return
        now = time.monotonic()
        if (self.odom is None or now - self.odom_received > self.odom_timeout
                or now - self.goal_started > self.goal_timeout
                or self.odom.header.frame_id != self.goal.header.frame_id):
            self.get_logger().error('Goal cancelled: odometry unavailable/changed or goal timed out. Send a new goal.')
            self.goal = None
            self.stop()
            return
        p, q, t = self.odom.pose.pose.position, self.odom.pose.pose.orientation, self.goal.point
        velocity, distance = body_velocity(
            (p.x, p.y, p.z), (t.x, t.y, t.z), (q.x, q.y, q.z, q.w),
            self.gain, self.max_speed, self.tolerance)
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.linear.z = velocity
        self.publisher.publish(cmd)
        if distance <= self.tolerance:
            self.get_logger().info(f'Arrived: {distance:.3f} m from target.')
            self.goal = None


def main():
    # Keep ROS alive during Ctrl+C cleanup so a final stop can be published.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = DroneMotion()
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
