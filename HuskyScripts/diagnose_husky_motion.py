#!/usr/bin/env python3
"""Measure Husky cmd_vel angular response without running the goal controller."""

import argparse
import math
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def unwrap_delta(values: np.ndarray) -> np.ndarray:
    return np.unwrap(values) - values[0]


class RotationTest(Node):
    def __init__(self, command_topic: str, odom_topic: str) -> None:
        super().__init__('diagnose_husky_motion')
        self.publisher = self.create_publisher(Twist, command_topic, 10)
        self.lock = threading.Lock()
        self.yaw = None
        self.x = None
        self.y = None
        self.samples = []
        self.create_subscription(
            Odometry, odom_topic, self.on_odom, qos_profile_sensor_data
        )

    def on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        with self.lock:
            self.yaw = yaw_from_quaternion(pose.orientation)
            self.x = pose.position.x
            self.y = pose.position.y
            self.samples.append(
                (time.monotonic(), self.yaw, self.x, self.y)
            )

    def command(self, angular: float) -> None:
        message = Twist()
        message.angular.z = angular
        self.publisher.publish(message)

    def stop(self) -> None:
        self.publisher.publish(Twist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command-topic', default='/husky1/cmd_vel')
    parser.add_argument('--odom-topic', default='/husky1/odometry')
    parser.add_argument('--angular-speed', type=float, default=0.5)
    parser.add_argument('--duration', type=float, default=4.0)
    parser.add_argument(
        '--settle',
        type=float,
        default=1.0,
        help='stationary time between positive and negative tests',
    )
    args = parser.parse_args()
    if args.angular_speed <= 0 or args.duration <= 0 or args.settle < 0:
        parser.error('angular-speed and duration must be positive; settle cannot be negative')

    rclpy.init(args=[])
    node = RotationTest(args.command_topic, args.odom_topic)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while node.yaw is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if node.yaw is None:
            raise RuntimeError('No odometry received.')

        for command in (args.angular_speed, -args.angular_speed):
            end = time.monotonic() + args.duration
            while time.monotonic() < end:
                node.command(command)
                time.sleep(0.05)
            node.stop()
            time.sleep(args.settle)
        node.stop()
    finally:
        node.stop()
        rclpy.shutdown()
        ros_thread.join(timeout=2.0)

    samples = np.asarray(node.samples, dtype=float)
    if samples.ndim != 2 or len(samples) < 2:
        raise RuntimeError('Not enough odometry samples for a diagnostic.')
    relative_time = samples[:, 0] - samples[0, 0]
    yaw_delta = unwrap_delta(samples[:, 1])
    movement = np.hypot(
        samples[:, 2] - samples[0, 2],
        samples[:, 3] - samples[0, 3],
    )
    print(
        f'Yaw change: {math.degrees(yaw_delta[-1]):+.1f} deg; '
        f'position displacement: {movement[-1]:.3f} m'
    )
    print(
        'Interpretation: positive cmd_vel.angular.z should produce positive '
        'odometry yaw change. If the sign is opposite, fix the simulator '
        'interface or invert the command before tuning gains.'
    )

    figure, (yaw_axis, position_axis) = plt.subplots(2, 1, figsize=(10, 6))
    yaw_axis.plot(relative_time, np.degrees(yaw_delta))
    yaw_axis.set_ylabel('yaw change (degrees)')
    yaw_axis.set_title('Husky rotation response')
    position_axis.plot(relative_time, movement)
    position_axis.set_ylabel('displacement (m)')
    position_axis.set_xlabel('time (s)')
    for axis in (yaw_axis, position_axis):
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
