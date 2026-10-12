#!/usr/bin/env python3
"""Keyboard teleoperation for the simulated four-wheel Husky."""

import argparse
import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class KeyboardTeleop(Node):
    def __init__(
        self,
        topic: str,
        linear_speed: float,
        angular_speed: float,
        repeat_timeout: float,
        ramp_rate: float,
    ) -> None:
        super().__init__('husky_keyboard_teleop')
        self.publisher = self.create_publisher(Twist, topic, 10)
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.repeat_timeout = repeat_timeout
        self.ramp_rate = ramp_rate
        self.target_linear = 0.0
        self.target_angular = 0.0
        self.linear = 0.0
        self.angular = 0.0
        self.last_key_time = 0.0
        self.last_update = time.monotonic()
        self.timer = self.create_timer(0.05, self.publish_command)

    def set_target(self, linear: float, angular: float) -> None:
        self.target_linear = linear
        self.target_angular = angular
        self.last_key_time = time.monotonic()

    def stop_target(self) -> None:
        self.set_target(0.0, 0.0)

    def publish_command(self) -> None:
        now = time.monotonic()
        if now - self.last_key_time > self.repeat_timeout:
            self.target_linear = 0.0
            self.target_angular = 0.0
        elapsed = min(now - self.last_update, 0.2)
        self.last_update = now
        max_change = self.ramp_rate * elapsed
        self.linear += max(
            -max_change,
            min(max_change, self.target_linear - self.linear),
        )
        self.angular += max(
            -max_change,
            min(max_change, self.target_angular - self.angular),
        )
        message = Twist()
        message.linear.x = self.linear
        message.angular.z = self.angular
        self.publisher.publish(message)

    def stop(self) -> None:
        self.target_linear = 0.0
        self.target_angular = 0.0
        self.linear = 0.0
        self.angular = 0.0
        self.publisher.publish(Twist())


def read_key() -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], 0.05)
    if not readable:
        return None
    first = sys.stdin.read(1)
    if first != '\x1b':
        return first.lower()
    if select.select([sys.stdin], [], [], 0.01)[0]:
        second = sys.stdin.read(1)
        if second == '[' and select.select([sys.stdin], [], [], 0.01)[0]:
            return {'A': 'w', 'B': 's', 'C': 'd', 'D': 'a'}.get(
                sys.stdin.read(1)
            )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='/husky1/cmd_vel')
    parser.add_argument('--linear-speed', type=float, default=0.5)
    parser.add_argument('--angular-speed', type=float, default=0.5)
    parser.add_argument(
        '--repeat-timeout',
        type=float,
        default=0.3,
        help='stop after this long without a repeated key (default: 0.3)',
    )
    parser.add_argument(
        '--ramp-rate',
        type=float,
        default=1.5,
        help='linear/angular command ramp rate per second (default: 1.5)',
    )
    args = parser.parse_args()
    if (
        args.linear_speed <= 0
        or args.angular_speed <= 0
        or args.repeat_timeout <= 0
        or args.ramp_rate <= 0
    ):
        parser.error('speed, repeat-timeout, and ramp-rate must be positive')

    print(
        'Husky keyboard teleop: W/S or Up/Down drive, '
        'A/D or Left/Right turn, Space stop, X/Q quit.'
    )
    print('Hold a key to move; releasing it stops automatically.')
    if not sys.stdin.isatty():
        raise SystemExit('Keyboard teleop requires an interactive terminal.')

    original_terminal = termios.tcgetattr(sys.stdin)
    rclpy.init(args=[])
    node = KeyboardTeleop(
        args.topic,
        args.linear_speed,
        args.angular_speed,
        args.repeat_timeout,
        args.ramp_rate,
    )
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            key = read_key()
            if key in ('x', 'q'):
                break
            if key in ('w', 's'):
                node.set_target(
                    args.linear_speed if key == 'w' else -args.linear_speed,
                    0.0,
                )
            elif key in ('a', 'd'):
                node.set_target(
                    0.0,
                    args.angular_speed if key == 'a' else -args.angular_speed,
                )
            elif key == ' ':
                node.stop_target()
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal)
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
