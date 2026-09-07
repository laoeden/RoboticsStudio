#!/usr/bin/env python3

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class SpinForScan(Node):
    def __init__(
        self,
        topic: str = '/husky1/cmd_vel',
        scan_topic: str = '/husky1/scan',
        linear_speed: float = 0.35,
        angular_speed: float = 1.0,
        side_length: float = 5.0,
        turn_duration: float = 1.57,
        safe_distance: float = 1.0,
        max_total_time: float = 60.0,
    ):
        super().__init__('spin_for_scan')
        self.publisher_ = self.create_publisher(Twist, topic, 10)
        self._scan_subscription = self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.side_length = side_length
        self.turn_duration = turn_duration
        self.safe_distance = safe_distance
        self.max_total_time = max_total_time

        self._scan_ranges = []
        self._start_time = time.monotonic()
        self._phase = 'drive'
        self._phase_start = self._start_time
        self._turn_count = 0
        self._drive_sign = 1.0
        self._turn_sign = 1.0
        self._reverse_distance = 0.5
        self._obstacle_stop_time = 0.0
        self._turn_after_obstacle = False
        self._avoid_turn_dir = 5.0
        self._timer = self.create_timer(0.05, self._publish_motion)

    def _on_scan(self, msg: LaserScan) -> None:
        ranges = [float(v) for v in msg.ranges if math.isfinite(v)]
        self._scan_ranges = ranges

    def _front_clearance(self) -> float:
        if not self._scan_ranges:
            return float('inf')

        center_index = len(self._scan_ranges) // 2
        window = self._scan_ranges[max(0, center_index - 15): min(len(self._scan_ranges), center_index + 16)]
        if not window:
            return float('inf')
        return min(window)

    def _side_clearance(self) -> tuple[float, float]:
        if not self._scan_ranges:
            return float('inf'), float('inf')

        n = len(self._scan_ranges)
        left_window = self._scan_ranges[: max(1, n // 3)]
        right_window = self._scan_ranges[min(n, 2 * n // 3):]
        left_clearance = min(left_window) if left_window else float('inf')
        right_clearance = min(right_window) if right_window else float('inf')
        return left_clearance, right_clearance

    def _start_avoidance(self, now: float) -> None:
        left_clearance, right_clearance = self._side_clearance()
        self._avoid_turn_dir = 1.0 if left_clearance >= right_clearance else -1.0
        self._phase = 'avoid_reverse'
        self._phase_start = now
        self.get_logger().warning(
            f'Obstacle ahead; reversing then turning {"left" if self._avoid_turn_dir > 0 else "right"}.'
        )

    def _publish_motion(self) -> None:
        now = time.monotonic()
        elapsed = now - self._start_time
        if elapsed > self.max_total_time:
            self._stop_and_exit('Scan routine complete.')
            return

        msg = Twist()
        front_clearance = self._front_clearance()

        if self._phase in {'avoid_reverse', 'avoid_turn'}:
            if self._phase == 'avoid_reverse':
                msg.linear.x = -0.25
                if now - self._phase_start >= 0.8:
                    self._phase = 'avoid_turn'
                    self._phase_start = now
            elif self._phase == 'avoid_turn':
                msg.angular.z = self.angular_speed * self._avoid_turn_dir
                if now - self._phase_start >= 1.1:
                    self._phase = 'drive'
                    self._phase_start = now
                    self.get_logger().info('Obstacle avoidance complete; resuming patrol.')
            self.publisher_.publish(msg)
            return

        if front_clearance < self.safe_distance:
            self._start_avoidance(now)
            self.publisher_.publish(msg)
            return

        phase_time = now - self._phase_start
        drive_duration = self.side_length / max(self.linear_speed, 0.05)
        reverse_duration = self._reverse_distance / max(self.linear_speed, 0.05)

        if self._phase == 'drive':
            msg.linear.x = self.linear_speed * self._drive_sign
            if phase_time >= drive_duration:
                self._phase = 'reverse'
                self._phase_start = now
                self.get_logger().info('Drive leg complete; reversing 0.5 m before turning.')

        elif self._phase == 'reverse':
            msg.linear.x = -self.linear_speed * 0.8
            if phase_time >= reverse_duration:
                self._phase = 'turn'
                self._phase_start = now
                self._turn_count += 1
                self.get_logger().info(f'Reverse complete {self._turn_count}; turning 90 degrees.')

        elif self._phase == 'turn':
            msg.angular.z = self.angular_speed * self._turn_sign
            if phase_time >= self.turn_duration:
                self._turn_sign *= -1.0
                self._drive_sign *= -1.0
                self._phase = 'drive'
                self._phase_start = now
                self.get_logger().info(f'Finished turn {self._turn_count}; starting next square leg.')

        self.publisher_.publish(msg)

    def _stop_and_exit(self, reason: str) -> None:
        stop = Twist()
        self.publisher_.publish(stop)
        self.destroy_timer(self._timer)
        self.get_logger().info(reason)
        rclpy.shutdown()


def main() -> None:
    rclpy.init()
    node = SpinForScan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
