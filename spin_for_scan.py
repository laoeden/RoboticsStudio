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
        area_size: float = 5.0,
        lane_spacing: float = 1.0,
        turn_duration: float = 3.14,
        safe_distance: float = 1.5,
        obstacle_reverse_duration: float = 1.4,
        obstacle_turn_duration: float = 3.14,
        obstacle_shift_duration: float = 2.0,
        max_total_time: float = 240.0,
    ):
        super().__init__('spin_for_scan')
        self.publisher_ = self.create_publisher(Twist, topic, 10)
        self._scan_subscription = self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.area_size = area_size
        self.lane_spacing = lane_spacing
        self.turn_duration = turn_duration
        self.safe_distance = safe_distance
        self.obstacle_reverse_duration = obstacle_reverse_duration
        self.obstacle_turn_duration = obstacle_turn_duration
        self.obstacle_shift_duration = obstacle_shift_duration
        self.max_total_time = max_total_time

        self._scan_ranges = []
        self._scan_angle_min = 0.0
        self._scan_angle_increment = 0.0
        self._scan_range_min = 0.0
        self._scan_range_max = float('inf')
        self._scan_received = False
        self._start_time = time.monotonic()
        self._phase = 'drive'
        self._phase_start = self._start_time
        self._lane_index = 0
        self._lane_count = max(1, int(math.floor(area_size / lane_spacing)) + 1)
        self._lane_turn_dir = 1.0
        self._avoid_turn_dir = 1.0
        self._resume_phase = 'drive'
        self._timer = self.create_timer(0.05, self._publish_motion)

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan_ranges = [float(v) for v in msg.ranges]
        self._scan_angle_min = float(msg.angle_min)
        self._scan_angle_increment = float(msg.angle_increment)
        self._scan_range_min = float(msg.range_min)
        self._scan_range_max = float(msg.range_max) if msg.range_max > 0.0 else float('inf')
        self._scan_received = True

    def _sector_clearance(self, minimum_angle: float, maximum_angle: float) -> float:
        if not self._scan_received or not self._scan_ranges:
            return float('inf')

        values = []
        for index, value in enumerate(self._scan_ranges):
            angle = self._scan_angle_min + index * self._scan_angle_increment
            if (
                minimum_angle <= angle <= maximum_angle
                and math.isfinite(value)
                and self._scan_range_min <= value <= self._scan_range_max
            ):
                values.append(value)
        return min(values) if values else float('inf')

    def _front_clearance(self) -> float:
        return self._sector_clearance(math.radians(-45.0), math.radians(45.0))

    def _side_clearance(self) -> tuple[float, float]:
        left = self._sector_clearance(math.radians(35.0), math.radians(145.0))
        right = self._sector_clearance(math.radians(-145.0), math.radians(-35.0))
        return left, right

    def _start_avoidance(self, now: float) -> None:
        left_clearance, right_clearance = self._side_clearance()
        self._avoid_turn_dir = 1.0 if left_clearance >= right_clearance else -1.0
        self._resume_phase = self._phase
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
        if not self._scan_received:
            self.publisher_.publish(msg)
            return

        front_clearance = self._front_clearance()

        if self._phase == 'avoid_reverse':
            msg.linear.x = -0.30
            if now - self._phase_start >= self.obstacle_reverse_duration:
                self._phase = 'avoid_turn_out'
                self._phase_start = now
        elif self._phase == 'avoid_turn_out':
            msg.angular.z = self.angular_speed * self._avoid_turn_dir
            if now - self._phase_start >= self.obstacle_turn_duration:
                self._phase = 'avoid_shift'
                self._phase_start = now
        elif self._phase == 'avoid_shift':
            msg.linear.x = self.linear_speed
            if now - self._phase_start >= self.obstacle_shift_duration:
                self._phase = 'avoid_turn_back'
                self._phase_start = now
        elif self._phase == 'avoid_turn_back':
            msg.angular.z = -self.angular_speed * self._avoid_turn_dir
            if now - self._phase_start >= self.obstacle_turn_duration:
                self._phase = self._resume_phase
                self._phase_start = now
                self.get_logger().info('Obstacle bypass complete; resuming lawn-mower coverage.')

        if self._phase.startswith('avoid_'):
            self.publisher_.publish(msg)
            return

        elif self._phase in {'drive', 'shift'} and front_clearance < self.safe_distance:
            self._start_avoidance(now)
            self.publisher_.publish(msg)
            return

        if front_clearance < self.safe_distance:
            self.publisher_.publish(msg)
            return

        phase_time = now - self._phase_start
        drive_duration = self.area_size / max(self.linear_speed, 0.05)
        shift_duration = self.lane_spacing / max(self.linear_speed, 0.05)

        if self._phase == 'drive':
            msg.linear.x = self.linear_speed
            if phase_time >= drive_duration:
                if self._lane_index >= self._lane_count - 1:
                    self._stop_and_exit(f'{self.area_size:g} m lawn-mower scan complete.')
                    return
                self._lane_turn_dir = 1.0 if self._lane_index % 2 == 0 else -1.0
                self._phase = 'turn_to_shift'
                self._phase_start = now
                self.get_logger().info(
                    f'Lane {self._lane_index + 1} complete; turning '
                    f'{"left" if self._lane_turn_dir > 0 else "right"} toward next strip.'
                )
        elif self._phase == 'turn_to_shift':
            msg.angular.z = self.angular_speed * self._lane_turn_dir
            if phase_time >= self.turn_duration:
                self._phase = 'shift'
                self._phase_start = now
        elif self._phase == 'shift':
            msg.linear.x = self.linear_speed
            if phase_time >= shift_duration:
                self._phase = 'turn_to_next_lane'
                self._phase_start = now
        elif self._phase == 'turn_to_next_lane':
            msg.angular.z = self.angular_speed * self._lane_turn_dir
            if phase_time >= self.turn_duration:
                self._lane_index += 1
                self._phase = 'drive'
                self._phase_start = now
                self.get_logger().info(f'Starting lawn-mower lane {self._lane_index + 1}.')

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
