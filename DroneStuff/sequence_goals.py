#!/usr/bin/env python3
"""Send a named sequence of absolute goals to DroneMotion.py."""

import argparse
import math
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


@dataclass(frozen=True)
class Waypoint:
    name: str
    x: float
    y: float
    z: float


class GoalSequence(Node):
    def __init__(self, namespace: str, frame: str) -> None:
        super().__init__('sequence_goals', namespace=namespace)
        self.frame = frame
        self.odometry: Odometry | None = None
        self.publisher = self.create_publisher(PointStamped, 'goal', 10)
        self.create_subscription(
            Odometry,
            'odometry',
            self._on_odometry,
            qos_profile_sensor_data,
        )

    def _on_odometry(self, message: Odometry) -> None:
        self.odometry = message

    def wait_for_connections(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while self.publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    'No goal subscriber found. Start DroneMotion.py first.'
                )
            rclpy.spin_once(self, timeout_sec=0.1)

        while self.odometry is None:
            if time.monotonic() >= deadline:
                raise RuntimeError('No odometry received from the Parrot.')
            rclpy.spin_once(self, timeout_sec=0.1)

    def send_waypoint(
        self,
        waypoint: Waypoint,
        tolerance: float,
        timeout: float,
        publish_period: float,
    ) -> None:
        message = PointStamped()
        message.header.frame_id = self.frame
        message.point.x = waypoint.x
        message.point.y = waypoint.y
        message.point.z = waypoint.z

        self.get_logger().info(
            f'Starting {waypoint.name}: '
            f'({waypoint.x:.2f}, {waypoint.y:.2f}, {waypoint.z:.2f})'
        )
        deadline = time.monotonic() + timeout
        next_publish = 0.0
        while True:
            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError(f"Movement '{waypoint.name}' timed out.")

            if now >= next_publish:
                self.publisher.publish(message)
                next_publish = now + publish_period

            rclpy.spin_once(self, timeout_sec=0.05)
            odometry = self.odometry
            if odometry is None or odometry.header.frame_id != self.frame:
                continue

            position = odometry.pose.pose.position
            distance = math.sqrt(
                (position.x - waypoint.x) ** 2
                + (position.y - waypoint.y) ** 2
                + (position.z - waypoint.z) ** 2
            )
            if distance <= tolerance:
                self.get_logger().info(
                    f"Completed {waypoint.name}: {distance:.3f} m from goal."
                )
                return


def parse_waypoint(values: list[str]) -> Waypoint:
    name, x, y, z = values
    try:
        coordinates = tuple(float(value) for value in (x, y, z))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            'waypoint coordinates must be numbers'
        ) from error
    if not all(math.isfinite(value) for value in coordinates):
        raise argparse.ArgumentTypeError('waypoint coordinates must be finite')
    return Waypoint(name, *coordinates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--waypoint',
        action='append',
        nargs=4,
        metavar=('NAME', 'X', 'Y', 'Z'),
        required=True,
        help='named absolute goal; repeat for each movement in order',
    )
    parser.add_argument('--namespace', default='parrot1')
    parser.add_argument('--frame', default='parrot1_odom')
    parser.add_argument('--tolerance', type=float, default=0.15)
    parser.add_argument('--timeout', type=float, default=120.0)
    parser.add_argument('--connect-timeout', type=float, default=10.0)
    parser.add_argument('--publish-period', type=float, default=1.0)
    args = parser.parse_args()

    if any(value <= 0 for value in (
        args.tolerance,
        args.timeout,
        args.connect_timeout,
        args.publish_period,
    )):
        parser.error('timeouts, tolerance, and publish period must be positive')

    waypoints = [parse_waypoint(values) for values in args.waypoint]
    rclpy.init(args=[])
    node = GoalSequence(args.namespace, args.frame)
    try:
        node.wait_for_connections(args.connect_timeout)
        for waypoint in waypoints:
            node.send_waypoint(
                waypoint,
                args.tolerance,
                args.timeout,
                args.publish_period,
            )
        node.get_logger().info('Goal sequence completed.')
    except (KeyboardInterrupt, RuntimeError) as error:
        if str(error):
            node.get_logger().error(str(error))
        raise SystemExit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
