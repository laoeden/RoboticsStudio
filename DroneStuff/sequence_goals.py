#!/usr/bin/env python3
"""Send a named sequence of absolute goals to a robot motion controller."""

import argparse
import math
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix


EARTH_RADIUS_METERS = 6378137.0


def matching_frame(first: str, second: str) -> bool:
    if first == second:
        return True
    return {first, second} <= {'husky1/odom', 'husky1_odom'}


@dataclass(frozen=True)
class Waypoint:
    name: str
    x: float
    y: float
    z: float


class GoalSequence(Node):
    def __init__(self, robot: str, namespace: str, frame: str) -> None:
        super().__init__('sequence_goals', namespace=namespace)
        self.robot = robot
        self.frame = frame
        self.odometry: Odometry | None = None
        self.gps_position: tuple[float, float] | None = None
        self.publisher = self.create_publisher(PointStamped, 'goal', 10)
        self.create_subscription(
            Odometry,
            'odometry',
            self._on_odometry,
            qos_profile_sensor_data,
        )
        if robot == 'husky':
            self.create_subscription(
                NavSatFix,
                'gps/fix',
                self._on_gps,
                qos_profile_sensor_data,
            )

    def _on_odometry(self, message: Odometry) -> None:
        self.odometry = message

    def _on_gps(self, message: NavSatFix) -> None:
        if not all(math.isfinite(value) for value in (
            message.latitude,
            message.longitude,
        )):
            return
        self.gps_position = (
            math.radians(message.longitude) * EARTH_RADIUS_METERS,
            math.radians(message.latitude) * EARTH_RADIUS_METERS,
        )

    def wait_for_connections(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while self.publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f'No goal subscriber found. Start the {self.robot} motion '
                    'controller first.'
                )
            rclpy.spin_once(self, timeout_sec=0.1)

        while self.odometry is None:
            if time.monotonic() >= deadline:
                raise RuntimeError(f'No odometry received from the {self.robot}.')
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.robot == 'husky':
            while self.gps_position is None:
                if time.monotonic() >= deadline:
                    raise RuntimeError('No GPS received from the Husky.')
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

        target_text = f'({waypoint.x:.2f}, {waypoint.y:.2f}, {waypoint.z:.2f})'
        if self.robot == 'husky':
            target_text += ' [Z ignored]'
        self.get_logger().info(f'Starting {waypoint.name}: {target_text}')
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
            if odometry is None or not matching_frame(
                odometry.header.frame_id,
                self.frame,
            ):
                continue

            if self.robot == 'husky':
                if self.gps_position is None:
                    continue
                position_x, position_y = self.gps_position
            else:
                position = odometry.pose.pose.position
                position_x, position_y = position.x, position.y
            distance = math.hypot(
                position_x - waypoint.x,
                position_y - waypoint.y,
            )
            if self.robot == 'parrot':
                distance = math.sqrt(
                    distance ** 2 + (position.z - waypoint.z) ** 2
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
        '--robot',
        choices=('parrot', 'husky'),
        default='parrot',
        help='robot to control; Husky ignores waypoint Z (default: parrot)',
    )
    parser.add_argument(
        '--waypoint',
        action='append',
        nargs=4,
        metavar=('NAME', 'X', 'Y', 'Z'),
        required=True,
        help='named absolute goal; repeat for each movement in order',
    )
    parser.add_argument('--namespace')
    parser.add_argument('--frame')
    parser.add_argument(
        '--tolerance',
        type=float,
        help='arrival distance in metres (default: 0.15 Parrot, 0.20 Husky)',
    )
    parser.add_argument('--timeout', type=float, default=120.0)
    parser.add_argument('--connect-timeout', type=float, default=10.0)
    parser.add_argument('--publish-period', type=float, default=1.0)
    args = parser.parse_args()

    if any(value <= 0 for value in (
        args.timeout,
        args.connect_timeout,
        args.publish_period,
    )) or args.tolerance is not None and args.tolerance <= 0:
        parser.error('timeouts, tolerance, and publish period must be positive')

    waypoints = [parse_waypoint(values) for values in args.waypoint]
    default_namespace = {'parrot': 'parrot1', 'husky': 'husky1'}[args.robot]
    default_frame = {'parrot': 'parrot1_odom', 'husky': 'husky1/odom'}[args.robot]
    tolerance = args.tolerance
    if tolerance is None:
        tolerance = {'parrot': 0.15, 'husky': 0.2}[args.robot]
    namespace = args.namespace or default_namespace
    frame = args.frame or default_frame
    rclpy.init(args=[])
    node = GoalSequence(args.robot, namespace, frame)
    try:
        node.wait_for_connections(args.connect_timeout)
        for waypoint in waypoints:
            node.send_waypoint(
                waypoint,
                tolerance,
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
