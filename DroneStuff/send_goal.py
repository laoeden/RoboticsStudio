#!/usr/bin/env python3
"""Send one coordinate to the running movement controller."""

import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for axis in ('x', 'y', 'z'):
        parser.add_argument(axis, type=float)
    parser.add_argument('--namespace', default='parrot1')
    parser.add_argument('--frame', default='parrot1_odom')
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.x, args.y, args.z)):
        parser.error('Coordinates must be finite.')
    rclpy.init(args=[])
    node = Node('send_drone_goal', namespace=args.namespace)
    try:
        publisher = node.create_publisher(PointStamped, 'goal', 10)
        deadline = time.monotonic() + 5.0
        while publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                parser.exit(1, 'No goal subscriber found. Start DroneMotion.py first.\n')
            rclpy.spin_once(node, timeout_sec=0.1)
        goal = PointStamped()
        goal.header.frame_id = args.frame
        goal.point.x, goal.point.y, goal.point.z = args.x, args.y, args.z
        publisher.publish(goal)
        # Allow reliable delivery before destroying the publisher.
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        print(f'Sent ({args.x}, {args.y}, {args.z}) in {args.frame}. '
              'Check the controller for acceptance and arrival.')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
