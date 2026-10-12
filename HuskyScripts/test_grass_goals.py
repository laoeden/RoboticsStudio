#!/usr/bin/env python3
"""Manually send saved grass pins to the Husky; never starts automatically.

Commands: list, reload, select ID, go, done, quit.
Use this instead of other goal senders during the test. 'done' only unlocks
the next test goal; it does not stop the robot or mark a patch cleared.
"""

import argparse
import json
import math
from pathlib import Path
import select
import sys
import time

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point


def read_pins(path):
    data = json.loads(path.read_text())
    if data.get('version') != 1:
        raise ValueError('Expected pinpoint JSON version 1')
    pins = {}
    for pin in data['pins']:
        if not pin['frame_id'] or not all(
            math.isfinite(float(pin[key])) for key in ('x', 'y', 'z')
        ):
            raise ValueError('Invalid pin coordinates or frame')
        identifier = int(pin['id'])
        if identifier in pins:
            raise ValueError(f'Duplicate pin ID {identifier}')
        pins[identifier] = pin
    return pins


class GrassGoalTest(Node):
    def __init__(self, args):
        super().__init__('manual_grass_goal_test')
        self.args = args
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.publisher = self.create_publisher(PointStamped, '/husky1/goal', 10)
        self.odom = None
        self.received = 0.0
        self.create_subscription(Odometry, '/husky1/odometry', self.on_odom,
                                 qos_profile_sensor_data)

    def on_odom(self, message):
        self.odom = message
        self.received = time.monotonic()

    def goal_for(self, pin):
        if self.odom is None or time.monotonic() - self.received > 2.0:
            raise ValueError('Waiting for fresh Husky odometry')
        target = self.odom.header.frame_id
        if not target:
            raise ValueError('Husky odometry has no frame ID')
        point = PointStamped()
        point.header.frame_id = pin['frame_id']
        point.point.x, point.point.y, point.point.z = (
            float(pin[key]) for key in ('x', 'y', 'z'))
        if point.header.frame_id != target:
            if self.args.shared_axes:
                # Explicit user assertion: source axes equal target axes,
                # apart from this translation. Never infer this from names.
                point.point.x += self.args.offset_x
                point.point.y += self.args.offset_y
                point.header.frame_id = target
            else:
                transform = self.buffer.lookup_transform(
                    target, point.header.frame_id, Time())
                point = do_transform_point(point, transform)
        point.header.stamp = self.get_clock().now().to_msg()
        point.point.z = 0.0  # HuskyMotion uses XY only.
        return point


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pins', type=Path, default=
                        Path(__file__).resolve().parents[1] / 'UI/dry_grass_pinpoints.json')
    parser.add_argument('--shared-axes', action='store_true', help=
                        'Explicitly assert drone and Husky odometry axes align; bypass TF')
    parser.add_argument('--offset-x', type=float, default=0.0,
                        help='With --shared-axes: add to pin X to obtain Husky X')
    parser.add_argument('--offset-y', type=float, default=0.0,
                        help='With --shared-axes: add to pin Y to obtain Husky Y')
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.offset_x, args.offset_y)):
        parser.error('Offsets must be finite')
    if (args.offset_x or args.offset_y) and not args.shared_axes:
        parser.error('Offsets require --shared-axes')
    if not sys.stdin.isatty():
        parser.error('Run this script in an interactive terminal')
    rclpy.init(args=[])
    node = GrassGoalTest(args)
    pins, selected, active = {}, None, False
    print(__doc__)
    print('Goals on /husky1/goal replace the controller goal. Pause other goal senders.')
    if args.shared_axes:
        print('Using explicitly supplied shared axes and offsets instead of TF.')
    try:
        try:
            pins = read_pins(args.pins)
            print(f'Loaded {len(pins)} pins. Type list, then select ID.')
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f'Cannot load pins: {error}. Save detections, then type reload.')
        print('> ', end='', flush=True)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if not select.select([sys.stdin], [], [], 0)[0]:
                continue
            line = sys.stdin.readline()
            if not line:
                break
            words = line.strip().split()
            try:
                if words == ['quit']:
                    break
                elif words == ['list']:
                    for identifier, pin in pins.items():
                        print(f"{identifier}: ({pin['x']:.2f}, {pin['y']:.2f}) in {pin['frame_id']}")
                    if not pins:
                        print('No saved pins. Confirm grass in the UI, then reload.')
                elif words == ['reload']:
                    pins = read_pins(args.pins)
                    selected = None
                    print(f'Loaded {len(pins)} pins; selection cleared.')
                elif len(words) == 2 and words[0] == 'select':
                    selected = None
                    candidate = pins[int(words[1])]
                    goal = node.goal_for(candidate)
                    selected = candidate
                    print(f'Preview: ({goal.point.x:.2f}, {goal.point.y:.2f}) '
                          f'in {goal.header.frame_id}. Type go to send.')
                elif words == ['go']:
                    if active:
                        raise ValueError('A test goal was already sent; type done when ready for another')
                    if selected is None:
                        raise ValueError('Select a pin first')
                    if node.publisher.get_subscription_count() == 0:
                        raise ValueError('No Husky goal subscriber. Start HuskyMotion.py first')
                    goal = node.goal_for(selected)
                    node.publisher.publish(goal)
                    active = True
                    print(f"Published pin {selected['id']} at "
                          f'({goal.point.x:.2f}, {goal.point.y:.2f}). '
                          'Check controller logs for acceptance and arrival.')
                elif words == ['done']:
                    active, selected = False, None
                    print('Ready to select another pin. No robot command sent.')
                else:
                    print('Commands: list, reload, select ID, go, done, quit')
            except (OSError, ValueError, KeyError, TypeError, TransformException) as error:
                print(f'No goal sent: {error}')
            print('> ', end='', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print('\nClosing test sender; an already sent goal remains with HuskyMotion.')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
