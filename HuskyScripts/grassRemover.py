#!/usr/bin/env python3
"""Remove static dry-grass models near the Husky in the running simulation.

Run from RoboticsStudio: python3 HuskyScripts/grassRemover.py
The default world and odometry axes match the current zero-XY Husky spawn.
Restart this node after restarting the simulation (removed models are remembered).
"""

import math
from pathlib import Path
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def load_grass(world_file):
    """Read model names and world XY positions from top-level SDF includes."""
    world = ET.parse(world_file).getroot().find('world')
    if world is None:
        raise ValueError('SDF must contain a world')
    patches = {}
    for include in world.findall('include'):
        name = include.findtext('name', '')
        if not re.fullmatch(r'dry_grass(?:_\d+)?', name):
            continue
        pose = include.find('pose')
        if pose is None or pose.get('relative_to'):
            raise ValueError(f'{name} needs an explicit world-frame pose')
        values = [float(value) for value in pose.text.split()]
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise ValueError(f'Invalid pose for {name}')
        patches[name] = tuple(values[:2])
    if not patches:
        raise ValueError('No dry_grass models found in the world file')
    return world.attrib['name'], patches


class GrassRemover(Node):
    def __init__(self):
        super().__init__('grass_remover')
        defaults = {
            'world_file': str(Path(__file__).resolve().parents[1] / 'world/worldterrian.sdf'),
            'odom_topic': '/husky1/odometry',

            'removal_radius': 2.5, #radius adjustment


            # Transform odometry XY into SDF world XY, if the spawn changes.
            'world_offset_x': 0.0,
            'world_offset_y': 0.0,
            'world_offset_yaw': 0.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.world_name, self.patches = load_grass(self.get_parameter('world_file').value)
        self.radius = float(self.get_parameter('removal_radius').value)
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError('removal_radius must be finite and positive')
        self.offset_x = float(self.get_parameter('world_offset_x').value)
        self.offset_y = float(self.get_parameter('world_offset_y').value)
        self.offset_yaw = float(self.get_parameter('world_offset_yaw').value)
        if not all(math.isfinite(v) for v in (self.offset_x, self.offset_y, self.offset_yaw)):
            raise ValueError('World offsets must be finite')
        self.command = shutil.which('ign') or shutil.which('gz')
        if self.command is None:
            raise RuntimeError('Gazebo command-line tool (ign or gz) is required')
        self.msg_prefix = 'ignition.msgs' if Path(self.command).name == 'ign' else 'gz.msgs'
        self.position = None
        self.received_at = 0.0
        self.retry_after = {}
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value,
            self.on_odom, qos_profile_sensor_data,
        )
        self.create_timer(0.25, self.check_grass)
        self.get_logger().info(
            f'Watching {len(self.patches)} grass patches in {self.world_name}; '
            f'removal radius {self.radius} m'
        )

    def on_odom(self, message):
        p = message.pose.pose.position
        if not all(math.isfinite(v) for v in (p.x, p.y)):
            return
        c, s = math.cos(self.offset_yaw), math.sin(self.offset_yaw)
        self.position = (self.offset_x + c * p.x - s * p.y,
                         self.offset_y + s * p.x + c * p.y)
        self.received_at = time.monotonic()

    def check_grass(self):
        now = time.monotonic()
        if self.position is None or now - self.received_at > 2.0:
            self.get_logger().warning(
                'Waiting for fresh Husky odometry on '
                + self.get_parameter('odom_topic').value,
                throttle_duration_sec=10.0,
            )
            return
        x, y = self.position
        for name, (gx, gy) in list(self.patches.items()):
            if math.hypot(x - gx, y - gy) > self.radius:
                continue
            if now < self.retry_after.get(name, 0.0):
                continue
            # Execute at most one bounded service call per timer tick.
            try:
                result = subprocess.run(
                    [self.command, 'service', '-s', f'/world/{self.world_name}/remove',
                     '--reqtype', f'{self.msg_prefix}.Entity',
                     '--reptype', f'{self.msg_prefix}.Boolean', '--timeout', '1000',
                     '--req', f'name: "{name}", type: MODEL'],
                    capture_output=True, text=True, timeout=2.0,
                )
                success = result.returncode == 0 and re.search(r'\bdata:\s*true\b', result.stdout)
                detail = (result.stderr or result.stdout).strip()[:300]
            except (OSError, subprocess.TimeoutExpired) as error:
                success, detail = False, str(error)
            if success:
                del self.patches[name]
                self.retry_after.pop(name, None)
                self.get_logger().info(f'Removed {name}')
            else:
                self.retry_after[name] = time.monotonic() + 5.0
                self.get_logger().warning(f'Could not remove {name}; retrying in 5 s. {detail}')
            break


def main():
    rclpy.init()
    node = None
    try:
        node = GrassRemover()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
