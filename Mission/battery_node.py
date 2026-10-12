#!/usr/bin/env python3
"""ROS 2 node that simulates the Husky battery.

Subscribes: /husky1/odometry (nav_msgs/Odometry), /husky1/cutting (std_msgs/Bool)
Publishes:  /husky1/battery      (std_msgs/Float32, percent)
            /husky1/battery_low  (std_msgs/Bool, latched low-battery flag)
            /husky1/charging     (std_msgs/Bool)

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Float32

from Mission.battery_model import BatteryConfig, BatteryModel


class BatteryNode(Node):
    def __init__(self) -> None:
        super().__init__('husky_battery', namespace='husky1')
        config = BatteryConfig()
        for name in vars(config):
            self.declare_parameter(name, getattr(config, name))
            setattr(config, name, float(self.get_parameter(name).value))
        initial = self.declare_parameter('initial_level', config.capacity).value
        self.model = BatteryModel(config, level=float(initial))
        self.last_stamp: float | None = None
        self.charging = False
        self.next_publish = 0.0

        self.level_pub = self.create_publisher(Float32, 'battery', 10)
        self.low_pub = self.create_publisher(Bool, 'battery_low', 10)
        self.charging_pub = self.create_publisher(Bool, 'charging', 10)
        self.create_subscription(Odometry, 'odometry', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(Bool, 'cutting', self.on_cutting, 10)
        self.get_logger().info(
            f'Battery model ready: {self.model.level:.0f}%, low at '
            f'{config.low_threshold:.0f}%, dock ({config.dock_x}, {config.dock_y}).'
        )

    def on_cutting(self, message: Bool) -> None:
        self.model.cutting = bool(message.data)

    def on_odom(self, message: Odometry) -> None:
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        dt = 0.0 if self.last_stamp is None else stamp - self.last_stamp
        self.last_stamp = stamp
        if dt < 0 or dt > 1.0:  # clock reset or long gap
            dt = 0.0
        was_low = self.model.low
        position = message.pose.pose.position
        before = self.model.level
        level = self.model.update(position.x, position.y, dt)
        self.charging = self.model.at_dock(position.x, position.y) and self.model.speed <= self.model.config.moving_speed and level < self.model.config.capacity

        # Odometry can arrive at ~1 kHz in sim time; publish status at 10 Hz.
        if stamp >= self.next_publish or stamp < self.next_publish - 1.0:
            self.next_publish = stamp + 0.1
            self.level_pub.publish(Float32(data=float(level)))
            self.low_pub.publish(Bool(data=self.model.low))
            self.charging_pub.publish(Bool(data=self.charging))
        if self.model.low and not was_low:
            self.get_logger().warn(f'LOW BATTERY: {level:.1f}%')
        if self.model.depleted:
            self.get_logger().error('Battery depleted.', throttle_duration_sec=5.0)


def main() -> None:
    rclpy.init()
    node = BatteryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
