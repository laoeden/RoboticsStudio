#!/usr/bin/env python3
"""Visualize Husky lidar and an obstacle-aware path to its latest goal."""

import argparse
import threading

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

try:
    from grid_path_planner import plan_path
except ImportError:
    from HuskyScripts.grid_path_planner import plan_path


class LidarVisualizer(Node):
    def __init__(self, topic: str, goal_topic: str, odom_topic: str, max_range: float) -> None:
        super().__init__('husky_lidar_visualizer')
        self._lock = threading.Lock()
        self._max_range = max_range
        self._angles = np.empty(0, dtype=np.float32)
        self._ranges = np.empty(0, dtype=np.float32)
        self._goal: tuple[float, float] | None = None
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self._odom_received = False
        self._range_min = 0.2
        self._range_max = 40.0
        self._scan_received = False
        self.create_subscription(
            LaserScan,
            topic,
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            odom_topic,
            self._on_odom,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointStamped,
            goal_topic,
            self._on_goal,
            10,
        )
        self.get_logger().info(f'Subscribed to {topic}, {odom_topic}, and {goal_topic}')

    def _on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        q = pose.orientation
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        with self._lock:
            self._robot_x = float(pose.position.x)
            self._robot_y = float(pose.position.y)
            self._robot_yaw = float(yaw)
            self._odom_received = True

    def _on_goal(self, message: PointStamped) -> None:
        goal = (float(message.point.x), float(message.point.y))
        with self._lock:
            self._goal = goal

    def _on_scan(self, message: LaserScan) -> None:
        angles = message.angle_min + np.arange(
            len(message.ranges), dtype=np.float32
        ) * message.angle_increment
        ranges = np.asarray(message.ranges, dtype=np.float32)
        with self._lock:
            self._angles = angles
            self._ranges = ranges
            self._range_min = float(message.range_min)
            self._range_max = float(message.range_max)
            self._scan_received = True

    def snapshot(self):
        with self._lock:
            return (
                self._angles.copy(),
                self._ranges.copy(),
                self._range_min,
                self._range_max,
                self._scan_received,
                None if self._goal is None else np.asarray(self._goal, dtype=np.float32),
                self._robot_x,
                self._robot_y,
                self._robot_yaw,
                self._odom_received,
            )


def update_plot(
    node: LidarVisualizer,
    points,
    front_points,
    status,
    scan_points,
    goal_points,
    path_points,
    robot_point,
    robot_heading,
    map_axis,
    max_range: float,
    collision_distance: float,
    front_angle: float,
) -> None:
    (
        angles, ranges, range_min, range_max, received, goal,
        robot_x, robot_y, robot_yaw, odom_received,
    ) = node.snapshot()
    if not received:
        status.set_text('Waiting for /husky1/scan...')
        return

    valid = np.isfinite(ranges)
    valid &= ranges >= range_min
    valid &= ranges <= min(range_max, max_range)
    front = valid & (np.abs((angles + np.pi) % (2.0 * np.pi) - np.pi) <= front_angle)
    collision = front & (ranges <= collision_distance)

    points.set_data(angles[valid], ranges[valid])
    front_points.set_data(angles[front], ranges[front])
    status.set_text(
        f'valid rays: {int(np.count_nonzero(valid))} | '
        f'nearest front: '
        f'{np.min(ranges[front]):.2f} m' if np.any(front)
        else 'valid rays: 0 | nearest front: none'
    )
    if np.any(collision):
        status.set_color('red')
        status.set_text(status.get_text() + ' | COLLISION ZONE')
    else:
        status.set_color('black')

    if odom_received and np.any(valid):
        local_x = ranges[valid] * np.cos(angles[valid])
        local_y = ranges[valid] * np.sin(angles[valid])
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        world_points = np.column_stack((
            robot_x + cos_yaw * local_x - sin_yaw * local_y,
            robot_y + sin_yaw * local_x + cos_yaw * local_y,
        ))
    else:
        world_points = np.empty((0, 2), dtype=np.float32)
    if world_points.size:
        scan_points.set_data(world_points[:, 0], world_points[:, 1])
    else:
        scan_points.set_data([], [])
    if goal is not None:
        goal_points.set_data([goal[0]], [goal[1]])
        path = plan_path(robot_x, robot_y, goal, world_points)
        if path.size:
            path_points.set_data(path[:, 0], path[:, 1])
            status.set_text(status.get_text() + ' | path planned')
        else:
            path_points.set_data([], [])
            status.set_text(status.get_text() + ' | no collision-free path')
    else:
        goal_points.set_data([], [])
        path_points.set_data([], [])
    robot_point.set_data([robot_x], [robot_y])
    robot_heading.set_data(
        [robot_x, robot_x + np.cos(robot_yaw)],
        [robot_y, robot_y + np.sin(robot_yaw)],
    )
    map_axis.relim()
    map_axis.autoscale_view()
    if not odom_received:
        status.set_text(status.get_text() + ' | waiting for odometry')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='/husky1/scan')
    parser.add_argument('--goal-topic', default='/husky1/goal')
    parser.add_argument('--odom-topic', default='/husky1/odometry')
    parser.add_argument('--max-range', type=float, default=15.0)
    parser.add_argument('--collision-distance', type=float, default=1.0)
    parser.add_argument('--front-angle-degrees', type=float, default=45.0)
    args = parser.parse_args()
    if args.max_range <= 0 or args.collision_distance <= 0:
        parser.error('max-range and collision-distance must be positive')
    if not 0 < args.front_angle_degrees <= 180:
        parser.error('front-angle-degrees must be in (0, 180]')

    rclpy.init(args=[])
    node = LidarVisualizer(args.topic, args.goal_topic, args.odom_topic, args.max_range)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    figure = plt.figure(figsize=(13, 6))
    axis = figure.add_subplot(121, projection='polar')
    map_axis = figure.add_subplot(122)
    axis.set_theta_zero_location('E')
    axis.set_theta_direction(1)
    axis.set_title(
        f'Husky 360 lidar | front +/-{args.front_angle_degrees:.0f} deg | '
        f'collision distance {args.collision_distance:.1f} m'
    )
    axis.set_rmax(args.max_range)
    axis.set_rlabel_position(135)

    front_angle = np.deg2rad(args.front_angle_degrees)
    sector_angles = np.linspace(-front_angle, front_angle, 100)
    axis.fill_between(
        sector_angles,
        0.0,
        args.collision_distance,
        color='red',
        alpha=0.18,
        label='collision zone',
    )
    axis.plot(
        sector_angles,
        np.full_like(sector_angles, args.collision_distance),
        color='red',
        linewidth=1.5,
        label=f'{args.collision_distance:.1f} m boundary',
    )
    axis.plot([0.0, 0.0], [0.0, args.max_range], color='black', linewidth=1)
    points, = axis.plot([], [], '.', color='steelblue', markersize=3, label='lidar')
    front_points, = axis.plot([], [], '.', color='darkorange', markersize=5, label='front sector')
    map_axis.set_title('Current lidar obstacles and planned path')
    map_axis.set_xlabel('world X (m)')
    map_axis.set_ylabel('world Y (m)')
    map_axis.set_aspect('equal', adjustable='datalim')
    map_axis.grid(True, alpha=0.3)
    map_points, = map_axis.plot([], [], '.', color='slateblue', markersize=4, label='current lidar')
    goal_points, = map_axis.plot([], [], 'x', color='crimson', markersize=10, label='latest goal')
    path_points, = map_axis.plot([], [], '-', color='seagreen', linewidth=2, label='planned path')
    robot_point, = map_axis.plot([], [], 'o', color='black', markersize=7, label='Husky')
    robot_heading, = map_axis.plot([], [], color='black', linewidth=2)
    status = figure.text(0.02, 0.02, 'Waiting for scan...', color='black')
    axis.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    map_axis.legend(loc='upper right')

    try:
        while plt.fignum_exists(figure.number) and rclpy.ok():
            update_plot(
                node,
                points,
                front_points,
                status,
                map_points,
                goal_points,
                path_points,
                robot_point,
                robot_heading,
                map_axis,
                args.max_range,
                args.collision_distance,
                front_angle,
            )
            figure.canvas.draw_idle()
            plt.pause(0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        plt.close(figure)


if __name__ == '__main__':
    main()
