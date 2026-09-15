#!/usr/bin/env python3
"""Voxel-assisted frontier exploration through Nav2.

Run this from the workspace after starting the simulator, SLAM, and Nav2:

    python3 voxel_nav_explorer.py --robot husky1

The node reuses Lidar3DPlotter for scan/point-cloud filtering and accumulated
voxel storage. Nav2's OccupancyGrid remains the navigation authority; voxels
are used to reject goals close to recently observed 3D obstacles and to prefer
frontiers with more nearby voxel information.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Optional

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from lidar_3d_plot import Lidar3DPlotter


class VoxelNavExplorer(Node):
    def __init__(self, robot: str, goal_distance: float = 2.5, goal_timeout: float = 20.0) -> None:
        super().__init__('voxel_nav_explorer', namespace=robot)
        self.robot = robot.strip('/') or 'husky1'
        self.map_frame = f'{self.robot}_map'
        self.goal_distance = goal_distance
        self.goal_timeout = goal_timeout
        self.map_msg: Optional[OccupancyGrid] = None
        self.map_array: Optional[np.ndarray] = None
        self.goal_active = False
        self.goal_handle = None
        self.last_goal_xy: Optional[tuple[float, float]] = None
        self.next_goal_time = 0.0
        self.last_log_time = 0.0
        self.goal_started_at = 0.0

        # This is the perception/voxel implementation from lidar_3d_plot.py.
        # Absolute topics are supplied because this node is already namespaced.
        self.voxels = Lidar3DPlotter(
            scan_topic=f'/{self.robot}/scan',
            cloud_topic=f'/{self.robot}/camera/depth/points',
            odom_topic=f'/{self.robot}/odometry',
        )
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self._on_map, 1)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.timer = self.create_timer(1.0, self._tick)
        self.get_logger().info(f'Voxel Nav2 explorer started for {self.robot}')

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg
        self.map_array = np.asarray(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))

    def _tick(self) -> None:
        if self.goal_active:
            if time.monotonic() - self.goal_started_at > self.goal_timeout:
                self._cancel_stalled_goal()
            return
        if time.monotonic() < self.next_goal_time:
            return
        if not self.nav_client.server_is_ready():
            if not self.nav_client.wait_for_server(timeout_sec=0.1):
                self._log_waiting('Waiting for Nav2 navigate_to_pose...')
                return
        if self.map_msg is None or self.map_array is None:
            self._log_waiting('Waiting for SLAM/Nav2 map...')
            return

        scan_points = self.voxels.get_scan_points()
        cloud_points, _ = self.voxels.get_cloud()
        if scan_points.size or cloud_points.size:
            if scan_points.size and cloud_points.size:
                points = np.vstack((scan_points, cloud_points))
            else:
                points = scan_points if scan_points.size else cloud_points
            self.voxels.accumulate_voxels(points)

        goal = self._choose_frontier()
        if goal is None:
            self._log_waiting('No safe frontier found in the current map...')
            return
        self._send_goal(goal[0], goal[1])

    def _log_waiting(self, message: str) -> None:
        now = time.monotonic()
        if now - self.last_log_time > 5.0:
            self.last_log_time = now
            self.get_logger().info(message)

    def _choose_frontier(self) -> Optional[tuple[float, float]]:
        assert self.map_msg is not None
        assert self.map_array is not None
        grid = self.map_array
        free = (grid >= 0) & (grid <= 20)
        unknown = grid < 0
        if grid.shape[0] < 3 or grid.shape[1] < 3:
            return None

        # A frontier is a free cell adjacent to unknown space. Downsample it
        # before scoring so a large map does not create thousands of candidates.
        adjacent_unknown = np.zeros_like(unknown, dtype=bool)
        adjacent_unknown[1:, :] |= unknown[:-1, :]
        adjacent_unknown[:-1, :] |= unknown[1:, :]
        adjacent_unknown[:, 1:] |= unknown[:, :-1]
        adjacent_unknown[:, :-1] |= unknown[:, 1:]
        frontier = free & adjacent_unknown
        ys, xs = np.where(frontier)
        if xs.size == 0:
            return None

        stride = max(1, xs.size // 300)
        candidates: list[tuple[float, float, float]] = []
        for x_cell, y_cell in zip(xs[::stride], ys[::stride]):
            world_x = self.map_msg.info.origin.position.x + (x_cell + 0.5) * self.map_msg.info.resolution
            world_y = self.map_msg.info.origin.position.y + (y_cell + 0.5) * self.map_msg.info.resolution
            if self._near_obstacle_voxel(world_x, world_y):
                continue
            distance = self._distance_from_robot(world_x, world_y)
            if distance < self.goal_distance or distance > 12.0:
                continue
            local_unknown = unknown[max(0, y_cell - 2):y_cell + 3, max(0, x_cell - 2):x_cell + 3]
            frontier_score = float(np.count_nonzero(local_unknown))
            voxel_score = self._voxel_information_score(world_x, world_y)
            candidates.append((frontier_score + 0.25 * voxel_score - 0.15 * distance, world_x, world_y))

        if not candidates:
            return None
        _, world_x, world_y = max(candidates)
        return world_x, world_y

    def _distance_from_robot(self, world_x: float, world_y: float) -> float:
        dx = world_x - self.voxels._robot_x
        dy = world_y - self.voxels._robot_y
        return math.hypot(dx, dy)

    def _near_obstacle_voxel(self, world_x: float, world_y: float) -> bool:
        centers, counts, floor_mask = self.voxels.get_accumulated_voxels()
        if centers.size == 0:
            return False
        # get_accumulated_voxels returns robot-local coordinates. Convert the
        # candidate into that same frame before applying a 0.65 m clearance.
        dx = world_x - self.voxels._robot_x
        dy = world_y - self.voxels._robot_y
        c = math.cos(self.voxels._robot_yaw)
        s = math.sin(self.voxels._robot_yaw)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        observed = ~floor_mask & (counts >= 1.0)
        if not np.any(observed):
            return False
        distance = np.hypot(centers[observed, 0] - local_x, centers[observed, 1] - local_y)
        return bool(np.any(distance < 0.65))

    def _voxel_information_score(self, world_x: float, world_y: float) -> float:
        centers, _, _ = self.voxels.get_accumulated_voxels()
        if centers.size == 0:
            return 0.0
        dx = world_x - self.voxels._robot_x
        dy = world_y - self.voxels._robot_y
        c = math.cos(self.voxels._robot_yaw)
        s = math.sin(self.voxels._robot_yaw)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        distance = np.hypot(centers[:, 0] - local_x, centers[:, 1] - local_y)
        return float(np.count_nonzero(distance < 2.0))

    def _send_goal(self, world_x: float, world_y: float) -> None:
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = world_x
        goal.pose.pose.position.y = world_y
        goal.pose.pose.orientation.w = 1.0
        self.goal_active = True
        self.goal_started_at = time.monotonic()
        self.last_goal_xy = (world_x, world_y)
        future = self.nav_client.send_goal_async(goal, feedback_callback=self._on_feedback)
        future.add_done_callback(self._on_goal_response)
        self.get_logger().info(f'Exploring frontier at ({world_x:.2f}, {world_y:.2f})')

    def _cancel_stalled_goal(self) -> None:
        if self.goal_handle is not None:
            self.get_logger().warn(
                f'Goal exceeded {self.goal_timeout:.0f} seconds; cancelling and selecting another frontier'
            )
            self.goal_handle.cancel_goal_async()
        self.goal_active = False
        self.goal_handle = None
        self.next_goal_time = time.monotonic() + 1.0

    def _on_goal_response(self, future) -> None:
        self.goal_handle = future.result()
        if self.goal_handle is None or not self.goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected the exploration goal')
            self.goal_active = False
            self.next_goal_time = time.monotonic() + 2.0
            return
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_feedback(self, feedback_msg) -> None:
        distance = feedback_msg.feedback.distance_remaining
        if distance is not None and distance < 0.5:
            self.get_logger().debug(f'Approaching frontier, {distance:.2f} m remaining')

    def _on_result(self, future) -> None:
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Frontier reached; selecting the next one')
        else:
            self.get_logger().warn(f'Exploration goal ended with status {status}')
        self.goal_active = False
        self.next_goal_time = time.monotonic() + 1.0

    def destroy_node(self) -> bool:
        self.voxels.destroy_node()
        return super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser(description='Explore with Nav2 using lidar_3d_plot voxels')
    parser.add_argument('--robot', default='husky1', choices=('husky1', 'parrot1'))
    parser.add_argument('--goal-distance', type=float, default=2.5)
    parser.add_argument('--goal-timeout', type=float, default=20.0)
    args = parser.parse_args(remove_ros_args()[1:])

    rclpy.init()
    explorer = VoxelNavExplorer(args.robot, args.goal_distance, args.goal_timeout)
    executor = SingleThreadedExecutor()
    executor.add_node(explorer)
    executor.add_node(explorer.voxels)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.remove_node(explorer.voxels)
        executor.remove_node(explorer)
        explorer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
