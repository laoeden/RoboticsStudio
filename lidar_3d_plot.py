#!/usr/bin/env python3

from __future__ import annotations

import threading
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2


@dataclass
class ScanCache:
    points: np.ndarray
    stamp: float


@dataclass
class CloudCache:
    points: np.ndarray
    depth: np.ndarray
    stamp: float


class Lidar3DPlotter(Node):
    def __init__(
        self,
        scan_topic: str = '/husky1/scan',
        cloud_topic: str = '/husky1/camera/depth/points',
        max_cloud_points: int = 7000,
    ) -> None:
        super().__init__('lidar_3d_plotter')
        self._lock = threading.Lock()
        self._max_cloud_points = max_cloud_points
        self._scan_cache = ScanCache(points=np.empty((0, 3), dtype=np.float32), stamp=0.0)
        self._cloud_cache = CloudCache(
            points=np.empty((0, 3), dtype=np.float32),
            depth=np.empty((0,), dtype=np.float32),
            stamp=0.0,
        )
        self._scan_subscription = self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        self._cloud_subscription = self.create_subscription(PointCloud2, cloud_topic, self._on_cloud, 10)
        self.get_logger().info(f'Subscribed to {scan_topic}')
        self.get_logger().info(f'Subscribed to {cloud_topic}')

    def _on_scan(self, msg: LaserScan) -> None:
        angles = msg.angle_min + np.arange(len(msg.ranges), dtype=np.float32) * msg.angle_increment
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        valid = np.isfinite(ranges)
        valid &= ranges >= msg.range_min
        if msg.range_max > 0.0:
            valid &= ranges <= msg.range_max

        angles = angles[valid]
        ranges = ranges[valid]
        if ranges.size == 0:
            return

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        z = np.zeros_like(x)
        points = np.column_stack((x, y, z))

        with self._lock:
            self._scan_cache = ScanCache(points=points, stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

    def _on_cloud(self, msg: PointCloud2) -> None:
        raw_points = list(point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
        if not raw_points:
            return

        # read_points can return structured rows on some ROS builds; unpack explicitly.
        xyz = np.array([[point[0], point[1], point[2]] for point in raw_points], dtype=np.float32)
        if xyz.size == 0:
            return

        finite_xyz = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite_xyz]
        if xyz.size == 0:
            return

        if xyz.shape[0] > self._max_cloud_points:
            index = np.random.choice(xyz.shape[0], size=self._max_cloud_points, replace=False)
            xyz = xyz[index]

        # Heatmap by depth from the sensor origin.
        depth = np.linalg.norm(xyz, axis=1)
        finite_depth = np.isfinite(depth)
        xyz = xyz[finite_depth]
        depth = depth[finite_depth]
        if depth.size == 0:
            return

        with self._lock:
            self._cloud_cache = CloudCache(
                points=xyz,
                depth=depth,
                stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            )

    def get_scan_points(self) -> np.ndarray:
        with self._lock:
            return self._scan_cache.points.copy()

    def get_cloud(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._cloud_cache.points.copy(), self._cloud_cache.depth.copy()


def main() -> None:
    rclpy.init()
    node = Lidar3DPlotter()

    plt.ion()
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection='3d')
    axis.set_title('Husky Lidar + Depth Point Cloud (Heatmap)')
    axis.set_xlabel('X (m)')
    axis.set_ylabel('Y (m)')
    axis.set_zlabel('Z (m)')
    axis.set_zlim(-2.0, 2.0)

    lidar_scatter = axis.scatter([], [], [], s=5, c='white', alpha=0.8, label='Lidar scan')
    cloud_scatter = axis.scatter([], [], [], s=2, cmap='turbo', alpha=0.75, label='Depth cloud')
    colorbar = figure.colorbar(cloud_scatter, ax=axis, shrink=0.7, pad=0.1)
    colorbar.set_label('Depth (m)')
    axis.legend(loc='upper right')

    try:
        while plt.fignum_exists(figure.number):
            rclpy.spin_once(node, timeout_sec=0.01)
            scan_points = node.get_scan_points()
            cloud_points, cloud_depth = node.get_cloud()

            if cloud_points.size:
                cloud_scatter.remove()
                cloud_scatter = axis.scatter(
                    cloud_points[:, 0],
                    cloud_points[:, 1],
                    cloud_points[:, 2],
                    s=2,
                    c=cloud_depth,
                    cmap='turbo',
                    alpha=0.75,
                )
                colorbar.update_normal(cloud_scatter)

            if scan_points.size:
                lidar_scatter.remove()
                lidar_scatter = axis.scatter(
                    scan_points[:, 0],
                    scan_points[:, 1],
                    scan_points[:, 2],
                    s=5,
                    c='white',
                    alpha=0.8,
                )

            if cloud_points.size or scan_points.size:
                xy = cloud_points[:, :2] if cloud_points.size else scan_points[:, :2]
                finite_xy = np.isfinite(xy).all(axis=1)
                xy = xy[finite_xy]
                if xy.size:
                    span = max(5.0, float(np.max(np.abs(xy))))
                else:
                    span = 5.0
                axis.set_xlim(-span, span)
                axis.set_ylim(-span, span)
            plt.pause(0.02)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as exc:
        # Handles forced shutdown while blocked in spin_once.
        if 'context is not valid' not in str(exc):
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
