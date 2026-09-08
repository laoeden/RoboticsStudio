#!/usr/bin/env python3

from __future__ import annotations

import threading
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nav_msgs.msg import Odometry
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
        odom_topic: str = '/husky1/odometry',
        max_cloud_points: int = 7000,
    ) -> None:
        super().__init__('lidar_3d_plotter')
        self._lock = threading.Lock()
        self._max_cloud_points = max_cloud_points
        self._voxel_size = 0.25
        self._max_local_radius = 5.0
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self._voxel_map: dict[tuple[int, int, int], int] = {}
        self._scan_cache = ScanCache(points=np.empty((0, 3), dtype=np.float32), stamp=0.0)
        self._cloud_cache = CloudCache(
            points=np.empty((0, 3), dtype=np.float32),
            depth=np.empty((0,), dtype=np.float32),
            stamp=0.0,
        )
        self._scan_subscription = self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        self._cloud_subscription = self.create_subscription(PointCloud2, cloud_topic, self._on_cloud, 10)
        self._odom_subscription = self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.get_logger().info(f'Subscribed to {scan_topic}')
        self.get_logger().info(f'Subscribed to {cloud_topic}')
        self.get_logger().info(f'Subscribed to {odom_topic}')

    def _on_odom(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        self._robot_x = float(pose.position.x)
        self._robot_y = float(pose.position.y)
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w
        self._robot_yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))

    def _transform_points_to_world(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points.copy()

        cos_yaw = np.cos(self._robot_yaw)
        sin_yaw = np.sin(self._robot_yaw)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        world_x = x * cos_yaw - y * sin_yaw + self._robot_x
        world_y = x * sin_yaw + y * cos_yaw + self._robot_y
        return np.column_stack((world_x, world_y, z)).astype(np.float32)

    def _transform_world_to_local(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points.copy()

        centered = points - np.array([self._robot_x, self._robot_y, 0.0], dtype=np.float32)
        cos_yaw = np.cos(-self._robot_yaw)
        sin_yaw = np.sin(-self._robot_yaw)
        x = centered[:, 0]
        y = centered[:, 1]
        local_x = x * cos_yaw - y * sin_yaw
        local_y = x * sin_yaw + y * cos_yaw
        return np.column_stack((local_x, local_y, centered[:, 2])).astype(np.float32)

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

    def accumulate_voxels(self, points: np.ndarray, voxel_size: float | None = None) -> None:
        if points.size == 0:
            return

        voxel_size = self._voxel_size if voxel_size is None else voxel_size
        finite = np.isfinite(points).all(axis=1)
        points = np.asarray(points[finite], dtype=np.float32)
        if points.size == 0:
            return

        world_points = self._transform_points_to_world(points)
        dist_to_robot = np.hypot(world_points[:, 0] - self._robot_x, world_points[:, 1] - self._robot_y)
        world_points = world_points[dist_to_robot <= self._max_local_radius]
        if world_points.size == 0:
            return

        voxel_coords = np.floor(world_points / voxel_size).astype(np.int32)
        unique_voxels, counts = np.unique(voxel_coords, axis=0, return_counts=True)

        for voxel, count in zip(unique_voxels, counts):
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            if count > self._voxel_map.get(key, 0):
                self._voxel_map[key] = int(count)

    def get_accumulated_voxels(self, voxel_size: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        voxel_size = self._voxel_size if voxel_size is None else voxel_size
        if not self._voxel_map:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=bool),
            )

        voxel_keys = np.asarray(list(self._voxel_map.keys()), dtype=np.int32)
        counts = np.asarray([self._voxel_map[tuple(key)] for key in voxel_keys], dtype=np.float32)
        voxel_centers = (voxel_keys.astype(np.float32) + 0.5) * voxel_size
        floor_mask = self._classify_floor_voxels(voxel_centers, counts, voxel_centers, voxel_size)
        local_voxels = self._transform_world_to_local(voxel_centers)
        return local_voxels, counts, floor_mask

    def voxelize_points(
        self,
        points: np.ndarray,
        voxel_size: float = 0.25,
        density_threshold: int = 1,
        max_voxels: int = 20000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if points.size == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=bool),
            )

        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if points.size == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=bool),
            )

        min_xyz = np.min(points, axis=0)
        voxel_coords = np.floor((points - min_xyz) / voxel_size).astype(np.int32)
        unique_voxels, counts = np.unique(voxel_coords, axis=0, return_counts=True)

        keep = counts >= density_threshold
        unique_voxels = unique_voxels[keep]
        counts = counts[keep]

        if unique_voxels.size == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=bool),
            )

        if unique_voxels.shape[0] > max_voxels:
            order = np.argsort(counts)[::-1][:max_voxels]
            unique_voxels = unique_voxels[order]
            counts = counts[order]

        centers = (unique_voxels.astype(np.float32) + 0.5) * voxel_size + min_xyz
        floor_mask = self._classify_floor_voxels(centers, counts, points, voxel_size)
        return centers, counts.astype(np.float32), floor_mask

    def _classify_floor_voxels(
        self,
        voxel_centers: np.ndarray,
        voxel_counts: np.ndarray,
        points: np.ndarray,
        voxel_size: float = 0.25,
        floor_quantile: float = 0.15,
        plane_std_limit: float = 0.05,
        floor_height_tolerance: float = 2.0,
        min_voxels_per_cell: int = 1,
    ) -> np.ndarray:
        if voxel_centers.size == 0:
            return np.empty((0,), dtype=bool)

        # In this scene, the floor is the horizontal plane in the Z axis.
        # Vertical walls are not flat in Z over a local XY cell, so reject anything
        # whose local Z variance or height above the floor band is too large.
        xy_cells = np.floor(voxel_centers[:, :2] / voxel_size).astype(np.int32)
        if points.shape[0] == 0:
            return np.zeros(voxel_centers.shape[0], dtype=bool)

        floor_reference = float(np.quantile(points[:, 2], floor_quantile))
        min_z = float(np.min(points[:, 2]))
        max_floor_height = floor_reference + floor_height_tolerance
        labels = np.zeros(voxel_centers.shape[0], dtype=bool)

        for x_idx, y_idx in np.unique(xy_cells, axis=0):
            mask = (xy_cells[:, 0] == x_idx) & (xy_cells[:, 1] == y_idx)
            if np.count_nonzero(mask) < min_voxels_per_cell:
                continue

            z_values = voxel_centers[mask, 2]
            z_mean = float(np.mean(z_values))
            z_std = float(np.std(z_values))
            z_min_local = float(np.min(z_values))
            z_max_local = float(np.max(z_values))

            # Treat only low, flat voxels in the lower Z band as floor.
            if (
                z_std <= plane_std_limit
                and z_mean <= max_floor_height
                and z_mean <= floor_reference + 0.5 * floor_height_tolerance
                and z_min_local >= min_z - 0.05
                and (z_max_local - z_min_local) <= 2.0 * plane_std_limit
            ):
                labels[mask] = True

        return labels

    @staticmethod
    def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
        if points.shape[0] <= 1:
            return points.copy()
        if points.shape[0] == 2:
            return points.copy()

        ordered = points[np.lexsort((points[:, 1], points[:, 0]))]

        def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: list[np.ndarray] = []
        for point in ordered:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)

        upper: list[np.ndarray] = []
        for point in reversed(ordered):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)

        hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)
        if hull.shape[0] >= 3:
            return hull
        return ordered[: min(4, ordered.shape[0])].astype(np.float32)

    def _floor_polygon(self, floor_points: np.ndarray) -> np.ndarray | None:
        if floor_points.size == 0:
            return None

        xy = floor_points[:, :2]
        z_level = float(np.median(floor_points[:, 2]))

        if xy.shape[0] >= 3:
            hull = self._convex_hull_2d(xy)
            if hull.shape[0] >= 3:
                return np.column_stack((hull[:, 0], hull[:, 1], np.full(hull.shape[0], z_level, dtype=np.float32)))

        min_xy = np.min(xy, axis=0)
        max_xy = np.max(xy, axis=0)
        corners = np.array(
            [
                [min_xy[0], min_xy[1]],
                [max_xy[0], min_xy[1]],
                [max_xy[0], max_xy[1]],
                [min_xy[0], max_xy[1]],
            ],
            dtype=np.float32,
        )
        return np.column_stack((corners[:, 0], corners[:, 1], np.full(corners.shape[0], z_level, dtype=np.float32)))


def main() -> None:
    rclpy.init()
    node = Lidar3DPlotter()

    def safe_remove_artist(artist: object | None) -> None:
        if artist is None:
            return
        try:
            artist.remove()
        except ValueError:
            pass

    plt.ion()
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection='3d')
    axis.set_title('RGB-D + LiDAR Planar Voxel Map')
    axis.set_xlabel('X (m)')
    axis.set_ylabel('Y (m)')
    axis.set_zlabel('Z (m)')
    axis.set_zlim(-2.0, 2.0)

    floor_surface = None
    object_scatter = None
    object_scatter = axis.scatter([], [], [], s=20, cmap='turbo', alpha=0.8, label='Object voxels')
    colorbar = figure.colorbar(object_scatter, ax=axis, shrink=0.7, pad=0.1)
    colorbar.set_label('Object voxel density')
    axis.legend(loc='upper right')

    try:
        while plt.fignum_exists(figure.number):
            rclpy.spin_once(node, timeout_sec=0.01)
            scan_points = node.get_scan_points()
            cloud_points, _ = node.get_cloud()

            if cloud_points.size or scan_points.size:
                merged_points = np.vstack((scan_points, cloud_points)) if cloud_points.size and scan_points.size else (
                    scan_points if scan_points.size else cloud_points
                )
                node.accumulate_voxels(merged_points)
                voxel_centers, voxel_density, floor_mask = node.get_accumulated_voxels()
            else:
                voxel_centers = np.empty((0, 3), dtype=np.float32)
                voxel_density = np.empty((0,), dtype=np.float32)
                floor_mask = np.empty((0,), dtype=bool)

            safe_remove_artist(floor_surface)
            safe_remove_artist(object_scatter)
            floor_surface = None
            object_scatter = None

            if voxel_centers.size:
                floor_mask = floor_mask & np.isfinite(voxel_centers).all(axis=1)
                floor_points = voxel_centers[floor_mask]
                floor_polygon = node._floor_polygon(floor_points)
                if floor_polygon is not None:
                    floor_surface = Poly3DCollection(
                        [floor_polygon],
                        facecolors='deepskyblue',
                        edgecolors='navy',
                        linewidths=0.5,
                        alpha=0.35,
                    )
                    axis.add_collection3d(floor_surface)

                object_mask = ~floor_mask
                if np.any(object_mask):
                    object_scatter = axis.scatter(
                        voxel_centers[object_mask, 0],
                        voxel_centers[object_mask, 1],
                        voxel_centers[object_mask, 2],
                        s=20,
                        c=voxel_density[object_mask],
                        cmap='turbo',
                        alpha=0.8,
                    )
                    colorbar.update_normal(object_scatter)
                else:
                    object_scatter = axis.scatter([], [], [], s=20, cmap='turbo', alpha=0.8)
                    colorbar.update_normal(object_scatter)
            else:
                object_scatter = axis.scatter([], [], [], s=20, cmap='turbo', alpha=0.8)
                colorbar.update_normal(object_scatter)

            axis.set_xlim(-5.0, 5.0)
            axis.set_ylim(-5.0, 5.0)
            axis.set_zlim(-5.0, 5.0)
            axis.set_box_aspect((1, 1, 1))
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
