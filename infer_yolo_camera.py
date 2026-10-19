#!/usr/bin/env python3
"""Display a trained dry-grass YOLO model on the drone ROS camera."""

import argparse
import math

import cv2
import numpy as np
from geometry_msgs.msg import PointStamped
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


FX = 207.85
FY = 207.85
CX = 360.0
CY = 240.0
MIN_DEPTH_M = 0.4
MAX_DEPTH_M = 10.0


def camera_to_base(point: np.ndarray) -> np.ndarray:
    """Transform a point from the depth optical frame into the parrot base frame."""
    def rotation_x(angle: float) -> np.ndarray:
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ])

    def rotation_y(angle: float) -> np.ndarray:
        return np.array([
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ])

    def rotation_z(angle: float) -> np.ndarray:
        return np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])

    rotation = (
        rotation_y(math.pi / 2)
        @ rotation_z(-math.pi / 2)
        @ rotation_x(-math.pi / 2)
    )
    translation = (
        np.array([0.0, 0.0, -0.25])
        + rotation_y(math.pi / 2) @ np.array([0.005, 0.028, 0.013])
    )
    return rotation @ point + translation


def rotate_by_quaternion(point: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    rotation = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * x * x - 2 * z * z, 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * x * x - 2 * y * y],
    ])
    return rotation @ point


def depth_at_detection(
    depth: np.ndarray,
    box: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Return the median-depth camera point for a YOLO box."""
    if depth.ndim != 2 or depth.size == 0:
        return None
    x1, y1, x2, y2 = box
    left = max(0, int(math.floor(x1)))
    top = max(0, int(math.floor(y1)))
    right = min(depth.shape[1], int(math.ceil(x2)))
    bottom = min(depth.shape[0], int(math.ceil(y2)))
    if left >= right or top >= bottom:
        return None

    region = depth[top:bottom, left:right]
    valid_depth = region[np.isfinite(region)]
    valid_depth = valid_depth[
        (valid_depth >= MIN_DEPTH_M) & (valid_depth <= MAX_DEPTH_M)
    ]
    if valid_depth.size == 0:
        return None

    depth_m = float(np.median(valid_depth))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return np.array([
        (center_x - CX) * depth_m / FX,
        (center_y - CY) * depth_m / FY,
        depth_m,
    ])


def world_position(
    camera_point: np.ndarray,
    odometry: Odometry,
) -> np.ndarray:
    pose = odometry.pose.pose
    quaternion = np.array([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ])
    return (
        rotate_by_quaternion(camera_to_base(camera_point), quaternion)
        + np.array([pose.position.x, pose.position.y, pose.position.z])
    )


class CameraInference(Node):
    def __init__(
        self,
        model_path: str,
        topic: str,
        depth_topic: str,
        odom_topic: str,
        confidence: float,
    ) -> None:
        super().__init__("dry_grass_yolo_camera")
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.confidence = confidence
        self.latest_depth: np.ndarray | None = None
        self.latest_odom: Odometry | None = None
        self.world_position_publisher = self.create_publisher(
            PointStamped,
            "/parrot1/yolo/world_position",
            10,
        )
        self.annotated_image_publisher = self.create_publisher(
            Image,
            "/parrot1/yolo/image",
            10,
        )
        self.subscription = self.create_subscription(Image, topic, self.callback, 10)
        self.depth_subscription = self.create_subscription(
            Image, depth_topic, self.depth_callback, 10
        )
        self.odom_subscription = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10
        )

    def callback(self, message: Image) -> None:
        image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        result = self.model.predict(image, conf=self.confidence, verbose=False)[0]
        display = result.plot()
        if self.latest_depth is not None and self.latest_odom is not None:
            for box in result.boxes.xyxy.cpu().numpy():
                camera_point = depth_at_detection(
                    self.latest_depth,
                    tuple(float(value) for value in box),
                )
                if camera_point is None:
                    label = "world: depth unavailable"
                else:
                    point = world_position(camera_point, self.latest_odom)
                    label = f"world: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}) m"
                    world_message = PointStamped()
                    world_message.header = self.latest_odom.header
                    world_message.point.x = float(point[0])
                    world_message.point.y = float(point[1])
                    world_message.point.z = float(point[2])
                    self.world_position_publisher.publish(world_message)
                x1, y1 = int(box[0]), int(box[1])
                cv2.putText(
                    display,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
        annotated_message = self.bridge.cv2_to_imgmsg(display, encoding="bgr8")
        annotated_message.header = message.header
        self.annotated_image_publisher.publish(annotated_message)

    def depth_callback(self, message: Image) -> None:
        try:
            self.latest_depth = self._depth_from_message(message)
        except ValueError as error:
            self.get_logger().error(str(error), throttle_duration_sec=2.0)

    def odom_callback(self, message: Odometry) -> None:
        self.latest_odom = message

    def _depth_from_message(self, message: Image) -> np.ndarray:
        if message.encoding not in ("32FC1", "16UC1"):
            raise ValueError(
                f"Unsupported depth encoding: {message.encoding!r}. "
                "Expected 32FC1 or 16UC1."
            )
        dtype = np.float32 if message.encoding == "32FC1" else np.uint16
        bytes_per_pixel = np.dtype(dtype).itemsize
        row_bytes = message.width * bytes_per_pixel
        if message.step < row_bytes:
            raise ValueError(
                f"Invalid depth image step {message.step} for width {message.width}."
            )
        depth = np.frombuffer(message.data, dtype=dtype).reshape(
            message.height, message.step // bytes_per_pixel
        )[:, :message.width].copy()
        if message.encoding == "16UC1":
            return depth.astype(np.float32) / 1000.0
        return depth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        default="my_model/my_model.pt",
        nargs="?",
    )
    parser.add_argument("--topic", default="/parrot1/camera/image")
    parser.add_argument("--depth-topic", default="/parrot1/camera/depth/image")
    parser.add_argument("--odom-topic", default="/parrot1/odometry")
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()
    rclpy.init()
    node = CameraInference(
        args.model,
        args.topic,
        args.depth_topic,
        args.odom_topic,
        args.confidence,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
