#!/usr/bin/env python3
"""Display a trained dry-grass YOLO model on the drone ROS camera."""

import argparse
import math
import threading
from collections import deque

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
# How far from the image's timestamp the paired depth frame / odometry may be. Odometry
# decides where the drone was, so it must be close; depth only gives the range to the
# ground, which barely changes over half a second.
DEPTH_TOLERANCE_S = 0.5
ODOM_TOLERANCE_S = 0.15
# How long (in message time) depth and odometry are kept for pairing. Images are paired the
# moment they arrive, so a few seconds is plenty; the frame cap bounds memory (about 1.4 MB each).
DEPTH_HISTORY_S = 3.0
ODOM_HISTORY_S = 5.0
MAX_DEPTH_FRAMES = 40


def stamp_seconds(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def nearest_sample(samples, stamp: float, tolerance: float):
    """Return the (stamp, value) closest in time to `stamp`, or None if none is within tolerance."""
    best = min(samples, key=lambda sample: abs(sample[0] - stamp), default=None)
    if best is None or abs(best[0] - stamp) > tolerance:
        return None
    return best


def remember(buffer: deque, stamp: float, value, history: float, limit: int = 100000) -> None:
    """Append a sample, dropping those older than `history` (or all, if the clock restarted)."""
    if buffer and stamp < buffer[-1][0] - 1.0:
        buffer.clear()          # simulation restarted: old samples belong to another timeline
    buffer.append((stamp, value))
    while buffer and (stamp - buffer[0][0] > history or len(buffer) > limit):
        buffer.popleft()


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
        # Recent depth frames and odometry, so each image is paired with the depth and
        # drone pose from the moment it was captured: the drone keeps moving, and at
        # 2 m/s every 0.25 s of mismatch is 0.5 m of position error.
        self.depth_buffer: deque = deque()
        self.odom_buffer: deque = deque()
        # YOLO runs on its own thread. On a CPU one frame can take seconds; run inside the
        # ROS callback it would stop this node from receiving odometry and depth meanwhile,
        # and the samples belonging to the image would be gone by the time it finished.
        self._pending = None            # newest (image, depth, odometry); older frames are dropped
        self._wake = threading.Condition()
        self._stopping = False
        self._worker = threading.Thread(
            target=self._inference_loop, daemon=True, name="YOLO_INFERENCE"
        )
        self._worker.start()
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
        """Pair the image with depth and odometry right away, then hand it to the YOLO thread."""
        stamp = stamp_seconds(message.header)
        depth_sample = nearest_sample(self.depth_buffer, stamp, DEPTH_TOLERANCE_S)
        odom_sample = nearest_sample(self.odom_buffer, stamp, ODOM_TOLERANCE_S)
        if depth_sample is None or odom_sample is None:
            def gap(buffer):
                return (
                    "none received" if not buffer
                    else f"closest {min(abs(s - stamp) for s, _ in buffer):.2f} s away"
                )
            # Still show the annotated image, but publish no world positions: a position
            # from mismatched samples is wrong by the distance the drone moved meanwhile.
            self.get_logger().warn(
                f"No synced depth/odometry for the image at {stamp:.2f}: "
                f"depth {gap(self.depth_buffer)}, odometry {gap(self.odom_buffer)}.",
                throttle_duration_sec=5.0,
            )
        job = (
            message,
            None if depth_sample is None else depth_sample[1],
            None if odom_sample is None else odom_sample[1],
        )
        with self._wake:
            self._pending = job
            self._wake.notify()

    def _inference_loop(self) -> None:
        while True:
            with self._wake:
                while self._pending is None and not self._stopping:
                    self._wake.wait()
                if self._stopping:
                    return
                job, self._pending = self._pending, None
            try:
                self.process(*job)
            except Exception as error:  # keep the thread alive whatever one frame does
                self.get_logger().error(f"YOLO inference failed: {error}", throttle_duration_sec=2.0)

    def shutdown_worker(self) -> None:
        with self._wake:
            self._stopping = True
            self._wake.notify()
        self._worker.join(timeout=2.0)

    def process(self, message: Image, depth, odometry) -> None:
        image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        result = self.model.predict(image, conf=self.confidence, verbose=False)[0]
        display = result.plot()
        if depth is not None and odometry is not None:
            for box in result.boxes.xyxy.cpu().numpy():
                camera_point = depth_at_detection(
                    depth,
                    tuple(float(value) for value in box),
                )
                if camera_point is None:
                    label = "world: depth unavailable"
                else:
                    point = world_position(camera_point, odometry)
                    label = f"world: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}) m"
                    world_message = PointStamped()
                    # Frame of the odometry, time of the image the detection came from.
                    world_message.header.frame_id = odometry.header.frame_id
                    world_message.header.stamp = message.header.stamp
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
            remember(
                self.depth_buffer, stamp_seconds(message.header),
                self._depth_from_message(message), DEPTH_HISTORY_S, MAX_DEPTH_FRAMES,
            )
        except ValueError as error:
            self.get_logger().error(str(error), throttle_duration_sec=2.0)

    def odom_callback(self, message: Odometry) -> None:
        remember(self.odom_buffer, stamp_seconds(message.header), message, ODOM_HISTORY_S)

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
        default="my_model (1)/my_model.pt",
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
        node.shutdown_worker()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
