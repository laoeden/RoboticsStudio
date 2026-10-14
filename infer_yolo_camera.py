#!/usr/bin/env python3
"""Display a trained dry-grass YOLO model on the drone ROS camera."""

import argparse

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


class CameraInference(Node):
    def __init__(self, model_path: str, topic: str, confidence: float) -> None:
        super().__init__("dry_grass_yolo_camera")
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.confidence = confidence
        self.subscription = self.create_subscription(Image, topic, self.callback, 10)

    def callback(self, message: Image) -> None:
        image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        result = self.model.predict(image, conf=self.confidence, verbose=False)[0]
        cv2.imshow("YOLO dry grass", result.plot())
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        default="runs/detect/weights/dry_grass/weights/best.pt",
        nargs="?",
    )
    parser.add_argument("--topic", default="/parrot1/camera/image")
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()
    rclpy.init()
    node = CameraInference(args.model, args.topic, args.confidence)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
