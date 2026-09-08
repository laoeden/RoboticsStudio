import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer')
        self.bridge = CvBridge()
        self.depth_subscription = self.create_subscription(
            Image,
            '/parrot1/camera/depth/image',
            self.depth_callback,
            10)
        self.rgb_subscription = self.create_subscription(
            Image,
            '/parrot1/camera/image',
            self.rgb_callback,
            10)

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
            return

        finite_depth = depth_image[np.isfinite(depth_image)]
        if finite_depth.size == 0:
            return

        display_depth = np.zeros(depth_image.shape, dtype=np.uint8)
        near = np.percentile(finite_depth, 5)
        far = np.percentile(finite_depth, 95)
        if far > near:
            normalized = np.clip((depth_image - near) / (far - near), 0.0, 1.0)
            display_depth = (normalized * 255).astype(np.uint8)

        colorized_depth = cv2.applyColorMap(display_depth, cv2.COLORMAP_TURBO)
        cv2.imshow("Depth Image", colorized_depth)
        cv2.waitKey(1)

    def rgb_callback(self, msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f'Error converting RGB image: {e}')
            return

        cv2.imshow("RGB Image", rgb_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ImageViewer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()   