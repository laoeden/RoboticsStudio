import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

class StampRelay(Node):
    def __init__(self):
        super().__init__('stamp_relay')
        self.sub = self.create_subscription(
            PointCloud2, '/husky1/camera/depth/points', self.cb, 10)
        self.pub = self.create_publisher(
            PointCloud2, '/husky1/camera/depth/points_stamped', 10)

    def cb(self, msg):
        msg.header.frame_id = 'husky1_odom'
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = StampRelay()
    rclpy.spin(node)

if __name__ == '__main__':
    main()   