"""Controller checks that run without Gazebo or ROS discovery."""
import math
import time
import unittest
from unittest.mock import Mock

from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry

from DroneMotion import DroneMotion, body_velocity


class MovementTests(unittest.TestCase):
    def test_speed_limit_and_climb(self):
        velocity, _ = body_velocity((0, 0, 0), (0, 0, 10), (0, 0, 0, 1), .8, 1, .15)
        self.assertEqual(velocity, (0, 0, 1))

    def test_rotated_drone(self):
        s = math.sqrt(.5)
        velocity, _ = body_velocity((0, 0, 0), (10, 0, 0), (0, 0, s, s), .8, 1, .15)
        self.assertAlmostEqual(velocity[0], 0)
        self.assertAlmostEqual(velocity[1], -1)

    def test_converges_in_three_dimensions(self):
        position = (0, 0, 0)
        target = (3, -2, 4)
        for _ in range(1000):
            velocity, distance = body_velocity(position, target, (0, 0, 0, 1), .8, 1, .15)
            position = tuple(p + .05*v for p, v in zip(position, velocity))
        self.assertLessEqual(distance, .15)
        self.assertEqual(velocity, (0, 0, 0))

    def controller(self):
        node = Mock()
        node.odom = Odometry()
        node.odom.header.frame_id = 'parrot1_odom'
        node.odom.pose.pose.orientation.w = 1.0
        node.odom_received = time.monotonic()
        node.odom_timeout, node.goal_timeout = 1.0, 120.0
        node.gain, node.max_speed, node.tolerance = .8, 1.0, .15
        node.goal = PointStamped()
        node.goal.header.frame_id = 'parrot1_odom'
        node.goal.point.z = 3.0
        node.goal_started = time.monotonic()
        return node

    def test_stale_odometry_cancels_goal(self):
        node = self.controller()
        node.odom_received -= 2
        DroneMotion.tick(node)
        self.assertIsNone(node.goal)
        node.stop.assert_called_once()

    def test_timeout_cancels_goal(self):
        node = self.controller()
        node.goal_started -= 121
        DroneMotion.tick(node)
        self.assertIsNone(node.goal)
        node.stop.assert_called_once()

    def test_arrival_publishes_zero(self):
        node = self.controller()
        node.odom.pose.pose.position.z = 3.0
        DroneMotion.tick(node)
        self.assertIsNone(node.goal)
        self.assertEqual(node.publisher.publish.call_args.args[0].linear.z, 0.0)

    def test_wrong_frame_and_nonfinite_goal_rejected(self):
        node = self.controller()
        previous = node.goal
        bad = PointStamped()
        bad.header.frame_id = 'map'
        DroneMotion.on_goal(node, bad)
        self.assertIs(node.goal, previous)
        bad.header.frame_id = 'parrot1_odom'
        bad.point.x = float('nan')
        DroneMotion.on_goal(node, bad)
        self.assertIs(node.goal, previous)


if __name__ == '__main__':
    unittest.main()
