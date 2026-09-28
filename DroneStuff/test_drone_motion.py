"""Controller checks that run without Gazebo or ROS discovery."""
import math
import time
import unittest
from unittest.mock import Mock

from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from DroneMotion import (
    DroneMotion,
    avoidance_velocity,
    body_velocity,
    horizontal_speed_limit,
)


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


class AvoidanceTests(unittest.TestCase):
    def test_clear_scan_has_no_effect(self):
        ranges = [10.0] * 8
        vx, vy = avoidance_velocity(ranges, 0.0, math.pi / 4, 1.5, 1.0)
        self.assertEqual((vx, vy), (0.0, 0.0))

    def test_obstacle_ahead_pushes_backward(self):
        # A single close reading dead ahead (angle 0) should push -x.
        vx, vy = avoidance_velocity([0.5], 0.0, 0.0, 1.5, 1.0)
        self.assertLess(vx, 0.0)
        self.assertAlmostEqual(vy, 0.0)

    def test_obstacle_to_the_left_pushes_right(self):
        # angle_min=pi/2 means the single reading is dead to the left (+y).
        vx, vy = avoidance_velocity([0.5], math.pi / 2, 0.0, 1.5, 1.0)
        self.assertAlmostEqual(vx, 0.0)
        self.assertLess(vy, 0.0)

    def test_non_finite_and_far_readings_ignored(self):
        ranges = [float('inf'), float('nan'), -1.0, 5.0, 1.5]
        vx, vy = avoidance_velocity(ranges, 0.0, math.pi / 4, 1.5, 1.0)
        self.assertEqual((vx, vy), (0.0, 0.0))

    def test_closer_obstacle_pushes_harder(self):
        near_x, _ = avoidance_velocity([0.2], 0.0, 0.0, 1.5, 1.0)
        far_x, _ = avoidance_velocity([1.4], 0.0, 0.0, 1.5, 1.0)
        self.assertLess(near_x, far_x)

    def test_horizontal_speed_limit_clamps_magnitude(self):
        vx, vy = horizontal_speed_limit(3.0, 4.0, 1.0)
        self.assertAlmostEqual(math.hypot(vx, vy), 1.0)
        self.assertAlmostEqual(vx / vy, 3.0 / 4.0)

    def test_horizontal_speed_limit_leaves_slow_commands_alone(self):
        self.assertEqual(horizontal_speed_limit(0.1, -0.2, 1.0), (0.1, -0.2))


class ControllerAvoidanceTests(unittest.TestCase):
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
        # No scan received yet: avoidance is skipped, matching pre-avoidance
        # behaviour, which the tests inherited by this helper rely on.
        node.scan = None
        node.scan_timeout = 1.0
        node.safe_distance, node.avoid_gain = 1.5, 1.5
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

    def test_stale_scan_is_ignored(self):
        # Goal is straight up (x=y=0), so with no avoidance vx == vy == 0.
        node = self.controller()
        node.scan = LaserScan(ranges=[0.1])
        node.scan.angle_min = 0.0
        node.scan.angle_increment = 0.0
        node.scan_received = time.monotonic() - 5.0  # older than scan_timeout
        DroneMotion.tick(node)
        cmd = node.publisher.publish.call_args.args[0]
        self.assertAlmostEqual(cmd.linear.x, 0.0)
        self.assertAlmostEqual(cmd.linear.y, 0.0)

    def test_fresh_close_scan_steers_around_obstacle(self):
        # Same straight-up goal, but now with an obstacle dead ahead (+x).
        node = self.controller()
        node.scan = LaserScan(ranges=[0.1])
        node.scan.angle_min = 0.0
        node.scan.angle_increment = 0.0
        node.scan_received = time.monotonic()
        DroneMotion.tick(node)
        cmd = node.publisher.publish.call_args.args[0]
        # With no goal-seeking pull in x/y, the only contribution is
        # avoidance, which should push straight backward (-x).
        self.assertLess(cmd.linear.x, 0.0)
        self.assertAlmostEqual(cmd.linear.y, 0.0)


if __name__ == '__main__':
    unittest.main()
