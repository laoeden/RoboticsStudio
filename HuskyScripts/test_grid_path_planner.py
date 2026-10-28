"""Path planner checks that run without ROS or Gazebo."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_path_planner import HARD_CLEARANCE, plan_path

NO_OBSTACLES = np.empty((0, 2), dtype=np.float32)


def nearest_obstacle(path, obstacles):
    return min(math.dist(p, o) for p in path for o in obstacles)


class PlannerTests(unittest.TestCase):
    def test_open_field_is_a_straight_line(self):
        path = plan_path(0, 0, np.array([5.0, 0.0]), NO_OBSTACLES)
        self.assertGreater(len(path), 0)
        self.assertAlmostEqual(float(path[-1][0]), 5.0, places=3)
        self.assertTrue(np.allclose(path[:, 1], 0.0))

    def test_robot_parked_inside_the_clearance_can_still_leave(self):
        # Obstacle 1.3 m ahead, clearance 2.0: used to wall the robot in completely.
        obstacles = np.array([[3.0, 0.0]], dtype=np.float32)
        path = plan_path(1.7, 0.0, np.array([11.2, -11.7]), obstacles, robot_clearance=2.0)
        self.assertGreater(len(path), 0)
        self.assertAlmostEqual(float(path[-1][0]), 11.2, delta=0.2)

    def test_escape_never_cuts_through_the_hard_core(self):
        obstacles = np.array([[3.0, 0.0]], dtype=np.float32)
        path = plan_path(1.7, 0.0, np.array([6.0, 0.0]), obstacles, robot_clearance=2.0)
        self.assertGreater(len(path), 0)
        self.assertGreaterEqual(nearest_obstacle(path, obstacles), HARD_CLEARANCE - 0.2)

    def test_goal_near_an_obstacle_is_reachable(self):
        obstacles = np.array([[8.0, 0.0]], dtype=np.float32)
        path = plan_path(0, 0, np.array([7.0, 0.0]), obstacles, robot_clearance=2.0)
        self.assertGreater(len(path), 0)

    def test_robot_hemmed_in_by_the_hard_core_has_no_path(self):
        ring = np.array([[0.4 * math.cos(a), 0.4 * math.sin(a)]
                         for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)],
                        dtype=np.float32)
        path = plan_path(0, 0, np.array([5.0, 0.0]), ring, robot_clearance=2.0)
        self.assertEqual(len(path), 0)

    def test_wall_across_the_way_is_still_avoided(self):
        wall = np.array([[3.0, y] for y in np.arange(-4.0, 4.01, 0.2)], dtype=np.float32)
        path = plan_path(0, 0, np.array([6.0, 0.0]), wall, robot_clearance=1.0)
        self.assertGreater(len(path), 0)
        self.assertGreaterEqual(nearest_obstacle(path, wall), HARD_CLEARANCE - 0.2)

    def test_no_goal_gives_empty_path(self):
        self.assertEqual(len(plan_path(0, 0, None, NO_OBSTACLES)), 0)

    def test_clear_start_behaves_as_before(self):
        # Obstacle well outside the clearance: nothing is relaxed, path stays clear of it.
        obstacles = np.array([[3.0, 3.0]], dtype=np.float32)
        path = plan_path(0, 0, np.array([6.0, 0.0]), obstacles, robot_clearance=1.0)
        self.assertGreater(len(path), 0)
        self.assertGreaterEqual(nearest_obstacle(path, obstacles), 1.0 - 0.25)


if __name__ == '__main__':
    unittest.main()
