"""Coverage planner checks (no ROS needed): python3 -m unittest Mission.test_coverage_planner"""
import math
import random
import re
import unittest
from pathlib import Path

import numpy as np

from Mission.coverage_planner import (
    CAMERA_DROP, CoverageGrid, FX, HORIZONTAL_FOV, IMAGE_HEIGHT, IMAGE_WIDTH,
    Rect, SurveyConfig, VERTICAL_FOV, footprint, gap_waypoints, order_visits,
    path_length, plan_survey,
)

XACRO = (Path(__file__).resolve().parents[1]
         / '41068_ignition_bringup' / 'urdf_parrot' / 'parrot.gazebo.xacro')


def swept_fraction(plan, cfg, step=0.4, cell=0.25):
    """Independent raster check: fraction of the area inside any footprint on the path."""
    area = plan.area
    grid = CoverageGrid(area, cell)
    agl = plan.agl_min
    pts = plan.waypoints
    if len(pts) == 1:
        grid.mark(pts[0][0], pts[0][1], agl)
    for p, q in zip(pts, pts[1:]):
        n = max(2, int(math.dist(p[:2], q[:2]) / step) + 1)
        for t in np.linspace(0.0, 1.0, n):
            grid.mark(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), agl)
    return grid.fraction


class CameraModelTests(unittest.TestCase):
    def test_constants_match_the_simulator_files(self):
        text = XACRO.read_text()
        self.assertAlmostEqual(
            float(re.search(r'<horizontal_fov>([\d.]+)</horizontal_fov>', text).group(1)),
            HORIZONTAL_FOV)
        self.assertEqual(int(re.search(r'<width>(\d+)</width>', text).group(1)), IMAGE_WIDTH)
        self.assertEqual(int(re.search(r'<height>(\d+)</height>', text).group(1)), IMAGE_HEIGHT)

    def test_derived_values(self):
        self.assertAlmostEqual(FX, 207.85, places=1)
        self.assertAlmostEqual(math.degrees(VERTICAL_FOV), 98.2, places=1)
        x_extent, y_extent = footprint(1.0)
        self.assertAlmostEqual(y_extent, 3.4641, places=3)   # wide image axis -> world Y
        self.assertAlmostEqual(x_extent, 2.3094, places=3)
        self.assertEqual(CAMERA_DROP, 0.255)


class PlanTests(unittest.TestCase):
    def test_every_random_area_is_fully_covered(self):
        rng = random.Random(7)
        cfg = SurveyConfig()
        for _ in range(60):
            w, h = rng.uniform(2, 50), rng.uniform(2, 50)
            x0, y0 = rng.uniform(-25, 0), rng.uniform(-25, 0)
            plan = plan_survey(Rect(x0, y0, x0 + w, y0 + h), cfg, (2.0, 0.0))
            self.assertEqual(swept_fraction(plan, cfg), 1.0,
                             f'{w:.1f}x{h:.1f} axis={plan.axis} rows={plan.rows}')

    def test_each_axis_covers(self):
        for axis in ('x', 'y'):
            cfg = SurveyConfig(sweep_axis=axis)
            plan = plan_survey(Rect(-10, -10, 25, 18), cfg)
            self.assertEqual(plan.axis, axis)
            self.assertEqual(swept_fraction(plan, cfg), 1.0)

    def test_side_overlap_meets_configuration(self):
        cfg = SurveyConfig(overlap=0.3, min_overlap_m=3.0)
        plan = plan_survey(Rect(0, 0, 50, 50), cfg)
        ys = sorted({round(w[1], 6) for w in plan.waypoints})
        self.assertGreater(len(ys), 1)
        swath = footprint(plan.agl_min)[1]
        gaps = [b - a for a, b in zip(ys, ys[1:])]
        for gap in gaps:
            self.assertGreaterEqual(swath - gap, 3.0 - 1e-6)
            self.assertGreaterEqual(swath - gap, 0.3 * swath * 0.95 - 1e-6)

    def test_flight_stays_inside_depth_range(self):
        cfg = SurveyConfig()
        for size in (5, 20, 30, 40, 50):
            plan = plan_survey(Rect(0, 0, size, size), cfg)
            self.assertLessEqual(plan.agl_max, 9.5 + 1e-6)
            self.assertGreaterEqual(plan.agl_min, cfg.min_agl - 1e-6)
            self.assertAlmostEqual(plan.flight_z, plan.agl_min + CAMERA_DROP + cfg.terrain_max)
            self.assertFalse(plan.warnings)

    def test_altitude_solver_finds_the_minimum_row_count_and_lowest_height(self):
        cfg = SurveyConfig()
        plan = plan_survey(Rect(0, 0, 40, 40), cfg)
        highest = cfg.max_agl - cfg.relief
        # At the highest allowed altitude no axis needs fewer rows than the plan.
        from Mission import coverage_planner as cp
        fewest = min(cp._rows_at(highest, 40, axis, cfg) for axis in ('x', 'y'))
        self.assertEqual(plan.rows, fewest)
        # A noticeably lower flight would need more rows.
        self.assertGreater(cp._rows_at(plan.agl_min - 0.3, 40, plan.axis, cfg), plan.rows)

    def test_fixed_low_altitude_is_never_shorter_than_solved_altitude(self):
        cfg = SurveyConfig()
        solved = plan_survey(Rect(0, 0, 40, 40), cfg)
        naive = plan_survey(Rect(0, 0, 40, 40), cfg, fixed_z=cfg.flight_z(cfg.min_agl))
        self.assertLessEqual(solved.path_length, naive.path_length)

    def test_wide_image_axis_is_used_across_the_track(self):
        plan = plan_survey(Rect(0, 0, 20, 50), SurveyConfig())
        self.assertEqual(plan.axis, 'x')       # 3 rows along X beats 2 longer rows along Y
        self.assertLess(plan.path_length, 61.0)

    def test_area_smaller_than_footprint_is_one_hover_point(self):
        plan = plan_survey(Rect(0, 0, 4, 4), SurveyConfig())
        self.assertEqual(plan.rows, 1)
        self.assertEqual(len(plan.waypoints), 1)
        self.assertAlmostEqual(plan.waypoints[0][0], 2.0)
        self.assertAlmostEqual(plan.waypoints[0][1], 2.0)

    def test_rows_do_not_stray_outside_the_area_by_more_than_the_footprint(self):
        plan = plan_survey(Rect(-12, -9, 30, 21), SurveyConfig())
        for x, y, _ in plan.waypoints:
            self.assertTrue(plan.area.contains(x, y))

    def test_route_starts_at_the_nearest_end(self):
        area = Rect(0, 0, 50, 50)
        near_origin = plan_survey(area, SurveyConfig(), (0.0, 0.0))
        near_far = plan_survey(area, SurveyConfig(), (50.0, 50.0))
        self.assertLess(math.dist(near_origin.waypoints[0][:2], (0, 0)),
                        math.dist(near_far.waypoints[0][:2], (0, 0)))
        self.assertLess(near_far.transit_length, 25.0)

    def test_eta_includes_travel_and_waypoint_time(self):
        cfg = SurveyConfig(cruise_speed=2.0, waypoint_time=3.0)
        plan = plan_survey(Rect(0, 0, 40, 40), cfg, (0, 0))
        expected = (plan.path_length + plan.transit_length) / 2.0 + 3.0 * len(plan.waypoints)
        self.assertAlmostEqual(plan.eta_s, expected)

    def test_bad_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            SurveyConfig(overlap=1.0)
        with self.assertRaises(ValueError):
            SurveyConfig(sweep_axis='z')
        with self.assertRaises(ValueError):
            Rect(0, 0, float('nan'), 1)
        with self.assertRaises(ValueError):
            Rect(5, 0, 1, 1)

    def test_rect_helpers(self):
        r = Rect.from_corners(4, 6, -2, 0)
        self.assertEqual((r.xmin, r.ymin, r.xmax, r.ymax), (-2, 0, 4, 6))
        self.assertEqual(r.center, (1.0, 3.0))
        self.assertEqual(r.area, 36.0)
        self.assertTrue(r.contains(0, 0))
        self.assertFalse(r.contains(5, 0))
        self.assertTrue(r.contains(4.4, 0, margin=0.5))
        clamped = Rect(-40, -40, 40, 40).clamp(Rect(-25, -25, 25, 25))
        self.assertEqual(clamped.bounds, (-25, -25, 25, 25))


class CoverageGridTests(unittest.TestCase):
    def test_mark_covers_the_footprint(self):
        grid = CoverageGrid(Rect(0, 0, 20, 20), cell=0.5)
        grid.mark(10, 10, 5.0)
        x_ext, y_ext = footprint(5.0)
        self.assertAlmostEqual(
            grid.fraction, min(1, x_ext / 20) * min(1, y_ext / 20), delta=0.03)
        self.assertTrue(grid.seen[20, 20])
        self.assertFalse(grid.seen[0, 0])

    def test_yaw_rotates_the_footprint(self):
        a, b = CoverageGrid(Rect(0, 0, 40, 40)), CoverageGrid(Rect(0, 0, 40, 40))
        a.mark(20, 20, 5.0, 0.0)
        b.mark(20, 20, 5.0, math.pi / 2)
        self.assertEqual(int(a.seen.sum()), int(b.seen.sum()))
        self.assertTrue(np.array_equal(a.seen, b.seen.T))

    def test_invalid_marks_are_ignored(self):
        grid = CoverageGrid(Rect(0, 0, 10, 10))
        self.assertEqual(grid.mark(float('nan'), 0, 5), 0)
        self.assertEqual(grid.mark(5, 5, -1), 0)
        self.assertEqual(grid.mark(500, 500, 5), 0)
        self.assertEqual(grid.fraction, 0.0)
        self.assertEqual(grid.version, 0)

    def test_gaps_find_unseen_regions_and_ignore_specks(self):
        grid = CoverageGrid(Rect(0, 0, 20, 20), cell=1.0)
        grid.seen[:] = True
        grid.seen[2:6, 3:9] = False          # 4 rows x 6 columns of unseen cells
        grid.seen[15, 15] = False            # single-cell speck
        gaps = grid.gaps(min_cells=4)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].bounds, (3.0, 2.0, 9.0, 6.0))

    def test_gap_waypoints_close_a_gap(self):
        cfg = SurveyConfig()
        plan = plan_survey(Rect(0, 0, 40, 40), cfg, (0, 0))
        grid = CoverageGrid(plan.area)
        pts = plan.waypoints
        for p, q in zip(pts, pts[1:]):
            for t in np.linspace(0, 1, 40):
                grid.mark(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), plan.agl_min)
        self.assertEqual(grid.fraction, 1.0)
        grid.seen[10:30, 50:70] = False       # something was missed
        self.assertLess(grid.fraction, 1.0)
        extra = gap_waypoints(grid, cfg, plan.flight_z, (0, 0))
        self.assertTrue(extra)
        for p, q in zip(extra, extra[1:]):
            for t in np.linspace(0, 1, 40):
                grid.mark(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), plan.agl_min)
        if len(extra) == 1:
            grid.mark(extra[0][0], extra[0][1], plan.agl_min)
        self.assertEqual(grid.fraction, 1.0)
        for wp in extra:
            self.assertAlmostEqual(wp[2], plan.flight_z)


class VisitOrderTests(unittest.TestCase):
    def test_trivial_cases(self):
        self.assertEqual(order_visits((0, 0), []), [])
        self.assertEqual(order_visits((0, 0), [(1, 1)]), [0])

    def test_exact_order_beats_input_order(self):
        points = [(10, 0), (2, 0), (6, 0)]
        order = order_visits((0, 0), points, end=(12, 0))
        self.assertEqual(order, [1, 2, 0])
        self.assertLess(path_length((0, 0), [points[i] for i in order], (12, 0)),
                        path_length((0, 0), points, (12, 0)))

    def test_end_point_influences_the_order(self):
        points = [(5, 0), (5, 10)]
        self.assertEqual(order_visits((0, 0), points, end=(0, 20)), [0, 1])
        self.assertEqual(order_visits((0, 10), points, end=(0, 0)), [1, 0])

    def test_heuristic_is_a_permutation_and_close_to_exact(self):
        rng = random.Random(3)
        points = [(rng.uniform(-20, 20), rng.uniform(-20, 20)) for _ in range(12)]
        order = order_visits((0, 0), points, end=(0, 0), exact_limit=0)
        self.assertEqual(sorted(order), list(range(12)))
        arbitrary = path_length((0, 0), points, (0, 0))
        tuned = path_length((0, 0), [points[i] for i in order], (0, 0))
        self.assertLess(tuned, arbitrary)


if __name__ == '__main__':
    unittest.main()