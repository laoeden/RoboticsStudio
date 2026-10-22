"""Survey mission checks (no ROS needed): python3 -m unittest Mission.test_survey_mission"""
import math
import unittest

from Mission.coverage_planner import Rect
from Mission.survey_mission import (
    ACTIVE, CLEARED, FALSE_POSITIVE, PENDING, UNREACHABLE, VISITED,
    MissionConfig, MissionIO, MissionState, SurveyMission, Telemetry,
    parse_husky_status,
)

DOCK = (0.0, 0.0)


class FakeWorld(MissionIO):
    """A tiny stand-in for the simulator, the UI glue and grassRemover."""

    def __init__(self, patches=(), pins=None, husky=(-5.0, 0.0), drone=(2.0, 0.0, 3.0),
                 battery=None, remover_online=True, blind_to=None):
        self.patches = [tuple(p) for p in patches]          # real hazards in the world
        self.pins = [dict(p, status=PENDING) for p in (
            pins if pins is not None else [{'id': i + 1, 'x': x, 'y': y}
                                           for i, (x, y) in enumerate(self.patches)])]
        self.drone = list(drone)
        self.husky = list(husky)
        self.drone_goal_xyz = None
        self.husky_goal_xy = None
        self.battery = battery
        self.remover_online = remover_online
        self.blind_to = blind_to          # callable(x, y) -> True if the camera misses that spot
        self.log_lines = []
        self.drone_goals = []
        self.husky_goals = []
        self.commands = []
        self.survey_prepared = 0
        self.survey_ended = 0
        self.drone_speed = 6.0            # fast so tests run quickly; no effect on logic
        self.husky_speed = 3.0
        self.now = 0.0
        self.mission = None
        self.drone_stuck = False
        self.husky_events = []

    # --- MissionIO
    def prepare_survey(self, area): self.survey_prepared += 1
    def end_survey(self): self.survey_ended += 1
    def drone_goal(self, x, y, z):
        self.drone_goals.append((x, y, z))
        self.drone_goal_xyz = (x, y, z)
    def husky_goal(self, x, y):
        self.husky_goals.append((x, y))
        self.husky_goal_xy = (x, y)
    def command(self, text):
        self.commands.append(text)
        if text == 'RETURN_TO_BASE':
            self.husky_goal_xy = DOCK
        elif text in ('STOP', 'CLEAR_GOAL'):
            self.husky_goal_xy = None
    def log(self, source, message): self.log_lines.append((source, message))
    def list_pois(self, area):
        return [dict(p) for p in self.pins if area.contains(p['x'], p['y'])
                and p['status'] in (PENDING, ACTIVE)]
    def mark_poi(self, poi_id, status):
        for p in self.pins:
            if p['id'] == poi_id:
                p['status'] = status

    # --- simulation
    def telemetry(self):
        low = self.battery is not None and self.battery <= 20.0
        return Telemetry(drone=tuple(self.drone), husky=tuple(self.husky),
                         battery=self.battery, battery_low=low,
                         remover_online=self.remover_online)

    def step(self, dt=0.5):
        self.now += dt
        if self.drone_goal_xyz is not None and not self.drone_stuck:
            self.drone = self._move(self.drone, self.drone_goal_xyz, self.drone_speed * dt)
        if self.mission is not None and not (
                self.blind_to and self.blind_to(self.drone[0], self.drone[1])):
            self.mission.observe_frame(self.drone[0], self.drone[1], self.drone[2], 0.0)
        if self.husky_goal_xy is not None:
            before = tuple(self.husky)
            self.husky = list(self._move(self.husky, self.husky_goal_xy, self.husky_speed * dt))
            if self.battery is not None:
                self.battery -= 0.25 * math.dist(before, self.husky)
        elif self.battery is not None and math.dist(self.husky, DOCK) <= 1.0:
            self.battery = min(100.0, self.battery + 2.0 * dt)
        if self.remover_online:
            for patch in list(self.patches):
                if math.dist(self.husky, patch) <= 2.5:
                    self.patches.remove(patch)
                    self.mission.on_hazard_removed(*patch)
        for text in self.husky_events:
            self.mission.on_husky_status(text)
        self.husky_events = []
        if (self.husky_goal_xy is not None
                and math.dist(self.husky, self.husky_goal_xy) < 0.1):
            self.husky_goal_xy = None
        self.mission.update(self.now, self.telemetry())

    @staticmethod
    def _move(pos, goal, distance):
        d = math.dist(pos, goal)
        if d <= distance:
            return list(goal)
        return [p + (g - p) * distance / d for p, g in zip(pos, goal)]

    def run(self, max_time=4000.0, until=None):
        while self.now < max_time:
            self.step()
            if until is not None and until():
                return
            if self.mission.state in (MissionState.COMPLETE, MissionState.ABORTED):
                return
        raise AssertionError(f'mission still {self.mission.state} after {max_time} s')

    def start(self, area, cfg=None):
        self.mission = SurveyMission(self, cfg)
        ok = self.mission.start(area, self.now, tuple(self.drone), tuple(self.husky))
        return ok

    def messages(self):
        return '\n'.join(f'{s}: {m}' for s, m in self.log_lines)


AREA = Rect(-15, -15, 15, 15)


class HappyPathTests(unittest.TestCase):
    def test_full_mission_clears_hazards_and_returns_to_base(self):
        world = FakeWorld(patches=[(-10, -10), (10, 10), (-10, 10), (10, -10)])
        self.assertTrue(world.start(AREA))
        world.run()
        m = world.mission
        self.assertEqual(m.state, MissionState.COMPLETE, world.messages())
        self.assertEqual(world.patches, [])
        self.assertEqual([p['status'] for p in world.pins], [CLEARED] * 4)
        self.assertGreater(m.grid.fraction, 0.99)
        self.assertEqual(world.survey_prepared, 1)
        self.assertEqual(world.survey_ended, 1)
        # Drone went home and the Husky was sent home last.
        self.assertEqual(world.drone_goals[-1], (0.0, 0.0, 10.0))
        self.assertEqual(world.commands[-1], 'RETURN_TO_BASE')
        self.assertLess(math.dist(world.husky, DOCK), 0.5)
        # The Husky does not move until the sweep has finished.
        first_husky_goal = next(i for i, (s, msg) in enumerate(world.log_lines)
                                if s == 'HUSKY')
        sweep_done = next(i for i, (s, msg) in enumerate(world.log_lines)
                          if 'Survey complete' in msg)
        self.assertGreater(first_husky_goal, sweep_done)
        self.assertEqual(m.status()['progress'], 1.0)

    def test_drone_flies_the_planned_route(self):
        world = FakeWorld(patches=[])
        world.start(AREA)
        plan = world.mission.plan
        world.run()
        flown = world.drone_goals
        # takeoff goal, then every planned waypoint in order, then home
        self.assertEqual(flown[0][2], plan.flight_z)
        self.assertEqual(flown[1:1 + len(plan.waypoints)], plan.waypoints)
        self.assertEqual(flown[-1], (0.0, 0.0, 10.0))

    def test_no_pois_means_the_husky_never_moves(self):
        world = FakeWorld(patches=[])
        world.start(AREA)
        world.run()
        self.assertEqual(world.mission.state, MissionState.COMPLETE)
        self.assertEqual(world.husky_goals, [])
        self.assertEqual(world.commands, [])
        self.assertIn('No points of interest', world.messages())

    def test_pois_outside_the_area_are_ignored(self):
        world = FakeWorld(patches=[(5, 5), (22, 22)])
        world.start(AREA)
        world.run()
        self.assertEqual(world.patches, [(22, 22)])
        self.assertEqual(world.husky_goals, [(5.0, 5.0)])

    def test_visit_order_is_the_shortest_tour(self):
        points = [(14.0, 0.0), (-14.0, 2.0), (1.0, 12.0), (-3.0, -13.0)]
        world = FakeWorld(patches=points, husky=(0.0, 0.0))
        world.start(AREA)
        world.run()
        self.assertEqual(sorted(world.husky_goals), sorted(points))
        from itertools import permutations
        from Mission.coverage_planner import path_length
        best = min(path_length((0.0, 0.0), list(order), DOCK) for order in permutations(points))
        actual = path_length((0.0, 0.0), world.husky_goals, DOCK)
        self.assertAlmostEqual(actual, best, places=6)


class DetectionOutcomeTests(unittest.TestCase):
    def test_false_positive_is_reported_after_a_visit(self):
        world = FakeWorld(patches=[(8, 8)],
                          pins=[{'id': 1, 'x': 8, 'y': 8}, {'id': 2, 'x': -8, 'y': -8}])
        world.start(AREA)
        world.run()
        statuses = {p['id']: p['status'] for p in world.pins}
        self.assertEqual(statuses, {1: CLEARED, 2: FALSE_POSITIVE})
        self.assertEqual(world.mission.state, MissionState.COMPLETE)

    def test_unconfirmed_when_the_remover_is_offline(self):
        world = FakeWorld(patches=[(8, 8)], remover_online=False)
        world.start(AREA)
        world.run()
        self.assertEqual(world.pins[0]['status'], VISITED)
        self.assertIn('not confirmed', world.messages())

    def test_hazard_removed_on_the_way_is_skipped(self):
        # Two patches side by side: driving to one removes both, so the second visit is skipped.
        world = FakeWorld(patches=[(10, 0.5), (10, -0.5)], husky=(0.0, 0.0))
        world.start(AREA)
        world.run()
        self.assertEqual([p['status'] for p in world.pins], [CLEARED, CLEARED])
        self.assertEqual(len(world.husky_goals), 1)

    def test_removal_not_on_the_list_is_counted_not_fatal(self):
        world = FakeWorld(patches=[(8, 8), (-12, 5)],
                          pins=[{'id': 1, 'x': 8, 'y': 8}], husky=(-12.0, 4.0))
        world.start(AREA)
        world.run()
        self.assertEqual(world.mission.state, MissionState.COMPLETE)
        self.assertEqual(world.mission.status()['extra_removed'], 1)


class SurveyQualityTests(unittest.TestCase):
    def test_missed_region_triggers_a_gap_fill_pass(self):
        blind_on = {'on': True}
        # The camera "fails" over the eastern half while the drone is on the first sweep.
        world = FakeWorld(patches=[], blind_to=lambda x, y: blind_on['on'] and x > 0)
        world.start(AREA)
        world.run(until=lambda: world.mission.state == MissionState.GAP_FILL)
        self.assertEqual(world.mission.state, MissionState.GAP_FILL)
        self.assertLess(world.mission.grid.fraction, 0.9)
        blind_on['on'] = False
        world.run()
        self.assertEqual(world.mission.state, MissionState.COMPLETE)
        self.assertEqual(world.mission.status()['gap_rounds'], 1)
        self.assertGreater(world.mission.grid.fraction, 0.99)

    def test_persistently_missed_region_gives_up_and_warns(self):
        world = FakeWorld(patches=[], blind_to=lambda x, y: x > 0)
        world.start(AREA)
        world.run()
        self.assertEqual(world.mission.state, MissionState.COMPLETE)
        self.assertEqual(world.mission.status()['gap_rounds'], 2)
        self.assertIn('could not be surveyed', world.messages())

    def test_survey_is_declared_failed_if_almost_nothing_was_seen(self):
        world = FakeWorld(patches=[(8, 8)], blind_to=lambda x, y: True)
        world.start(AREA)
        world.run()
        self.assertEqual(world.mission.state, MissionState.ABORTED)
        self.assertEqual(world.husky_goals, [])


class SettleTests(unittest.TestCase):
    def test_a_pin_confirmed_just_after_the_sweep_is_still_investigated(self):
        world = FakeWorld(patches=[(8, 8)], pins=[])
        world.start(AREA)
        world.run(until=lambda: world.mission.state == MissionState.SETTLE)
        # The camera pipeline confirms the last sighting a moment later.
        world.pins.append({'id': 1, 'x': 8.0, 'y': 8.0, 'status': PENDING})
        world.run()
        self.assertEqual(world.mission.state, MissionState.COMPLETE)
        self.assertEqual(world.pins[0]['status'], CLEARED)

    def test_husky_waits_for_the_settle_period(self):
        world = FakeWorld(patches=[(8, 8)])
        world.start(AREA)
        world.run(until=lambda: world.mission.state == MissionState.SETTLE)
        started = world.now
        world.run(until=lambda: bool(world.husky_goals))
        self.assertGreaterEqual(world.now - started, MissionConfig().settle_time - 0.5)


class FailureTests(unittest.TestCase):
    def test_unreachable_drone_aborts_after_repeated_timeouts(self):
        world = FakeWorld(patches=[(8, 8)])
        world.start(AREA)
        # Climb normally, then freeze the drone.
        world.run(until=lambda: world.mission.state == MissionState.SWEEP)
        world.drone_stuck = True
        world.run(max_time=2000)
        self.assertEqual(world.mission.state, MissionState.ABORTED)
        self.assertIn('waypoints', world.mission.reason)
        self.assertEqual(world.husky_goals, [])

    def test_no_drone_odometry_refuses_to_start(self):
        world = FakeWorld()
        world.mission = SurveyMission(world)
        self.assertFalse(world.mission.start(AREA, 0.0, None))
        self.assertEqual(world.mission.state, MissionState.IDLE)
        self.assertEqual(world.drone_goals, [])

    def test_cannot_start_twice(self):
        world = FakeWorld()
        self.assertTrue(world.start(AREA))
        self.assertFalse(world.mission.start(AREA, 0.0, (0, 0, 3)))
        self.assertEqual(world.survey_prepared, 1)

    def test_abort_hovers_the_drone_and_stops_the_husky(self):
        world = FakeWorld(patches=[(8, 8), (-8, -8)])
        world.start(AREA)
        world.run(until=lambda: world.mission.state == MissionState.HUSKY_TRANSIT)
        position = tuple(world.drone)
        world.mission.abort('operator stop')
        self.assertEqual(world.mission.state, MissionState.ABORTED)
        self.assertEqual(world.drone_goals[-1], position)
        self.assertEqual(world.commands[-1], 'STOP')
        self.assertNotIn(ACTIVE, [p['status'] for p in world.pins])
        self.assertEqual(world.survey_ended, 1)
        world.mission.abort('again')          # second abort is harmless
        self.assertEqual(world.survey_ended, 1)
        world.step()
        self.assertEqual(world.mission.state, MissionState.ABORTED)

    def test_restart_after_abort_is_allowed(self):
        world = FakeWorld(patches=[])
        world.start(AREA)
        world.mission.abort('test')
        self.assertTrue(world.mission.start(AREA, world.now, tuple(world.drone), tuple(world.husky)))
        world.run()
        self.assertEqual(world.mission.state, MissionState.COMPLETE)

    def test_husky_cancel_is_retried_once_then_unreachable(self):
        world = FakeWorld(patches=[])
        world.pins = [{'id': 1, 'x': 8.0, 'y': 8.0, 'status': PENDING}]
        world.start(AREA)
        world.run(until=lambda: world.mission.state == MissionState.HUSKY_TRANSIT)
        world.husky_goal_xy = None            # Husky gives up
        world.husky_events.append('CANCELLED 8.0 8.0 no path')
        world.step()
        self.assertEqual(world.husky_goals, [(8.0, 8.0), (8.0, 8.0)])
        world.husky_goal_xy = None
        world.husky_events.append('CANCELLED 8.0 8.0 no path')
        world.step()
        self.assertEqual(world.pins[0]['status'], UNREACHABLE)
        world.run()
        self.assertEqual(world.mission.state, MissionState.COMPLETE)

    def test_husky_that_never_arrives_times_out(self):
        world = FakeWorld(patches=[])
        world.pins = [{'id': 1, 'x': 8.0, 'y': 8.0, 'status': PENDING}]
        world.husky_speed = 0.0               # wedged
        world.start(AREA)
        world.run(until=lambda: world.mission.state == MissionState.HUSKY_TRANSIT)
        world.run(max_time=5000, until=lambda: world.pins[0]['status'] == UNREACHABLE)
        self.assertEqual(len(world.husky_goals), 2)       # one retry
        self.assertIn('CLEAR_GOAL', world.commands)

    def test_stale_arrival_for_another_goal_is_ignored(self):
        world = FakeWorld(patches=[])
        world.pins = [{'id': 1, 'x': 8.0, 'y': 8.0, 'status': PENDING}]
        world.husky_speed = 0.0
        world.start(AREA)
        world.run(until=lambda: world.mission.state == MissionState.HUSKY_TRANSIT)
        world.husky_events.append('ARRIVED -3.0 4.0')
        world.step()
        self.assertEqual(world.mission.state, MissionState.HUSKY_TRANSIT)
        world.husky_events.append('ARRIVED 8.0 8.0')
        world.step()
        self.assertEqual(world.mission.state, MissionState.INVESTIGATE)


class BatteryTests(unittest.TestCase):
    def test_low_battery_sends_the_husky_to_recharge_then_it_resumes(self):
        # Parked just off the pad so it does not charge during the sweep.
        world = FakeWorld(patches=[(12, 12), (-12, -12), (12, -12)], husky=(0.0, 3.0),
                          battery=34.0)
        world.start(AREA)
        world.run()
        self.assertEqual(world.mission.state, MissionState.COMPLETE, world.messages())
        self.assertIn('Returning to recharge', world.messages())
        self.assertIn('Recharged; resuming', world.messages())
        self.assertEqual([p['status'] for p in world.pins], [CLEARED] * 3)
        self.assertGreaterEqual(world.battery, 20.0)

    def test_full_battery_needs_no_recharge(self):
        world = FakeWorld(patches=[(12, 12)], husky=(0.0, 0.0), battery=100.0)
        world.start(AREA)
        world.run()
        self.assertNotIn('recharge', world.messages().lower())
        self.assertEqual(world.pins[0]['status'], CLEARED)

    def test_poi_beyond_one_charge_is_unreachable_not_a_loop(self):
        area = Rect(-80, -80, 80, 80)
        world = FakeWorld(patches=[], husky=(0.0, 0.0), battery=100.0)
        world.pins = [{'id': 1, 'x': 70.0, 'y': 70.0, 'status': PENDING}]
        world.start(area)
        world.run()
        self.assertEqual(world.pins[0]['status'], UNREACHABLE)
        self.assertEqual(world.mission.state, MissionState.COMPLETE)
        self.assertEqual(world.husky_goals, [])

    def test_estimates_are_skipped_without_a_battery_node(self):
        world = FakeWorld(patches=[(12, 12)], husky=(0.0, 0.0), battery=None)
        world.start(AREA)
        world.run()
        self.assertEqual(world.pins[0]['status'], CLEARED)


class HelperTests(unittest.TestCase):
    def test_parse_husky_status(self):
        self.assertEqual(parse_husky_status('ARRIVED 1.5 -2.0'), ('ARRIVED', 1.5, -2.0, ''))
        self.assertEqual(parse_husky_status('cancelled 1 2 no path found'),
                         ('CANCELLED', 1.0, 2.0, 'no path found'))
        for bad in ('', 'ARRIVED', 'ARRIVED x y', 'ARRIVED nan 1'):
            self.assertIsNone(parse_husky_status(bad), bad)

    def test_status_is_available_before_and_after_a_run(self):
        world = FakeWorld(patches=[(8, 8)])
        world.mission = SurveyMission(world)
        before = world.mission.status()
        self.assertEqual(before['state'], MissionState.IDLE)
        self.assertEqual(before['progress'], 0.0)
        world.mission.start(AREA, 0.0, tuple(world.drone), tuple(world.husky))
        world.run()
        after = world.mission.status()
        self.assertEqual(after['pois_cleared'], 1)
        self.assertEqual(after['pois_total'], 1)
        self.assertEqual(after['progress'], 1.0)

    def test_observe_frame_ignores_frames_outside_the_survey(self):
        world = FakeWorld()
        world.mission = SurveyMission(world)
        world.mission.observe_frame(0, 0, 9, 0)             # no grid yet: harmless
        world.start(AREA)
        world.mission.observe_frame(0, 0, 40.0, 0)           # far beyond the depth range
        self.assertEqual(world.mission.grid.fraction, 0.0)


if __name__ == '__main__':
    unittest.main()