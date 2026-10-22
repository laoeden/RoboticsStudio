"""Survey-then-investigate mission logic (pure Python, no ROS or Tk).

Sequence
--------
1. TAKEOFF   climb straight up to the survey altitude.
2. SWEEP     fly the lawnmower rows from ``coverage_planner`` while the UI's
             camera pipeline pins dry grass; every processed frame is fed to
             ``observe_frame`` so the area actually seen is tracked.
3. GAP_FILL  re-fly any region the camera missed (at most ``max_gap_rounds``).
4. SETTLE    hover a few seconds so the last frames finish pinning, then the
             POI list is frozen and the drone flies home while the Husky works.
5. HUSKY_TRANSIT / INVESTIGATE  visit each pending POI in a short order. A
   POI is "cleared" when ``grassRemover`` reports a removal near it, or a
   "false_positive" when the Husky arrives, dwells, and nothing is removed.
6. RECHARGE   go home and recharge when the battery cannot cover the next
   leg plus the trip back.
7. HUSKY_RETURN, then COMPLETE.

The class never touches ROS: everything it does to the outside world goes
through a ``MissionIO``; time is passed in, so tests can drive it with a fake
clock.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from Mission.battery_model import BatteryConfig
from Mission.coverage_planner import (
    CoverageGrid, DEPTH_MAX, Rect, SurveyConfig, SweepPlan, gap_waypoints,
    order_visits, plan_survey,
)

# POI statuses (also stored on the grass pins).
PENDING = 'pending'
ACTIVE = 'active'
CLEARED = 'cleared'
FALSE_POSITIVE = 'false_positive'
UNREACHABLE = 'unreachable'
VISITED = 'visited'          # reached, but nothing was there to confirm removal
OPEN_STATUSES = (PENDING, ACTIVE)


class MissionState(str, Enum):
    IDLE = 'IDLE'
    TAKEOFF = 'TAKEOFF'
    SWEEP = 'SURVEYING'
    GAP_FILL = 'FILLING GAPS'
    SETTLE = 'PROCESSING SURVEY'
    HUSKY_TRANSIT = 'HUSKY EN ROUTE'
    INVESTIGATE = 'INVESTIGATING'
    RECHARGE = 'RECHARGING'
    HUSKY_RETURN = 'HUSKY RETURNING'
    COMPLETE = 'COMPLETE'
    ABORTED = 'ABORTED'


ACTIVE_STATES = (
    MissionState.TAKEOFF, MissionState.SWEEP, MissionState.GAP_FILL,
    MissionState.SETTLE, MissionState.HUSKY_TRANSIT, MissionState.INVESTIGATE, MissionState.RECHARGE,
    MissionState.HUSKY_RETURN,
)


@dataclass
class MissionConfig:
    survey: SurveyConfig = field(default_factory=SurveyConfig)
    drone_base: tuple[float, float, float] = (0.0, 0.0, 10.0)
    dock: tuple[float, float] = (0.0, 0.0)
    drone_tolerance: float = 0.3          # waypoint counts as reached inside this
    takeoff_tolerance: float = 0.5
    drone_base_tolerance: float = 0.6
    waypoint_timeout: float = 90.0        # s per drone goal
    max_waypoint_failures: int = 3        # consecutive timeouts before aborting
    min_gap_cells: int = 4
    settle_time: float = 4.0              # s of hovering so the last frames finish pinning
    max_gap_rounds: int = 2
    min_coverage_for_pois: float = 0.5    # below this the survey is declared failed
    poi_match_radius: float = 3.0         # removal event -> POI match (m)
    investigate_dwell: float = 3.0        # s to wait at a POI for a removal event
    husky_tolerance: float = 0.35         # arrival if closer than this to the goal
    dock_tolerance: float = 0.5
    husky_speed_estimate: float = 1.0     # m/s, for leg timeouts
    husky_leg_timeout_min: float = 60.0
    return_timeout: float = 300.0
    recharge_timeout: float = 600.0
    detour_factor: float = 1.5            # real path / straight line
    battery_margin: float = 5.0           # percent kept in reserve
    battery: BatteryConfig = field(default_factory=BatteryConfig)


@dataclass
class Telemetry:
    drone: tuple[float, float, float] | None = None
    husky: tuple[float, float] | None = None
    battery: float | None = None          # percent; None if no battery node
    battery_low: bool = False
    remover_online: bool = True           # is grassRemover publishing hazard events?


class MissionIO:
    """Everything the mission does to the outside world. Override in the UI."""

    def prepare_survey(self, area: Rect) -> None: ...
    def end_survey(self) -> None: ...
    def drone_goal(self, x: float, y: float, z: float) -> None: ...
    def husky_goal(self, x: float, y: float) -> None: ...
    def command(self, text: str) -> None: ...
    def log(self, source: str, message: str) -> None: ...
    def list_pois(self, area: Rect) -> list[dict]: return []
    def mark_poi(self, poi_id, status: str) -> None: ...


def parse_husky_status(text: str):
    """Parse ``'EVENT x y detail'`` from ``/husky1/goal_status``; None if malformed."""
    parts = text.strip().split(None, 3)
    if len(parts) < 3:
        return None
    try:
        x, y = float(parts[1]), float(parts[2])
    except ValueError:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return parts[0].upper(), x, y, parts[3] if len(parts) > 3 else ''


class SurveyMission:
    def __init__(self, io: MissionIO, config: MissionConfig | None = None):
        self.io = io
        self.cfg = config or MissionConfig()
        self._lock = threading.RLock()
        self.state = MissionState.IDLE
        self.reason = ''
        self.area: Rect | None = None
        self.plan: SweepPlan | None = None
        self.grid: CoverageGrid | None = None
        self.tel = Telemetry()
        self._reset_run()

    # ------------------------------------------------------------------ public
    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    def start(self, area: Rect, now: float, drone_pos, husky_pos=None) -> bool:
        with self._lock:
            if self.active:
                self.io.log('MISSION', 'A survey is already running')
                return False
            if drone_pos is None:
                self.io.log('MISSION', 'Cannot start: no drone odometry yet')
                return False
            self._reset_run()
            self.area = area
            self.plan = plan_survey(area, self.cfg.survey, tuple(drone_pos[:2]))
            self.grid = CoverageGrid(area)
            self.tel = Telemetry(drone=tuple(drone_pos),
                                 husky=None if husky_pos is None else tuple(husky_pos[:2]))
            self.io.prepare_survey(area)
            p = self.plan
            self.io.log('MISSION',
                        f'Survey {area.width:.0f}x{area.height:.0f} m: {p.rows} rows, '
                        f'{len(p.waypoints)} waypoints, {p.path_length:.0f} m, '
                        f'altitude {p.flight_z:.1f} m, ETA {p.eta_s:.0f} s')
            for warning in p.warnings:
                self.io.log('MISSION', f'WARNING: {warning}')
            self.state = MissionState.TAKEOFF
            self.reason = ''
            self._send_drone((drone_pos[0], drone_pos[1], p.flight_z), now)
            return True

    def abort(self, reason: str, hover: bool = True) -> None:
        with self._lock:
            if not self.active:
                return
            self.state = MissionState.ABORTED
            self.reason = reason
            self.io.log('MISSION', f'ABORTED: {reason}')
            if hover and self.tel.drone is not None:
                self.io.drone_goal(*self.tel.drone)
            if self._husky_busy:
                self.io.command('STOP')
            for poi in self.pois.values():
                if poi['status'] == ACTIVE:
                    self._set_status(poi, PENDING)
            self.io.end_survey()

    def observe_frame(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        """Record what a processed camera frame saw. Safe to call from any thread."""
        with self._lock:
            if self.grid is None or self.state not in (
                    MissionState.TAKEOFF, MissionState.SWEEP, MissionState.GAP_FILL):
                return
            cfg = self.cfg.survey
            # Be conservative: assume the highest terrain, so the footprint is the smallest.
            agl = cfg.agl_min_for_z(z)
            if agl <= 0 or z - cfg.camera_drop - cfg.terrain_min > DEPTH_MAX:
                return          # too close, or low ground beyond the depth range
            self.grid.mark(x, y, agl, yaw)

    def on_hazard_removed(self, x: float, y: float) -> None:
        with self._lock:
            best, best_d = None, self.cfg.poi_match_radius
            for poi in self.pois.values():
                if poi['status'] not in OPEN_STATUSES:
                    continue
                d = math.hypot(poi['x'] - x, poi['y'] - y)
                if d <= best_d:
                    best, best_d = poi, d
            if best is None:
                if self.active:
                    self.extra_removed += 1
                    self.io.log('HAZARD', f'Hazard removed at ({x:.1f}, {y:.1f}) '
                                          'that was not on the POI list')
                return
            self._set_status(best, CLEARED)
            self.io.log('HAZARD', f'POI {best["id"]} cleared at ({x:.1f}, {y:.1f})')

    def on_husky_status(self, text: str) -> None:
        parsed = parse_husky_status(text)
        if parsed is not None:
            with self._lock:
                self._husky_events.append(parsed)

    def update(self, now: float, tel: Telemetry) -> None:
        with self._lock:
            self.tel = tel
            if not self.active:
                self._husky_events.clear()
                return
            events = list(self._husky_events)
            self._husky_events.clear()
            self._check_drone_return(now)
            handler = {
                MissionState.TAKEOFF: self._tick_takeoff,
                MissionState.SWEEP: self._tick_sweep,
                MissionState.GAP_FILL: self._tick_sweep,
                MissionState.SETTLE: self._tick_settle,
                MissionState.HUSKY_TRANSIT: self._tick_transit,
                MissionState.INVESTIGATE: self._tick_investigate,
                MissionState.RECHARGE: self._tick_recharge,
                MissionState.HUSKY_RETURN: self._tick_return,
            }[self.state]
            handler(now, events)

    def status(self) -> dict:
        with self._lock:
            counts = {s: 0 for s in (PENDING, ACTIVE, CLEARED, FALSE_POSITIVE,
                                      UNREACHABLE, VISITED)}
            for poi in self.pois.values():
                counts[poi['status']] = counts.get(poi['status'], 0) + 1
            total = len(self.pois)
            done = total - counts[PENDING] - counts[ACTIVE]
            coverage = self.grid.fraction if self.grid is not None else 0.0
            if self.state in (MissionState.TAKEOFF, MissionState.SWEEP,
                              MissionState.GAP_FILL, MissionState.SETTLE):
                progress = 0.5 * coverage
            elif self.state in (MissionState.IDLE,):
                progress = 0.0
            elif self.state == MissionState.COMPLETE:
                progress = 1.0
            else:
                progress = 0.5 + 0.5 * ((done / total) if total else 1.0)
                if self.state == MissionState.HUSKY_RETURN:
                    progress = max(progress, 0.95)
            return {
                'state': self.state, 'reason': self.reason, 'progress': progress,
                'coverage': coverage, 'waypoint': self.wp_index,
                'waypoints': len(self.waypoints), 'pois_total': total,
                'pois_cleared': counts[CLEARED], 'pois_false': counts[FALSE_POSITIVE],
                'pois_unreachable': counts[UNREACHABLE], 'pois_visited': counts[VISITED],
                'pois_open': counts[PENDING] + counts[ACTIVE],
                'extra_removed': self.extra_removed, 'gap_rounds': self.gap_rounds,
                'coverage_version': self.grid.version if self.grid is not None else 0,
                'pois_found': self.pois_frozen,
            }

    def coverage_snapshot(self):
        """(version, seen-array in screen order, area) for drawing; None before a survey."""
        with self._lock:
            if self.grid is None:
                return None
            return self.grid.version, self.grid.image().copy(), self.grid.area

    # ----------------------------------------------------------------- helpers
    def _reset_run(self) -> None:
        self.waypoints: list[tuple[float, float, float]] = []
        self.wp_index = 0
        self.gap_rounds = 0
        self.pois: dict = {}
        self.pois_frozen = False
        self.order: list = []
        self.current = None
        self.extra_removed = 0
        self._husky_events: deque = deque()
        self._husky_busy = False
        self._drone_goal = None
        self._drone_goal_time = 0.0
        self._failures = 0
        self._drone_returning = False
        self._drone_home = False
        self._drone_return_time = 0.0
        self._husky_goal = None
        self._husky_goal_time = 0.0
        self._husky_retries = 0
        self._dwell_start = 0.0
        self._phase_start = 0.0

    def _set_status(self, poi: dict, status: str) -> None:
        poi['status'] = status
        self.io.mark_poi(poi['id'], status)

    def _send_drone(self, wp, now: float) -> None:
        self._drone_goal = wp
        self._drone_goal_time = now
        self.io.drone_goal(*wp)

    def _drone_distance(self, goal) -> float:
        d = self.tel.drone
        if d is None:
            return math.inf
        return math.dist(d, goal)

    def _husky_distance(self, goal) -> float:
        h = self.tel.husky
        if h is None:
            return math.inf
        return math.hypot(h[0] - goal[0], h[1] - goal[1])

    def _matching(self, events, kinds, goal):
        for kind, x, y, detail in events:
            if kind in kinds and math.hypot(x - goal[0], y - goal[1]) <= 0.6:
                return kind, detail
        return None

    # --------------------------------------------------------------- drone side
    def _tick_takeoff(self, now, events):
        goal = self._drone_goal
        if self._drone_distance(goal) <= self.cfg.takeoff_tolerance:
            self.state = MissionState.SWEEP
            self.waypoints = list(self.plan.waypoints)
            self.wp_index = 0
            self._failures = 0
            self.io.log('UAV', 'At survey altitude; starting sweep')
            self._send_drone(self.waypoints[0], now)
        elif now - self._drone_goal_time > self.cfg.waypoint_timeout:
            self.abort('drone did not reach survey altitude')

    def _tick_sweep(self, now, events):
        goal = self._drone_goal
        arrived = self._drone_distance(goal) <= self.cfg.drone_tolerance
        timed_out = now - self._drone_goal_time > self.cfg.waypoint_timeout
        if not (arrived or timed_out):
            return
        if arrived:
            self._failures = 0
        else:
            self._failures += 1
            self.io.log('UAV', f'Waypoint {self.wp_index + 1}/{len(self.waypoints)} '
                               'timed out; skipping')
            if self._failures >= self.cfg.max_waypoint_failures:
                self.abort('drone is not reaching its waypoints')
                return
        self.wp_index += 1
        if self.wp_index < len(self.waypoints):
            self._send_drone(self.waypoints[self.wp_index], now)
        else:
            self._finish_sweep(now)

    def _finish_sweep(self, now) -> None:
        cfg = self.cfg
        gaps = self.grid.gaps(cfg.min_gap_cells)
        if gaps and self.gap_rounds < cfg.max_gap_rounds:
            here = (self.tel.drone[0], self.tel.drone[1]) if self.tel.drone else (0.0, 0.0)
            waypoints = gap_waypoints(self.grid, cfg.survey, self.plan.flight_z, here,
                                      cfg.min_gap_cells)
            if waypoints:
                self.gap_rounds += 1
                self.state = MissionState.GAP_FILL
                self.waypoints = waypoints
                self.wp_index = 0
                self._failures = 0
                self.io.log('UAV', f'{len(gaps)} unseen region(s) '
                                   f'({100 * (1 - self.grid.fraction):.1f}% of the area); '
                                   f'gap-fill pass {self.gap_rounds}')
                self._send_drone(waypoints[0], now)
                return
        left = 100.0 * (1.0 - self.grid.fraction)
        if gaps:
            self.io.log('UAV', f'WARNING: {left:.1f}% of the area could not be surveyed')
        else:
            self.io.log('UAV', f'Survey complete: {100 * self.grid.fraction:.1f}% covered')
        if self.grid.fraction < cfg.min_coverage_for_pois:
            self.abort('too little of the area was surveyed')
            return
        # Hover while the camera pipeline works through the last frames; a grass
        # patch seen at the end of the sweep still needs its confirming sightings.
        self.state = MissionState.SETTLE
        self._phase_start = now

    def _tick_settle(self, now, events):
        if now - self._phase_start >= self.cfg.settle_time:
            self._begin_husky_phase(now)

    def _check_drone_return(self, now) -> None:
        if not self._drone_returning or self._drone_home:
            return
        if (self._drone_distance(self.cfg.drone_base) <= self.cfg.drone_base_tolerance
                or now - self._drone_return_time > self.cfg.return_timeout):
            self._drone_home = True
            self.io.log('UAV', 'Drone back at base')

    # --------------------------------------------------------------- husky side
    def _begin_husky_phase(self, now) -> None:
        cfg = self.cfg
        # The drone goes home while the Husky works.
        self._drone_returning = True
        self._drone_return_time = now
        self.io.drone_goal(*cfg.drone_base)
        self.io.log('UAV', 'Returning to base')
        found = self.io.list_pois(self.area)
        self.pois = {p['id']: {'id': p['id'], 'x': float(p['x']), 'y': float(p['y']),
                               'status': PENDING} for p in found}
        for poi in self.pois.values():
            self.io.mark_poi(poi['id'], PENDING)
        self.pois_frozen = True
        if not self.pois:
            self.io.log('MISSION', 'No points of interest found; the Husky stays at base')
            self._finish(now)
            return
        self.io.log('MISSION', f'{len(self.pois)} point(s) of interest found')
        self._phase_start = now
        self._plan_order()
        self._next_poi(now)

    def _husky_start(self) -> tuple[float, float]:
        return self.tel.husky if self.tel.husky is not None else self.cfg.dock

    def _plan_order(self) -> None:
        open_pois = [p for p in self.pois.values() if p['status'] in OPEN_STATUSES]
        idx = order_visits(self._husky_start(), [(p['x'], p['y']) for p in open_pois],
                           end=self.cfg.dock)
        self.order = [open_pois[i]['id'] for i in idx]

    def _battery_needed(self, poi) -> float:
        """Percent needed to reach ``poi`` from here and then get back to the dock."""
        cfg = self.cfg
        leg = math.dist(self._husky_start(), (poi['x'], poi['y']))
        back = math.dist((poi['x'], poi['y']), cfg.dock)
        return ((leg + back) * cfg.detour_factor * cfg.battery.drain_per_metre
                + cfg.battery.low_threshold + cfg.battery_margin)

    def _next_poi(self, now) -> None:
        cfg = self.cfg
        while True:
            self.order = [i for i in self.order if self.pois[i]['status'] in OPEN_STATUSES]
            if not self.order:
                self._begin_return(now)
                return
            poi = self.pois[self.order[0]]
            level = self.tel.battery
            if level is not None:
                from_dock = ((math.dist(cfg.dock, (poi['x'], poi['y'])) * 2)
                             * cfg.detour_factor * cfg.battery.drain_per_metre
                             + cfg.battery.low_threshold + cfg.battery_margin)
                if from_dock > cfg.battery.resume_level:
                    self.io.log('HUSKY', f'POI {poi["id"]} is too far for one battery charge')
                    self.order.pop(0)
                    self._set_status(poi, UNREACHABLE)
                    continue
                # A freshly charged Husky at the dock leaves even if its estimate is tight,
                # otherwise it would loop between the dock and the POI forever.
                fresh = (self._husky_distance(cfg.dock) <= cfg.dock_tolerance
                         and level >= cfg.battery.resume_level)
                if not fresh and (level < self._battery_needed(poi) or self.tel.battery_low):
                    self._begin_recharge(now, f'battery {level:.0f}% is too low for POI {poi["id"]}')
                    return
            self.current = poi
            self._set_status(poi, ACTIVE)
            self._husky_retries = 0
            self._send_husky(poi, now)
            self.state = MissionState.HUSKY_TRANSIT
            return

    def _send_husky(self, poi, now) -> None:
        self._husky_goal = (poi['x'], poi['y'])
        self._husky_goal_time = now
        self._husky_busy = True
        self.io.husky_goal(*self._husky_goal)
        self.io.log('HUSKY', f'Heading to POI {poi["id"]} at ({poi["x"]:.1f}, {poi["y"]:.1f})')

    def _leg_timeout(self, goal) -> float:
        cfg = self.cfg
        leg = math.dist(self._husky_start(), goal) * cfg.detour_factor
        return max(cfg.husky_leg_timeout_min, 3.0 * leg / cfg.husky_speed_estimate)

    def _tick_transit(self, now, events):
        poi = self.current
        goal = self._husky_goal
        if poi['status'] not in OPEN_STATUSES:       # cleared on the way past
            self._husky_busy = False
            self.io.command('CLEAR_GOAL')
            self._next_poi(now)
            return
        if self.tel.battery_low:
            self._set_status(poi, PENDING)
            self._begin_recharge(now, 'battery low')
            return
        failed = self._matching(events, ('CANCELLED', 'REJECTED'), goal)
        arrived = (self._husky_distance(goal) <= self.cfg.husky_tolerance
                   or self._matching(events, ('ARRIVED',), goal) is not None)
        if arrived:
            self.state = MissionState.INVESTIGATE
            self._dwell_start = now
            self.io.log('HUSKY', f'At POI {poi["id"]}; investigating')
            return
        timed_out = now - self._husky_goal_time > self._leg_timeout(goal)
        if failed or timed_out:
            why = f'{failed[0].lower()}: {failed[1]}' if failed else 'timed out'
            if self._husky_retries < 1:
                self._husky_retries += 1
                self.io.log('HUSKY', f'POI {poi["id"]} goal {why}; retrying once')
                self._send_husky(poi, now)
                return
            self.io.log('HUSKY', f'POI {poi["id"]} unreachable ({why})')
            self._husky_busy = False
            self.io.command('CLEAR_GOAL')
            self._set_status(poi, UNREACHABLE)
            self._next_poi(now)

    def _tick_investigate(self, now, events):
        poi = self.current
        if poi['status'] == CLEARED:
            self._husky_busy = False
            self._next_poi(now)
            return
        if poi['status'] not in OPEN_STATUSES:
            self._next_poi(now)
            return
        if now - self._dwell_start < self.cfg.investigate_dwell:
            return
        self._husky_busy = False
        if self.tel.remover_online:
            self._set_status(poi, FALSE_POSITIVE)
            self.io.log('HUSKY', f'POI {poi["id"]}: nothing to remove (false positive)')
        else:
            self._set_status(poi, VISITED)
            self.io.log('HUSKY', f'POI {poi["id"]} visited; hazard remover offline, '
                                 'removal not confirmed')
        self._next_poi(now)

    def _begin_recharge(self, now, why: str) -> None:
        self.state = MissionState.RECHARGE
        self._phase_start = now
        self._husky_busy = True
        self.io.log('HUSKY', f'Returning to recharge ({why})')
        self.io.command('RETURN_TO_BASE')

    def _tick_recharge(self, now, events):
        cfg = self.cfg
        docked = self._husky_distance(cfg.dock) <= cfg.dock_tolerance
        level = self.tel.battery
        charged = (not self.tel.battery_low and
                   (level is None or level >= cfg.battery.resume_level))
        if docked and charged:
            self.io.log('HUSKY', 'Recharged; resuming')
            self._husky_busy = False
            self._plan_order()
            self._next_poi(now)
        elif now - self._phase_start > cfg.recharge_timeout:
            self.abort('Husky did not finish recharging')

    def _begin_return(self, now) -> None:
        self.state = MissionState.HUSKY_RETURN
        self._phase_start = now
        self._husky_busy = True
        self.io.log('HUSKY', 'All points handled; returning to base')
        self.io.command('RETURN_TO_BASE')

    def _tick_return(self, now, events):
        cfg = self.cfg
        docked = (self._husky_distance(cfg.dock) <= cfg.dock_tolerance
                  or any(e[0] == 'DOCKED' for e in events))
        if docked:
            self._husky_busy = False
            drone_wait = self._drone_returning and not self._drone_home
            if drone_wait:
                return           # let the drone finish landing at base
            self._finish(now)
        elif now - self._phase_start > cfg.return_timeout:
            self.abort('Husky did not get back to base')

    def _finish(self, now) -> None:
        self.state = MissionState.COMPLETE
        s = self.status()
        self.io.log('MISSION', f'COMPLETE: {s["pois_cleared"]} cleared, '
                               f'{s["pois_false"]} false positive, '
                               f'{s["pois_unreachable"]} unreachable, '
                               f'{s["pois_visited"]} visited (unconfirmed)')
        self.io.end_survey()