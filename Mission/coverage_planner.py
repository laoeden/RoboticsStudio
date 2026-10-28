"""Survey coverage planning for the Parrot's downward-looking camera.

Pure Python + numpy: no ROS, Tk or OpenCV imports, so it can be unit tested
without the simulator.

Camera facts (see ``41068_ignition_bringup/urdf_parrot/parrot.gazebo.xacro``
and ``parrot.urdf.xacro``; ``test_coverage_planner.py`` checks the constants
below against those files):

* 720 x 480 px, horizontal FOV 120 deg, so fx = fy = 207.85 px and the
  vertical FOV is about 98.2 deg.
* Mounted looking straight down. Image "up" is the drone's +X axis and image
  "right" is the drone's -Y axis, so the wide (120 deg) image axis lies along
  the drone's Y axis.
* The camera sits 0.255 m below the odometry origin.
* Depth is only valid from 0.4 m to 10 m.

The drone never yaws (``DroneMotion`` only commands linear velocity), so the
footprint is axis aligned with the world: ``2 * agl * tan(60 deg)`` metres
along world Y and ``2 * agl * 1.1547`` metres along world X.

Coverage strategy: parallel back-and-forth ("lawnmower") rows. Rows run along
the axis that puts the wide image dimension across the track, the number of
rows is the minimum that still gives the requested overlap, and the flight
altitude is the *lowest* one that still achieves that row count (same flight
distance, best resolution).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

# --- Camera model -----------------------------------------------------------
IMAGE_WIDTH = 720
IMAGE_HEIGHT = 480
HORIZONTAL_FOV = 2.0944  # rad (120 degrees)
FX = IMAGE_WIDTH / (2.0 * math.tan(HORIZONTAL_FOV / 2.0))
FY = FX
# Half-angle tangents: "across" is drone Y (image width), "along" is drone X.
TAN_HALF_ACROSS = math.tan(HORIZONTAL_FOV / 2.0)
TAN_HALF_ALONG = (IMAGE_HEIGHT / 2.0) / FY
VERTICAL_FOV = 2.0 * math.atan(TAN_HALF_ALONG)
CAMERA_DROP = 0.255  # camera is this far below the odometry origin (m)
DEPTH_MIN = 0.4
DEPTH_MAX = 10.0


def footprint(agl: float) -> tuple[float, float]:
    """Ground footprint (x_extent, y_extent) in metres at height ``agl``.

    Extents are along the drone's X and Y axes, which equal world X and Y
    while the drone's yaw is zero.
    """
    return 2.0 * agl * TAN_HALF_ALONG, 2.0 * agl * TAN_HALF_ACROSS


def ground_sample_distance(agl: float) -> float:
    """Metres of ground per pixel at height ``agl``."""
    return agl / FX


# --- Geometry ---------------------------------------------------------------
@dataclass(frozen=True)
class Rect:
    """Axis aligned rectangle in world metres."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        values = (self.xmin, self.ymin, self.xmax, self.ymax)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Rect coordinates must be finite')
        if self.xmax < self.xmin or self.ymax < self.ymin:
            raise ValueError('Rect max must not be below min')

    @staticmethod
    def from_corners(x0: float, y0: float, x1: float, y1: float) -> 'Rect':
        return Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @property
    def width(self) -> float:  # extent along X
        return self.xmax - self.xmin

    @property
    def height(self) -> float:  # extent along Y
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.xmin, self.ymin, self.xmax, self.ymax

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.xmin - margin <= x <= self.xmax + margin
                and self.ymin - margin <= y <= self.ymax + margin)

    def clamp(self, limit: 'Rect') -> 'Rect':
        return Rect(
            min(max(self.xmin, limit.xmin), limit.xmax),
            min(max(self.ymin, limit.ymin), limit.ymax),
            min(max(self.xmax, limit.xmin), limit.xmax),
            min(max(self.ymax, limit.ymin), limit.ymax),
        )

    def padded(self, amount: float) -> 'Rect':
        return Rect(self.xmin - amount, self.ymin - amount,
                    self.xmax + amount, self.ymax + amount)


# --- Survey planning --------------------------------------------------------
@dataclass(frozen=True)
class SurveyConfig:
    overlap: float = 0.2            # side overlap as a fraction of the swath
    min_overlap_m: float = 2.5      # side overlap is never below one grass patch
    safety_margin: float = 0.05     # footprint is shrunk by this fraction
    max_agl: float = 9.5            # highest camera height above ground (depth clip 10 m)
    min_agl: float = 3.0            # lowest camera height above ground
    terrain_min: float = 0.0        # terrain height range in the world (m)
    terrain_max: float = 2.4
    camera_drop: float = CAMERA_DROP
    cruise_speed: float = 2.0       # m/s, used for the ETA only
    waypoint_time: float = 3.0      # s lost slowing to each waypoint (ETA only)
    sweep_axis: str = 'auto'        # 'auto', 'x' (rows along X) or 'y'

    def __post_init__(self) -> None:
        if not 0.0 <= self.overlap < 0.9:
            raise ValueError('overlap must be in [0, 0.9)')
        if not 0.0 <= self.safety_margin < 0.5:
            raise ValueError('safety_margin must be in [0, 0.5)')
        if self.sweep_axis not in ('auto', 'x', 'y'):
            raise ValueError("sweep_axis must be 'auto', 'x' or 'y'")
        if self.min_agl <= 0 or self.max_agl < self.min_agl:
            raise ValueError('need 0 < min_agl <= max_agl')
        if self.terrain_max < self.terrain_min:
            raise ValueError('terrain_max must not be below terrain_min')
        if self.cruise_speed <= 0:
            raise ValueError('cruise_speed must be positive')

    @property
    def relief(self) -> float:
        return self.terrain_max - self.terrain_min

    def flight_z(self, agl_min: float) -> float:
        """Odometry z that puts the camera ``agl_min`` above the highest terrain."""
        return agl_min + self.camera_drop + self.terrain_max

    def agl_min_for_z(self, z: float) -> float:
        return z - self.camera_drop - self.terrain_max


@dataclass
class SweepPlan:
    area: Rect
    axis: str
    rows: int
    flight_z: float
    agl_min: float
    agl_max: float
    footprint: tuple[float, float]       # full (x_extent, y_extent) at agl_min
    waypoints: list[tuple[float, float, float]]
    path_length: float                   # horizontal length of the sweep (m)
    transit_length: float                # start -> first waypoint (m)
    eta_s: float
    gsd_cm: float                        # worst-case cm per pixel
    warnings: list[str] = field(default_factory=list)


def _row_count(span: float, swath: float, step_max: float) -> int:
    if span <= swath + 1e-9:
        return 1
    return int(math.ceil((span - swath) / step_max - 1e-9)) + 1


def _step_max(swath: float, cfg: SurveyConfig) -> float:
    step = min(swath * (1.0 - cfg.overlap), swath - cfg.min_overlap_m)
    return max(step, 0.25 * swath)


def _rows_at(agl: float, cross_span: float, axis: str, cfg: SurveyConfig) -> int:
    fx_, fy_ = footprint(agl)
    cross_fp = (fy_ if axis == 'x' else fx_) * (1.0 - cfg.safety_margin)
    return _row_count(cross_span, cross_fp, _step_max(cross_fp, cfg))


def _row_centres(lo: float, span: float, swath: float, n: int) -> list[float]:
    if n == 1:
        return [lo + span / 2.0]
    step = (span - swath) / (n - 1)
    return [lo + swath / 2.0 + i * step for i in range(n)]


def _solve_agl(area: Rect, axis: str, cfg: SurveyConfig, warnings: list[str]) -> float:
    """Lowest minimum-AGL that still gives the fewest rows allowed by ``max_agl``."""
    cross_span = area.height if axis == 'x' else area.width
    agl_hi = cfg.max_agl - cfg.relief
    agl_lo = cfg.min_agl
    if agl_hi < agl_lo:
        warnings.append(
            'Terrain relief leaves no altitude band inside the depth range; '
            'flying at the minimum altitude.')
        return agl_lo
    target = _rows_at(agl_hi, cross_span, axis, cfg)
    if _rows_at(agl_lo, cross_span, axis, cfg) == target:
        return agl_lo
    lo, hi = agl_lo, agl_hi  # rows(lo) > target == rows(hi)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _rows_at(mid, cross_span, axis, cfg) <= target:
            hi = mid
        else:
            lo = mid
    return min(agl_hi, hi + 0.01)


def _build_waypoints(area: Rect, axis: str, agl_min: float, z: float,
                     cfg: SurveyConfig, reverse_rows: bool, flip_first: bool
                     ) -> tuple[list[tuple[float, float, float]], int]:
    fx_, fy_ = footprint(agl_min)
    keep = 1.0 - cfg.safety_margin
    if axis == 'x':
        a_lo, a_hi, c_lo, c_hi = area.xmin, area.xmax, area.ymin, area.ymax
        along_fp, cross_fp = fx_ * keep, fy_ * keep
    else:
        a_lo, a_hi, c_lo, c_hi = area.ymin, area.ymax, area.xmin, area.xmax
        along_fp, cross_fp = fy_ * keep, fx_ * keep
    n = _row_count(c_hi - c_lo, cross_fp, _step_max(cross_fp, cfg))
    centres = _row_centres(c_lo, c_hi - c_lo, cross_fp, n)
    if a_hi - a_lo <= along_fp:
        ends = [(a_lo + a_hi) / 2.0]
    else:
        ends = [a_lo + along_fp / 2.0, a_hi - along_fp / 2.0]
    if reverse_rows:
        centres.reverse()
    points: list[tuple[float, float, float]] = []
    forward = not flip_first
    for c in centres:
        row_ends = ends if forward else list(reversed(ends))
        for a in row_ends:
            points.append((a, c, z) if axis == 'x' else (c, a, z))
        forward = not forward
    return points, n


def _polyline_length(points: list[tuple[float, float, float]]) -> float:
    return sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(points, points[1:]))


def _plan_axis(area: Rect, axis: str, cfg: SurveyConfig, start_xy,
               fixed_z: float | None) -> SweepPlan:
    warnings: list[str] = []
    if fixed_z is None:
        agl_min = _solve_agl(area, axis, cfg, warnings)
        z = cfg.flight_z(agl_min)
    else:
        z = float(fixed_z)
        agl_min = cfg.agl_min_for_z(z)
        if agl_min < DEPTH_MIN:
            raise ValueError('fixed_z leaves the camera too close to the terrain')
    agl_max = agl_min + cfg.relief
    if agl_max > DEPTH_MAX:
        warnings.append(
            f'Camera may be {agl_max:.1f} m above low ground, beyond the '
            f'{DEPTH_MAX:.0f} m depth range; pins there may be missed.')

    best = None
    for reverse_rows, flip_first in itertools.product((False, True), repeat=2):
        wps, n = _build_waypoints(area, axis, agl_min, z, cfg, reverse_rows, flip_first)
        transit = (math.hypot(wps[0][0] - start_xy[0], wps[0][1] - start_xy[1])
                   if start_xy is not None else 0.0)
        # Prefer the closest start; ties keep the earlier (canonical) variant.
        if best is None or transit < best[0] - 1e-9:
            best = (transit, wps, n)
    transit, wps, n = best
    length = _polyline_length(wps)
    eta = (length + transit) / cfg.cruise_speed + len(wps) * cfg.waypoint_time
    return SweepPlan(
        area=area, axis=axis, rows=n, flight_z=z, agl_min=agl_min, agl_max=agl_max,
        footprint=footprint(agl_min), waypoints=wps, path_length=length,
        transit_length=transit, eta_s=eta,
        gsd_cm=100.0 * ground_sample_distance(agl_max), warnings=warnings)


def plan_survey(area: Rect, cfg: SurveyConfig | None = None,
                start_xy: tuple[float, float] | None = None,
                fixed_z: float | None = None) -> SweepPlan:
    """Plan a lawnmower sweep that covers ``area``.

    ``fixed_z`` pins the flight altitude (used when filling gaps so the
    footprint size matches the main sweep).
    """
    cfg = cfg or SurveyConfig()
    axes = ('x', 'y') if cfg.sweep_axis == 'auto' else (cfg.sweep_axis,)
    plans = [_plan_axis(area, axis, cfg, start_xy, fixed_z) for axis in axes]
    # Shortest total flight wins; near ties prefer rows along X (wide swath).
    chosen = plans[0]
    for candidate in plans[1:]:
        if candidate.path_length + candidate.transit_length < (
                chosen.path_length + chosen.transit_length) - 0.5:
            chosen = candidate
    return chosen


# --- Coverage tracking ------------------------------------------------------
class CoverageGrid:
    """Which parts of the survey area the camera has actually seen."""

    def __init__(self, area: Rect, cell: float = 0.5):
        if cell <= 0:
            raise ValueError('cell must be positive')
        self.area = area
        self.nx = max(1, int(math.ceil(area.width / cell - 1e-9)))
        self.ny = max(1, int(math.ceil(area.height / cell - 1e-9)))
        self.dx = area.width / self.nx if area.width > 0 else cell
        self.dy = area.height / self.ny if area.height > 0 else cell
        self.seen = np.zeros((self.ny, self.nx), dtype=bool)
        self._xs = area.xmin + (np.arange(self.nx) + 0.5) * self.dx
        self._ys = area.ymin + (np.arange(self.ny) + 0.5) * self.dy
        self.version = 0

    def mark(self, x: float, y: float, agl: float, yaw: float = 0.0,
             shrink: float = 0.0) -> int:
        """Mark cells inside the camera footprint; returns newly seen cells."""
        if not all(math.isfinite(v) for v in (x, y, agl, yaw)) or agl <= 0:
            return 0
        fx_, fy_ = footprint(agl)
        half_x, half_y = fx_ * (1.0 - shrink) / 2.0, fy_ * (1.0 - shrink) / 2.0
        reach = math.hypot(half_x, half_y)
        i0 = max(0, int(math.floor((x - reach - self.area.xmin) / self.dx)))
        i1 = min(self.nx, int(math.ceil((x + reach - self.area.xmin) / self.dx)) + 1)
        j0 = max(0, int(math.floor((y - reach - self.area.ymin) / self.dy)))
        j1 = min(self.ny, int(math.ceil((y + reach - self.area.ymin) / self.dy)) + 1)
        if i0 >= i1 or j0 >= j1:
            return 0
        gx, gy = np.meshgrid(self._xs[i0:i1] - x, self._ys[j0:j1] - y)
        c, s = math.cos(yaw), math.sin(yaw)
        along = gx * c + gy * s          # drone X axis
        across = -gx * s + gy * c        # drone Y axis
        inside = (np.abs(along) <= half_x) & (np.abs(across) <= half_y)
        window = self.seen[j0:j1, i0:i1]
        new = int(np.count_nonzero(inside & ~window))
        if new:
            window |= inside
            self.version += 1
        return new

    @property
    def fraction(self) -> float:
        return float(np.count_nonzero(self.seen)) / self.seen.size

    def gaps(self, min_cells: int = 4) -> list[Rect]:
        """Bounding boxes of connected unseen regions with at least ``min_cells`` cells."""
        unseen = ~self.seen
        visited = np.zeros_like(unseen)
        found: list[Rect] = []
        for j in range(self.ny):
            for i in range(self.nx):
                if not unseen[j, i] or visited[j, i]:
                    continue
                stack = [(j, i)]
                visited[j, i] = True
                count = 0
                jmin = jmax = j
                imin = imax = i
                while stack:
                    cj, ci = stack.pop()
                    count += 1
                    jmin, jmax = min(jmin, cj), max(jmax, cj)
                    imin, imax = min(imin, ci), max(imax, ci)
                    for nj, ni in ((cj + 1, ci), (cj - 1, ci), (cj, ci + 1), (cj, ci - 1)):
                        if (0 <= nj < self.ny and 0 <= ni < self.nx
                                and unseen[nj, ni] and not visited[nj, ni]):
                            visited[nj, ni] = True
                            stack.append((nj, ni))
                if count >= min_cells:
                    found.append(Rect(
                        self.area.xmin + imin * self.dx, self.area.ymin + jmin * self.dy,
                        self.area.xmin + (imax + 1) * self.dx,
                        self.area.ymin + (jmax + 1) * self.dy))
        return found

    def image(self) -> np.ndarray:
        """Boolean array, row 0 = highest Y (screen order) for drawing."""
        return self.seen[::-1]


def gap_waypoints(grid: CoverageGrid, cfg: SurveyConfig, flight_z: float,
                  start_xy: tuple[float, float], min_cells: int = 4
                  ) -> list[tuple[float, float, float]]:
    """Waypoints that re-fly every unseen region at the main sweep's altitude."""
    remaining = [plan_survey(gap, cfg, None, fixed_z=flight_z).waypoints
                 for gap in grid.gaps(min_cells)]
    ordered: list[tuple[float, float, float]] = []
    here = start_xy
    while remaining:
        # Nearest region first; fly it from whichever end is closer.
        index = min(range(len(remaining)), key=lambda k: min(
            math.hypot(remaining[k][0][0] - here[0], remaining[k][0][1] - here[1]),
            math.hypot(remaining[k][-1][0] - here[0], remaining[k][-1][1] - here[1])))
        wps = remaining.pop(index)
        if (math.hypot(wps[-1][0] - here[0], wps[-1][1] - here[1])
                < math.hypot(wps[0][0] - here[0], wps[0][1] - here[1])):
            wps = list(reversed(wps))
        ordered.extend(wps)
        here = (wps[-1][0], wps[-1][1])
    return ordered


# --- Visiting points of interest -------------------------------------------
def path_length(start: tuple[float, float], points: list[tuple[float, float]],
                end: tuple[float, float] | None = None) -> float:
    """Length of start -> points (in order) -> end."""
    total = 0.0
    here = start
    for p in points:
        total += math.dist(here, p)
        here = p
    if end is not None:
        total += math.dist(here, end)
    return total


def order_visits(start: tuple[float, float], points: list[tuple[float, float]],
                 end: tuple[float, float] | None = None, exact_limit: int = 8
                 ) -> list[int]:
    """Indices of ``points`` in a short visiting order from ``start`` to ``end``.

    Exact for up to ``exact_limit`` points, otherwise nearest neighbour
    followed by 2-opt.
    """
    n = len(points)
    if n <= 1:
        return list(range(n))

    def cost(order):
        return path_length(start, [points[i] for i in order], end)

    if n <= exact_limit:
        return list(min(itertools.permutations(range(n)), key=cost))

    order: list[int] = []
    left = set(range(n))
    here = start
    while left:
        nearest = min(left, key=lambda k: math.dist(here, points[k]))
        order.append(nearest)
        left.remove(nearest)
        here = points[nearest]
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if cost(candidate) < cost(order) - 1e-9:
                    order = candidate
                    improved = True
    return order