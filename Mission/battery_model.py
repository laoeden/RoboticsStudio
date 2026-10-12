"""Battery model for the ground robot (pure Python, no ROS imports).

Drain comes from distance driven, idle time and (optionally) the cutter.
The battery charges while parked on the charging pad. A low-battery flag
latches when the level falls to ``low_threshold`` and only clears once the
battery has recharged to ``resume_level``.
"""

import math
from dataclasses import dataclass


@dataclass
class BatteryConfig:
    capacity: float = 100.0  # percent
    drain_per_metre: float = 0.25  # percent per metre driven
    idle_drain_per_s: float = 0.005 # percent per second
    cutting_drain_per_s: float = 0.05 # percent per second while cutting
    charge_per_s: float = 2.0   # percent per second on the pad
    low_threshold: float = 20.0  # percent; triggers return to base
    resume_level: float = 95.0  # percent; low flag clears above this
    dock_x: float = 0.0   # charging pad centre, husky1_odom frame
    dock_y: float = 0.0
    dock_radius: float = 1.0  # metres; inside this the robot may charge
    window: float = 0.25   # s; motion is measured over this window, not per sample
    deadband: float = 0.02   # m; displacement per window below this is noise
    moving_speed: float = 0.15  # m/s; faster than this counts as driving (no charging)
    max_valid_speed: float = 3.0  # m/s; faster is an odometry glitch/teleport, ignored


class BatteryModel:
    def __init__(self, config: BatteryConfig | None = None, level: float | None = None):
        self.config = config or BatteryConfig()
        self.level = self.config.capacity if level is None else float(level)
        self.low = self.level <= self.config.low_threshold
        self.cutting = False
        self.distance_driven = 0.0
        self.speed = 0.0
        self._anchor: tuple[float, float] | None = None
        self._window_t = 0.0

    def at_dock(self, x: float, y: float) -> bool:
        c = self.config
        return math.hypot(x - c.dock_x, y - c.dock_y) <= c.dock_radius

    def update(self, x: float, y: float, dt: float) -> float:
        """Advance the model by ``dt`` seconds with the robot at (x, y).

        Motion is judged over ``window`` seconds so that millimetre-level
        odometry noise (physics jitter, a robot settling on a slope) is not
        mistaken for driving, and single-sample position spikes are dropped.
        """
        c = self.config
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(dt)) or dt < 0:
            return self.level

        if self._anchor is None:
            self._anchor = (x, y)
        self._window_t += dt

        window_distance = 0.0
        if self._window_t >= c.window:
            d = math.hypot(x - self._anchor[0], y - self._anchor[1])
            speed = d / self._window_t
            if speed > c.max_valid_speed: # glitch or teleport
                d, speed = 0.0, 0.0
            if d < c.deadband:  # jitter, not travel
                d = 0.0
            self.speed = speed if d > 0 else 0.0
            window_distance = d
            self.distance_driven += d
            self._anchor = (x, y)
            self._window_t = 0.0

        moving = self.speed > c.moving_speed
        if self.at_dock(x, y) and not moving:
            self.level += c.charge_per_s * dt
        else:
            self.level -= c.drain_per_metre * window_distance + c.idle_drain_per_s * dt
            if self.cutting:
                self.level -= c.cutting_drain_per_s * dt

        self.level = min(c.capacity, max(0.0, self.level))
        if self.level <= c.low_threshold:
            self.low = True
        elif self.level >= c.resume_level:
            self.low = False
        return self.level

    @property
    def depleted(self) -> bool:
        return self.level <= 0.0
