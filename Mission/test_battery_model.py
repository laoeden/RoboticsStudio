"""Battery model checks (no ROS needed): python3 -m unittest Mission.test_battery_model"""
import unittest

from Mission.battery_model import BatteryConfig, BatteryModel


class BatteryTests(unittest.TestCase):
    def drive(self, model, metres, speed=1.0, x0=5.0):
        x = x0
        model.update(x, 0.0, 0.0)
        for _ in range(int(metres / speed * 10)):
            x += speed * 0.1
            model.update(x, 0.0, 0.1)
        return x

    def test_starts_full_and_not_low(self):
        m = BatteryModel()
        self.assertEqual(m.level, 100.0)
        self.assertFalse(m.low)

    def test_driving_drains_by_distance(self):
        m = BatteryModel(BatteryConfig(idle_drain_per_s=0))
        self.drive(m, 10)
        self.assertAlmostEqual(m.level, 100 - 10 * 0.25, delta=0.2)

    def test_low_flag_latches_and_returns_only_after_recharge(self):
        m = BatteryModel(BatteryConfig(), level=21.0)
        x = self.drive(m, 10)
        self.assertTrue(m.low)
        # Back at the dock, stationary: charges; flag holds until resume_level.
        m.update(0.0, 0.0, 0.0)
        m.update(0.0, 0.0, 1.0)
        self.assertTrue(m.low)
        for _ in range(60):
            m.update(0.0, 0.0, 1.0)
        self.assertGreaterEqual(m.level, 95.0)
        self.assertFalse(m.low)

    def test_no_charge_away_from_dock(self):
        m = BatteryModel(level=50.0)
        m.update(10.0, 0.0, 0.0)
        m.update(10.0, 0.0, 10.0)
        self.assertLess(m.level, 50.0)

    def test_no_charge_while_driving_through_dock(self):
        m = BatteryModel(level=50.0)
        m.update(-0.5, 0.0, 0.0)
        m.update(0.5, 0.0, 1.0)  # 1 m/s inside the dock radius
        self.assertLess(m.level, 50.0)

    def test_cutting_adds_drain(self):
        a, b = BatteryModel(level=80), BatteryModel(level=80)
        b.cutting = True
        for m in (a, b):
            m.update(5, 0, 0)
            m.update(5, 0, 100)
        self.assertLess(b.level, a.level)

    def test_jitter_on_dock_does_not_drain(self):
        # ~1 ms odometry samples with 1 mm noise and a 3.5 cm spike, as seen in the sim.
        import random
        random.seed(1)
        m = BatteryModel(level=50.0)
        m.update(0.0, 0.0, 0.0)
        previous = m.level
        for i in range(5000):
            x = random.uniform(-0.001, 0.001) + (0.035 if i == 2500 else 0.0)
            m.update(x, random.uniform(-0.001, 0.001), 0.001)
            self.assertGreaterEqual(m.level, previous - 1e-9)
            previous = m.level
        self.assertGreater(m.level, 50.0)

    def test_level_clamped_and_jumps_ignored(self):
        m = BatteryModel(level=1.0)
        m.update(5, 0, 0)
        m.update(500, 0, 0.1)  # teleport, not counted as distance
        self.assertEqual(m.distance_driven, 0.0)
        self.drive(m, 20, x0=5)
        self.assertEqual(m.level, 0.0)
        self.assertTrue(m.depleted)


if __name__ == '__main__':
    unittest.main()
