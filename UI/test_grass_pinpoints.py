import unittest

from grass_pinpoints import GrassPinpoints


class GrassPinpointTests(unittest.TestCase):
    def test_topic_point_creates_pin_without_persistence(self):
        tracker = GrassPinpoints()
        added = tracker.observe([(2, 3, .8)], 'parrot1_odom', 1)
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]['x'], 2)
        self.assertEqual(GrassPinpoints().pins, [])

    def test_repeated_messages_do_not_create_duplicate_pins(self):
        tracker = GrassPinpoints()
        tracker.observe([(2, 3, 0)], 'odom', 1)
        self.assertEqual(tracker.observe([(2.1, 3.1, 0)], 'odom', 10), [])
        self.assertEqual(len(tracker.pins), 1)

    def test_invalid_points_and_missing_frames_are_ignored(self):
        tracker = GrassPinpoints()
        tracker.observe([(float('nan'), 0, 0), (0, float('inf'), 0)], 'odom', 1)
        tracker.observe([(2, 3, 0)], '', 1)
        self.assertEqual(tracker.pins, [])

    def test_separate_patches_and_frames_are_preserved(self):
        tracker = GrassPinpoints()
        tracker.observe([(2, 3, 0), (6, 3, 0)], 'odom', 1)
        tracker.observe([(2, 3, 0)], 'different_odom', 2)
        self.assertEqual([p['id'] for p in tracker.pins], [1, 2, 3])

    def test_clear_resets_session(self):
        tracker = GrassPinpoints()
        tracker.observe([(2, 3, 0)], 'odom', 1)
        tracker.clear()
        self.assertEqual(tracker.pins, [])
        self.assertEqual(tracker.candidates, [])
        added = tracker.observe([(2, 3, 0)], 'odom', 2)
        self.assertEqual(added[0]['id'], 1)


if __name__ == '__main__':
    unittest.main()
