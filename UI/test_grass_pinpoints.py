import json
import tempfile
import unittest
from pathlib import Path

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


class SurveyStatusTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'pins.json'
        self.tracker = GrassPinpoints(self.path)
        for x, y in ((2, 3), (8, 3), (-6, -6)):
            for stamp in (1, 1.1, 1.2):
                self.tracker.observe([(x, y, 0)], 'odom', stamp)

    def test_new_pins_start_pending(self):
        self.assertEqual([p['status'] for p in self.tracker.pins], ['pending'] * 3)

    def test_open_in_filters_by_area_and_status(self):
        area = (0, 0, 10, 10)
        self.assertEqual([p['id'] for p in self.tracker.open_in(area)], [1, 2])
        self.tracker.set_status(1, 'cleared')
        self.assertEqual([p['id'] for p in self.tracker.open_in(area)], [2])
        self.tracker.set_status(2, 'active')
        self.assertEqual([p['id'] for p in self.tracker.open_in(area)], [2])

    def test_status_is_saved_and_reloaded(self):
        self.assertTrue(self.tracker.set_status(2, 'false_positive'))
        self.assertFalse(self.tracker.set_status(99, 'cleared'))
        loaded = GrassPinpoints(self.path)
        self.assertEqual({p['id']: p['status'] for p in loaded.pins},
                         {1: 'pending', 2: 'false_positive', 3: 'pending'})

    def test_unknown_status_is_rejected(self):
        with self.assertRaises(ValueError):
            self.tracker.set_status(1, 'banana')

    def test_files_without_status_load_as_pending(self):
        self.path.write_text(json.dumps({'version': 1, 'pins': [
            {'id': 4, 'x': 1.0, 'y': 1.0, 'z': 0.0, 'frame_id': 'odom'}]}))
        self.assertEqual(GrassPinpoints(self.path).pins[0]['status'], 'pending')

    def test_resurvey_discards_old_results_but_keeps_false_positives(self):
        self.tracker.set_status(1, 'cleared')
        self.tracker.set_status(3, 'false_positive')
        gone = self.tracker.discard_for_resurvey((-10, -10, 10, 10))
        self.assertEqual(gone, [1, 2])
        self.assertEqual([p['id'] for p in self.tracker.pins], [3])
        for stamp in (5, 5.1, 5.2):
            self.tracker.observe([(20, 20, 0)], 'odom', stamp)
        self.assertEqual([p['id'] for p in self.tracker.pins], [3, 4])   # ids never repeat
        self.assertEqual(len(GrassPinpoints(self.path).pins), 2)

    def test_a_cleared_spot_can_be_pinned_again(self):
        self.tracker.set_status(1, 'cleared')
        for stamp in (7, 7.1, 7.2):
            self.tracker.observe([(2.2, 3.1, 0)], 'odom', stamp)
        self.assertEqual([p['status'] for p in self.tracker.pins],
                         ['cleared', 'pending', 'pending', 'pending'])

    def test_a_handled_spot_is_not_pinned_again(self):
        self.tracker.set_status(1, 'false_positive')
        for stamp in (6, 6.1, 6.2):
            self.tracker.observe([(2.2, 3.1, 0)], 'odom', stamp)
        self.assertEqual(len(self.tracker.pins), 3)

    def test_discard_with_nothing_to_discard_is_a_no_op(self):
        self.assertEqual(self.tracker.discard_for_resurvey((100, 100, 110, 110)), [])
        self.assertEqual(len(self.tracker.pins), 3)


if __name__ == '__main__':
    unittest.main()