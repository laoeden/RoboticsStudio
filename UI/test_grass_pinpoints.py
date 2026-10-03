import json
from pathlib import Path
import tempfile
import unittest

from grass_pinpoints import GrassPinpoints


class GrassPinpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'pins.json'
        self.tracker = GrassPinpoints(self.path)

    def test_repeated_sightings_save_once_and_survive_restart(self):
        for stamp in (1.0, 1.1, 1.2):
            self.tracker.observe([(2, 3, .8)], 'parrot1_odom', stamp)
        self.tracker.observe([(2.1, 3.1, .9)], 'parrot1_odom', 1.3)
        loaded = GrassPinpoints(self.path)
        self.assertEqual(len(loaded.pins), 1)
        self.assertEqual(loaded.pins[0]['frame_id'], 'parrot1_odom')
        self.assertEqual(loaded.pins[0]['x'], 2)

    def test_multiple_contours_in_one_frame_are_not_multiple_confirmations(self):
        self.tracker.observe([(2, 3, 0), (2.1, 3, 0), (2.2, 3, 0)], 'odom', 1)
        self.assertEqual(self.tracker.pins, [])
        self.assertEqual(self.tracker.candidates[0]['count'], 1)

    def test_transient_and_invalid_detections_are_not_saved(self):
        self.tracker.observe([(2, 3, 0), (float('nan'), 0, 0)], 'odom', 1)
        self.tracker.observe([(2, 3, 0)], 'odom', 4)
        self.assertEqual(self.tracker.pins, [])
        self.assertFalse(self.path.exists())

    def test_separate_patches_and_frames_are_preserved(self):
        for stamp in (1, 1.1, 1.2):
            self.tracker.observe([(2, 3, 0), (6, 3, 0)], 'odom', stamp)
        for stamp in (1.3, 1.4, 1.5):
            self.tracker.observe([(2, 3, 0)], 'different_odom', stamp)
        self.assertEqual(len(self.tracker.pins), 3)
        self.assertEqual([p['id'] for p in self.tracker.pins], [1, 2, 3])

    def test_clear_removes_saved_and_pending_sightings(self):
        for stamp in (1, 1.1, 1.2):
            self.tracker.observe([(2, 3, 0)], 'odom', stamp)
        self.tracker.observe([(6, 3, 0)], 'odom', 1.3)
        self.tracker.clear()
        self.assertFalse(self.path.exists())
        self.assertEqual(self.tracker.pins, [])
        self.assertEqual(self.tracker.candidates, [])
        self.assertEqual(GrassPinpoints(self.path).pins, [])
        self.tracker.clear()  # Clearing an empty survey is safe.
        for stamp in (2, 2.1, 2.2):
            self.tracker.observe([(2, 3, 0)], 'odom', stamp)
        self.assertEqual(self.tracker.pins[0]['id'], 1)

    def test_clear_recovers_invalid_saved_file(self):
        self.path.write_text('{broken')
        tracker = GrassPinpoints(self.path)
        tracker.clear()
        self.assertIsNone(tracker.load_error)
        self.assertFalse(self.path.exists())

    def test_invalid_saved_file_is_not_overwritten(self):
        self.path.write_text('{broken')
        tracker = GrassPinpoints(self.path)
        self.assertIsNotNone(tracker.load_error)
        for stamp in (1, 1.1, 1.2):
            tracker.observe([(2, 3, 0)], 'odom', stamp)
        self.assertEqual(self.path.read_text(), '{broken')


if __name__ == '__main__':
    unittest.main()
