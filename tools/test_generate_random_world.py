"""Regression checks for vegetation placement on scaled, translated terrain."""
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import generate_random_world as generator


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'world').mkdir()
        (self.root / 'ground.obj').write_text(
            'v 0 0 0\nv 100 0 10\nv 0 100 20\nv 100 100 30\n'
        )
        self.source = self.root / 'world/RandomWorld.sdf'
        self.source.write_text('''<sdf version="1.8"><world name="test">
          <model name="terrain"><pose>-25 -25 1 0 0 0</pose>
            <link name="ground"><pose>0 0 0.5 0 0 0</pose>
              <collision name="ground"><pose>0 0 0.25 0 0 0</pose>
                <geometry><mesh><uri>../ground.obj</uri>
                  <scale>0.5 0.5 0.2</scale>
                </mesh></geometry>
              </collision>
            </link>
          </model>
          <include><uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Oak%20tree</uri>
            <name>oak</name><pose>0 0 1000 0 0 0</pose><static>true</static></include>
          <include><uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Pine%20tree</uri>
            <name>pine</name><pose>0 0 42 0 0 0</pose><static>true</static></include>
          <include><uri>model://Dry_Grass</uri>
            <name>dry_grass</name><pose>0 0 -500 0 0 0</pose><static>true</static></include>
          <model name="charging_station"><pose>0 0 3 0 0 0</pose><static>true</static></model>
        </world></sdf>''')
        self.reference = self.root / 'world/worldterrian.sdf'
        self.reference.write_text(self.source.read_text())
        self.output = self.root / 'world/generated.sdf'

    def generate(self, source=None):
        with patch.object(generator, 'ROOT', self.root):
            generator.generate(source or self.source, self.output, seed=23)

    def test_scaled_terrain_includes_model_link_and_collision_translation(self):
        world = ET.parse(self.source).getroot().find('world')
        terrain = generator.Terrain(world, self.source)
        self.assertEqual(terrain.xs, [-25.0, 25.0])
        self.assertEqual(terrain.ys, [-25.0, 25.0])
        self.assertAlmostEqual(terrain.height(-25, -25), 1.75)
        self.assertAlmostEqual(terrain.height(0, 0), 4.75)
        self.assertAlmostEqual(terrain.height(25, 25), 7.75)

    def test_visible_bases_are_grounded_despite_bad_reference_heights(self):
        # Use known offsets for this geometry check; users can tune the defaults.
        with patch.dict(generator.MODEL_BASE_OFFSETS,
                        {'oak': -0.03, 'pine': -0.03, 'grass': 0.06}):
            self.generate()
        world = ET.parse(self.output).getroot().find('world')
        # Approximate asset base heights after their internal transforms.
        bases = {'oak_1': -0.0703, 'pine_2': 0.0, 'dry_grass': -0.057}
        self.assertEqual(len(world.findall('include')), 3)
        for include in world.findall('include'):
            x, y, z, *_ = map(float, include.findtext('pose').split())
            ground = 0.04 * x + 0.08 * y + 4.75
            name = include.findtext('name')
            expected_base = ground + {
                'oak_1': -0.10, 'pine_2': -0.03, 'dry_grass': 0.0,
            }[name]
            self.assertAlmostEqual(z + bases[name], expected_base,
                                   delta=0.004)
        self.assertEqual(world.find("model[@name='charging_station']/pose").text,
                         '0 0 3 0 0 0')

    def test_regeneration_and_reference_edits_do_not_change_placement(self):
        self.generate()
        first = self.output.read_text()
        reference = ET.parse(self.reference)
        reference.getroot().find('world/model/pose').text = '0 0 20 0 0 0'
        for include in reference.getroot().findall('world/include'):
            include.find('pose').text = '0 0 999 0 0 0'
        reference.write(self.reference)
        self.generate(self.output)
        self.assertEqual(self.output.read_text(), first)

    def test_grass_footprints_do_not_overlap(self):
        # The Dry_Grass mesh extends about 2.2 m from its origin in XY,
        # including its plane. Cover the footprint for every random yaw.
        for seed in range(10):
            with self.subTest(seed=seed), patch.object(generator, 'ROOT', self.root):
                generator.generate(self.source, self.output, trees=0, grass=20, seed=seed)
                world = ET.parse(self.output).getroot().find('world')
                points = [list(map(float, include.findtext('pose').split()))[:2]
                          for include in world.findall('include')
                          if include.findtext('uri', '').strip() == 'model://Dry_Grass']
                self.assertEqual(len(points), 20)
                for index, (x, y) in enumerate(points):
                    for other_x, other_y in points[index + 1:]:
                        self.assertGreaterEqual(math.hypot(x - other_x, y - other_y), 4.4)


if __name__ == '__main__':
    unittest.main()
