#!/usr/bin/env python3
"""Randomize RandomWorld vegetation: python3 tools/generate_random_world.py.

Use --seed NUMBER to repeat a layout. Only vegetation includes are replaced;
terrain, walls, charging station and other objects are preserved.
Use --launch to regenerate and then open the resulting world in Gazebo.
"""
import argparse
import bisect
import copy
import math
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

# Placement offsets above the ground, in metres.
# Oak bark extends about 0.0703 m below its origin. Dry_Grass extends about
# 0.057 m below its origin after its DAE transforms and visual pose are applied.
# Negative values lower the model origin below the sampled ground height.
# Change these values here; they are applied whenever a new world is generated.
MODEL_BASE_OFFSETS = {'oak': -0.15, 'pine': -0.03, 'grass': 0.06}

# Dry_Grass reaches about 2.2 m from its origin, including its ground plane.
# Reserve 2.5 m for any yaw plus a gap: patch centres stay at least 5 m apart.
GRASS_CLEARANCE_RADIUS = 3


class Terrain:
    """Sample the project's regular OBJ grid using its triangle diagonals."""

    def __init__(self, world, source):
        model = world.find("model[@name='terrain']")
        link = model.find('link')
        collision = link.find('collision')
        mesh = collision.find('geometry/mesh')
        scale = list(map(float, mesh.findtext('scale', '1 1 1').split()))
        translation = [0.0, 0.0, 0.0]
        for element in (model, link, collision):
            pose = list(map(float, element.findtext('pose', '0 0 0 0 0 0').split()))
            if any(abs(angle) > 1e-9 for angle in pose[3:]):
                raise ValueError('Terrain sampling requires an unrotated terrain mesh')
            translation = [a + b for a, b in zip(translation, pose[:3])]
        self.points = {}
        path = (source.parent / mesh.findtext('uri')).resolve()
        
        # Load mesh vertices and apply the terrain scale.
        for line in path.read_text().splitlines():
            if line.startswith('v '):

                x, y, z = (float(v) * s + offset for v, s, offset in
                           zip(line.split()[1:], scale, translation))
                self.points[x, y] = z
        self.xs = sorted({x for x, y in self.points})
        self.ys = sorted({y for x, y in self.points})

    def height(self, x, y):

        # Find the grid cell containing this position.
        i = max(0, min(len(self.xs) - 2, bisect.bisect_right(self.xs, x) - 1))
        j = max(0, min(len(self.ys) - 2, bisect.bisect_right(self.ys, y) - 1))
        x0, x1 = self.xs[i:i + 2]
        y0, y1 = self.ys[j:j + 2]
        u, v = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
        a, b = self.points[x0, y0], self.points[x1, y0]
        c, d = self.points[x0, y1], self.points[x1, y1]

        # Interpolate ground height using the matching mesh triangle.
        if v >= u:
            return a * (1 - v) + c * (v - u) + d * u
        return a * (1 - u) + b * (u - v) + d * v


def generate(source, output, trees=None, grass=None, seed=None):

    # Reuse model definitions, but not their manually edited reference heights.
    reference = ET.parse(ROOT / 'world/worldterrian.sdf').getroot().find('world')
    templates = {}

    for include in reference.findall('include'):
        uri = include.findtext('uri', '').strip()
        kind = ('grass' if uri == 'model://Dry_Grass' else
                'oak' if uri.endswith('/Oak%20tree') else
                'pine' if uri.endswith('/Pine%20tree') else None)
        
        if kind:
            templates[kind] = include
            
    document = ET.parse(source)
    world = document.getroot().find('world')
    terrain = Terrain(world, source)

    # Preserve the number and species already present in this world.
    existing_trees = []
    existing_grass = 0
    for include in world.findall('include'):
        uri = include.findtext('uri', '').strip()
        for kind, template in templates.items():
            if uri == template.findtext('uri', '').strip():
                if kind == 'grass':
                    existing_grass += 1
                else:
                    existing_trees.append(kind)
    tree_kinds = existing_trees if trees is None else [rng_kind for rng_kind in
        (['oak'] * max(0, trees - 1) + ['pine'] if trees else [])]
    trees = len(tree_kinds)
    grass = existing_grass if grass is None else grass
    # Remove old vegetation before adding the new layout.
    for include in list(world.findall('include')):
        uri = include.findtext('uri', '').strip()
        if uri in {t.findtext('uri', '').strip() for t in templates.values()}:
            world.remove(include)
    rng = random.Random(seed)

    # Clearance radii for charging pad, Husky, drone/kangaroo and falling rock.
    occupied = [(0, 0, 3), (-5, 0, 2), (2, 0, 2), (-3, -2, 2)]
    for category, count, radius in [('tree', trees, 2.5),
                                    ('grass', grass, GRASS_CLEARANCE_RADIUS)]:
        for index in range(count):
            kind = tree_kinds[index] if category == 'tree' else 'grass'
            for _ in range(10000):

                # Keep models away from the inward-sloping perimeter walls.
                x = rng.uniform(terrain.xs[0] + 5, terrain.xs[-1] - 5)
                y = rng.uniform(terrain.ys[0] + 5, terrain.ys[-1] - 5)

                # Accept only positions with enough space around other objects.
                if all(math.hypot(x - ox, y - oy) >= radius + r for ox, oy, r in occupied):
                    break
            else:
                raise ValueError('Cannot fit requested vegetation counts with safe spacing')
            
            occupied.append((x, y, radius))

            # Copy the model, then set its name, ground height and random rotation.
            include = copy.deepcopy(templates[kind])

            name = f'{kind}_{index + 1}' if category == 'tree' else ('dry_grass' if index == 0 else f'dry_grass_{index + 1}')
            include.find('name').text = name
            z = terrain.height(x, y) + MODEL_BASE_OFFSETS[kind]
            include.find('pose').text = f'{x:.6f} {y:.6f} {z:.6f} 0 0 {rng.uniform(-math.pi, math.pi):.6f}'
            world.append(include)

    ET.indent(document, space='  ')
    document.write(output, encoding='utf-8', xml_declaration=True)
    return output


def main():
    # Read optional vegetation counts, seed and output path.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trees', type=int)
    parser.add_argument('--grass', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', type=Path, default=ROOT / 'world/RandomWorld.sdf')
    parser.add_argument('--launch', action='store_true',
                        help='Open the generated world in Gazebo (world only)')
    args = parser.parse_args()
    
    if any(n is not None and n < 0 for n in (args.trees, args.grass)):
        parser.error('Vegetation counts must be nonnegative')
    if args.output.resolve().parent != ROOT / 'world':
        parser.error('Output must be in world/ so relative terrain paths remain valid')
    try:
        result = generate(ROOT / 'world/RandomWorld.sdf', args.output, args.trees, args.grass, args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(f'Randomized vegetation in {result} (seed={args.seed})', flush=True)
    print('Applied height offsets (m): ' + ', '.join(
        f'{kind}={offset:+.3f}' for kind, offset in MODEL_BASE_OFFSETS.items()
    ), flush=True)
    if args.launch:
        environment = os.environ.copy()
        for name in ('IGN_GAZEBO_RESOURCE_PATH', 'GZ_SIM_RESOURCE_PATH'):
            existing = environment.get(name)
            environment[name] = str(ROOT / 'models') + (os.pathsep + existing if existing else '')
        environment.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
        os.execvpe('ign', ['ign', 'gazebo', '-r', str(result.resolve())], environment)


if __name__ == '__main__':
    main()
