#!/usr/bin/env python3
"""Randomize RandomWorld vegetation: python3 tools/generate_random_world.py.

Use --seed NUMBER to repeat a layout. Only vegetation includes are replaced;
terrain, walls, charging station and other objects are preserved.
"""
import argparse
import bisect
import copy
import math
from pathlib import Path
import random
import statistics
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class Terrain:
    """Sample the project's regular OBJ grid using its triangle diagonals."""

    def __init__(self, world, source):
        mesh = world.find("model[@name='terrain']/link/collision/geometry/mesh")
        scale = list(map(float, mesh.findtext('scale', '1 1 1').split()))
        self.points = {}
        path = (source.parent / mesh.findtext('uri')).resolve()
        
        # Load mesh vertices and apply the terrain scale.
        for line in path.read_text().splitlines():
            if line.startswith('v '):

                x, y, z = (float(v) * s for v, s in zip(line.split()[1:], scale))
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

    # Read offsets from the fixed reference world so repeated generation cannot
    # accumulate rounding errors or move the model bases relative to the ground.
    reference = ET.parse(ROOT / 'world/worldterrian.sdf').getroot().find('world')
    reference_terrain = Terrain(reference, ROOT / 'world/worldterrian.sdf')
    templates, offsets = {}, {}

    for include in reference.findall('include'):
        uri = include.findtext('uri', '').strip()
        kind = ('grass' if uri == 'model://Dry_Grass' else
                'oak' if uri.endswith('/Oak%20tree') else
                'pine' if uri.endswith('/Pine%20tree') else None)
        
        if kind:
            templates[kind] = include
            x, y, z, *_ = map(float, include.findtext('pose').split())
            offsets.setdefault(kind, []).append(z - reference_terrain.height(x, y))
            
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
    for category, count, radius in [('tree', trees, 2.5), ('grass', grass, 0.7)]:
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
            z = terrain.height(x, y) + statistics.median(offsets[kind])
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
    args = parser.parse_args()
    
    if any(n is not None and n < 0 for n in (args.trees, args.grass)):
        parser.error('Vegetation counts must be nonnegative')
    if args.output.resolve().parent != ROOT / 'world':
        parser.error('Output must be in world/ so relative terrain paths remain valid')
    try:
        result = generate(ROOT / 'world/RandomWorld.sdf', args.output, args.trees, args.grass, args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(f'Randomized vegetation in {result} (seed={args.seed})')


if __name__ == '__main__':
    main()
