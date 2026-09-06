#!/usr/bin/env python3
"""
Convert a grayscale heightmap into a lightweight, textured OBJ mesh and MTL.

The OBJ is used as the visible terrain because Ignition Gazebo 6's
heightmap renderers produced crashes and visual corruption in this
project's WSL graphics environment. The original heightmap is still
used for terrain collision.

Run from the RoboticsStudio project directory:

python3 tools/heightmap_to_obj.py \
  models/GreyHeatMapV2/e9836408f061fe241792508fc50c1535dea7810a.png \
  --output models/GreyHeatMapV2/GreyMapV2Terrain.obj \
  --texture models/GreyHeatMapV2/terrain_bright.png
"""

import argparse
import os
from pathlib import Path
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
GRID_SIZE = 129
WIDTH_METRES = 100.0
LENGTH_METRES = 100.0
HEIGHT_METRES = 15.0

parser = argparse.ArgumentParser()
parser.add_argument("source", nargs="?", default="models/GreyHeatMap/GreyMap513.png")
parser.add_argument("--output", default="models/GreyHeatMap/GreyMapTerrain.obj")
parser.add_argument(
    "--texture",
    default="models/forest_plane/materials/textures/forest_overhead.png",
)
args = parser.parse_args()

source = (ROOT / args.source).resolve()
output = (ROOT / args.output).resolve()
material = output.with_suffix(".mtl")
texture = (ROOT / args.texture).resolve()
output.parent.mkdir(parents=True, exist_ok=True)

image = Image.open(source).convert("L").resize((GRID_SIZE, GRID_SIZE))
pixels = image.load()

with output.open("w", encoding="ascii") as obj:
    obj.write(f"# Generated from {source.name}\n")
    obj.write(f"mtllib {material.name}\n")

    for row in range(GRID_SIZE):
        y = LENGTH_METRES / 2 - row * LENGTH_METRES / (GRID_SIZE - 1)
        for column in range(GRID_SIZE):
            x = -WIDTH_METRES / 2 + column * WIDTH_METRES / (GRID_SIZE - 1)
            z = pixels[column, row] / 255.0 * HEIGHT_METRES
            obj.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

    for row in range(GRID_SIZE):
        v = 1.0 - row / (GRID_SIZE - 1)
        for column in range(GRID_SIZE):
            u = column / (GRID_SIZE - 1)
            obj.write(f"vt {u:.6f} {v:.6f}\n")

    obj.write("usemtl terrain_ground\n")
    for row in range(GRID_SIZE - 1):
        for column in range(GRID_SIZE - 1):
            top_left = row * GRID_SIZE + column + 1
            top_right = top_left + 1
            bottom_left = top_left + GRID_SIZE
            bottom_right = bottom_left + 1
            obj.write(
                f"f {top_left}/{top_left} {bottom_left}/{bottom_left} "
                f"{top_right}/{top_right}\n"
            )
            obj.write(
                f"f {top_right}/{top_right} {bottom_left}/{bottom_left} "
                f"{bottom_right}/{bottom_right}\n"
            )

with material.open("w", encoding="ascii") as mtl:
    mtl.write("newmtl terrain_ground\n")
    mtl.write("Ka 0.6 0.6 0.6\n")
    mtl.write("Kd 1.0 1.0 1.0\n")
    mtl.write("Ks 0.05 0.05 0.05\n")
    mtl.write("Ns 8\n")
    texture_path = os.path.relpath(texture, material.parent)
    mtl.write(f"map_Kd {texture_path}\n")

print(f"Created {output}")
print(f"Created {material}")
