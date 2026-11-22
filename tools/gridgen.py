#!/usr/bin/env python3
"""Rasterize the arena into a compact 2 m grid shared by the bot and the simulator.

Usage: python3 -I tools/gridgen.py <arena_dir> <output_file>

File format: b"MGRD", then little-endian minX, minY, cell (float32), nx, ny (uint32), then a
zlib stream of five int16 centimeter layers in row-major order:
ground, solid_top, solid_base, bridge_base, bridge_top. NONE marks an empty cell.
"""
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

CELL = 2.0
NONE = -32768


def inside(px, py, flat):
    """Even-odd point-in-polygon test for arrays of points against a flat [x0, y0, ...] ring."""
    ring = np.asarray(flat, dtype=np.float64).reshape(-1, 2)
    result = np.zeros(px.shape, dtype=bool)
    x0, y0 = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for ax, ay, bx, by in zip(x0, y0, x1, y1):
        if ay == by:
            continue
        crosses = (ay > py) != (by > py)
        result ^= crosses & (px < ax + (py - ay) * (bx - ax) / (by - ay))
    return result


def sample_ground(terrain, raw, xs, ys):
    """Bilinear terrain height in meters at the given coordinates."""
    nx, ny, step = terrain["nx"], terrain["ny"], terrain["step"]
    fx = np.clip((xs - terrain["minX"]) / step, 0, nx - 1.001)
    fy = np.clip((ys - terrain["minY"]) / step, 0, ny - 1.001)
    i, j = fx.astype(int), fy.astype(int)
    u, v = fx - i, fy - j
    h00, h10 = raw[j, i], raw[j, i + 1]
    h01, h11 = raw[j + 1, i], raw[j + 1, i + 1]
    return ((1 - u) * (1 - v) * h00 + u * (1 - v) * h10 + (1 - u) * v * h01 + u * v * h11) * terrain["scale"]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    arena, output = Path(sys.argv[1]), Path(sys.argv[2])
    world = json.loads((arena / "v2/world.json").read_text())
    terrain = world["terrain"]
    raw = np.fromfile(arena / "v2/terrain.i16", dtype="<i2").reshape(terrain["ny"], terrain["nx"]).astype(np.float64)

    bounds = world["bounds"]
    min_x, min_y = float(bounds["minX"]), float(bounds["minY"])
    nx = int(np.ceil((bounds["maxX"] - min_x) / CELL))
    ny = int(np.ceil((bounds["maxY"] - min_y) / CELL))
    xs = min_x + (np.arange(nx) + 0.5) * CELL
    ys = min_y + (np.arange(ny) + 0.5) * CELL
    grid_x, grid_y = np.meshgrid(xs, ys)

    ground = np.round(sample_ground(terrain, raw, grid_x, grid_y) * 100).astype(np.int16)
    solid_top = np.full((ny, nx), NONE, dtype=np.int16)
    solid_base = np.full((ny, nx), NONE, dtype=np.int16)
    bridge_top = np.full((ny, nx), NONE, dtype=np.int16)
    bridge_base = np.full((ny, nx), NONE, dtype=np.int16)

    for prism in world["prisms"]:
        ring = np.asarray(prism["ring"], dtype=np.float64).reshape(-1, 2)
        i0 = max(int((ring[:, 0].min() - min_x) / CELL) - 1, 0)
        i1 = min(int((ring[:, 0].max() - min_x) / CELL) + 2, nx)
        j0 = max(int((ring[:, 1].min() - min_y) / CELL) - 1, 0)
        j1 = min(int((ring[:, 1].max() - min_y) / CELL) + 2, ny)
        if i0 >= i1 or j0 >= j1:
            continue
        px, py = grid_x[j0:j1, i0:i1], grid_y[j0:j1, i0:i1]
        mask = inside(px, py, prism["ring"])
        holes = prism.get("holes") or []
        if holes and isinstance(holes[0], (int, float)):
            holes = [holes]
        for hole in holes:
            if len(hole) >= 6:
                mask &= ~inside(px, py, hole)
        if not mask.any():
            continue
        top_cm, base_cm = int(round(prism["top"] * 100)), int(round(prism["base"] * 100))
        is_bridge = prism["kind"] == "bridge"
        top_layer = (bridge_top if is_bridge else solid_top)[j0:j1, i0:i1]
        base_layer = (bridge_base if is_bridge else solid_base)[j0:j1, i0:i1]
        empty = top_layer == NONE
        base_layer[mask & empty] = base_cm
        base_layer[mask & ~empty] = np.minimum(base_layer[mask & ~empty], base_cm)
        top_layer[mask] = np.maximum(top_layer[mask], top_cm)

    payload = b"".join(layer.astype("<i2").tobytes() for layer in (ground, solid_top, solid_base, bridge_base, bridge_top))
    blob = b"MGRD" + struct.pack("<fffII", min_x, min_y, CELL, nx, ny) + zlib.compress(payload, 9)
    output.write_bytes(blob)
    print(f"grid {nx}x{ny} cells at {CELL} m, solid {(solid_top != NONE).mean():.1%}, bridge {(bridge_top != NONE).sum()} cells, {len(blob) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
