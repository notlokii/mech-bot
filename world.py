"""Static arena queries for the bot and simulator. Standard library only."""
import heapq
import math
import os
import struct
import zlib
from array import array

NONE = -32768
STEP_UP_CM = 120
BODY_CM = 900
MAX_STRIDE_RISE_CM = 350
NEIGHBORS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))


class Grid:
    def __init__(self, path):
        with open(path, "rb") as handle:
            blob = handle.read()
        if blob[:4] != b"MGRD":
            raise ValueError("bad grid file")
        self.min_x, self.min_y, self.cell, self.nx, self.ny = struct.unpack_from("<fffII", blob, 4)
        payload = zlib.decompress(blob[24:])
        count = self.nx * self.ny
        layers = []
        for index in range(5):
            layer = array("h")
            layer.frombytes(payload[2 * count * index : 2 * count * (index + 1)])
            layers.append(layer)
        self.ground, self.top, self.base, self.bridge_base, self.bridge_top = layers
        self.max_x = self.min_x + self.nx * self.cell
        self.max_y = self.min_y + self.ny * self.cell
        self.walk = self._walkable()

    def _walkable(self):
        nx, ny = self.nx, self.ny
        ground, top, base, bridge_base, bridge_top = self.ground, self.top, self.base, self.bridge_base, self.bridge_top
        raw = bytearray(nx * ny)
        for k in range(nx * ny):
            floor = ground[k]
            solid_clear = top[k] == NONE or base[k] > floor + BODY_CM or top[k] <= floor + STEP_UP_CM
            bridge_clear = bridge_top[k] == NONE or bridge_base[k] > floor + BODY_CM or bridge_top[k] <= floor + STEP_UP_CM
            raw[k] = 1 if solid_clear and bridge_clear else 0
        out = bytearray(raw)
        for j in range(ny):
            for i in range(nx):
                if raw[j * nx + i]:
                    if i < 2 or j < 2 or i >= nx - 2 or j >= ny - 2:
                        out[j * nx + i] = 0
                    continue
                for dj in (-1, 0, 1):
                    for di in (-1, 0, 1):
                        a, b = i + di, j + dj
                        if 0 <= a < nx and 0 <= b < ny:
                            out[b * nx + a] = 0
        return out

    def index(self, x, y):
        i = min(max(int((x - self.min_x) / self.cell), 0), self.nx - 1)
        j = min(max(int((y - self.min_y) / self.cell), 0), self.ny - 1)
        return j * self.nx + i

    def ground_at(self, x, y):
        return self.ground[self.index(x, y)] / 100.0

    def walkable(self, x, y):
        return bool(self.walk[self.index(x, y)])

    def surface(self, x, y, z):
        """Highest standable height at (x, y) for feet at z."""
        k = self.index(x, y)
        height = self.ground[k]
        feet = z * 100 + STEP_UP_CM
        if self.top[k] != NONE and self.top[k] <= feet:
            height = max(height, self.top[k])
        if self.bridge_top[k] != NONE and self.bridge_top[k] <= feet:
            height = max(height, self.bridge_top[k])
        return height / 100.0

    def wall(self, x, y, z, body_cm=BODY_CM):
        """True if a body with feet at z cannot occupy this column."""
        if x <= self.min_x or y <= self.min_y or x >= self.max_x or y >= self.max_y:
            return True
        k = self.index(x, y)
        feet = int(z * 100) + STEP_UP_CM
        head = int(z * 100) + body_cm
        if self.ground[k] > feet:
            return True
        if self.top[k] != NONE and self.top[k] > feet and self.base[k] < head:
            return True
        return self.bridge_top[k] != NONE and self.bridge_top[k] > feet and self.bridge_base[k] < head

    def solid_at(self, x, y, z):
        k = self.index(x, y)
        cm = z * 100
        if cm < self.ground[k]:
            return True
        if self.top[k] != NONE and self.base[k] <= cm <= self.top[k]:
            return True
        return self.bridge_top[k] != NONE and self.bridge_base[k] <= cm <= self.bridge_top[k]

    def los(self, ax, ay, az, bx, by, bz):
        """True if no building, bridge, or terrain blocks the straight segment."""
        distance = math.hypot(bx - ax, by - ay)
        steps = int(distance / self.cell) + 1
        if steps <= 1:
            return True
        ground, top, base, bridge_base, bridge_top = self.ground, self.top, self.base, self.bridge_base, self.bridge_top
        nx, ny, cell, min_x, min_y = self.nx, self.ny, self.cell, self.min_x, self.min_y
        dx, dy, dz = (bx - ax) / steps, (by - ay) / steps, (bz - az) / steps
        for step in range(1, steps):
            i = int((ax + dx * step - min_x) / cell)
            j = int((ay + dy * step - min_y) / cell)
            if not (0 <= i < nx and 0 <= j < ny):
                return False
            k = j * nx + i
            cm = (az + dz * step) * 100
            if cm < ground[k]:
                return False
            if top[k] != NONE and base[k] <= cm <= top[k]:
                return False
            if bridge_top[k] != NONE and bridge_base[k] <= cm <= bridge_top[k]:
                return False
        return True

    def nearest_walkable(self, x, y, max_radius=16):
        ci = min(max(int((x - self.min_x) / self.cell), 0), self.nx - 1)
        cj = min(max(int((y - self.min_y) / self.cell), 0), self.ny - 1)
        if self.walk[cj * self.nx + ci]:
            return ci, cj
        for radius in range(1, max_radius + 1):
            for di in range(-radius, radius + 1):
                for dj in (-radius, radius):
                    a, b = ci + di, cj + dj
                    if 0 <= a < self.nx and 0 <= b < self.ny and self.walk[b * self.nx + a]:
                        return a, b
            for dj in range(-radius + 1, radius):
                for di in (-radius, radius):
                    a, b = ci + di, cj + dj
                    if 0 <= a < self.nx and 0 <= b < self.ny and self.walk[b * self.nx + a]:
                        return a, b
        return None

    def center(self, i, j):
        return self.min_x + (i + 0.5) * self.cell, self.min_y + (j + 0.5) * self.cell

    def path(self, sx, sy, gx, gy, max_expand=1400, stride=2):
        """A* over walkable ground. Returns waypoints, or a partial route if the cap is reached."""
        start = self.nearest_walkable(sx, sy)
        goal = self.nearest_walkable(gx, gy)
        if start is None or goal is None:
            return []
        nx, ny, walk, ground = self.nx, self.ny, self.walk, self.ground
        goal_i, goal_j = goal
        start_key = start[1] * nx + start[0]

        def heuristic(i, j):
            di, dj = abs(i - goal_i), abs(j - goal_j)
            return (max(di, dj) + 0.414 * min(di, dj)) / stride

        frontier = [(heuristic(*start), 0.0, start[0], start[1])]
        came = {start_key: -1}
        best_cost = {start_key: 0.0}
        best_key, best_h = start_key, heuristic(*start)
        expanded = 0
        while frontier and expanded < max_expand:
            _, cost, i, j = heapq.heappop(frontier)
            key = j * nx + i
            if cost > best_cost.get(key, 1e18):
                continue
            expanded += 1
            h_now = heuristic(i, j)
            if h_now < best_h:
                best_h, best_key = h_now, key
            if abs(i - goal_i) <= stride and abs(j - goal_j) <= stride:
                best_key = key
                break
            for di, dj, weight in NEIGHBORS:
                a, b = i + di * stride, j + dj * stride
                if not (0 <= a < nx and 0 <= b < ny):
                    continue
                next_key = b * nx + a
                mid_key = (j + dj * (stride // 2)) * nx + i + di * (stride // 2)
                if not walk[next_key] or not walk[mid_key] or abs(ground[next_key] - ground[key]) > MAX_STRIDE_RISE_CM:
                    continue
                new_cost = cost + weight
                if new_cost < best_cost.get(next_key, 1e18):
                    best_cost[next_key] = new_cost
                    came[next_key] = key
                    heapq.heappush(frontier, (new_cost + heuristic(a, b), new_cost, a, b))
        route = []
        key = best_key
        while key != -1:
            route.append(self.center(key % nx, key // nx))
            key = came[key]
        route.reverse()
        return route[1:] if len(route) > 1 else route


_GRID = None


def load_grid(path=None):
    """Load the arena grid once per process."""
    global _GRID
    if _GRID is None:
        _GRID = Grid(path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "navgrid.bin"))
    return _GRID
