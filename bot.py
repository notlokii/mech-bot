"""Survival-first mech pilot for mechfight/1."""
import math
import random

import mechfight as mf
from config import Params
from world import load_grid


def wrap(angle):
    return (angle + math.pi) % math.tau - math.pi


def unit(x, y):
    length = math.hypot(x, y)
    return (x / length, y / length, length) if length > 1e-6 else (0.0, 0.0, 0.0)


def lob_pitch(horizontal, rise, speed, gravity):
    """Low-arc launch pitch that reaches a target, or None if it is out of range."""
    if horizontal < 1e-3:
        return None
    v2 = speed * speed
    disc = v2 * v2 - gravity * (gravity * horizontal * horizontal + 2 * rise * v2)
    if disc < 0:
        return None
    return math.atan((v2 - math.sqrt(disc)) / (gravity * horizontal))


class Bot:
    name = "mech-bot"

    def __init__(self, hello, params=None):
        self.p = params or Params()
        self.cfg = hello["config"]
        self.me_id = hello["slot"]
        self.rng = random.Random(hello.get("seed", 0))
        self.grid = load_grid()
        self.retreating = False
        self.strafe_sign = 1.0
        self.next_strafe = 0.0
        self.route = []
        self.route_goal = None
        self.next_route = 0.0
        self.cover = None
        self.next_cover = 0.0
        self.stuck_since = None
        self.last_hit_t = -99.0
        self.last_attacker = None
        self.last_move = [0.0, 0.0]
        self.last_aim = [0.0, 0.0]

    def act(self, obs):
        try:
            action = self.decide(obs)
        except Exception as exc:  # A bad tick must never stop the mech.
            mf.log(f"decide failed at tick {obs.get('tick')}: {exc!r}")
            action = self.fallback(obs)
        action["tick"] = obs["tick"]
        self.last_move = action.get("move", self.last_move)
        self.last_aim = action.get("aim", self.last_aim)
        return action

    def fallback(self, obs):
        me, zone = obs["self"], obs["zone"]
        dx, dy, _ = unit(zone["next_center"][0] - me["pos"][0], zone["next_center"][1] - me["pos"][1])
        return {"move": [dx, dy], "aim": self.last_aim, "boost": not me["in_zone"], "fire": [False, False]}

    # ------------------------------------------------------------------ decision
    def decide(self, obs):
        me = obs["self"]
        now = obs["t"]
        self.read_events(obs, now)
        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        target = self.pick_target(me, enemies)
        hp_frac = me["hp"] / max(me["max_hp"], 1.0)
        self.update_retreat(hp_frac, enemies, now)

        action = {"fire": [False, False]}
        self.choose_upgrade(obs, action)
        self.choose_pickup(me, obs, action)
        self.aim_and_fire(me, target, npcs, obs, hp_frac, enemies, action)
        move, boost, jump = self.plan_move(me, obs, target, enemies, npcs, hp_frac, now)
        action["move"] = [round(move[0], 4), round(move[1], 4)]
        action["boost"] = boost
        action["jump"] = jump
        return action

    def read_events(self, obs, now):
        for event in obs["events"]:
            if event["type"] == "damaged" and event.get("by") is not None:
                self.last_hit_t = now
                self.last_attacker = event["by"]

    def pick_target(self, me, enemies):
        if not enemies:
            return None
        x, y, z = me["pos"]
        best, best_score = None, 1e18
        for enemy in enemies:
            ex, ey, ez = enemy["pos"]
            distance = math.hypot(ex - x, ey - y)
            clear = self.grid.los(x, y, z + 7, ex, ey, ez + 4.5)
            hp = enemy.get("hp", 1000) / max(enemy.get("max_hp", 1400), 1)
            score = distance + 90 * hp + (0 if clear else 120)
            if hp < self.p.finish_target_hp:
                score -= 70
            if enemy["id"] == self.last_attacker:
                score -= 35
            if score < best_score:
                best, best_score = enemy, score
        return best

    def update_retreat(self, hp_frac, enemies, now):
        if hp_frac < self.p.retreat_hp and (enemies or now - self.last_hit_t < 3.0):
            self.retreating = True
        elif hp_frac >= self.p.reengage_hp:
            self.retreating = False

    # ------------------------------------------------------------------ loot and upgrades
    def choose_upgrade(self, obs, action):
        offer = obs.get("offer")
        if not offer:
            return
        ranks = {name: index for index, name in enumerate(self.p.upgrade_order)}
        choices = offer["choices"]
        action["pick"] = min(range(len(choices)), key=lambda c: ranks.get(choices[c]["id"], 999))

    def choose_pickup(self, me, obs, action):
        x, y, z = me["pos"]
        reach = self.cfg["loot"]["pickup_radius"]
        held = [w["kind"] if w else None for w in me["weapons"]]
        rank = {kind: index for index, kind in enumerate(self.p.weapon_rank)}
        best, best_rank = None, 99
        for item in obs["pickups"]:
            if math.hypot(item["pos"][0] - x, item["pos"][1] - y) > reach or abs(item["pos"][2] - z) > 4:
                continue
            if item["kind"] in held or item["kind"] not in rank:
                continue
            if rank[item["kind"]] < best_rank:
                best, best_rank = item, rank[item["kind"]]
        if best is None:
            return
        if held[1] is None:
            action["take"] = best["id"]
        elif held[1] in rank and best_rank < rank[held[1]] and not me["weapons"][1].get("charge"):
            action["take"] = best["id"]
            action["drop"] = 1

    # ------------------------------------------------------------------ aiming and weapons
    def aim_and_fire(self, me, target, npcs, obs, hp_frac, enemies, action):
        x, y, z = me["pos"]
        muzzle = z + me["stats"].get("muzzle_height", 7)
        if target is None:
            soft = self.soft_target(me, npcs, obs, hp_frac, enemies)
            if soft is None:
                self.idle_aim(me, obs, action)
                return
            target = soft
        tx, ty, tz = target["pos"]
        height = 4.5 if target.get("kind") == "mech" else 1.2
        vel = target.get("vel", [0, 0, 0])
        distance = math.hypot(tx - x, ty - y)
        clear = self.grid.los(x, y, muzzle, tx, ty, tz + height)
        weapons = [(slot, w) for slot, w in enumerate(me["weapons"]) if w]
        gun = next(((s, w) for s, w in weapons if w["kind"] == "gun"), None)
        laser = next(((s, w) for s, w in weapons if w["kind"] == "laser"), None)
        bolt = next(((s, w) for s, w in weapons if w["kind"] == "lightning"), None)
        mortar = next(((s, w) for s, w in weapons if w["kind"] == "mortar"), None)
        speed = gun[1]["stats"]["speed"] if gun else 252.0
        flight = distance / max(speed, 1.0)
        for _ in range(2):
            flight = math.hypot(tx + vel[0] * flight - x, ty + vel[1] * flight - y) / max(speed, 1.0)
        aim_x, aim_y, aim_z = tx + vel[0] * flight, ty + vel[1] * flight, tz + height + vel[2] * flight
        yaw = math.atan2(aim_y - y, aim_x - x)
        pitch = math.atan2(aim_z - muzzle, max(math.hypot(aim_x - x, aim_y - y), 0.1))
        pitch = max(me["stats"].get("min_pitch", -0.7), min(me["stats"].get("max_pitch", 1.3), pitch))
        tolerance = max(self.p.min_aim_tolerance, math.atan2(2.5 * self.p.aim_tolerance_scale, max(distance, 1.0)))
        yaw_error = abs(wrap(me["aim"][0] - yaw))
        pitch_error = abs(me["aim"][1] - pitch)
        aimed = yaw_error <= tolerance and pitch_error <= max(tolerance, 0.03)

        if mortar and mortar[1]["cooldown"] <= 0 and target.get("kind") == "mech":
            target_speed = math.hypot(vel[0], vel[1])
            stats = mortar[1]["stats"]
            if self.p.mortar_min_range <= distance <= 215 and (target_speed <= self.p.mortar_max_target_speed or not clear):
                arc = lob_pitch(distance, tz + 1 - (z + me["stats"].get("height", 9) + 0.5), stats["speed"], stats["gravity"])
                if arc is not None and arc <= me["stats"].get("max_pitch", 1.3):
                    action["aim"] = [round(yaw, 4), round(arc, 4)]
                    if abs(wrap(me["aim"][0] - yaw)) <= 0.05 and abs(me["aim"][1] - arc) <= 0.05:
                        action["fire"][mortar[0]] = True
                    return

        action["aim"] = [round(yaw, 4), round(pitch, 4)]
        if bolt and clear:
            stats = bolt[1]["stats"]
            in_cone = yaw_error <= math.radians(stats["cone_deg"]) * 0.8
            if distance <= stats["range"] * self.p.lightning_range_margin and in_cone and me["energy"] >= stats["energy"] + 4:
                action["fire"][bolt[0]] = True
        if laser:
            self.drive_laser(laser, clear, aimed, distance, hp_frac, action)
        if gun and clear and aimed and distance <= gun[1]["stats"]["range"]:
            if gun[1].get("ammo", 0) > 0 and gun[1].get("reload", 0) <= 0:
                action["fire"][gun[0]] = True
        elif gun and not clear and gun[1].get("reload", 0) <= 0:
            if gun[1].get("ammo", 0) < gun[1]["stats"]["mag"] * self.p.reload_when_idle_below:
                action["reload"] = True

    def drive_laser(self, laser, clear, aimed, distance, hp_frac, action):
        slot, weapon = laser
        stats = weapon["stats"]
        charge = weapon.get("charge", 0) or 0
        if charge > 0:
            full = charge >= stats["charge"] * self.p.laser_release_frac
            action["fire"][slot] = not (full and aimed and clear)
            if not clear and charge >= stats["charge"]:
                action["fire"][slot] = False
            return
        usable = clear and self.p.laser_min_range <= distance <= stats["range"]
        if usable and weapon["cooldown"] <= 0 and hp_frac >= self.p.laser_max_hp_risk:
            action["fire"][slot] = True

    def soft_target(self, me, npcs, obs, hp_frac, enemies):
        """Pick a safe farming target when no enemy is in view."""
        if enemies or self.retreating:
            return None
        x, y, z = me["pos"]
        best, best_distance = None, 1e18
        if hp_frac >= self.p.npc_min_hp:
            for npc in npcs:
                distance = math.hypot(npc["pos"][0] - x, npc["pos"][1] - y)
                if distance <= self.p.npc_radius and distance < best_distance and self.grid.los(x, y, z + 7, npc["pos"][0], npc["pos"][1], npc["pos"][2] + 1.2):
                    best, best_distance = npc, distance
        if best is not None:
            return best
        for crate in obs["crates"]:
            distance = math.hypot(crate["pos"][0] - x, crate["pos"][1] - y)
            if distance <= self.p.crate_radius and distance < best_distance and self.grid.los(x, y, z + 7, crate["pos"][0], crate["pos"][1], crate["pos"][2] + 1.0):
                best, best_distance = crate, distance
        if best is None:
            return None
        return {"id": best["id"], "kind": "crate", "pos": best["pos"], "vel": [0, 0, 0]}

    def idle_aim(self, me, obs, action):
        x, y, _ = me["pos"]
        now = obs["t"]
        threat_yaw = None
        for event in obs["events"]:
            if event["type"] == "damaged" and event.get("bearing") is not None:
                threat_yaw = event["bearing"]
        if threat_yaw is None:
            for sound in obs["sounds"]:
                if sound["kind"] in ("gun", "laser", "laser_charge", "lightning", "boost", "step") and sound["dist"] != "far":
                    threat_yaw = sound["bearing"]
                    break
        if threat_yaw is None:
            center = obs["zone"]["next_center"]
            threat_yaw = math.atan2(center[1] - y, center[0] - x) + 0.9 * math.sin(now * 0.9)
        action["aim"] = [round(wrap(threat_yaw), 4), 0.0]
        gun = next((w for w in me["weapons"] if w and w["kind"] == "gun"), None)
        if gun and gun.get("reload", 0) <= 0 and gun.get("ammo", 0) < gun["stats"]["mag"] * self.p.reload_when_idle_below:
            action["reload"] = True

    # ------------------------------------------------------------------ movement
    def plan_move(self, me, obs, target, enemies, npcs, hp_frac, now):
        x, y, _ = me["pos"]
        zone_goal, zone_urgent = self.zone_goal(me, obs, now)
        dodge = self.dodge_vector(me, obs, enemies)
        boost = False
        if zone_urgent:
            direction = self.navigate(x, y, zone_goal[0], zone_goal[1], now)
            boost = True
        elif self.retreating and enemies:
            direction, boost = self.retreat_vector(me, obs, enemies, zone_goal, now)
        elif target is not None:
            direction, boost = self.combat_vector(me, target, enemies, zone_goal, now)
        else:
            goal = self.objective(me, obs, npcs, hp_frac, zone_goal, now)
            direction = self.navigate(x, y, goal[0], goal[1], now)
        if dodge is not None:
            direction = unit(direction[0] + dodge[0] * 1.6, direction[1] + dodge[1] * 1.6)[:2]
            boost = True
        direction = self.avoid_walls(x, y, direction)
        if me["energy"] <= self.p.energy_reserve and not zone_urgent:
            boost = False
        jump = self.update_stuck(me, direction, now)
        if jump:
            boost = me["energy"] > 8
        return direction, boost, jump

    def zone_goal(self, me, obs, now):
        x, y, _ = me["pos"]
        zone = obs["zone"]
        cx, cy = zone["next_center"]
        radius = zone["next_radius"]
        safe_radius = max(radius - self.p.zone_margin, min(radius, self.p.zone_center_radius) * 0.4)
        dx, dy, distance = unit(x - cx, y - cy)
        if distance <= safe_radius:
            goal = (x, y)
            overshoot = 0.0
        else:
            goal = (cx + dx * safe_radius, cy + dy * safe_radius)
            overshoot = distance - safe_radius
        speed = max(me["stats"].get("walk_speed", 11.7), 1.0)
        deadline = max(zone["shrink_end"] - now, 0.0)
        urgent = not me["in_zone"] or (overshoot > 0 and overshoot / speed + self.p.zone_rush_buffer_s >= deadline)
        return goal, urgent

    def dodge_vector(self, me, obs, enemies):
        x, y, z = me["pos"]
        vx, vy, _ = me["vel"]
        gravity = self.cfg["weapons"]["mortar"]["gravity"]
        best = None
        best_time = 1e9
        for shot in obs["projectiles"]:
            px, py, pz = shot["pos"]
            qx, qy, qz = shot["vel"]
            if shot["kind"] == "mortar":
                disc = qz * qz + 2 * gravity * max(pz - z, 0.0)
                land = (qz + math.sqrt(disc)) / gravity if disc >= 0 else 0.0
                lx, ly = px + qx * land, py + qy * land
                dx, dy, distance = unit(x - lx, y - ly)
                if distance < self.p.shell_danger_radius and land < best_time:
                    best_time = land
                    best = (dx, dy) if distance > 0.5 else (-qy, qx)
                continue
            rx, ry = x - px, y - py
            wx, wy = qx - vx, qy - vy
            speed2 = wx * wx + wy * wy
            if speed2 < 1:
                continue
            when = (rx * wx + ry * wy) / speed2
            if 0 < when < self.p.dodge_horizon_s:
                miss = math.hypot(rx - wx * when, ry - wy * when)
                if miss < self.p.dodge_radius and when < best_time:
                    best_time = when
                    side = 1.0 if (wx * ry - wy * rx) >= 0 else -1.0
                    nx, ny, _ = unit(-wy * side, wx * side)
                    best = (nx, ny)
        for enemy in enemies:
            if not enemy.get("charging"):
                continue
            ex, ey, _ = enemy["pos"]
            bearing = math.atan2(y - ey, x - ex)
            if abs(wrap(enemy["aim"][0] - bearing)) < 0.09:
                nx, ny, _ = unit(-(y - ey) * self.strafe_sign, (x - ex) * self.strafe_sign)
                return (nx, ny)
        return best

    def combat_vector(self, me, target, enemies, zone_goal, now):
        x, y, _ = me["pos"]
        tx, ty, _ = target["pos"]
        ux, uy, distance = unit(tx - x, ty - y)
        if now >= self.next_strafe:
            self.strafe_sign = -self.strafe_sign if self.rng.random() < 0.8 else self.strafe_sign
            self.next_strafe = now + self.rng.uniform(self.p.strafe_min_s, self.p.strafe_max_s)
        preferred = self.p.preferred_range
        if target.get("hp", 1400) / max(target.get("max_hp", 1400), 1) < self.p.finish_target_hp:
            preferred *= 0.7
        if len(enemies) >= self.p.outnumbered_count:
            preferred *= 1.25
        radial = max(-1.0, min(1.0, (distance - preferred) / self.p.range_band)) * self.p.radial_gain
        if distance < self.p.close_range:
            radial = -1.0
        mx = -uy * self.strafe_sign + ux * radial
        my = ux * self.strafe_sign + uy * radial
        gx, gy = zone_goal
        if math.hypot(gx - x, gy - y) > 3:
            zx, zy, _ = unit(gx - x, gy - y)
            mx += zx * 0.9
            my += zy * 0.9
        dx, dy, _ = unit(mx, my)
        boost = distance < self.p.close_range or len(enemies) >= self.p.outnumbered_count
        return (dx, dy), boost

    def retreat_vector(self, me, obs, enemies, zone_goal, now):
        x, y, z = me["pos"]
        if now >= self.next_cover or self.cover is None:
            self.cover = self.find_cover(x, y, z, enemies, obs)
            self.next_cover = now + self.p.cover_refresh_s
        if self.cover is not None:
            return self.navigate(x, y, self.cover[0], self.cover[1], now), True
        ax, ay = 0.0, 0.0
        for enemy in enemies:
            ex, ey, distance = unit(x - enemy["pos"][0], y - enemy["pos"][1])
            weight = 1.0 / max(distance, 8.0)
            ax += ex * weight
            ay += ey * weight
        gx, gy = zone_goal
        zx, zy, _ = unit(gx - x, gy - y)
        dx, dy, _ = unit(ax * 60 + zx * 0.6, ay * 60 + zy * 0.6)
        return (dx, dy), True

    def find_cover(self, x, y, z, enemies, obs):
        grid = self.grid
        zone = obs["zone"]
        threats = sorted(enemies, key=lambda m: math.hypot(m["pos"][0] - x, m["pos"][1] - y))[:3]
        cx, cy = zone["center"]
        limit = zone["radius"] - 4
        best, best_score = None, 1e18
        for ring in self.p.cover_rings:
            for index in range(self.p.cover_angles):
                angle = index * math.tau / self.p.cover_angles + ring
                px, py = x + math.cos(angle) * ring, y + math.sin(angle) * ring
                if not grid.walkable(px, py) or math.hypot(px - cx, py - cy) > limit:
                    continue
                floor = grid.ground_at(px, py)
                if any(grid.los(t["pos"][0], t["pos"][1], t["pos"][2] + 8, px, py, floor + 4.5) for t in threats):
                    continue
                nearest = min(math.hypot(t["pos"][0] - px, t["pos"][1] - py) for t in threats)
                score = ring - 0.25 * min(nearest, 80)
                if score < best_score:
                    best, best_score = (px, py), score
            if best is not None:
                return best
        return best

    def objective(self, me, obs, npcs, hp_frac, zone_goal, now):
        x, y, z = me["pos"]
        if self.retreating:
            return zone_goal
        drop = self.best_drop(me, obs)
        if drop is not None:
            return drop
        if hp_frac >= self.p.loot_min_hp:
            held = [w["kind"] if w else None for w in me["weapons"]]
            best, best_distance = None, self.p.loot_radius
            for cube in obs["cubes"]:
                if abs(cube["pos"][2] - z) <= 4:
                    distance = math.hypot(cube["pos"][0] - x, cube["pos"][1] - y)
                    if distance < best_distance and self.in_next_zone(cube["pos"], obs):
                        best, best_distance = cube["pos"], distance
            if best is None:
                ranks = {kind: index for index, kind in enumerate(self.p.weapon_rank)}
                held_rank = ranks.get(held[1], 99) if held[1] else 99
                for item in obs["pickups"]:
                    if item["kind"] in held or ranks.get(item["kind"], 99) >= held_rank or abs(item["pos"][2] - z) > 4:
                        continue
                    distance = math.hypot(item["pos"][0] - x, item["pos"][1] - y)
                    if distance < best_distance and self.in_next_zone(item["pos"], obs):
                        best, best_distance = item["pos"], distance
            if best is not None:
                return (best[0], best[1])
        return zone_goal

    def in_next_zone(self, pos, obs):
        zone = obs["zone"]
        return math.hypot(pos[0] - zone["next_center"][0], pos[1] - zone["next_center"][1]) <= zone["next_radius"]

    def best_drop(self, me, obs):
        x, y, _ = me["pos"]
        speed = max(me["stats"].get("boost_speed", 19.5), 1.0)
        best, best_distance = None, self.p.drop_max_travel
        for drop in obs.get("drops", []):
            px, py, _ = drop["pos"]
            distance = math.hypot(px - x, py - y)
            if distance / speed + self.p.drop_lead_s < drop["lands_in"]:
                continue
            contested = any(m["kind"] == "mech" and math.hypot(m["pos"][0] - px, m["pos"][1] - py) < self.p.drop_contest_radius for m in obs["mechs"])
            if contested or distance >= best_distance:
                continue
            best, best_distance = (px, py), distance
        return best

    def navigate(self, x, y, gx, gy, now):
        if math.hypot(gx - x, gy - y) < 2.0:
            return (0.0, 0.0)
        if self.grid.los(x, y, self.grid.ground_at(x, y) + 4.5, gx, gy, self.grid.ground_at(gx, gy) + 4.5) and self.grid.walkable(gx, gy):
            self.route = []
            self.route_goal = None
            return unit(gx - x, gy - y)[:2]
        moved = self.route_goal is None or math.hypot(self.route_goal[0] - gx, self.route_goal[1] - gy) > 8
        if moved or now >= self.next_route or not self.route:
            self.route = self.grid.path(x, y, gx, gy)
            self.route_goal = (gx, gy)
            self.next_route = now + self.p.path_refresh_s
        while self.route and math.hypot(self.route[0][0] - x, self.route[0][1] - y) < self.p.waypoint_reach:
            self.route.pop(0)
        point = self.route[0] if self.route else (gx, gy)
        return unit(point[0] - x, point[1] - y)[:2]

    def avoid_walls(self, x, y, direction):
        dx, dy = direction[0], direction[1]
        if abs(dx) + abs(dy) < 1e-3:
            return (0.0, 0.0)
        if self.grid.walkable(x + dx * 5, y + dy * 5):
            return (dx, dy)
        for offset in (0.6, -0.6, 1.2, -1.2, 1.9, -1.9):
            c, s = math.cos(offset), math.sin(offset)
            rx, ry = dx * c - dy * s, dx * s + dy * c
            if self.grid.walkable(x + rx * 5, y + ry * 5):
                self.strafe_sign = -self.strafe_sign if abs(offset) > 1.5 else self.strafe_sign
                return (rx, ry)
        return (dx, dy)

    def update_stuck(self, me, direction, now):
        moving = abs(direction[0]) + abs(direction[1]) > 0.1
        speed = math.hypot(me["vel"][0], me["vel"][1])
        if not moving or speed >= self.p.stuck_speed:
            self.stuck_since = None
            return False
        if self.stuck_since is None:
            self.stuck_since = now
        if now - self.stuck_since >= self.p.stuck_time_s:
            self.route = []
            self.next_route = 0.0
            return bool(me["grounded"]) or me["energy"] > 10
        return False


if __name__ == "__main__":
    mf.run(Bot)
