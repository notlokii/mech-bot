"""Approximate offline mechfight simulator for comparing bot versions.

This is not the official server. It follows protocol.md where the rules are specified and models the
house squad from the track description: 15 bots that share vision, never hit each other, and send one
hunter early, more as they are thinned out, and all of them late. Tier skill is an estimate.
"""
import gzip
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from world import load_grid  # noqa: E402

DT = 0.05
SQUAD = 1
PLAYER = 0
NPC_TEAM = 2
TIER_ORDER = ["easy"] * 4 + ["medium"] * 4 + ["hard"] * 4 + ["expert"] * 3
TIERS = {
    "easy": {"sigma": 0.060, "lead": 0.45, "react": 0.85, "pref": 120.0, "strafe": 0.35, "loot": 0.35},
    "medium": {"sigma": 0.034, "lead": 0.72, "react": 0.55, "pref": 125.0, "strafe": 0.60, "loot": 0.55},
    "hard": {"sigma": 0.018, "lead": 0.90, "react": 0.35, "pref": 130.0, "strafe": 0.85, "loot": 0.75},
    "expert": {"sigma": 0.009, "lead": 1.00, "react": 0.20, "pref": 135.0, "strafe": 1.00, "loot": 0.90},
}
HUNT = {"base": 1, "per_kill": 1.0, "late_start": 150.0, "late_every": 22.0}


def wrap(angle):
    return (angle + math.pi) % math.tau - math.pi


def load_config(stream_path):
    with gzip.open(stream_path, "rt") as handle:
        return json.loads(handle.readline())["config"]


def new_weapon(kind, stats):
    weapon = {"kind": kind, "cooldown": 0.0}
    if kind == "gun":
        weapon.update(ammo=int(stats["gun"]["mag"]), reload=0.0)
    if kind == "laser":
        weapon["charge"] = 0.0
    return weapon


class Mech:
    def __init__(self, mech_id, x, y, z, cfg, team, tier):
        self.id, self.x, self.y, self.z = mech_id, x, y, z
        self.vx = self.vy = self.vz = 0.0
        self.yaw, self.pitch = 0.0, 0.0
        self.cmd_yaw, self.cmd_pitch = 0.0, 0.0
        self.stats = dict(cfg["mech"])
        self.wstats = {kind: dict(values) for kind, values in cfg["weapons"].items()}
        self.hp = self.max_hp = float(self.stats["hp"])
        self.energy = float(self.stats["energy"])
        self.grounded = True
        self.cubes = 0
        self.upgrades = []
        self.weapons = [new_weapon("gun", self.wstats), None]
        self.team, self.tier = team, tier
        self.alive = True
        self.place = None
        self.last_combat = -99.0
        self.last_spend = -99.0
        self.short_until = -99.0
        self.events, self.sounds, self.beams = [], [], []
        self.seen = {}
        self.action = {}
        self.offer = None
        self.offer_queue = 0
        self.offer_counter = 0
        self.thresholds_hit = 0
        self.killer = None
        self.route, self.next_route = [], 0.0
        self.strafe, self.next_strafe = 1.0, 0.0
        self.acquired = None
        self.loot_at = None

    def charging(self):
        return any(w and w["kind"] == "laser" and w.get("charge", 0) > 0 for w in self.weapons)


class Sim:
    def __init__(self, cfg, seed, make_bot, hunt=None, keep_going=False):
        self.cfg, self.rng, self.grid = cfg, random.Random(seed), load_grid()
        self.hunt = dict(HUNT, **(hunt or {}))
        self.keep_going = keep_going
        self.t, self.tick_no = 0.0, 0
        self.bullets, self.shells = [], []
        self.upgrade_defs = {u["id"]: u for u in cfg["upgrades"]}
        self.next_id = {"crate": 1000, "cube": 5000, "pickup": 8000, "drop": 20000, "shot": 1}
        self.player_slot = self.rng.randrange(16)
        self._build_zone()
        self._spawn_mechs()
        self._spawn_world()
        self.player = self.mechs[self.player_slot]
        hello = {"type": "hello", "protocol": "mechfight/1", "slot": self.player_slot, "seed": self.rng.randrange(2 ** 31),
                 "map": {"dir": "/arena", "files": ["v2/world.json", "v2/terrain.i16"]}, "config": cfg}
        self.bot = make_bot(hello)
        self.player_kills = 0
        self.squad_sees = False
        self.hunters = set()
        self.drops = []
        self.drop_times = list(cfg.get("drops", {}).get("times", []))

    # ------------------------------------------------------------------ setup
    def _build_zone(self):
        zone = self.cfg["zone"]
        center, radius, clock = list(zone["start_center"]), float(zone["start_radius"]), 0.0
        self.phases = []
        for phase in zone["phases"]:
            slack = max(radius - phase["radius"], 0.0)
            angle, reach = self.rng.uniform(0, math.tau), slack * math.sqrt(self.rng.random()) * 0.9
            next_center = [center[0] + math.cos(angle) * reach, center[1] + math.sin(angle) * reach]
            start, end = clock + phase["hold_s"], clock + phase["hold_s"] + phase["shrink_s"]
            self.phases.append((start, end, center, radius, next_center, float(phase["radius"]), float(phase["dps"])))
            center, radius, clock = next_center, float(phase["radius"]), end

    def zone_now(self):
        for start, end, center, radius, next_center, next_radius, dps in self.phases:
            if self.t < end:
                if self.t <= start:
                    now_center, now_radius = center, radius
                else:
                    f = (self.t - start) / max(end - start, 1e-6)
                    now_center = [center[0] + (next_center[0] - center[0]) * f, center[1] + (next_center[1] - center[1]) * f]
                    now_radius = radius + (next_radius - radius) * f
                return {"center": [round(now_center[0], 2), round(now_center[1], 2)], "radius": round(now_radius, 2),
                        "next_center": [round(next_center[0], 2), round(next_center[1], 2)], "next_radius": next_radius,
                        "shrink_start": start, "shrink_end": end, "dps": dps}
        start, end, center, radius, next_center, next_radius, dps = self.phases[-1]
        return {"center": list(next_center), "radius": next_radius, "next_center": list(next_center), "next_radius": next_radius,
                "shrink_start": start, "shrink_end": end, "dps": dps}

    def _random_ground(self, center=None, radius=None):
        grid = self.grid
        for _ in range(400):
            if center is None:
                x, y = self.rng.uniform(grid.min_x + 30, grid.max_x - 30), self.rng.uniform(grid.min_y + 30, grid.max_y - 30)
            else:
                angle, reach = self.rng.uniform(0, math.tau), radius * math.sqrt(self.rng.random())
                x, y = center[0] + math.cos(angle) * reach, center[1] + math.sin(angle) * reach
            if grid.walkable(x, y):
                return x, y, grid.ground_at(x, y)
        cell = grid.nearest_walkable(0.0, 0.0, 120)
        x, y = grid.center(*cell)
        return x, y, grid.ground_at(x, y)

    def _spawn_mechs(self):
        match, zone = self.cfg["match"], self.cfg["zone"]
        radius = zone["start_radius"] * match["spawn_zone_fraction"]
        spots = []
        for _ in range(16):
            candidate = None
            for separation in (match["spawn_min_separation"], 90, 50, 0):
                for _ in range(60):
                    point = self._random_ground(zone["start_center"], radius)
                    if all(math.hypot(point[0] - s[0], point[1] - s[1]) >= separation for s in spots):
                        candidate = point
                        break
                if candidate:
                    break
            spots.append(candidate)
        tiers = list(TIER_ORDER)
        self.mechs = []
        for mech_id, (x, y, z) in enumerate(spots):
            if mech_id == self.player_slot:
                self.mechs.append(Mech(mech_id, x, y, z, self.cfg, PLAYER, None))
            else:
                mech = Mech(mech_id, x, y, z, self.cfg, SQUAD, tiers.pop(0))
                if self.rng.random() < TIERS[mech.tier]["loot"]:
                    mech.loot_at = self.rng.uniform(4.0, 45.0)
                self.mechs.append(mech)
        self.squad_order = sorted((m for m in self.mechs if m.team == SQUAD), key=lambda m: list(TIERS).index(m.tier))

    def _spawn_world(self):
        cfg = self.cfg
        self.crates, self.cubes, self.pickups, self.npcs = [], [], [], []
        for _ in range(int(cfg["crates"]["count"])):
            x, y, z = self._random_ground()
            self.crates.append({"id": self._id("crate"), "x": x, "y": y, "z": z, "hp": float(cfg["crates"]["hp"])})
        kinds = [k for k, weight in cfg["loot"]["weights"].items() if weight > 0]
        for _ in range(int(cfg["loot"]["floor_count"])):
            x, y, z = self._random_ground()
            self.pickups.append({"id": self._id("pickup"), "kind": self.rng.choice(kinds), "x": x, "y": y, "z": z})
        for index in range(int(cfg["npc"]["count"])):
            x, y, z = self._random_ground()
            self.npcs.append({"id": 100 + index, "x": x, "y": y, "z": z, "hx": x, "hy": y, "hp": float(cfg["npc"]["hp"]), "cd": 0.0, "angry": False})

    def _id(self, kind):
        value = self.next_id[kind]
        self.next_id[kind] += 1
        return value

    # ------------------------------------------------------------------ main loop
    def run(self):
        duration = self.cfg["match"]["duration_s"]
        while self.t < duration:
            living = [m for m in self.mechs if m.alive]
            if len(living) <= 1 or (not self.player.alive and not self.keep_going):
                break
            self._update_hunters(living)
            for mech in living:
                if mech.team == PLAYER:
                    action = self.bot.act(self._observe(mech)) or {}
                    self._apply_commands(mech, action)
                else:
                    self._apply_commands(mech, self._house_action(mech))
            for _ in range(int(self.cfg["match"]["decide_every"])):
                self._tick()
        living = [m for m in self.mechs if m.alive]
        if self.player.alive:
            ranked = sorted(living, key=lambda m: -m.hp)
            self.player.place = ranked.index(self.player) + 1
        return {"place": self.player.place, "score": 16 - self.player.place, "death_t": round(self.t, 1),
                "kills": self.player_kills, "killer": self.player.killer, "cubes": self.player.cubes,
                "survived": self.player.alive}

    def _tick(self):
        self.t = round(self.t + DT, 3)
        self.tick_no += 1
        for mech in self.mechs:
            if mech.alive:
                self._step_mech(mech)
        self._step_bullets()
        self._step_shells()
        if self.tick_no % 2 == 0:
            self._step_npcs()
        self._step_zone_and_regen()
        self._step_loot()
        self._step_drops()
        self._resolve_deaths()

    def _update_hunters(self, living):
        hunt = self.hunt
        late = max(0.0, self.t - hunt["late_start"]) / hunt["late_every"]
        count = int(hunt["base"] + self.player_kills * hunt["per_kill"] + late)
        squad = [m for m in self.squad_order if m.alive]
        self.hunters = {m.id for m in squad[:count]}
        player = self.player
        self.squad_sees = player.alive and any(self._can_see(m, player) for m in squad)

    # ------------------------------------------------------------------ physics
    def _apply_commands(self, mech, action):
        mech.action = action
        aim = action.get("aim")
        if aim:
            mech.cmd_yaw, mech.cmd_pitch = float(aim[0]), float(aim[1])
        if "pick" in action and mech.offer:
            choices = mech.offer["choices"]
            index = int(action["pick"])
            if 0 <= index < len(choices):
                self._apply_upgrade(mech, choices[index]["id"])
                mech.offer = None
        if "take" in action:
            self._take(mech, action["take"], action.get("drop"))

    def _step_mech(self, mech):
        stats, action, grid = mech.stats, mech.action, self.grid
        turn = stats["turn_rate"] * DT
        mech.yaw = wrap(mech.yaw + max(-turn, min(turn, wrap(mech.cmd_yaw - mech.yaw))))
        pitch_step = stats["pitch_rate"] * DT
        target_pitch = max(stats["min_pitch"], min(stats["max_pitch"], mech.cmd_pitch))
        mech.pitch += max(-pitch_step, min(pitch_step, target_pitch - mech.pitch))

        mx, my = action.get("move") or (0.0, 0.0)
        length = math.hypot(mx, my)
        if length > 1:
            mx, my, length = mx / length, my / length, 1.0
        wants_boost = bool(action.get("boost")) and mech.energy > 0
        spent = False
        speed = stats["walk_speed"]
        if wants_boost and mech.grounded and length > 0.05:
            speed = stats["boost_speed"]
            mech.energy = max(0.0, mech.energy - stats["boost_drain"] * DT)
            spent = True
        if mech.charging():
            speed *= stats["charge_move_mult"]
        accel = (stats["ground_accel"] if mech.grounded else stats["air_accel"]) * DT
        dvx, dvy = mx * speed - mech.vx, my * speed - mech.vy
        change = math.hypot(dvx, dvy)
        if change > accel:
            dvx, dvy = dvx / change * accel, dvy / change * accel
        mech.vx += dvx
        mech.vy += dvy
        if action.get("jump") and mech.grounded:
            mech.vz = stats["jump_speed"]
            mech.grounded = False
        if not mech.grounded:
            if wants_boost:
                mech.vz = min(stats["max_rise_speed"], mech.vz + stats["jet_accel"] * DT)
                mech.energy = max(0.0, mech.energy - stats["jet_drain"] * DT)
                spent = True
            mech.vz -= stats["gravity"] * DT
        if spent:
            mech.last_spend = self.t
        elif self.t - mech.last_spend >= stats["energy_regen_delay"]:
            mech.energy = min(stats["energy"], mech.energy + stats["energy_regen"] * DT)

        radius = stats["radius"] * (1 + self.cfg["cubes"]["size_per_cube"] * mech.cubes)
        nx = mech.x + mech.vx * DT
        if self._body_blocked(nx, mech.y, mech.z, radius):
            mech.vx, nx = 0.0, mech.x
        ny = mech.y + mech.vy * DT
        if self._body_blocked(nx, ny, mech.z, radius):
            mech.vy, ny = 0.0, mech.y
        mech.x, mech.y = nx, ny
        mech.z += mech.vz * DT
        floor = grid.surface(mech.x, mech.y, mech.z)
        if mech.z <= floor or (mech.grounded and mech.z - floor < 0.8):
            mech.z, mech.vz, mech.grounded = floor, 0.0, True
        else:
            mech.grounded = False
        self._step_weapons(mech)

    def _body_blocked(self, x, y, z, radius):
        wall = self.grid.wall
        return wall(x, y, z) or wall(x + radius, y, z) or wall(x - radius, y, z) or wall(x, y + radius, z) or wall(x, y - radius, z)

    def _step_weapons(self, mech):
        triggers = mech.action.get("fire") or (False, False)
        for slot, weapon in enumerate(mech.weapons):
            if weapon is None:
                continue
            weapon["cooldown"] = max(0.0, weapon["cooldown"] - DT)
            held = bool(triggers[slot]) if slot < len(triggers) else False
            kind = weapon["kind"]
            stats = mech.wstats[kind]
            if kind == "gun":
                if weapon["reload"] > 0:
                    weapon["reload"] = max(0.0, weapon["reload"] - DT)
                    if weapon["reload"] == 0:
                        weapon["ammo"] = int(stats["mag"])
                elif mech.action.get("reload") and weapon["ammo"] < stats["mag"]:
                    weapon["reload"] = stats["reload"]
                elif held and weapon["cooldown"] <= 0 and weapon["ammo"] > 0:
                    self._fire_gun(mech, stats)
                    weapon["ammo"] -= 1
                    weapon["cooldown"] = stats["cooldown"]
                    if weapon["ammo"] == 0:
                        weapon["reload"] = stats["reload"]
            elif kind == "laser":
                if self.t < mech.short_until:
                    weapon["charge"] = 0.0
                elif weapon["charge"] > 0:
                    weapon["charge"] = min(stats["charge"] + stats["hold_s"], weapon["charge"] + DT)
                    release = not held and weapon["charge"] >= stats["min_charge"]
                    if release or weapon["charge"] >= stats["charge"] + stats["hold_s"]:
                        self._fire_laser(mech, stats, min(weapon["charge"], stats["charge"]))
                        weapon["charge"] = 0.0
                        weapon["cooldown"] = stats["cooldown"]
                elif held and weapon["cooldown"] <= 0:
                    weapon["charge"] = DT
            elif kind == "lightning" and held and weapon["cooldown"] <= 0 and mech.energy >= stats["energy"]:
                mech.energy -= stats["energy"]
                mech.last_spend = self.t
                weapon["cooldown"] = stats["cooldown"]
                self._fire_lightning(mech, stats)
            elif kind == "mortar" and held and weapon["cooldown"] <= 0:
                weapon["cooldown"] = stats["cooldown"]
                cp = math.cos(mech.pitch)
                self.shells.append({"id": self._id("shot"), "x": mech.x + math.cos(mech.yaw) * 3, "y": mech.y + math.sin(mech.yaw) * 3,
                                    "z": mech.z + mech.stats["height"] + 0.5, "vx": stats["speed"] * cp * math.cos(mech.yaw),
                                    "vy": stats["speed"] * cp * math.sin(mech.yaw), "vz": stats["speed"] * math.sin(mech.pitch),
                                    "ttl": stats["ttl"], "owner": mech.id, "team": mech.team, "stats": stats})
                self._sound("mortar", mech)

    def _fire_gun(self, mech, stats):
        spread = math.radians(stats["spread_deg"])
        yaw = mech.yaw + self.rng.gauss(0, spread)
        pitch = mech.pitch + self.rng.gauss(0, spread)
        cp = math.cos(pitch)
        dx, dy, dz = cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)
        self.bullets.append({"id": self._id("shot"), "x": mech.x + dx * 3, "y": mech.y + dy * 3, "z": mech.z + mech.stats["muzzle_height"] + dz * 3,
                             "vx": dx * stats["speed"], "vy": dy * stats["speed"], "vz": dz * stats["speed"],
                             "left": stats["range"], "damage": stats["damage"], "owner": mech.id, "team": mech.team, "kind": "gun"})
        self._sound("gun", mech)

    def _fire_laser(self, mech, stats, charge):
        frac = stats["min_frac"] + (1 - stats["min_frac"]) * min(charge / stats["charge"], 1.0)
        amount = stats["damage"] * frac
        cp = math.cos(mech.pitch)
        dx, dy, dz = cp * math.cos(mech.yaw), cp * math.sin(mech.yaw), math.sin(mech.pitch)
        x, y, z = mech.x, mech.y, mech.z + mech.stats["muzzle_height"]
        self._sound("laser", mech)
        travelled = 3.0
        while travelled < stats["range"]:
            px, py, pz = x + dx * travelled, y + dy * travelled, z + dz * travelled
            if self.grid.solid_at(px, py, pz):
                return
            for other in self.mechs:
                if other.alive and other.team != mech.team and self._in_body(other, px, py, pz, 0.6):
                    extra = 1 + stats["cookoff"] if any(w and w["kind"] == "mortar" for w in other.weapons) else 1.0
                    self._damage(other, amount * extra, mech, "laser", mech.x, mech.y)
                    return
            travelled += 2.0

    def _fire_lightning(self, mech, stats):
        best, best_distance = None, stats["range"]
        cone = math.radians(stats["cone_deg"])
        for other in self.mechs:
            if not other.alive or other.team == mech.team:
                continue
            distance = math.hypot(other.x - mech.x, other.y - mech.y)
            if distance > best_distance or abs(wrap(math.atan2(other.y - mech.y, other.x - mech.x) - mech.yaw)) > cone:
                continue
            if self.grid.los(mech.x, mech.y, mech.z + 7, other.x, other.y, other.z + 4.5):
                best, best_distance = other, distance
        self._sound("lightning", mech)
        if best is not None:
            self._damage(best, stats["damage"], mech, "lightning", mech.x, mech.y)
            if any(w and w["kind"] == "laser" for w in best.weapons):
                best.short_until = self.t + stats["short_s"]

    def _in_body(self, mech, x, y, z, pad):
        radius = mech.stats["radius"] * (1 + self.cfg["cubes"]["size_per_cube"] * mech.cubes) + pad
        height = mech.stats["height"] * (1 + self.cfg["cubes"]["size_per_cube"] * mech.cubes)
        return abs(mech.x - x) <= radius and abs(mech.y - y) <= radius and mech.z <= z <= mech.z + height and (mech.x - x) ** 2 + (mech.y - y) ** 2 <= radius * radius

    def _step_bullets(self):
        keep = []
        for bullet in self.bullets:
            speed = math.sqrt(bullet["vx"] ** 2 + bullet["vy"] ** 2 + bullet["vz"] ** 2)
            parts = max(1, int(speed * DT / 3.0))
            step = DT / parts
            done = False
            for _ in range(parts):
                bullet["x"] += bullet["vx"] * step
                bullet["y"] += bullet["vy"] * step
                bullet["z"] += bullet["vz"] * step
                bullet["left"] -= speed * step
                if bullet["left"] <= 0 or self.grid.solid_at(bullet["x"], bullet["y"], bullet["z"]):
                    done = True
                    break
                if self._bullet_hits(bullet):
                    done = True
                    break
            if not done:
                keep.append(bullet)
        self.bullets = keep

    def _bullet_hits(self, bullet):
        x, y, z = bullet["x"], bullet["y"], bullet["z"]
        shooter = self.mechs[bullet["owner"]] if bullet["owner"] < 16 else None
        for mech in self.mechs:
            if mech.alive and mech.team != bullet["team"] and self._in_body(mech, x, y, z, 0.3):
                self._damage(mech, bullet["damage"], shooter, bullet["kind"], x - bullet["vx"] * 0.02, y - bullet["vy"] * 0.02)
                return True
        if bullet["team"] == PLAYER:
            for npc in self.npcs:
                if npc["hp"] > 0 and (npc["x"] - x) ** 2 + (npc["y"] - y) ** 2 <= 4.5 and npc["z"] <= z <= npc["z"] + 3.5:
                    self._hit_npc(npc, bullet["damage"], shooter)
                    return True
            for crate in self.crates:
                if crate["hp"] > 0 and (crate["x"] - x) ** 2 + (crate["y"] - y) ** 2 <= 3.0 and crate["z"] <= z <= crate["z"] + 2.4:
                    self._hit_crate(crate, bullet["damage"], shooter)
                    return True
        return False

    def _step_shells(self):
        keep = []
        for shell in self.shells:
            shell["ttl"] -= DT
            shell["vz"] -= shell["stats"]["gravity"] * DT
            shell["x"] += shell["vx"] * DT
            shell["y"] += shell["vy"] * DT
            shell["z"] += shell["vz"] * DT
            direct = next((m for m in self.mechs if m.alive and m.team != shell["team"] and self._in_body(m, shell["x"], shell["y"], shell["z"], 0.5)), None)
            if direct is not None or shell["ttl"] <= 0 or self.grid.solid_at(shell["x"], shell["y"], shell["z"]):
                self._explode(shell, direct)
            else:
                keep.append(shell)
        self.shells = keep

    def _explode(self, shell, direct):
        stats = shell["stats"]
        owner = self.mechs[shell["owner"]]
        x, y, z = shell["x"], shell["y"], shell["z"]
        self._sound_at("explosion", x, y, None)
        if direct is not None:
            self._damage(direct, stats["damage"], owner, "mortar", x, y)
        for mech in self.mechs:
            if not mech.alive or mech is direct or (mech.team == shell["team"] and mech is not owner):
                continue
            distance = math.hypot(mech.x - x, mech.y - y)
            if distance > stats["splash_radius"] or not self.grid.los(x, y, z + 0.5, mech.x, mech.y, mech.z + 4.5):
                continue
            falloff = 1 - (1 - stats["splash_edge"]) * distance / stats["splash_radius"]
            amount = stats["splash"] * falloff * (mech.stats["self_damage_mult"] if mech is owner else 1.0)
            self._damage(mech, amount, owner, "mortar", x, y)
            ux, uy = (mech.x - x) / max(distance, 0.5), (mech.y - y) / max(distance, 0.5)
            mech.vx += ux * stats["knock"] * falloff
            mech.vy += uy * stats["knock"] * falloff
            mech.vz = stats["knock_up"] * falloff
            mech.grounded = False
            mech.energy = max(0.0, mech.energy - stats["drain"] * falloff)

    def _damage(self, target, amount, attacker, weapon, ox, oy):
        if not target.alive or target.hp <= 0:
            return
        if attacker is not None:
            amount *= 1 + self.cfg["cubes"]["damage_per_cube"] * min(attacker.cubes, self.cfg["cubes"]["stat_cap"])
            attacker.last_combat = self.t
        bearing = math.atan2(oy - target.y, ox - target.x)
        rear = abs(wrap(bearing - target.yaw)) > math.radians(target.stats["front_arc_deg"] / 2)
        if rear:
            amount *= target.stats["rear_mult"]
        amount *= max(0.0, 1 - target.stats["armor"])
        target.hp -= amount
        target.last_combat = self.t
        if attacker is not None:
            target.last_attacker = attacker.id
        target.events.append({"type": "damaged", "amount": round(amount, 1), "weapon": weapon, "bearing": round(bearing, 3),
                              "by": attacker.id if attacker else None, "rear": rear})
        if attacker is not None and attacker.team == PLAYER:
            attacker.events.append({"type": "hit", "target": target.id, "amount": round(amount, 1), "weapon": weapon})

    def _hit_npc(self, npc, amount, shooter):
        npc["hp"] -= amount
        npc["angry"] = True
        if npc["hp"] <= 0 and shooter is not None:
            for _ in range(int(self.cfg["npc"]["cubes"])):
                self._drop_cube(npc["x"], npc["y"], npc["z"])

    def _hit_crate(self, crate, amount, shooter):
        crate["hp"] -= amount
        if crate["hp"] <= 0:
            self._drop_cube(crate["x"], crate["y"], crate["z"])
            self._sound_at("crate", crate["x"], crate["y"], None)
            if self.rng.random() < self.cfg["loot"]["crate_weapon_chance"]:
                kinds = [k for k, w in self.cfg["loot"]["weights"].items() if w > 0]
                self.pickups.append({"id": self._id("pickup"), "kind": self.rng.choice(kinds), "x": crate["x"], "y": crate["y"], "z": crate["z"]})

    def _drop_cube(self, x, y, z):
        self.cubes.append({"id": self._id("cube"), "x": x + self.rng.uniform(-2, 2), "y": y + self.rng.uniform(-2, 2), "z": z})

    def _step_npcs(self):
        npc_cfg = self.cfg["npc"]
        player = self.player
        if not player.alive:
            return
        for npc in self.npcs:
            if npc["hp"] <= 0:
                continue
            npc["cd"] = max(0.0, npc["cd"] - DT * 2)
            distance = math.hypot(player.x - npc["x"], player.y - npc["y"])
            if distance > npc_cfg["deaggro_range"]:
                npc["angry"] = False
                continue
            if not npc["angry"] and distance > npc_cfg["aggro_range"]:
                continue
            if npc["cd"] > 0 or not self.grid.los(npc["x"], npc["y"], npc["z"] + 3, player.x, player.y, player.z + 4.5):
                continue
            npc["cd"] = npc_cfg["cooldown"]
            yaw = math.atan2(player.y - npc["y"], player.x - npc["x"]) + self.rng.gauss(0, math.radians(npc_cfg["spread_deg"]))
            pitch = math.atan2(player.z + 4.5 - npc["z"] - 3, max(distance, 1.0))
            speed = npc_cfg["projectile_speed"]
            self.bullets.append({"id": self._id("shot"), "x": npc["x"], "y": npc["y"], "z": npc["z"] + 3,
                                 "vx": speed * math.cos(pitch) * math.cos(yaw), "vy": speed * math.cos(pitch) * math.sin(yaw), "vz": speed * math.sin(pitch),
                                 "left": 120.0, "damage": npc_cfg["damage"], "owner": npc["id"], "team": NPC_TEAM, "kind": "npc"})

    def _step_zone_and_regen(self):
        zone = self.zone_now()
        cx, cy = zone["center"]
        for mech in self.mechs:
            if not mech.alive:
                continue
            outside = math.hypot(mech.x - cx, mech.y - cy) > zone["radius"]
            if outside:
                mech.hp -= zone["dps"] * DT
                mech.last_combat = self.t
                if mech.team == PLAYER:
                    mech.events.append({"type": "damaged", "amount": round(zone["dps"] * DT, 2), "weapon": "zone", "bearing": None, "by": None})
                    mech.last_attacker = None
            elif self.t - mech.last_combat >= mech.stats["regen_delay"]:
                mech.hp = min(mech.max_hp, mech.hp + mech.stats["regen_per_s"] * DT)

    def _step_loot(self):
        cube_cfg = self.cfg["cubes"]
        keep = []
        for cube in self.cubes:
            best, best_distance = None, cube_cfg["pickup_radius"]
            for mech in self.mechs:
                if mech.alive and abs(mech.z - cube["z"]) <= 4:
                    distance = math.hypot(mech.x - cube["x"], mech.y - cube["y"])
                    if distance <= best_distance:
                        best, best_distance = mech, distance
            if best is None:
                keep.append(cube)
            else:
                self._gain_cubes(best, 1)
                best.events.append({"type": "pickup", "cubes": best.cubes})
        self.cubes = keep
        for mech in self.mechs:
            if mech.alive and mech.team == SQUAD and mech.loot_at is not None and self.t >= mech.loot_at:
                mech.weapons[1] = new_weapon(self.rng.choice(["mortar", "laser", "lightning"]), mech.wstats)
                mech.loot_at = None
            if mech.alive and mech.offer and self.t >= mech.offer["expires"]:
                self._apply_upgrade(mech, mech.offer["choices"][0]["id"])
                mech.offer = None
            if mech.alive and mech.team == PLAYER and mech.offer is None and mech.offer_queue > 0:
                self._make_offer(mech)

    def _gain_cubes(self, mech, count):
        cube_cfg = self.cfg["cubes"]
        old_max = mech.max_hp
        mech.cubes += count
        mech.max_hp = mech.stats["hp"] * (1 + cube_cfg["hp_per_cube"] * min(mech.cubes, cube_cfg["stat_cap"]))
        mech.hp += mech.max_hp - old_max
        while mech.thresholds_hit < len(cube_cfg["upgrade_thresholds"]) and mech.cubes >= cube_cfg["upgrade_thresholds"][mech.thresholds_hit]:
            mech.thresholds_hit += 1
            mech.offer_queue += 1

    def _make_offer(self, mech):
        held = {w["kind"] for w in mech.weapons if w}
        options = [u for u in self.cfg["upgrades"] if mech.upgrades.count(u["id"]) < u["max"] and (not u.get("requires") or u["requires"] in held)]
        if not options:
            mech.offer_queue = 0
            return
        picks = self.rng.sample(options, min(3, len(options)))
        mech.offer_counter += 1
        mech.offer_queue -= 1
        mech.offer = {"id": mech.offer_counter, "expires": self.t + self.cfg["bot"]["offer_timeout_s"],
                      "choices": [{"id": u["id"], "name": u["name"], "desc": u["desc"]} for u in picks]}

    def _apply_upgrade(self, mech, upgrade_id):
        upgrade = self.upgrade_defs[upgrade_id]
        mech.upgrades.append(upgrade_id)
        for effect in upgrade["effects"]:
            parts = effect["path"].split(".")
            store = mech.stats if parts[0] == "mech" else mech.wstats[parts[1]]
            key = parts[-1]
            if "mul" in effect:
                store[key] *= effect["mul"]
            if "add" in effect:
                store[key] += effect["add"]
        cube_cfg = self.cfg["cubes"]
        old_max = mech.max_hp
        mech.max_hp = mech.stats["hp"] * (1 + cube_cfg["hp_per_cube"] * min(mech.cubes, cube_cfg["stat_cap"]))
        mech.hp += mech.max_hp - old_max
        mech.events.append({"type": "upgrade", "id": upgrade_id})

    def _take(self, mech, pickup_id, drop_slot):
        item = next((p for p in self.pickups if p["id"] == pickup_id), None)
        if item is None or math.hypot(item["x"] - mech.x, item["y"] - mech.y) > self.cfg["loot"]["pickup_radius"] or abs(item["z"] - mech.z) > 4:
            mech.events.append({"type": "take_failed", "id": pickup_id, "reason": "range" if item else "gone"})
            return
        if any(w and w["kind"] == item["kind"] for w in mech.weapons):
            mech.events.append({"type": "take_failed", "id": pickup_id, "reason": "have"})
            return
        slot = next((s for s, w in enumerate(mech.weapons) if w is None), None)
        if slot is None:
            if drop_slot not in (0, 1):
                mech.events.append({"type": "take_failed", "id": pickup_id, "reason": "full"})
                return
            slot = int(drop_slot)
            old = mech.weapons[slot]
            if old.get("charge"):
                mech.events.append({"type": "take_failed", "id": pickup_id, "reason": "busy"})
                return
            self.pickups.append({"id": self._id("pickup"), "kind": old["kind"], "x": mech.x, "y": mech.y, "z": mech.z})
        self.pickups.remove(item)
        weapon = new_weapon(item["kind"], mech.wstats)
        weapon["cooldown"] = max(self.cfg["loot"]["equip_s"], mech.wstats[item["kind"]].get("cooldown", 0))
        mech.weapons[slot] = weapon
        mech.events.append({"type": "took", "id": pickup_id, "kind": item["kind"], "slot": slot})

    def _step_drops(self):
        drop_cfg = self.cfg.get("drops")
        if not drop_cfg:
            return
        while self.drop_times and self.t >= self.drop_times[0]:
            self.drop_times.pop(0)
            zone = self.zone_now()
            x, y, z = self._random_ground(zone["next_center"], zone["next_radius"] * drop_cfg["spread"])
            self.drops.append({"id": self._id("drop"), "x": x, "y": y, "z": z, "land": self.t + drop_cfg["fall_s"], "progress": 0.0, "holder": None})
            self._sound_at("drop", x, y, None)
        keep = []
        for drop in self.drops:
            if self.t < drop["land"]:
                keep.append(drop)
                continue
            inside = [m for m in self.mechs if m.alive and math.hypot(m.x - drop["x"], m.y - drop["y"]) <= drop_cfg["radius"]]
            if len(inside) == 1:
                if drop["holder"] != inside[0].id:
                    drop["holder"], drop["progress"] = inside[0].id, 0.0
                drop["progress"] += DT
                if drop["progress"] >= drop_cfg["capture_s"]:
                    mech = inside[0]
                    self._gain_cubes(mech, int(drop_cfg["cubes"]))
                    mech.hp = mech.max_hp
                    mech.energy = mech.stats["energy"]
                    mech.events.append({"type": "drop", "id": drop["id"], "cubes": mech.cubes})
                    continue
            elif not inside:
                drop["holder"], drop["progress"] = None, 0.0
            keep.append(drop)
        self.drops = keep

    def _resolve_deaths(self):
        living = [m for m in self.mechs if m.alive]
        dead = [m for m in living if m.hp <= 0]
        if not dead:
            return
        place = len(living)
        for mech in dead:
            mech.alive = False
            mech.place = place
            mech.killer = getattr(mech, "last_attacker", None)
            self._sound_at("death", mech.x, mech.y, None)
            for _ in range(mech.cubes + int(self.cfg["cubes"]["death_drop_extra"])):
                self._drop_cube(mech.x, mech.y, mech.z)
            for weapon in mech.weapons:
                if weapon and weapon["kind"] != "gun":
                    self.pickups.append({"id": self._id("pickup"), "kind": weapon["kind"], "x": mech.x, "y": mech.y, "z": mech.z})
            if mech.team == SQUAD and mech.killer == self.player_slot:
                self.player_kills += 1
                self.player.events.append({"type": "kill", "target": mech.id})

    # ------------------------------------------------------------------ perception
    def _can_see(self, viewer, target):
        dx, dy = target.x - viewer.x, target.y - viewer.y
        distance = math.hypot(dx, dy)
        vision = self.cfg["vision"]
        near = vision["near_range"] + viewer.stats.get("near_range_bonus", 0)
        far = vision["range"] + viewer.stats.get("vision_bonus", 0)
        if distance > far:
            return False
        if distance > near and abs(wrap(math.atan2(dy, dx) - viewer.yaw)) > math.radians(vision["fov_deg"] / 2):
            return False
        return self.grid.los(viewer.x, viewer.y, viewer.z + 8, target.x, target.y, target.z + 4.5)

    def _item_visible(self, viewer, x, y, z):
        dx, dy = x - viewer.x, y - viewer.y
        distance = math.hypot(dx, dy)
        vision = self.cfg["vision"]
        if distance > vision["range"]:
            return False
        if distance > vision["near_range"] and abs(wrap(math.atan2(dy, dx) - viewer.yaw)) > math.radians(vision["fov_deg"] / 2):
            return False
        return self.grid.los(viewer.x, viewer.y, viewer.z + 8, x, y, z + 1.0)

    def _sound(self, kind, source):
        self._sound_at(kind, source.x, source.y, source)

    def _sound_at(self, kind, x, y, source):
        player = self.player
        if not player.alive or source is player:
            return
        carry = self.cfg["sound"]["range"].get(kind, 0)
        distance = math.hypot(x - player.x, y - player.y)
        if distance > carry:
            return
        noise = math.radians(self.cfg["sound"]["bearing_noise_deg"]) * player.stats.get("hearing", 1)
        label = "near" if distance < self.cfg["sound"]["near"] else "mid" if distance < self.cfg["sound"]["mid"] else "far"
        player.sounds.append({"kind": kind, "bearing": round(wrap(math.atan2(y - player.y, x - player.x) + self.rng.gauss(0, noise)), 3), "dist": label, "age": 0.05})

    def _observe(self, mech):
        visible = []
        for other in self.mechs:
            if other is mech or not other.alive:
                continue
            if self._can_see(mech, other):
                mech.seen[other.id] = self.t
            if self.t - mech.seen.get(other.id, -99) <= self.cfg["vision"]["memory_s"]:
                visible.append({"id": other.id, "kind": "mech", "pos": [round(other.x, 2), round(other.y, 2), round(other.z, 2)],
                                "vel": [round(other.vx, 2), round(other.vy, 2), round(other.vz, 2)], "aim": [round(other.yaw, 3), round(other.pitch, 3)],
                                "hp": round(other.hp, 1), "max_hp": other.max_hp, "cubes": other.cubes,
                                "weapons": [w["kind"] for w in other.weapons if w], "charging": other.charging()})
        for npc in self.npcs:
            if npc["hp"] > 0 and self._item_visible(mech, npc["x"], npc["y"], npc["z"]):
                visible.append({"id": npc["id"], "kind": "npc", "pos": [round(npc["x"], 2), round(npc["y"], 2), round(npc["z"], 2)],
                                "vel": [0, 0, 0], "aim": [0, 0], "hp": npc["hp"], "max_hp": self.cfg["npc"]["hp"]})
        projectiles = [{"id": b["id"], "kind": b["kind"], "pos": [round(b["x"], 2), round(b["y"], 2), round(b["z"], 2)], "vel": [round(b["vx"], 2), round(b["vy"], 2), round(b["vz"], 2)]}
                       for b in self.bullets if b["team"] != mech.team and abs(b["x"] - mech.x) < 120 and abs(b["y"] - mech.y) < 120]
        projectiles += [{"id": s["id"], "kind": "mortar", "pos": [round(s["x"], 2), round(s["y"], 2), round(s["z"], 2)], "vel": [round(s["vx"], 2), round(s["vy"], 2), round(s["vz"], 2)]}
                        for s in self.shells if s["team"] != mech.team and abs(s["x"] - mech.x) < 200 and abs(s["y"] - mech.y) < 200]
        crates = [{"id": c["id"], "pos": [c["x"], c["y"], c["z"]], "hp": c["hp"]} for c in self.crates if c["hp"] > 0 and abs(c["x"] - mech.x) < 90 and abs(c["y"] - mech.y) < 90 and self._item_visible(mech, c["x"], c["y"], c["z"])]
        cubes = [{"id": c["id"], "pos": [c["x"], c["y"], c["z"]]} for c in self.cubes if abs(c["x"] - mech.x) < 110 and abs(c["y"] - mech.y) < 110 and self._item_visible(mech, c["x"], c["y"], c["z"])]
        pickups = [{"id": p["id"], "kind": p["kind"], "pos": [p["x"], p["y"], p["z"]]} for p in self.pickups if abs(p["x"] - mech.x) < 110 and abs(p["y"] - mech.y) < 110 and self._item_visible(mech, p["x"], p["y"], p["z"])]
        drop_cfg = self.cfg.get("drops", {})
        drops = [{"id": d["id"], "pos": [d["x"], d["y"], d["z"]], "radius": drop_cfg.get("radius", 8), "lands_in": round(max(d["land"] - self.t, 0.0), 2),
                  "capture_s": drop_cfg.get("capture_s", 4), "progress": round(d["progress"], 2), "holder": d["holder"]} for d in self.drops]
        weapons = []
        for weapon in mech.weapons:
            if weapon is None:
                weapons.append(None)
            else:
                weapons.append(dict(weapon, stats=mech.wstats[weapon["kind"]]))
        zone = self.zone_now()
        in_zone = math.hypot(mech.x - zone["center"][0], mech.y - zone["center"][1]) <= zone["radius"]
        obs = {"type": "obs", "tick": self.tick_no, "t": self.t, "alive": sum(1 for m in self.mechs if m.alive),
               "self": {"id": mech.id, "pos": [round(mech.x, 2), round(mech.y, 2), round(mech.z, 2)], "vel": [round(mech.vx, 2), round(mech.vy, 2), round(mech.vz, 2)],
                        "aim": [round(mech.yaw, 3), round(mech.pitch, 3)], "hp": round(mech.hp, 1), "max_hp": mech.max_hp, "energy": round(mech.energy, 1),
                        "max_energy": mech.stats["energy"], "grounded": mech.grounded, "cubes": mech.cubes, "upgrades": list(mech.upgrades), "in_zone": in_zone,
                        "weapons": weapons, "stats": mech.stats},
               "mechs": visible, "projectiles": projectiles, "crates": crates, "cubes": cubes, "pickups": pickups, "sounds": mech.sounds, "beams": mech.beams,
               "events": mech.events, "drops": drops, "zone": zone, "offer": mech.offer}
        mech.sounds, mech.events, mech.beams = [], [], []
        return obs

    # ------------------------------------------------------------------ house squad
    def _house_action(self, mech):
        tier = TIERS[mech.tier]
        player = self.player
        action = {"fire": [False, False], "aim": [mech.cmd_yaw, mech.cmd_pitch], "move": [0.0, 0.0], "boost": False}
        zone = self.zone_now()
        cx, cy = zone["next_center"]
        to_center = math.hypot(cx - mech.x, cy - mech.y)
        outside = to_center > zone["next_radius"] * 0.82
        hunting = mech.id in self.hunters and player.alive
        dx, dy = player.x - mech.x, player.y - mech.y
        distance = math.hypot(dx, dy)
        clear = hunting and distance <= 230 and self.grid.los(mech.x, mech.y, mech.z + 7, player.x, player.y, player.z + 4.5)
        if outside and (not hunting or distance > 70 or zone["next_radius"] < 80):
            self._house_go(mech, cx, cy, action, boost=True)
        elif hunting:
            if distance > tier["pref"] * 1.05 or not clear:
                self._house_go(mech, player.x, player.y, action, boost=False)
            else:
                if self.t >= mech.next_strafe:
                    mech.strafe = -mech.strafe if self.rng.random() < 0.7 else mech.strafe
                    mech.next_strafe = self.t + self.rng.uniform(0.6, 1.6)
                ux, uy = dx / max(distance, 1), dy / max(distance, 1)
                radial = -0.6 if distance < tier["pref"] * 0.6 else 0.0
                action["move"] = [(-uy * mech.strafe + ux * radial) * tier["strafe"], (ux * mech.strafe + uy * radial) * tier["strafe"]]
        elif to_center > 22:
            self._house_go(mech, cx + math.cos(mech.id) * 15, cy + math.sin(mech.id) * 15, action, boost=False)
        if not clear:
            mech.acquired = None
            for slot, weapon in enumerate(mech.weapons):
                if weapon and weapon["kind"] == "laser" and weapon["charge"] > 0:
                    action["fire"][slot] = False
            return action
        if mech.acquired is None:
            mech.acquired = self.t
        flight = distance / 252.0 * tier["lead"]
        aim_x, aim_y = player.x + player.vx * flight, player.y + player.vy * flight
        yaw = math.atan2(aim_y - mech.y, aim_x - mech.x) + self.rng.gauss(0, tier["sigma"])
        pitch = math.atan2(player.z + 4.5 - mech.z - 7, max(distance, 1.0)) + self.rng.gauss(0, tier["sigma"] * 0.5)
        action["aim"] = [yaw, pitch]
        if self.t - mech.acquired < tier["react"] or abs(wrap(mech.yaw - yaw)) > 0.12:
            return action
        for slot, weapon in enumerate(mech.weapons):
            if weapon is None:
                continue
            kind = weapon["kind"]
            if kind == "gun":
                action["fire"][slot] = True
            elif kind == "laser":
                stats = mech.wstats["laser"]
                if distance <= stats["range"]:
                    action["fire"][slot] = weapon["charge"] < stats["charge"] * 0.97
            elif kind == "lightning":
                action["fire"][slot] = distance <= mech.wstats["lightning"]["range"]
            elif kind == "mortar" and weapon["cooldown"] <= 0 and 40 <= distance <= 200:
                stats = mech.wstats["mortar"]
                v2 = stats["speed"] ** 2
                disc = v2 * v2 - stats["gravity"] * (stats["gravity"] * distance * distance + 2 * (player.z - mech.z - 9.5) * v2)
                if disc >= 0:
                    action["aim"] = [yaw, math.atan((v2 - math.sqrt(disc)) / (stats["gravity"] * distance))]
                    action["fire"] = [False, False]
                    action["fire"][slot] = abs(mech.pitch - action["aim"][1]) < 0.08
                    return action
        return action

    def _house_go(self, mech, gx, gy, action, boost):
        if self.t >= mech.next_route or not mech.route:
            mech.route = self.grid.path(mech.x, mech.y, gx, gy, max_expand=450)
            mech.next_route = self.t + 1.3
        while mech.route and math.hypot(mech.route[0][0] - mech.x, mech.route[0][1] - mech.y) < 5:
            mech.route.pop(0)
        px, py = mech.route[0] if mech.route else (gx, gy)
        dx, dy = px - mech.x, py - mech.y
        length = max(math.hypot(dx, dy), 1e-3)
        action["move"] = [dx / length, dy / length]
        action["boost"] = boost and mech.energy > 25
        if abs(mech.vx) + abs(mech.vy) < 1.0 and mech.grounded and self.tick_no % 20 == 0:
            action["jump"] = True
