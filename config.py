"""Every tunable strategy constant, in one place."""
from dataclasses import dataclass


@dataclass
class Params:
    # Health thresholds, as a fraction of max HP.
    retreat_hp: float = 0.42
    reengage_hp: float = 0.80
    finish_target_hp: float = 0.25
    loot_min_hp: float = 0.45
    npc_min_hp: float = 0.75

    # Fight spacing and evasion, in meters and seconds.
    preferred_range: float = 85.0
    range_band: float = 35.0
    radial_gain: float = 0.65
    close_range: float = 28.0
    strafe_min_s: float = 0.45
    strafe_max_s: float = 1.15
    dodge_horizon_s: float = 0.9
    dodge_radius: float = 5.0
    shell_danger_radius: float = 10.0
    outnumbered_count: int = 3

    # Weapons.
    aim_tolerance_scale: float = 1.15
    min_aim_tolerance: float = 0.012
    laser_min_range: float = 18.0
    laser_max_hp_risk: float = 0.20
    laser_release_frac: float = 0.97
    mortar_min_range: float = 34.0
    mortar_max_target_speed: float = 7.0
    lightning_range_margin: float = 0.93
    reload_when_idle_below: float = 0.7

    # Zone safety.
    zone_margin: float = 14.0
    zone_rush_buffer_s: float = 7.0
    zone_center_radius: float = 26.0
    zone_late_radius: float = 80.0

    # Looting and objectives.
    loot_radius: float = 70.0
    crate_radius: float = 42.0
    npc_radius: float = 42.0
    threat_clear_radius: float = 95.0
    drop_max_travel: float = 210.0
    drop_lead_s: float = 3.0
    drop_contest_radius: float = 34.0

    # Movement and cover.
    energy_reserve: float = 22.0
    path_refresh_s: float = 0.7
    waypoint_reach: float = 5.0
    stuck_speed: float = 1.5
    stuck_time_s: float = 0.9
    cover_refresh_s: float = 0.8
    cover_rings: tuple = (14.0, 24.0, 36.0, 50.0, 66.0)
    cover_angles: int = 12

    # Priority orders; earlier entries are preferred.
    upgrade_order: tuple = (
        "armor", "plating", "nanites", "gun_rate", "gun_damage", "laser_damage", "laser_charge",
        "lightning_efficiency", "gun_mag", "servos", "gyros", "gun_velocity", "lightning_range",
        "reactor", "jets", "capacitor", "sensors", "lightning_chain", "mortar_reload",
        "mortar_velocity", "mortar_splash", "laser_pierce",
    )
    weapon_rank: tuple = ("laser", "lightning", "mortar")
