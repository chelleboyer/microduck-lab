"""Hopscotch hops: hop forward from standing and land on BOTH feet, or on
the LEFT foot only, or on the RIGHT foot only (2026-09-28).

Three recipes, one per landing, the way the kicks are one per foot: each
exports as a zero-command 61-obs policy the robot runs as an episodic skill
(`robotctl policy add hop_left <repo>` then `robotctl robot do hop_left`),
and a hopscotch course is a brain sequencing them.

What is already known about hopping this robot, and what the pay is built on:

* It CAN leave the ground, barely. TorchRL's MicroDuck `jump_task` (0.55 N m
  servos clipped at 0.96, 0.74 kg) learned 2 Hz hops of ~2 cm, airborne 15%
  of the time, and only once a LAUNCH term paid upward trunk speed while the
  feet were still planted - the airborne-gated pay alone cannot see the
  take-off, and a rhythm-dominant recipe bobbed without ever leaving the
  floor. So `launch` is here, below the landing pay so a bob never outearns
  a hop.
* Whether a random policy's rollouts ever contain a flight is the question
  to ask before touching a weight (AGENTS.md). The ladder therefore opens on
  the strong xml servos, where a flight is samplable, and steps down to
  honest BAM - physics and strictness move, the pay never does.

The hop is a small state machine kept by `_hop_update` (the recipe's
`state_fn`), because "landed" only means something after a real flight:

  ground --both feet off--> air --touchdown after >= min flight--> landed
                              \\--touchdown too soon: a shuffle, back to ground

On touchdown the landing is latched: how far forward of the start it came
down, and whether it came down CLEAN - on exactly the feet the recipe names
(for `hop_both`, the second foot has LAND_WINDOW_S to follow the first).
After that the only big pay is `land_it`: a per-step salary for holding the
named stance, upright, scaled by the latched distance. A salary, not a
bonus - landing and falling over is worth almost nothing, which is the
point of a hopscotch landing.
"""

import math

import numpy as np

from .. import contract as C
from .core import (
    _BASE_REGULARIZERS,
    Behavior,
    CurriculumStage,
    RewardTerm,
    _base_vel,
    _face_home_pen,
    _foot_z,
    _limit_parking_pen,
    _register,
    _soft_landing_pen,
    _spawn_knob,
    _upright,
    _upright_term,
)

LANDINGS = ("both", "left", "right")

FOOT_Z_PLANTED = 0.0086   # m: a planted foot's collision-geom centre (locomotion.py's clearance note)
HOP_CLEARANCE = 0.02      # m of foot lift that pays get_air in full (TorchRL's hops were ~2 cm)
LAUNCH_SPEED = 0.4        # m/s of upward trunk speed, still on the floor, that pays launch in full
FLY_SPEED = 0.3           # m/s forward while airborne that pays fly_forward in full
LAND_WINDOW_S = 0.08      # s the second foot has to follow the first for a two-foot landing
LAND_FLOOR = 0.25         # share of land_it a clean landing earns with zero forward distance
LINE_SCALE = 6.0          # off_line saturates one unit at ~41 cm off the line (stay_home's scale)

# Strictness knobs (the ladder moves these, never the pay).
MIN_AIR_S = 0.05          # s of flight before a touchdown counts as a landing
TARGET_FWD = 0.04         # m forward of the start at touchdown that pays land_it in full


def _hop_knob(env, key: str, default: float) -> float:
    raw = _spawn_knob(env, key)
    return float(raw) if raw else default


def _fwd_lat(env) -> tuple[float, float]:
    """Trunk displacement since the start, along and across the starting heading (m)."""
    home = getattr(env, "home_xy", None)
    if home is None:
        return 0.0, 0.0
    yaw = getattr(env, "home_yaw", 0.0) or 0.0
    dx = float(env._trunk_xpos[0]) - home[0]
    dy = float(env._trunk_xpos[1]) - home[1]
    return (dx * math.cos(yaw) + dy * math.sin(yaw),
            -dx * math.sin(yaw) + dy * math.cos(yaw))


def _stance_ok(landing: str, contacts: dict) -> bool:
    """Is the duck standing on exactly the feet this landing names?"""
    left, right = bool(contacts["left"]), bool(contacts["right"])
    if landing == "both":
        return left and right
    if landing == "left":
        return left and not right
    return right and not left


def _hop_state(env) -> dict:
    s = getattr(env, "_hop", None)
    if s is None or s.get("episode") != getattr(env, "episode_id", None):
        s = _hop_fresh(env)
    return s


def _hop_fresh(env) -> dict:
    s = env._hop = {
        "episode": getattr(env, "episode_id", None),
        "phase": "ground",       # ground | air | pending | landed
        "air_s": 0.0,            # current flight time
        "flight_s": 0.0,         # the flight that ended in the landing
        "pending_s": 0.0,        # time since first touchdown (two-foot landings)
        "land_fwd": 0.0,         # m forward of the start at touchdown
        "clean": False,          # came down on exactly the named feet
        "takeoffs": 0,           # flights started (a shuffle counts; report only)
        "held": 0,               # steps in the named stance after landing (report only)
        "after": 0,              # steps after landing (report only)
        "min_air": _hop_knob(env, "MICRODUCK_HOP_MIN_AIR_S", MIN_AIR_S),
        "target_fwd": _hop_knob(env, "MICRODUCK_HOP_TARGET_FWD", TARGET_FWD),
    }
    return s


def _hop_update_for(landing: str):
    def _hop_update(env) -> None:
        s = _hop_state(env)
        c = env.foot_contact_state
        airborne = not (c["left"] or c["right"])
        if s["phase"] == "ground":
            if airborne:
                s["phase"], s["air_s"] = "air", C.CTRL_DT
                s["takeoffs"] += 1
        elif s["phase"] == "air":
            if airborne:
                s["air_s"] += C.CTRL_DT
            elif s["air_s"] < s["min_air"]:
                s["phase"], s["air_s"] = "ground", 0.0     # a shuffle, not a hop
            else:
                s["flight_s"] = s["air_s"]
                s["land_fwd"] = _fwd_lat(env)[0]
                if landing == "both" and not (c["left"] and c["right"]):
                    s["phase"], s["pending_s"] = "pending", 0.0
                else:
                    s["phase"], s["clean"] = "landed", _stance_ok(landing, c)
        elif s["phase"] == "pending":
            # Two-foot landing, first foot down: the second has a short window.
            s["pending_s"] += C.CTRL_DT
            if c["left"] and c["right"]:
                s["phase"], s["clean"] = "landed", True
            elif airborne or s["pending_s"] > LAND_WINDOW_S:
                s["phase"], s["clean"] = "landed", False
        if s["phase"] == "landed":
            s["after"] += 1
            if _stance_ok(landing, c):
                s["held"] += 1

    _hop_update.__name__ = f"_hop_update_{landing}"
    return _hop_update


def _hop_reset(env) -> None:
    """Fresh hop state at every reset, read once from this stage's knobs."""
    _hop_fresh(env)


def _pre_landing(env) -> bool:
    return _hop_state(env)["phase"] in ("ground", "air")


def _get_air(env) -> float:
    """Both feet off the floor before the landing, paid by clearance and uprightness (0..1)."""
    if not _pre_landing(env):
        return 0.0
    c = env.foot_contact_state
    if c["left"] or c["right"]:
        return 0.0
    lift = min(_foot_z(env, "left"), _foot_z(env, "right")) - FOOT_Z_PLANTED
    return float(np.clip(lift / HOP_CLEARANCE, 0.0, 1.0)) * _upright(env)


def _launch(env) -> float:
    """Upward trunk speed with a foot still on the floor, before the landing (0..1):
    the take-off, which the airborne pay cannot see (TorchRL's lesson)."""
    if not _pre_landing(env):
        return 0.0
    c = env.foot_contact_state
    if not (c["left"] or c["right"]):
        return 0.0
    vz = _base_vel(env)[2]
    return float(np.clip(vz / LAUNCH_SPEED, 0.0, 1.0)) * _upright(env)


def _fly_forward(env) -> float:
    """Forward speed while airborne, before the landing (0..1)."""
    if not _pre_landing(env):
        return 0.0
    c = env.foot_contact_state
    if c["left"] or c["right"]:
        return 0.0
    return float(np.clip(_base_vel(env)[0] / FLY_SPEED, 0.0, 1.0))


def _land_it_for(landing: str):
    def _land_it(env) -> float:
        """The salary: after a clean landing, every step on exactly the named
        feet, upright, scaled by how far forward it came down (0..1)."""
        s = _hop_state(env)
        if s["phase"] != "landed" or not s["clean"]:
            return 0.0
        if not _stance_ok(landing, env.foot_contact_state):
            return 0.0
        dist = float(np.clip(s["land_fwd"] / s["target_fwd"], 0.0, 1.0))
        return (LAND_FLOOR + (1.0 - LAND_FLOOR) * dist) * _upright(env)

    _land_it.__name__ = f"_land_it_{landing}"
    return _land_it


def _off_line_pen(env) -> float:
    """Sideways drift off the line the hop started on (<= 0, bounded)."""
    lat = _fwd_lat(env)[1]
    return -min(LINE_SCALE * lat * lat, 1.0)


def _hop_caption(env) -> str:
    s = _hop_state(env)
    c = env.foot_contact_state
    return (f"hop {s['phase']} air {s['air_s']:.2f}s "
            f"fwd {100.0 * _fwd_lat(env)[0]:+.1f}cm "
            f"feet {int(c['left'])}{int(c['right'])}"
            + (f" {'CLEAN' if s['clean'] else 'MESSY'}" if s["phase"] == "landed" else ""))


def _hop_report(env) -> list[str]:
    s = _hop_state(env)
    landed = s["phase"] == "landed"
    return [f"flights started: {s['takeoffs']}",
            f"landed: {'yes' if landed else 'no'}"
            + (f" ({'clean' if s['clean'] else 'wrong feet'}, "
               f"after {1000.0 * s['flight_s']:.0f} ms in the air)" if landed else ""),
            f"forward at touchdown: {100.0 * s['land_fwd']:+.1f} cm "
            f"(target {100.0 * s['target_fwd']:.1f} cm)" if landed else "forward at touchdown: -",
            f"held the landing stance: {s['held']}/{s['after']} steps" if landed else
            "held the landing stance: -"]


_FEET = {"both": "both feet", "left": "the left foot", "right": "the right foot"}
_ONLY = {"both": "both feet", "left": "ONLY the left foot", "right": "ONLY the right foot"}

_KEYWORDS = {
    # Registered first, so a bare "hop" / "hopscotch" / "jump forward" lands here.
    "both": ("hop", "hopscotch", "hop forward", "hop_both", "hop on both feet",
             "land on both feet", "two foot hop", "two-foot hop", "bunny hop",
             "jump forward"),
    "left": ("hop_left", "hop left", "left foot hop", "hop onto the left foot",
             "land on the left foot", "land on my left foot", "hop on one foot"),
    "right": ("hop_right", "hop right", "right foot hop", "hop onto the right foot",
              "land on the right foot", "land on my right foot"),
}

# THE LADDER. Identical terms in every stage (AGENTS.md: a stage may ladder
# physics, spawns and strictness, never the pay).
HOP_CURRICULUM = (
    CurriculumStage("finding the hop (strong servos)", 1_500_000,
                    {"MICRODUCK_ACTUATOR": "xml",
                     "MICRODUCK_HOP_MIN_AIR_S": "0.03",
                     "MICRODUCK_HOP_TARGET_FWD": "0.02"},
                    detail=("Phantom-strong servos, so a flight shows up in the rollouts "
                            "at all, and a short one counts: 30 ms off the floor and "
                            "2 cm forward pays the landing in full.")),
    CurriculumStage("on stepped-down servos", 1_500_000,
                    {"MICRODUCK_ACTUATOR": "bam",
                     "MICRODUCK_BAM_CURRENT_SCALE": "1.3",
                     "MICRODUCK_HOP_MIN_AIR_S": "0.04",
                     "MICRODUCK_HOP_TARGET_FWD": "0.03"},
                    detail=("The servos step down toward the real XL330s, and the hop "
                            "has to be a little longer and a little further.")),
    CurriculumStage("real servos, real hop", 2_000_000,
                    {"MICRODUCK_ACTUATOR": "bam",
                     "MICRODUCK_BAM_CURRENT_SCALE": "1.0",
                     "MICRODUCK_HOP_MIN_AIR_S": f"{MIN_AIR_S}",
                     "MICRODUCK_HOP_TARGET_FWD": f"{TARGET_FWD}"},
                    detail=("Honest XL330s: 50 ms in the air and 4 cm forward for the "
                            "full landing pay - the hop a real duck can do.")),
)

for _landing in LANDINGS:
    _register(Behavior(
        id=f"hop_{_landing}",
        emoji="🦆",
        title=f"Hop forward, land on {_FEET[_landing]}",
        description=(f"Hop forward from standing, hopscotch style, and land on "
                     f"{_ONLY[_landing]} - then hold it."),
        how_it_learns=(
            "Every attempt starts standing on both feet. It is paid for pushing up "
            "off the floor, for getting both feet off it, and for moving forward "
            "while it is up there. Once it comes down, the only thing that pays is "
            f"holding the landing on {_ONLY[_landing]}, upright - more the further "
            "forward it landed, and nothing at all if it came down on the wrong "
            "feet. It starts on extra-strong servos so a hop is possible at all, "
            "then the servos step down to the real ones."
        ),
        keywords=_KEYWORDS[_landing],
        terms=(
            RewardTerm("land_it", f"Big points every step it holds a clean landing on {_ONLY[_landing]}, "
                       "more the further forward it landed", 6.0, _land_it_for(_landing)),
            RewardTerm("get_air", "Points for both feet off the floor before landing, up to 2 cm",
                       3.0, _get_air),
            RewardTerm("launch", "Points for pushing the body up off the floor before the hop",
                       1.5, _launch),
            RewardTerm("fly_forward", "Points for moving forward while in the air",
                       2.0, _fly_forward),
            _upright_term(1.0),
            RewardTerm("soft_landings", "Penalty for slamming down hard",
                       0.75, _soft_landing_pen, is_penalty=True),
            RewardTerm("off_line", "Penalty for drifting sideways off the hop line",
                       1.0, _off_line_pen, is_penalty=True),
            RewardTerm("face_home", "Penalty for twisting away from the starting direction",
                       1.0, _face_home_pen, is_penalty=True),
            RewardTerm("no_limit_parking", "Penalty for cranking joints to their end stops",
                       1.0, _limit_parking_pen, is_penalty=True),
        ) + _BASE_REGULARIZERS,
        default_steps=sum(st.steps for st in HOP_CURRICULUM),
        success_metric=(f"clean landings on {_ONLY[_landing]}, held to the end of the "
                        "episode, and cm forward at touchdown - from standing, on BAM servos"),
        # The two-foot hop is sagittal; the one-foot landings name a side, so
        # the mirror loss would pull each toward its twin.
        symmetric=_landing == "both",
        episode_s=3.0,
        scene="walk",
        terminate_on_fall=True,
        state_fn=_hop_update_for(_landing),
        reset_fn=_hop_reset,
        caption_fn=_hop_caption,
        report_fn=_hop_report,
        curriculum=HOP_CURRICULUM,
    ))
