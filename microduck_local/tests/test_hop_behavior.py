"""The hopscotch hops (behaviors/hop.py): hop forward from standing and land
on both feet, on the left foot only, or on the right foot only.

The landing is a latch kept by the recipe's state_fn, so most of these drive
that state machine by hand - foot contacts set, a flight timed, the landing
spot placed - and check what each term pays around it. The generic contract
(61-dim obs, zero commands, penalties <= 0) is test_behaviors.py's, which
covers every registered id, these included.
"""

import math

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.behaviors import BEHAVIORS, BehaviorEnv, match_behavior
from microduck_local.behaviors.hop import (
    HOP_CURRICULUM,
    LAND_WINDOW_S,
    LANDINGS,
    MIN_AIR_S,
    TARGET_FWD,
    _hop_state,
)

AIR = {"left": False, "right": False}
BOTH = {"left": True, "right": True}
LEFT = {"left": True, "right": False}
RIGHT = {"left": False, "right": True}
STANCE = {"both": BOTH, "left": LEFT, "right": RIGHT}


def _env(landing: str, **overrides) -> BehaviorEnv:
    env = BehaviorEnv(f"hop_{landing}", spawn_overrides=overrides or None,
                      obs_noise=False, domain_rand=False, action_delay=False,
                      random_yaw=False, seed=0)
    env.reset(seed=0)
    for _ in range(10):   # settle the drop-in so the pose is a real stance
        env.step(np.zeros(14, np.float32))
    env.reset(seed=0)
    return env


def _tick(env, contacts: dict, n: int = 1) -> None:
    """Advance the recipe's hop state n control steps with these contacts."""
    for _ in range(n):
        env.foot_contact_state = dict(contacts)
        env.behavior.state_fn(env)


def _place_forward(env, metres: float) -> None:
    """Move the start anchor so the trunk reads `metres` forward of it."""
    yaw = env.home_yaw
    x, y = float(env._trunk_xpos[0]), float(env._trunk_xpos[1])
    env.home_xy = (x - metres * math.cos(yaw), y - metres * math.sin(yaw))


def _flight_steps(seconds: float) -> int:
    return int(math.ceil(seconds / C.CTRL_DT)) + 1


def _hop_and_land(env, touchdown: dict, fwd: float = TARGET_FWD) -> None:
    _tick(env, BOTH, 6)
    _tick(env, AIR, _flight_steps(MIN_AIR_S))
    _place_forward(env, fwd)
    _tick(env, touchdown)


def _pay(env, key: str) -> float:
    term = next(t for t in env.behavior.terms if t.key == key)
    return term.fn(env)


def test_three_recipes_one_per_landing():
    ids = {f"hop_{landing}" for landing in LANDINGS}
    assert ids <= set(BEHAVIORS)
    for bid in ids:
        b = BEHAVIORS[bid]
        assert b.scene == "walk" and b.terminate_on_fall
        assert b.state_fn is not None and b.reset_fn is not None
        assert b.report_fn is not None and b.caption_fn is not None
    # Sagittal two-foot hop keeps the mirror prior; the one-foot landings
    # name a side and must not be pulled toward their twin.
    assert BEHAVIORS["hop_both"].symmetric
    assert not BEHAVIORS["hop_left"].symmetric
    assert not BEHAVIORS["hop_right"].symmetric


def test_a_short_flight_is_a_shuffle_not_a_landing():
    env = _env("both")
    _tick(env, BOTH, 6)
    _tick(env, AIR, 1)                       # 20 ms: under the 50 ms floor
    _tick(env, BOTH)
    s = _hop_state(env)
    assert s["phase"] == "ground" and s["takeoffs"] == 1
    assert _pay(env, "land_it") == 0.0


@pytest.mark.parametrize("landing", LANDINGS)
def test_a_clean_landing_pays_the_salary_only_in_the_named_stance(landing):
    env = _env(landing)
    _hop_and_land(env, STANCE[landing])
    s = _hop_state(env)
    assert s["phase"] == "landed" and s["clean"]
    assert s["land_fwd"] == pytest.approx(TARGET_FWD, abs=1e-9)
    assert _pay(env, "land_it") > 0.5        # full distance x upright standing pose
    for other, contacts in STANCE.items():
        if other != landing:
            env.foot_contact_state = dict(contacts)
            assert _pay(env, "land_it") == 0.0, other


@pytest.mark.parametrize("landing,wrong", [("left", RIGHT), ("left", BOTH),
                                           ("right", LEFT), ("right", BOTH)])
def test_one_foot_landing_on_the_wrong_feet_is_latched_messy(landing, wrong):
    env = _env(landing)
    _hop_and_land(env, wrong)
    s = _hop_state(env)
    assert s["phase"] == "landed" and not s["clean"]
    # Shuffling onto the right stance afterwards earns nothing: the landing
    # was the test.
    env.foot_contact_state = dict(STANCE[landing])
    assert _pay(env, "land_it") == 0.0


def test_two_foot_landing_gives_the_second_foot_a_short_window():
    env = _env("both")
    _hop_and_land(env, LEFT)
    assert _hop_state(env)["phase"] == "pending"
    _tick(env, BOTH)
    assert _hop_state(env)["phase"] == "landed" and _hop_state(env)["clean"]

    late = _env("both")
    _hop_and_land(late, LEFT)
    _tick(late, LEFT, _flight_steps(LAND_WINDOW_S))
    _tick(late, BOTH)
    assert _hop_state(late)["phase"] == "landed" and not _hop_state(late)["clean"]


def test_distance_scales_the_salary_and_a_hop_in_place_still_earns_the_floor():
    near, far = _env("both"), _env("both")
    _hop_and_land(near, BOTH, fwd=0.0)
    _hop_and_land(far, BOTH, fwd=TARGET_FWD)
    assert 0.0 < _pay(near, "land_it") < _pay(far, "land_it")
    backwards = _env("both")
    _hop_and_land(backwards, BOTH, fwd=-TARGET_FWD)
    assert _pay(backwards, "land_it") == pytest.approx(_pay(near, "land_it"))


@pytest.mark.parametrize("key", ["get_air", "launch", "fly_forward"])
def test_the_hop_pay_stops_at_the_landing(key):
    """No bunny-hop farm: once it has landed, the take-off and flight terms
    pay nothing however high or fast it goes again."""
    env = _env("both")
    _hop_and_land(env, BOTH)
    for contacts in (AIR, BOTH, LEFT):
        env.foot_contact_state = dict(contacts)
        assert _pay(env, key) == 0.0


def test_get_air_needs_both_feet_off():
    env = _env("both")
    for contacts in (BOTH, LEFT, RIGHT):
        env.foot_contact_state = dict(contacts)
        assert _pay(env, "get_air") == 0.0
    assert _pay(env, "fly_forward") == 0.0


def test_one_foot_recipes_are_asymmetric_in_their_reward_not_just_their_flag():
    """The same body after the same flight, only the landing foot renamed:
    hop_left and hop_right score it in opposite directions, so the mirror
    loss really would fight them."""
    scores = {}
    for landing in ("left", "right"):
        for down, contacts in (("left", LEFT), ("right", RIGHT)):
            env = _env(landing)
            _hop_and_land(env, contacts)
            scores[landing, down] = _pay(env, "land_it")
    assert scores["left", "left"] > 0.5 and scores["left", "right"] == 0.0
    assert scores["right", "right"] > 0.5 and scores["right", "left"] == 0.0


@pytest.mark.parametrize("landing", LANDINGS)
def test_positive_terms_pay_at_most_their_weight_per_step(landing):
    """No jackpots: every positive hop term is in [0, 1] per step, under
    random actions and under a hand-driven hop."""
    env = BehaviorEnv(f"hop_{landing}", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=0)
    env.reset(seed=0)
    positives = [t for t in env.behavior.terms if not t.is_penalty]
    rng = np.random.default_rng(4)
    for _ in range(60):
        _, _, terminated, truncated, _ = env.step(rng.uniform(-0.5, 0.5, 14).astype(np.float32))
        for t in positives:
            assert 0.0 <= t.fn(env) <= 1.0 + 1e-9, t.key
        if terminated or truncated:
            env.reset(seed=5)
    env = _env(landing)
    _hop_and_land(env, STANCE[landing], fwd=10 * TARGET_FWD)
    for t in positives:
        assert 0.0 <= t.fn(env) <= 1.0 + 1e-9, t.key


def test_reset_clears_the_landing():
    env = _env("left")
    _hop_and_land(env, LEFT)
    assert _hop_state(env)["phase"] == "landed"
    env.reset(seed=1)
    s = _hop_state(env)
    assert s["phase"] == "ground" and s["takeoffs"] == 0 and not s["clean"]


def test_stage_knobs_set_the_strictness_per_instance():
    first = HOP_CURRICULUM[0].env
    env = _env("both", MICRODUCK_HOP_MIN_AIR_S=first["MICRODUCK_HOP_MIN_AIR_S"],
               MICRODUCK_HOP_TARGET_FWD=first["MICRODUCK_HOP_TARGET_FWD"])
    s = _hop_state(env)
    assert s["min_air"] == pytest.approx(float(first["MICRODUCK_HOP_MIN_AIR_S"]))
    assert s["target_fwd"] == pytest.approx(float(first["MICRODUCK_HOP_TARGET_FWD"]))
    # A 40 ms flight lands on the opening stage and is a shuffle on the last.
    _tick(env, BOTH, 6)
    _tick(env, AIR, 2)
    _tick(env, BOTH)
    assert _hop_state(env)["phase"] == "landed"
    strict = _env("both")
    _tick(strict, BOTH, 6)
    _tick(strict, AIR, 2)
    _tick(strict, BOTH)
    assert _hop_state(strict)["phase"] == "ground"


def test_the_ladder_moves_physics_and_strictness_never_the_pay():
    allowed = {"MICRODUCK_ACTUATOR", "MICRODUCK_BAM_CURRENT_SCALE",
               "MICRODUCK_HOP_MIN_AIR_S", "MICRODUCK_HOP_TARGET_FWD"}
    for landing in LANDINGS:
        b = BEHAVIORS[f"hop_{landing}"]
        assert b.curriculum == HOP_CURRICULUM
        assert b.default_steps == sum(st.steps for st in HOP_CURRICULUM)
    for st in HOP_CURRICULUM:
        assert set(st.env) <= allowed, st.label
    assert HOP_CURRICULUM[0].env["MICRODUCK_ACTUATOR"] == "xml"
    last = HOP_CURRICULUM[-1].env
    assert last["MICRODUCK_ACTUATOR"] == "bam"
    assert float(last["MICRODUCK_BAM_CURRENT_SCALE"]) == 1.0   # ships on honest servos
    assert float(last["MICRODUCK_HOP_MIN_AIR_S"]) == MIN_AIR_S
    assert float(last["MICRODUCK_HOP_TARGET_FWD"]) == TARGET_FWD
    airs = [float(st.env["MICRODUCK_HOP_MIN_AIR_S"]) for st in HOP_CURRICULUM]
    fwds = [float(st.env["MICRODUCK_HOP_TARGET_FWD"]) for st in HOP_CURRICULUM]
    assert airs == sorted(airs) and fwds == sorted(fwds)


def test_the_report_says_whether_it_landed_clean():
    env = _env("right")
    lines = env.behavior.report_fn(env)
    assert any(line == "landed: no" for line in lines)
    _hop_and_land(env, RIGHT)
    lines = env.behavior.report_fn(env)
    assert any(line.startswith("landed: yes (clean") for line in lines)
    assert "CLEAN" in env.behavior.caption_fn(env)


@pytest.mark.parametrize("text,bid", [
    ("hop", "hop_both"),
    ("let's play hopscotch", "hop_both"),
    ("hop forward and land on both feet", "hop_both"),
    ("do a two-foot hop", "hop_both"),
    ("hop and land on the left foot", "hop_left"),
    ("hop onto the right foot", "hop_right"),
    ("hop on 1 foot", "hop_left"),
    ("hop_right", "hop_right"),
])
def test_matcher_finds_the_hops(text, bid):
    assert match_behavior(text).id == bid


def test_the_hops_do_not_steal_the_neighbours_words():
    assert match_behavior("stand on one leg").id == "one_leg"
    assert match_behavior("balance on one foot").id == "one_leg"
    assert match_behavior("stand on both feet").id == "stand"
    assert match_behavior("do a jump backflip").id == "airflip"


def test_the_spawn_settle_is_not_a_hop():
    """Regression (measured 2026-09-28): the drop-in spawn settles through up
    to two airborne control steps, which cleared the opening stage's 30 ms
    flight floor - 36 of 64 zero-action episodes latched a 'landing' and
    collected the in-place salary without hopping. A take-off now only
    counts after the duck has stood on the floor for SETTLE_S."""
    free = 0
    for actuator in ("xml", "bam"):
        for seed in range(16):
            env = BehaviorEnv("hop_both", spawn_overrides={"MICRODUCK_HOP_MIN_AIR_S": "0.03"},
                              actuator_force=actuator, seed=seed)
            env.reset(seed=seed)
            for _ in range(40):
                env.step(np.zeros(14, np.float32))
            free += _hop_state(env)["phase"] == "landed"
    assert free == 0, f"{free}/32 zero-action episodes latched a landing"


def test_a_take_off_needs_the_settle_first():
    env = _env("both")
    _tick(env, BOTH, 2)                      # 40 ms on the floor: still settling
    _tick(env, AIR, 4)
    assert _hop_state(env)["phase"] == "ground" and _hop_state(env)["takeoffs"] == 0
    _tick(env, BOTH, 6)                      # 120 ms: settled
    _tick(env, AIR, 1)
    assert _hop_state(env)["phase"] == "air"


def test_the_salary_is_paid_facing_down_the_hop_line():
    """Regression (measured 2026-09-28): hop_left / hop_right learned to pivot
    on the stance foot after landing, ending a median -119 / +98 deg off the
    hop line, because the capped face_home penalty was cheap next to the
    salary. land_it now carries a heading factor."""
    env = _env("left")
    _hop_and_land(env, LEFT)
    straight = _pay(env, "land_it")
    assert straight > 0.5
    for deg, most in ((30.0, 0.9), (90.0, 0.05), (180.0, 0.01)):
        env.home_yaw = _trunk_yaw_of(env) - math.radians(deg)
        assert _pay(env, "land_it") < most * straight, deg
    env.home_yaw = _trunk_yaw_of(env) - math.radians(-90.0)
    assert _pay(env, "land_it") < 0.05 * straight                  # either direction


def _trunk_yaw_of(env) -> float:
    from microduck_local.behaviors.core import _trunk_yaw
    return _trunk_yaw(env)
