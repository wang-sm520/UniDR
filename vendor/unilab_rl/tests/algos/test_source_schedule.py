"""Scheduling invariants independent of simulator or policy implementation."""

from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from uni_rl.algos.source_schedule import SourceSchedule, _integers

SOURCES = ("isaacsim", "isaacgym", "genesis", "motrix")


def metrics(errors=(1.0, 0.0, 0.0, 0.0), survival=1.0, reward=1.0):
    return {
        s: {"error": e, "survival": survival, "return": reward} for s, e in zip(SOURCES, errors)
    }


def test_initial_and_direction_actual_step_bounds_and_fixed_baseline():
    schedule = SourceSchedule(SOURCES, {})
    assert schedule.quotas() == dict.fromkeys(SOURCES, 24)
    report = schedule.update(metrics())
    assert report["status"] == "updated"
    assert schedule.quotas()["isaacsim"] == 25
    assert sorted(schedule.quotas()[s] for s in SOURCES[1:]) == [23, 24, 24]
    assert report["baseline"] == 0.125
    assert report["targets"]["isaacsim"] == pytest.approx(0.27)
    for _ in range(100):
        previous = schedule.ratios.copy()
        report = schedule.update(metrics())
        assert sum(schedule.quotas().values()) == 96
        assert np.all(np.abs(schedule.ratios - previous) <= 0.02 + 1e-12)
        assert np.all(schedule.ratios >= 0.10)
        assert np.all(schedule.ratios <= 0.70)
        assert report["baseline"] == pytest.approx(0.125)
    assert schedule.quotas()["isaacsim"] == 66  # Other pools must each retain >= 10.


def test_ema_initialization_noise_and_fixed_mode():
    schedule = SourceSchedule(SOURCES, {})
    report = schedule.update(metrics((0.2, 0.21, 0.2, 0.21)))
    assert report["status"] == "noise_hold"
    assert schedule.quotas() == dict.fromkeys(SOURCES, 24)
    np.testing.assert_array_equal(schedule.ema[:, 0], [0.2, 0.21, 0.2, 0.21])
    schedule.update(metrics((0.3, 0.21, 0.2, 0.21)))
    assert schedule.ema[0, 0] == pytest.approx(0.22)
    fixed = SourceSchedule(SOURCES, {"enabled": False})
    before = fixed.state_dict()
    assert fixed.update(metrics())["status"] == "disabled"
    assert fixed.state_dict() == before


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1, None, "broken"])
def test_invalid_probe_holds_atomically(bad):
    schedule = SourceSchedule(SOURCES, {})
    schedule.update(metrics())
    before = schedule.state_dict()
    invalid = metrics()
    invalid["motrix"]["error"] = bad
    assert schedule.update(invalid)["status"] == "invalid"
    assert schedule.state_dict() == before
    invalid = metrics()
    del invalid["genesis"]
    assert schedule.update(invalid)["status"] == "invalid"
    assert schedule.state_dict() == before


def test_individual_noise_hold_and_integer_small_budget():
    schedule = SourceSchedule(SOURCES, {})
    before = schedule.ratios.copy()
    schedule.update(metrics((1.0, 0.0, 0.5, 0.5)))
    np.testing.assert_equal(schedule.ratios[2:], before[2:])
    small = SourceSchedule(SOURCES, {}, total_steps=8)
    assert small.update(metrics())["status"] == "quantization_hold"
    assert small.quotas() == dict.fromkeys(SOURCES, 2)
    with pytest.raises(ValueError, match="integer"):
        SourceSchedule(SOURCES, {}, total_steps=3)


def test_integer_solver_matches_bruteforce_objective():
    low, high = np.array([0.1] * 4), np.array([0.7] * 4)
    target = np.array([0.41, 0.32, 0.15, 0.12])
    allocation = _integers(target, low, high, 12)
    candidates = [np.array(q) for q in product(range(2, 9), repeat=4) if sum(q) == 12]
    best = min(float(((q - 12 * target) ** 2).sum()) for q in candidates)
    assert float(((allocation - 12 * target) ** 2).sum()) == pytest.approx(best)


def test_random_probe_sequence_conserves_budget_bounds_and_realized_step_limit():
    rng = np.random.default_rng(20260920)
    schedule = SourceSchedule(SOURCES, {"ema_alpha": 1.0})
    for _ in range(300):
        old = schedule.ratios.copy()
        old_targets = schedule.targets.copy()
        probe = {
            source: dict(zip(("error", "survival", "return"), rng.random(3))) for source in SOURCES
        }
        report = schedule.update(probe)
        assert sum(report["quotas"].values()) == 96
        assert np.all((schedule.ratios >= 0.1) & (schedule.ratios <= 0.7))
        assert np.max(np.abs(schedule.ratios - old)) <= 0.02 + 1e-12
        assert np.max(np.abs(schedule.targets - old_targets)) <= 0.02 + 1e-12
        for index, source in enumerate(SOURCES):
            difference = report["difficulty"][source] - report["baseline"]
            if abs(difference) <= 0.02:
                assert schedule.ratios[index] == old[index]
            elif difference > 0:
                assert schedule.ratios[index] >= old[index]
            else:
                assert schedule.ratios[index] <= old[index]


def test_reversal_limits_both_continuous_targets_and_integer_allocation():
    schedule = SourceSchedule(SOURCES, {"ema_alpha": 1.0})
    schedule.update(metrics())
    assert schedule.targets[0] == pytest.approx(0.27)
    assert schedule.ratios[0] == pytest.approx(25 / 96)
    old_targets, old_actual = schedule.targets.copy(), schedule.ratios.copy()
    schedule.update(metrics((0.0, 1.0, 1.0, 1.0)))
    assert np.max(np.abs(schedule.targets - old_targets)) <= 0.02 + 1e-12
    assert np.max(np.abs(schedule.ratios - old_actual)) <= 0.02 + 1e-12
    assert schedule.targets[0] == pytest.approx(0.25)
    assert schedule.ratios[0] < old_actual[0]


def test_metric_weight_policy_is_global_separate_and_has_positive_floor():
    schedule = SourceSchedule(SOURCES, {"weight_adaptation": True})
    good = metrics((0.1,) * 4)
    schedule.update(good)
    schedule.update(good)
    np.testing.assert_array_equal(schedule.weights, [0.5, 0.25, 0.25])
    schedule.update(good)
    np.testing.assert_allclose(schedule.weights, [0.51, 0.24, 0.25])
    for _ in range(100):
        schedule.update(good)
    np.testing.assert_allclose(schedule.weights, [0.70, 0.05, 0.25])
    assert schedule.quotas() == dict.fromkeys(SOURCES, 24)
    schedule.update(metrics((1.0,) * 4, survival=0.0))
    assert schedule.stable_count == 0


def test_resume_is_exact_and_rejects_bad_state_atomically():
    schedule = SourceSchedule(SOURCES, {"weight_adaptation": True})
    for _ in range(4):
        schedule.update(metrics())
    restored = SourceSchedule(SOURCES, {"weight_adaptation": True})
    restored.load_state_dict(schedule.state_dict())
    assert restored.update(metrics((0.0, 1.0, 0.0, 0.0))) == schedule.update(
        metrics((0.0, 1.0, 0.0, 0.0))
    )
    for key, value in (
        ("fingerprint", "wrong"),
        ("quotas", [24] * 4),
        ("ema", [[float("nan")] * 3] * 4),
        ("stable_count", -1),
        ("weights", [0.5, 0.0, 0.5]),
        ("targets", [0.1] * 4),
    ):
        before = restored.state_dict()
        broken = deepcopy(before)
        broken[key] = value
        with pytest.raises(ValueError):
            restored.load_state_dict(broken)
        assert restored.state_dict() == before
    with pytest.raises(ValueError, match="configuration"):
        SourceSchedule(SOURCES, {}).load_state_dict(schedule.state_dict())


@pytest.mark.parametrize(
    "config",
    [
        {"ema_alpha": 0},
        {"deadband": float("nan")},
        {"min_ratio": 0.3},
        {"max_ratio": 0.2},
        {"failure_weight_min": 0},
        {"stable_windows": 1.5},
        {"enabled": "true"},
        {"gain": -1},
        {"stable_survival": 2},
    ],
)
def test_invalid_config(config):
    with pytest.raises(ValueError):
        SourceSchedule(SOURCES, config)
