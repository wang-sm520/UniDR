"""Bounded source allocation from independent, normalized probe measurements."""

from __future__ import annotations

import hashlib
import json
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "ema_alpha": 0.2,
    "deadband": 0.02,
    "max_change": 0.02,
    "min_ratio": 0.10,
    "max_ratio": 0.70,
    "gain": 0.10,
    "weight_adaptation": False,
    "stable_error": 0.20,
    "stable_survival": 0.95,
    "stable_return": 0.80,
    "stable_windows": 3,
    "failure_weight_min": 0.05,
    "weight_step": 0.01,
}


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _project(target: FloatArray, low: FloatArray, high: FloatArray) -> FloatArray:
    """Euclidean projection onto the bounded simplex, including frozen entries."""
    _require(low.sum() <= 1 + 1e-12 and high.sum() >= 1 - 1e-12, "infeasible bounds")
    left, right = float(np.min(target - high)), float(np.max(target - low))
    for _ in range(80):
        middle = (left + right) / 2
        if np.clip(target - middle, low, high).sum() > 1:
            left = middle
        else:
            right = middle
    return cast(FloatArray, np.clip(target - (left + right) / 2, low, high))


def _integers(target: FloatArray, low: FloatArray, high: FloatArray, total: int):
    """Exact separable convex integer allocation with deterministic tie-breaking."""
    lower = np.ceil(low * total - 1e-10).astype(np.int64)
    upper = np.floor(high * total + 1e-10).astype(np.int64)
    _require(
        np.all(lower <= upper) and lower.sum() <= total <= upper.sum(),
        "no feasible integer source quota",
    )
    result = lower.copy()
    for _ in range(total - int(result.sum())):
        marginal = 2 * (result - total * target) + 1
        result[int(np.argmin(np.where(result < upper, marginal, np.inf)))] += 1
    return result


class SourceSchedule:
    """Keep pool capacity fixed; allocate whole-pool contiguous control steps.

    ``ratios`` always reports achieved integer allocation. Continuous targets are
    logged separately. An invalid probe leaves every scheduling state unchanged.
    The caller owns metric normalization, the probe interval, and persistence at
    completed learner iteration boundaries.
    """

    def __init__(self, sources: tuple[str, ...], config: dict, total_steps: int = 96):
        _require(
            len(sources) >= 2
            and len(set(sources)) == len(sources)
            and all(isinstance(s, str) and s for s in sources),
            "source names must be unique nonempty strings",
        )
        _require(type(total_steps) is int and total_steps > 0, "invalid total_steps")
        self.sources, self.total_steps = tuple(sources), total_steps
        self.config = DEFAULTS | config
        for key in ("enabled", "weight_adaptation"):
            _require(type(self.config[key]) is bool, f"{key} must be boolean")
        for key in DEFAULTS.keys() - {"enabled", "weight_adaptation"}:
            value = self.config[key]
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(value)
                and 0 <= value,
                f"invalid {key}",
            )
        c = self.config
        _require(0 < c["ema_alpha"] <= 1, "invalid ema_alpha")
        _require(
            0 < c["min_ratio"] <= 1 / len(sources) <= c["max_ratio"] <= 1,
            "uniform allocation outside bounds",
        )
        _require(c["max_change"] <= 1 and c["deadband"] <= 1, "invalid step/deadband")
        _require(
            type(c["stable_windows"]) is int and c["stable_windows"] > 0, "invalid stable_windows"
        )
        _require(
            all(c[key] <= 1 for key in ("stable_error", "stable_survival", "stable_return")),
            "invalid stability threshold",
        )
        _require(0 < c["failure_weight_min"] <= 0.25, "invalid failure_weight_min")
        self.fingerprint = hashlib.sha256(
            json.dumps(
                {"config": c, "sources": sources, "total_steps": total_steps},
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        self.targets = np.full(len(sources), 1 / len(sources))
        self._quotas = _integers(self.targets, *self._bounds(), total_steps)
        self.targets = self.ratios.copy()
        self.ema: FloatArray | None = None
        self.weights = np.array([0.50, 0.25, 0.25])
        self.valid_probes = self.stable_count = 0

    @property
    def ratios(self) -> FloatArray:
        return cast(FloatArray, self._quotas / self.total_steps)

    def quotas(self) -> dict[str, int]:
        return dict(zip(self.sources, (int(q) for q in self._quotas)))

    def _bounds(self) -> tuple[FloatArray, FloatArray]:
        return (
            np.full(len(self.sources), self.config["min_ratio"]),
            np.full(len(self.sources), self.config["max_ratio"]),
        )

    def _report(self, status: str, difficulty: FloatArray | None = None) -> dict:
        return {
            "status": status,
            "valid": status not in {"invalid", "disabled"},
            "ratios": dict(zip(self.sources, self.ratios.tolist())),
            "targets": dict(zip(self.sources, self.targets.tolist())),
            "quotas": self.quotas(),
            "ema": None
            if self.ema is None
            else {
                s: dict(zip(("error", "survival", "return"), row.tolist()))
                for s, row in zip(self.sources, self.ema)
            },
            "weights": dict(zip(("error", "failure", "return"), self.weights.tolist())),
            "difficulty": None
            if difficulty is None
            else dict(zip(self.sources, difficulty.tolist())),
            "baseline": None if difficulty is None else float(difficulty.mean()),
            "valid_probes": self.valid_probes,
            "stable_count": self.stable_count,
        }

    def update(self, metrics: dict[str, dict[str, float]]) -> dict:
        if not self.config["enabled"]:
            return self._report("disabled")
        try:
            _require(set(metrics) == set(self.sources), "missing or unexpected source")
            observed = np.array(
                [[metrics[s][k] for k in ("error", "survival", "return")] for s in self.sources],
                dtype=np.float64,
            )
            _require(
                observed.shape == (len(self.sources), 3)
                and np.isfinite(observed).all()
                and ((0 <= observed) & (observed <= 1)).all(),
                "invalid probe values",
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return self._report("invalid")
        c = self.config
        ema = (
            observed
            if self.ema is None
            else ((1 - c["ema_alpha"]) * self.ema + c["ema_alpha"] * observed)
        )
        self.ema = ema
        self.valid_probes += 1
        stable = bool(
            np.all(ema[:, 0] <= c["stable_error"])
            and np.all(ema[:, 1] >= c["stable_survival"])
            and np.all(ema[:, 2] >= c["stable_return"])
        )
        self.stable_count = self.stable_count + 1 if stable else 0
        if c["weight_adaptation"] and self.stable_count >= c["stable_windows"]:
            transfer = min(c["weight_step"], self.weights[1] - c["failure_weight_min"])
            self.weights += np.array([transfer, -transfer, 0])
        difficulty = (ema * np.array([1, -1, -1]) + np.array([0, 1, 1])) @ self.weights
        difference = difficulty - difficulty.mean()  # Always equal reference weights.
        harder, easier = difference > c["deadband"], difference < -c["deadband"]
        old = self.ratios
        if not (harder.any() and easier.any()):
            self.targets = old.copy()
            return self._report("noise_hold", difficulty)
        low, high = self._bounds()
        low, high = np.maximum(low, old - c["max_change"]), np.minimum(high, old + c["max_change"])
        low = np.maximum(low, self.targets - c["max_change"])
        high = np.minimum(high, self.targets + c["max_change"])
        # Only sources clearly harder/easier than the baseline may gain/lose.
        low = np.where(easier, low, old)
        high = np.where(harder, high, old)
        self.targets = _project(old + c["gain"] * difference, low, high)
        quotas = _integers(self.targets, low, high, self.total_steps)
        status = "updated" if np.any(quotas != self._quotas) else "quantization_hold"
        self._quotas = quotas
        return self._report(status, difficulty)

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "fingerprint": self.fingerprint,
            "sources": list(self.sources),
            "total_steps": self.total_steps,
            "quotas": self._quotas.tolist(),
            "ratios": self.ratios.tolist(),
            "targets": self.targets.tolist(),
            "weights": self.weights.tolist(),
            "ema": None if self.ema is None else self.ema.tolist(),
            "valid_probes": self.valid_probes,
            "stable_count": self.stable_count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Validate everything before replacing state; resume config must match."""
        _require(set(state) == set(self.state_dict()), "malformed scheduler state")
        _require(
            state["version"] == 1
            and state["fingerprint"] == self.fingerprint
            and state["sources"] == list(self.sources)
            and state["total_steps"] == self.total_steps,
            "scheduler configuration mismatch",
        )
        quota_list = state["quotas"]
        _require(
            isinstance(quota_list, list) and all(type(q) is int for q in quota_list),
            "noninteger quotas",
        )
        quotas = np.array(quota_list, dtype=np.int64)
        ratios, targets, weights = (
            np.array(state[k], dtype=np.float64) for k in ("ratios", "targets", "weights")
        )
        low, high = self._bounds()
        _require(
            quotas.shape == (len(self.sources),) and quotas.sum() == self.total_steps,
            "invalid quota budget",
        )
        for values in (ratios, targets):
            _require(
                values.shape == low.shape
                and np.isfinite(values).all()
                and np.all(values >= low - 1e-12)
                and np.all(values <= high + 1e-12)
                and abs(values.sum() - 1) < 1e-12,
                "invalid ratios",
            )
        _require(np.array_equal(ratios, quotas / self.total_steps), "quota/ratio mismatch")
        _require(
            np.all(np.abs(targets - ratios) <= self.config["max_change"] + 1e-12),
            "target/actual slew mismatch",
        )
        _require(
            weights.shape == (3,)
            and np.isfinite(weights).all()
            and abs(weights.sum() - 1) < 1e-12
            and weights[2] == 0.25
            and self.config["failure_weight_min"] - 1e-12 <= weights[1] <= 0.25,
            "invalid metric weights",
        )
        probes, stable = state["valid_probes"], state["stable_count"]
        _require(
            type(probes) is int and type(stable) is int and 0 <= stable <= probes,
            "invalid probe counters",
        )
        ema = None if state["ema"] is None else np.array(state["ema"], dtype=np.float64)
        _require((ema is None) == (probes == 0), "EMA/counter mismatch")
        if ema is not None:
            _require(
                ema.shape == (len(self.sources), 3)
                and np.isfinite(ema).all()
                and ((0 <= ema) & (ema <= 1)).all(),
                "invalid restored EMA",
            )
        self._quotas, self.targets, self.weights, self.ema = quotas, targets, weights, ema
        self.valid_probes, self.stable_count = probes, stable
