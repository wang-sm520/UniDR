"""Complete adaptive windows, independent probes and checkpoint-boundary recovery."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from examples.unidr_fake import FakeSource
from test_synchronous_runner import make_runner

from uni_rl.ipc.synchronous_env import SourceSpec, SynchronousEnv

CONFIG = {
    "enabled": True,
    "probe": {
        "interval": 1,
        "horizon": 4,
        "seed": 1,
        "error_cap": 0.5,
        "return_min": 0.0,
        "return_max": 4.0,
    },
}


class ProbeSource(FakeSource):
    probe = False

    def reset_probe(self, seed):
        self.probe = True
        self.tick = 0
        return self.reset(np.arange(self.num_envs))

    def finish_probe(self):
        self.probe = False

    def step(self, actions):
        state = super().step(actions)
        if self.probe:
            state.info["probe"] = {
                "error": np.full(self.num_envs, 0.5 if self.source == 0 else 0.0)
            }
        return state


def factory(num_envs, cfg):
    return ProbeSource(cfg["source"], num_envs)


def build(path, enabled=True):
    cfg = deepcopy(CONFIG)
    cfg["enabled"] = enabled
    return make_runner(
        path,
        adaptive=cfg,
        source=SynchronousEnv(
            [SourceSpec(f"source{i}", factory, 2, {"source": i}) for i in range(4)]
        ),
    )


def test_real_ppo_probe_quota_change_and_boundary_resume(tmp_path):
    runner = build(tmp_path / "first")
    runner.learn(2)
    assert runner.total_transitions == 384 and runner.optimizer_steps == 40
    assert runner.normalizer_version == 49
    assert runner.last_metrics["epoch_samples"] == [192] * 5
    assert runner.latest_window.spec.quotas["source0"] == 50
    assert sum(runner.latest_window.spec.quotas.values()) == 192
    assert runner.last_metrics["probe"]["schedule"]["status"] == "updated"
    assert runner.schedule.valid_probes == 2
    assert all(int(m.obs_normalizer.count) == 384 for m in (runner.alg.actor, runner.alg.critic))
    saved = torch.load(tmp_path / "first/model_1.pt", weights_only=False)
    resumed = build(tmp_path / "resume")
    resumed.load(str(tmp_path / "first/model_1.pt"))
    assert resumed.schedule.state_dict() == saved["synchronous"]["schedule"]
    assert resumed.source_transitions == runner.source_transitions
    assert resumed.generation == 1
    resumed.learn(1)
    assert resumed.latest_window.spec.stamp == (2, 2, 49)
    assert resumed.total_transitions == 576 and resumed.optimizer_steps == 60
    assert resumed.latest_window.spec.quotas["source0"] == 52
    assert resumed.schedule.valid_probes == 3
    assert resumed.last_metrics["epoch_samples"] == [192] * 5
    assert sum(resumed.source_transitions.values()) == 576


def test_fixed_switch_retains_probe_and_equal_budget(tmp_path):
    runner = build(tmp_path, enabled=False)
    runner.learn(2)
    assert runner.normalizer_version == 48
    assert set(runner.latest_window.spec.quotas.values()) == {48}
    assert runner.last_metrics["probe"]["schedule"]["status"] == "disabled"
    assert runner.total_transitions == 384


def test_failed_probe_cannot_save_checkpoint(tmp_path, monkeypatch):
    runner = build(tmp_path)

    def fail(*_):
        raise RuntimeError("probe failure")

    monkeypatch.setattr("uni_rl.algos.synchronous_runner.run_source_probe", fail)
    with pytest.raises(RuntimeError, match="probe failure"):
        runner.learn(1)
    assert runner._closed and not list(tmp_path.glob("*.pt"))
    with pytest.raises(RuntimeError, match="complete iteration"):
        runner.save(str(tmp_path / "invalid.pt"))


def test_corrupt_schedule_or_changed_mode_refuses_restore(tmp_path):
    runner = build(tmp_path / "first")
    runner.learn(1)
    path = tmp_path / "first/model_0.pt"
    saved = torch.load(path, weights_only=False)
    saved["synchronous"]["schedule"]["quotas"][0] += 1
    torch.save(saved, tmp_path / "corrupt.pt")
    resumed = build(tmp_path / "bad")
    with pytest.raises(ValueError, match="quota budget"):
        resumed.load(str(tmp_path / "corrupt.pt"))
    assert resumed._closed
    for fault in ("waves", "totals", "probes"):
        saved = torch.load(path, weights_only=False)
        meta = saved["synchronous"]
        if fault == "waves":
            meta["normalizer_version"] += 1
        elif fault == "totals":
            meta["source_transitions"]["source0"] += 1
            meta["source_transitions"]["source1"] -= 1
        else:
            meta["schedule"]["valid_probes"] += 1
        torch.save(saved, tmp_path / "corrupt.pt")
        broken = build(tmp_path / fault)
        with pytest.raises(ValueError, match="history/counters"):
            broken.load(str(tmp_path / "corrupt.pt"))
        assert broken._closed
    control = build(tmp_path / "control", enabled=False)
    with pytest.raises(ValueError, match="mismatch"):
        control.load(str(path))
    assert control._closed
