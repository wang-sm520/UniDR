"""Joint finals retain their own audit before reusing single-source rendering."""

import pytest
from omegaconf import OmegaConf

from unilab.visualization import single_reference as playback
from unilab.visualization.unidr_holdout import HoldoutPlan


@pytest.fixture
def joint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model_4999.pt"
    checkpoint.write_bytes(b"engineering fixture")
    audit = {
        "final_checkpoint": str(checkpoint),
        "final_checkpoint_sha256": "a" * 64,
        "generations": [0],
        "history": [{"resume_checkpoint": None}],
        "optimizer_steps": 100000,
        "source_order": ["isaacsim", "isaacgym", "genesis", "motrix"],
    }
    metadata = {"planned_iterations": 5000, "total_transitions": 491520000}
    plan = HoldoutPlan(checkpoint, "a" * 64, OmegaConf.create({}), None, metadata)
    calls = []

    def audit_run(run, *, expected_iterations, num_envs):
        assert run == tmp_path and expected_iterations == 5000 and num_envs == 1024
        calls.append("audit")
        return audit

    def preflight(path, root, *, expected_iterations):
        assert path == checkpoint and root == tmp_path and expected_iterations == 5000
        calls.append("strict_preflight")
        return plan

    def render(validated, output, *, root):
        assert validated.actor is plan.actor and validated.config is plan.config
        assert output == tmp_path / "video" and root == tmp_path
        calls.append("render")
        return validated.metadata

    monkeypatch.setattr(playback, "audit_joint_run", audit_run)
    monkeypatch.setattr(playback, "joint_preflight", preflight)
    monkeypatch.setattr(playback, "_record_plan", render)
    return tmp_path, checkpoint, audit, metadata, calls


def test_joint_uses_own_audit_and_shared_renderer(joint):
    root, checkpoint, audit, _, calls = joint
    result = playback.record_joint_reference(checkpoint, root / "video", root=root)
    assert calls == ["audit", "strict_preflight", "render"]
    assert result["training_audit"] == audit and result["source"] == "joint"
    assert result["training_num_envs"] == 4096 and result["training_envs_per_source"] == 1024
    assert result["optimizer_steps"] == 100000 and result["total_transitions"] == 491520000


@pytest.mark.parametrize(
    "damage", ["hash", "path", "resume", "history", "budget", "audit", "strict"]
)
def test_reject_before_render_or_output(joint, monkeypatch, damage):
    root, checkpoint, audit, metadata, calls = joint
    if damage == "hash":
        audit["final_checkpoint_sha256"] = "b" * 64
    elif damage == "path":
        audit["final_checkpoint"] = str(root / "different.pt")
    elif damage == "resume":
        audit["generations"] = [1]
    elif damage == "history":
        audit["history"][0]["resume_checkpoint"] = "old.pt"
    elif damage == "budget":
        metadata["planned_iterations"] = 10000
    else:

        def reject(*args, **kwargs):
            raise ValueError("invalid final")

        monkeypatch.setattr(
            playback, "audit_joint_run" if damage == "audit" else "joint_preflight", reject
        )
    with pytest.raises(ValueError):
        playback.record_joint_reference(checkpoint, root / "video", root=root)
    assert "render" not in calls and not (root / "video").exists()
