"""Joint admission requires completed single runs with the same task and assets."""

import fcntl
import hashlib
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base.config_adapter import BackendAdapter
from unilab.training import joint_comparison as gate
from unilab.training.synchronous import SOURCE_ORDER, _learner_config
from unilab.utils.sim2sim import extract_contract_snapshot

ROOT = Path(__file__).parents[2]


def write(path, value):
    path.write_text(json.dumps(value))


def mutate(path, change):
    value = json.loads(path.read_text())
    change(value)
    write(path, value)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    (tmp_path / "queue.lock").touch()
    (tmp_path / "status.tsv").write_text("date\tcompleted\tgenesis\tsim2sim\n")
    (tmp_path / "stage.pid").write_text("0\n")
    assets = {"g1.xml": "fixed", "flip.npz": "fixed"}
    audits = {}
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        joint = compose("config", ["task=g1_flip_tracking/unidr_comparison"])
        for source in SOURCE_ORDER:
            run_dir = tmp_path / source
            run_dir.mkdir()
            cfg = compose("config", [f"task=g1_flip_tracking/{source}_comparison"])
            env = BackendAdapter(cfg, root_dir=ROOT, algo_name="ppo").build_task_env_cfg_override()
            config = OmegaConf.to_container(cfg, resolve=True)
            manifest = dict(
                sources=[dict(source=source, env=env)],
                algorithm=_learner_config(cfg),
                assets=assets,
            )
            digest = hashlib.sha256(
                json.dumps(manifest, sort_keys=True, default=str).encode()
            ).hexdigest()
            manifest.update(source=source, env=env, config=config, digest=digest)
            write(run_dir / "single_manifest.json", manifest)
            write(
                run_dir / "run_config.json",
                dict(config=config, contract_snapshot=extract_contract_snapshot(cfg)),
            )
            audit = dict(source=source, final_checkpoint_sha256=source)
            audits[source] = audit
            folder = run_dir / "mujoco-front-reference"
            folder.mkdir()
            video = folder / "front-reference.mp4"
            video.write_bytes(b"test video; decoder mocked")
            write(
                folder / "verification.json",
                dict(
                    source=source,
                    checkpoint_sha256=source,
                    training_audit=audit,
                    manifest_digest=digest,
                    backend="mujoco",
                    strict_preflight=True,
                    actor_normalizer_unchanged=True,
                    fresh_policy_steps=1000,
                    text_overlays=False,
                    reference_rgba=[0.0, 0.85, 1.0, 0.55],
                    video=str(video),
                    video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                    format={"frames": 1000},
                    landing_candidates=0,
                    terminated=1000,
                ),
            )
    monkeypatch.setattr(
        gate, "build_manifest", lambda *args: dict(algorithm=_learner_config(joint), assets=assets)
    )
    monkeypatch.setattr(gate, "audit_single_run", lambda path, **kw: audits[path.name])
    monkeypatch.setattr(gate, "validate_video", lambda path: {"frames": 1000})
    return tmp_path


def test_joint_owner_is_fresh_5000_and_all_sources_are_required(completed):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose("config", ["task=g1_flip_tracking/unidr_comparison"])
    assert (cfg.algo.num_envs, cfg.algo.max_iterations, cfg.algo.num_steps_per_env) == (
        1024,
        5000,
        24,
    )
    assert cfg.algo.resume is False and cfg.algo.resume_path is None and cfg.algo.load_run == "-1"
    assert cfg.algo.algorithm.num_learning_epochs * cfg.algo.algorithm.num_mini_batches == 20
    result = gate.verify_singles(ROOT, completed)
    assert tuple(result["singles"]) == SOURCE_ORDER
    assert result["holdout_performance_gate"] is False  # Failure to flip must not tune selection.


def test_running_queue_or_held_lock_blocks(completed):
    with (completed / "queue.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            gate.verify_singles(ROOT, completed)
    (completed / "status.tsv").write_text("date\trunning\tgenesis\tsim2sim\n")
    with pytest.raises(ValueError, match="not completed"):
        gate.verify_singles(ROOT, completed)


@pytest.mark.parametrize("field", ["env", "algorithm", "assets", "config"])
def test_modified_manifest_or_effective_config_blocks(completed, field):
    mutate(
        completed / "motrix/single_manifest.json", lambda value: value[field].update(tampered=True)
    )
    with pytest.raises(ValueError, match="configuration/assets"):
        gate.verify_singles(ROOT, completed)


def test_raw_ppo_mismatch_with_manifest_blocks(completed):
    def change(value):
        value["config"]["algo"]["algorithm"]["gamma"] = 0.5

    for name in ("single_manifest.json", "run_config.json"):
        mutate(completed / "motrix" / name, change)
    with pytest.raises(ValueError, match="configuration/assets"):
        gate.verify_singles(ROOT, completed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_sha256", "other"),
        ("strict_preflight", False),
        ("fresh_policy_steps", 999),
        ("actor_normalizer_unchanged", False),
        ("video_sha256", "other"),
        ("text_overlays", True),
    ],
)
def test_incomplete_or_wrong_video_blocks(completed, field, value):
    mutate(
        completed / "isaacgym/mujoco-front-reference/verification.json",
        lambda v: v.update({field: value}),
    )
    with pytest.raises(ValueError, match="video verification"):
        gate.verify_singles(ROOT, completed)


def test_decode_failure_or_failed_native_audit_blocks(completed, monkeypatch):
    monkeypatch.setattr(gate, "validate_video", lambda path: {"frames": 999})
    with pytest.raises(ValueError, match="full decode"):
        gate.verify_singles(ROOT, completed)

    def failed(*args, **kwargs):
        raise ValueError("native budget incomplete")

    monkeypatch.setattr(gate, "audit_single_run", failed)
    with pytest.raises(ValueError, match="native budget incomplete"):
        gate.verify_singles(ROOT, completed)
