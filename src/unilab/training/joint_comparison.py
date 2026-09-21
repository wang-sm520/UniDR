"""Gate the fresh joint comparison on completed single-source experiments."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from uni_rl.logging.single_run_audit import audit_single_run

from unilab.base.config_adapter import BackendAdapter
from unilab.training.synchronous import _learner_config, build_manifest, build_sources
from unilab.utils.sim2sim import extract_contract_snapshot
from unilab.visualization.unidr_holdout import validate_video


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_singles(
    root: Path, single_root: Path, *, num_envs: int = 1024, single_iterations: int = 20000
) -> dict:
    """Check completion, fixed videos and identical current joint task provenance.

    No holdout performance criterion influences admission: only artifact integrity
    and successful completion are required. The shell separately checks unit exit.
    """
    single_root = single_root.resolve(strict=True)
    with (single_root / "queue.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _require(
            (single_root / "status.tsv").read_text().strip().split("\t")[1:]
            == ["completed", "genesis", "sim2sim"]
            and (single_root / "stage.pid").read_text().strip() == "0",
            "Single-source queue has not completed all four stages",
        )
        with initialize_config_dir(
            config_dir=str(root / "src/unilab/conf/ppo"), version_base="1.3"
        ):
            cfg = compose(
                "config",
                ["task=g1_flip_tracking/unidr_comparison", f"algo.num_envs={num_envs}"],
            )
            _, snapshots = build_sources(cfg, root)
        joint = build_manifest(cfg, root, snapshots)
        audits = {}
        for snapshot in snapshots:
            source = snapshot["source"]
            run_dir = single_root / source
            audit = audit_single_run(
                run_dir, expected_iterations=single_iterations, num_envs=num_envs
            )
            saved = _read(run_dir / "single_manifest.json")
            run = _read(run_dir / "run_config.json")
            native = OmegaConf.create(run["config"])
            if not isinstance(native, DictConfig):
                raise ValueError(f"{source}: expected a mapping training config")
            actual_env = BackendAdapter(
                native, root_dir=root, algo_name="ppo"
            ).build_task_env_cfg_override()
            behavior = {key: saved[key] for key in ("sources", "algorithm", "assets")}
            digest = hashlib.sha256(
                json.dumps(behavior, sort_keys=True, default=str).encode()
            ).hexdigest()
            _require(
                saved["digest"] == digest
                and saved["source"] == audit["source"] == source
                and saved["config"] == run["config"]
                and saved["env"] == actual_env == snapshot["env"]
                and saved["sources"] == [{"source": source, "env": actual_env}]
                and saved["algorithm"] == _learner_config(native) == joint["algorithm"]
                and saved["assets"] == joint["assets"]
                and run["contract_snapshot"]
                == extract_contract_snapshot(native)
                == extract_contract_snapshot(cfg),
                f"{source}: single-source configuration/assets differ from joint comparison",
            )
            folder = run_dir / "mujoco-front-reference"
            video = folder / "front-reference.mp4"
            verification = _read(folder / "verification.json")
            _require(
                verification["source"] == source
                and verification["checkpoint_sha256"] == audit["final_checkpoint_sha256"]
                and verification["training_audit"] == audit
                and verification["manifest_digest"] == digest
                and verification["backend"] == "mujoco"
                and verification["strict_preflight"] is True
                and verification["actor_normalizer_unchanged"] is True
                and verification["fresh_policy_steps"] == 1000
                and verification["text_overlays"] is False
                and verification["reference_rgba"] == [0.0, 0.85, 1.0, 0.55]
                and Path(verification["video"]).resolve() == video.resolve()
                and verification["video_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest(),
                f"{source}: fixed final reference video verification mismatch",
            )
            _require(
                verification["format"] == validate_video(video),
                f"{source}: reference video failed full decode validation",
            )
            audits[source] = audit
        return {"singles": audits, "joint_manifest": joint, "holdout_performance_gate": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("single_root", type=Path)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--single-iterations", type=int, default=20000)
    args = parser.parse_args()
    result = verify_singles(Path(__file__).resolve().parents[3], **vars(args))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
