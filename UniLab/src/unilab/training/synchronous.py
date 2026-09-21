"""Compose four task owners and inject their factories into the RL runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, cast

from hydra import compose
from omegaconf import DictConfig, OmegaConf
from uni_rl.algos.rsl_rl import normalize_ppo_train_cfg
from uni_rl.algos.synchronous_ppo import CentralVecEnv
from uni_rl.algos.synchronous_runner import SynchronousPPORunner
from uni_rl.ipc.synchronous_env import SourceSpec, SynchronousEnv
from uni_rl.utils.seed import apply_training_seed

from unilab.assets.hub import ensure_robot_assets_for_paths, resolve_motion_files
from unilab.base.config_adapter import BackendAdapter
from unilab.base.env_factory import registry_env_factory
from unilab.training.experiment import get_git_info, write_run_config_snapshot
from unilab.training.run import algo_config_dict
from unilab.utils.sim2sim import extract_contract_snapshot

SOURCE_ORDER = ("isaacsim", "isaacgym", "genesis", "motrix")


def _learner_config(cfg: DictConfig) -> dict[str, Any]:
    result = cast(dict[str, Any], normalize_ppo_train_cfg(algo_config_dict(cfg)))
    for key in ("resume", "resume_path", "load_run", "checkpoint", "max_iterations"):
        result.pop(key, None)
    return result


def _task_behavior(cfg: DictConfig) -> Any:
    env = OmegaConf.to_container(cfg.env, resolve=True)
    assert isinstance(env, dict)
    for backend in SOURCE_ORDER:
        for suffix in ("device_id", "worker_timeout_s", "integrator"):
            env.pop(f"{backend}_{suffix}", None)
    env["scene"]["entities"]["robot"].pop("geom_names", None)
    return env, OmegaConf.to_container(cfg.reward, resolve=True)


def build_sources(cfg: DictConfig, root: Path) -> tuple[list[SourceSpec], list[dict]]:
    """Resolve each backend's Hydra owner, validating the shared policy contract."""
    if tuple(source.name for source in cfg.unidr.sources) != SOURCE_ORDER:
        raise ValueError("four training sources must have the declared fixed order")
    if cfg.training.task_name != "G1FlipTracking" or not cfg.training.sim2sim_strict:
        raise ValueError("G1FlipTracking and strict policy contracts are required")
    if int(cfg.algo.num_envs) < 1 or cfg.training.devices is not None:
        raise ValueError("positive per-source capacity and a single learner are required")
    if str(cfg.training.device) != str(cfg.unidr.learner_device):
        raise ValueError("training.device must match the single learner device")
    if cfg.training.logger != "tensorboard":
        raise ValueError("this synchronous owner uses local TensorBoard logging")
    contract = extract_contract_snapshot(cfg)
    sources, snapshots = [], []
    for index, source in enumerate(cfg.unidr.sources):
        owner = compose("config", overrides=[f"task={source.task}"])
        if str(owner.training.sim_backend) != source.name:
            raise ValueError("source backend must be selected by its task owner")
        if extract_contract_snapshot(owner) != contract:
            raise ValueError("source task policy contract differs from the common owner")
        if _task_behavior(owner) != _task_behavior(cfg):
            raise ValueError("source task behavior differs from the common owner")
        expected_learner = _learner_config(owner)
        expected_learner["num_envs"] = int(cfg.algo.num_envs)
        if expected_learner != _learner_config(cfg):
            raise ValueError("PPO parameters must preserve the successful native owner")
        allowed = {f"{source.name}_device_id"} if source.name != "motrix" else set()
        overrides = OmegaConf.to_container(source.env_overrides, resolve=True)
        if not isinstance(overrides, dict) or set(overrides) - allowed:
            raise ValueError("source overrides may only bind physical devices")
        if source.name == "motrix":
            if source.device != "cpu":
                raise ValueError("Motrix physics requires CPU")
        elif source.device != f"cuda:{overrides.get(f'{source.name}_device_id')}":
            raise ValueError("source physical device and owner binding disagree")
        for key, value in overrides.items():
            OmegaConf.update(owner, f"env.{key}", value, force_add=True)
        override = BackendAdapter(
            owner, root_dir=root, algo_name="ppo"
        ).build_task_env_cfg_override()
        override["seed"] = int(cfg.algo.seed) + index + 1
        sources.append(
            SourceSpec(
                source.name,
                registry_env_factory("G1FlipTracking", source.name),
                int(cfg.algo.num_envs),
                override,
            )
        )
        snapshots.append({"source": source.name, "device": source.device, "env": override})
    return sources, snapshots


def build_manifest(cfg: DictConfig, root: Path, snapshots: list[dict]) -> dict[str, Any]:
    """Fingerprint the effective task/assets and record actual dependency provenance."""
    asset_root = root / "src/unilab/assets"
    ensure_robot_assets_for_paths([str(root / str(cfg.env.scene.model_file))])
    resolve_motion_files(str(cfg.env.commands.motion.params.motion_file))
    paths = list((asset_root / "robots/g1").rglob("*.xml"))
    paths += list((asset_root / "robots/g1/assets").rglob("*"))
    paths += [asset_root / str(cfg.env.commands.motion.params.motion_file)]
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
        if path.is_file()
    }
    motion_key = str(
        (asset_root / str(cfg.env.commands.motion.params.motion_file)).relative_to(root)
    )
    if hashes.get(motion_key) != "bd6785de42e569deb0f680057c0ff62449d3d99058e3c0c1ce437c373e8b604d":
        raise ValueError("the successful run's pinned flip motion is required")
    if (
        hashes.get("src/unilab/assets/robots/g1/g1.xml")
        != "bb6089243f8fe1c97a6410ccde8754eaa58e5d8e476aca69d790ce501cb4a48e"
    ):
        raise ValueError("the successful run's pinned G1 asset is required")
    algorithm = _learner_config(cfg)
    behavior = {"sources": snapshots, "algorithm": algorithm, "assets": hashes}
    digest = hashlib.sha256(json.dumps(behavior, sort_keys=True, default=str).encode()).hexdigest()
    return {
        "digest": digest,
        **behavior,
        "repositories": {
            name: get_git_info(path)
            for name, path in {
                "unilab": root,
                "uni_rl": root.parent / "unilab_rl",
                "unisim": root.parent / "unisim-unidr-backends",
            }.items()
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("unilab-rl", "rsl-rl-lib", "torch", "genesis-world", "motrixsim-core")
        },
    }


def run_synchronous_training(cfg: DictConfig, root: Path) -> None:
    """Wire task ownership to the runtime, preserving the final MuJoCo contract."""
    if cfg.algo.resume:
        if not cfg.algo.resume_path:
            raise ValueError("resume requires an explicit complete checkpoint path")
        cfg.algo.resume_path = str(Path(str(cfg.algo.resume_path)).resolve(strict=True))
    elif cfg.algo.resume_path:
        raise ValueError("checkpoint path supplied with resume disabled")
    sources, snapshots = build_sources(cfg, root)
    manifest = build_manifest(cfg, root, snapshots)
    if cfg.training.log_dir is None:
        raise ValueError("training.log_dir must name a fresh explicit experiment directory")
    log_dir = Path(str(cfg.training.log_dir)).resolve()
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError("use a fresh output directory, including for checkpoint recovery")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "sources_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    write_run_config_snapshot(
        log_dir,
        run_metadata={"mode": "synchronous_four_source", "manifest_digest": manifest["digest"]},
        full_cfg=cfg,
        contract_snapshot=extract_contract_snapshot(cfg),
    )
    apply_training_seed(int(cfg.algo.seed))
    env = SynchronousEnv(
        sources,
        timeout_s=float(cfg.unidr.timeout_s),
        startup_timeout_s=float(cfg.unidr.startup_timeout_s),
    )
    try:
        wrapped = CentralVecEnv(env, device=str(cfg.unidr.learner_device))
        train_cfg = _learner_config(cfg)
        runner = SynchronousPPORunner(
            wrapped,
            train_cfg,
            str(log_dir),
            str(cfg.unidr.learner_device),
            manifest_digest=manifest["digest"],
        )
        if cfg.algo.resume:
            runner.load(str(cfg.algo.resume_path))
        runner.learn(
            int(cfg.algo.max_iterations) - runner.next_iteration,
            init_at_random_ep_len=not cfg.algo.resume,
        )
    finally:
        env.close()
