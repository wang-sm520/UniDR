"""Hydra-owned composition of the fixed G1 multi-simulator experiment."""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from omegaconf import DictConfig, OmegaConf
from uni_rl.env_contract import EnvProtocol, get_algo_capabilities

from unilab.base.config_adapter import BackendAdapter
from unilab.base.env_factory import registry_env_factory
from unilab.training.experiment import get_git_info
from unilab.utils.sim2sim import extract_contract_snapshot

if TYPE_CHECKING:
    from uni_rl.ipc.multi_source_env import EnvSourceSpec, MultiSourceOptions


SOURCE_ORDER = ("isaacgym", "isaacsim", "motrix", "genesis")
SOURCE_NUM_ENVS = 2000
TOTAL_NUM_ENVS = SOURCE_NUM_ENVS * len(SOURCE_ORDER)
_RUNTIME_FIELDS = {
    "isaacgym": frozenset({"isaacgym_device_id", "isaacgym_worker_timeout_s"}),
    "isaacsim": frozenset(
        {"isaacsim_device_id", "isaacsim_worker_timeout_s", "isaacsim_render_mode"}
    ),
    "motrix": frozenset({"cpu_ids"}),
    "genesis": frozenset({"genesis_device_id", "genesis_integrator"}),
}


@dataclass(frozen=True)
class MultiSourceTrainingPlan:
    """Prepared factories and JSON-safe provenance; contains no live simulator."""

    sources: tuple[EnvSourceSpec, ...]
    options: MultiSourceOptions
    manifest: dict[str, Any]
    joint_names: tuple[str, ...]


def is_multi_source(cfg: DictConfig) -> bool:
    return str(OmegaConf.select(cfg, "training.sim_backend", default="")) == "multisim"


def source_timing_context(
    cfg: DictConfig, runner: Any, env: Any, *, log_dir: str | Path | None
) -> AbstractContextManager[None]:
    """Attach the algorithm-owned timing recorder only when explicitly configured."""
    if not OmegaConf.select(cfg, "training.source_timing.enabled", default=False):
        return nullcontext()
    if not is_multi_source(cfg) or log_dir is None:
        raise ValueError("source_timing requires multi-source training with a run directory")
    from uni_rl.algos.rsl_rl_source_timing import record_source_timing

    return cast(
        AbstractContextManager[None],
        record_source_timing(
            runner,
            output_path=Path(log_dir) / "source_timing.jsonl",
            source_statistics=lambda: env.source_statistics,
        ),
    )


def _validate_shared_task(cfg: DictConfig) -> None:
    owner_root = Path(__file__).resolve().parents[1] / "conf/ppo"
    expected = OmegaConf.merge(
        OmegaConf.load(owner_root / "config.yaml"),
        OmegaConf.load(owner_root / "task/g1_walk_flat/base.yaml"),
        OmegaConf.load(owner_root / "task/g1_walk_flat/multisim_base.yaml"),
    )
    actual_env = OmegaConf.to_container(cfg.env, resolve=True)
    if not isinstance(actual_env, dict):
        raise ValueError("multisim requires the shared G1 environment mapping")
    actual_env.pop("seed", None)
    if actual_env != OmegaConf.to_container(expected.env, resolve=True):
        raise ValueError(
            "multisim task semantics are fixed by multisim_base.yaml; only per-source runtime options may differ"
        )
    if OmegaConf.to_container(cfg.reward, resolve=True) != OmegaConf.to_container(
        expected.reward, resolve=True
    ):
        raise ValueError("multisim must retain the common G1 reward, including foot contacts")
    for field in ("empirical_normalization", "obs_groups"):
        if OmegaConf.select(cfg, f"algo.{field}") != OmegaConf.select(expected, f"algo.{field}"):
            raise ValueError(f"multisim requires the shared algo.{field} policy contract")


def validate_multi_source_topology(
    cfg: DictConfig, *, world_size: int = 1, devices: Sequence[int] | None = None
) -> None:
    """Reject unsupported experiment/topology choices before allocating resources."""
    declared = OmegaConf.select(cfg, "training.multi_source", default=None)
    if not is_multi_source(cfg):
        if declared is not None:
            raise ValueError("training.multi_source requires the multisim task owner")
        return
    if declared is None:
        raise ValueError("multisim requires training.multi_source.sources in its task owner")
    if str(OmegaConf.select(cfg, "training.task_name")) != "G1WalkFlat":
        raise ValueError("The multisim experiment only supports G1WalkFlat")
    if str(OmegaConf.select(cfg, "algo.algo", default="")) != "ppo":
        raise ValueError("The multisim experiment only supports PPO")
    if world_size != 1 or (devices is not None and tuple(devices) != (0,)):
        raise ValueError("multisim requires one learner on GPU 0; torchrun is unsupported")
    device = OmegaConf.select(cfg, "training.device", default=None)
    if device not in (None, "cuda", "cuda:0"):
        raise ValueError("multisim requires training.device=cuda:0")
    if OmegaConf.select(cfg, "training.play_only", default=False):
        raise ValueError("Evaluate multisim checkpoints with a real backend and --profile multisim")
    if (
        not OmegaConf.select(cfg, "training.no_play", default=False)
        or str(OmegaConf.select(cfg, "training.play_render_mode", default="auto")) != "none"
    ):
        raise ValueError("multisim requires training.no_play=true and play_render_mode=none")
    if OmegaConf.select(cfg, "algo.num_envs") != TOTAL_NUM_ENVS:
        raise ValueError(
            "multisim requires exactly 8000 environments; automatic resizing is disabled"
        )
    if OmegaConf.select(cfg, "algo.num_steps_per_env") != 24:
        raise ValueError("multisim requires a 24-step PPO rollout")
    if platform.system() != "Linux":
        raise ValueError("The multisim process-group runtime currently requires Linux")
    _validate_shared_task(cfg)


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in (
        "unilab",
        "unilab-rl",
        "unisim-core",
        "torch",
        "rsl-rl-lib",
        "genesis-world",
        "motrixsim-core",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _repository_versions() -> dict[str, Any]:
    repositories = {}
    for module in ("unilab", "uni_rl", "unisim"):
        spec = find_spec(module)
        if spec is not None and spec.origin is not None:
            origin = Path(spec.origin).resolve()
            repositories[module] = {"module_file": str(origin), **get_git_info(origin.parent)}
    return repositories


def build_multi_source_plan(cfg: DictConfig, *, root_dir: Path) -> MultiSourceTrainingPlan:
    """Resolve the common task into four factories without initializing an engine."""
    from uni_rl.ipc.multi_source_env import EnvSourceSpec, MultiSourceOptions

    validate_multi_source_topology(cfg)
    source_cfgs = OmegaConf.to_container(cfg.training.multi_source.sources, resolve=True)
    if not isinstance(source_cfgs, list) or len(source_cfgs) != len(SOURCE_ORDER):
        raise ValueError("multisim requires exactly four configured sources")
    names = tuple(
        source.get("name") if isinstance(source, dict) else None for source in source_cfgs
    )
    if names != SOURCE_ORDER:
        raise ValueError(f"multisim source order must be {SOURCE_ORDER}, got {names}")
    option_cfg = OmegaConf.to_container(cfg.training.multi_source.options, resolve=True)
    if not isinstance(option_cfg, dict) or not all(isinstance(key, str) for key in option_cfg):
        raise ValueError("multi_source.options must be a mapping")
    options = MultiSourceOptions(**cast(dict[str, Any], option_cfg))
    seed = int(cfg.algo.seed)
    common = OmegaConf.to_container(cfg, resolve=True)
    sources: list[EnvSourceSpec] = []
    manifests: list[dict[str, Any]] = []
    joint_names = tuple(str(name) for name in cfg.env.scene.entities.robot.joint_names)
    if len(joint_names) != 29 or len(set(joint_names)) != 29:
        raise ValueError("G1 multisim requires 29 distinct joints in action order")
    task_contract: dict[str, Any] | None = None
    for source_index, (name, source) in enumerate(zip(SOURCE_ORDER, source_cfgs, strict=True)):
        if not isinstance(source, dict):
            raise ValueError("Each multi-source declaration must be a mapping")
        unknown = set(source) - {"name", "backend", "num_envs", "seed_offset", "env"}
        if unknown:
            raise ValueError(f"Unknown source fields for {name}: {sorted(unknown)}")
        count = source.get("num_envs")
        if isinstance(count, bool) or not isinstance(count, int) or count != SOURCE_NUM_ENVS:
            raise ValueError(f"{name} requires exactly {SOURCE_NUM_ENVS} environments")
        if source.get("backend") != name:
            raise ValueError(f"Source {name} must select backend {name}")
        runtime = source.get("env") or {}
        if not isinstance(runtime, Mapping):
            raise ValueError(f"Source {name}.env must be a runtime-options mapping")
        unknown_runtime = set(runtime) - _RUNTIME_FIELDS[name]
        if unknown_runtime:
            raise ValueError(
                f"Source {name} cannot override shared task fields: {sorted(unknown_runtime)}"
            )
        for field, value in runtime.items():
            if field.endswith("_device_id") and (isinstance(value, bool) or value != 0):
                raise ValueError(f"Source {name} must use GPU 0")
        if name == "isaacsim" and runtime.get("isaacsim_render_mode", "none") != "none":
            raise ValueError("IsaacSim multi-source training must run without rendering")
        offset = source.get("seed_offset", source_index + 1)
        if isinstance(offset, bool) or not isinstance(offset, int) or seed + offset < 0:
            raise ValueError(f"Invalid source seed offset for {name}")
        source_seed = seed + offset
        resolved = cast(DictConfig, OmegaConf.create(common))
        resolved.training.sim_backend = name
        resolved.training.multi_source = None
        resolved.algo.num_envs = count
        resolved.algo.seed = source_seed
        resolved.env = OmegaConf.merge(resolved.env, runtime, {"seed": source_seed})
        env_cfg = BackendAdapter(
            resolved, root_dir=root_dir, algo_name="ppo"
        ).build_task_env_cfg_override()
        semantic_env = {
            key: value
            for key, value in env_cfg.items()
            if key not in _RUNTIME_FIELDS[name] | {"seed"}
        }
        contract = {"env": semantic_env, "policy": extract_contract_snapshot(resolved)}
        if task_contract is None:
            task_contract = contract
        elif contract != task_contract:
            raise ValueError(f"Source {name} does not match the shared task/observation contract")
        sources.append(
            EnvSourceSpec(
                name=name,
                factory=registry_env_factory(str(resolved.training.task_name), name),
                num_envs=count,
                env_cfg_override=env_cfg,
                seed=source_seed,
            )
        )
        manifests.append(
            {
                "name": name,
                "backend": name,
                "num_envs": count,
                "seed": source_seed,
                "slice": [source_index * count, (source_index + 1) * count],
                "fraction": count / TOTAL_NUM_ENVS,
                "resolved_config": OmegaConf.to_container(resolved, resolve=True),
                "contract_snapshot": extract_contract_snapshot(resolved),
            }
        )
    return MultiSourceTrainingPlan(
        sources=tuple(sources),
        options=options,
        manifest={
            "sources": manifests,
            "packages": _package_versions(),
            "repositories": _repository_versions(),
            "synchronization": "step_barrier",
            "failure_policy": "abort_all",
            "total_num_envs": TOTAL_NUM_ENVS,
            "samples_per_iteration": TOTAL_NUM_ENVS * 24,
        },
        joint_names=joint_names,
    )


def create_multi_source_training_env(plan: MultiSourceTrainingPlan) -> EnvProtocol:
    """Construct the composite env and guard the actual G1 policy interface."""
    from uni_rl.ipc.multi_source_env import make_multi_source_env

    env = make_multi_source_env(plan.sources, options=plan.options)
    try:
        if env.obs_groups_spec != {"obs": 98, "critic": 101}:
            raise ValueError(
                f"G1 multisim observation groups must be obs=98/critic=101, got {env.obs_groups_spec}"
            )
        if env.action_space.shape != (29,):
            raise ValueError("G1 multisim requires 29-dimensional actions")
        if get_algo_capabilities(env).joint_names != plan.joint_names:
            raise ValueError("G1 multisim action joint order does not match the task owner")
        return env
    except BaseException:
        env.close()
        raise
