"""Snapshot a fresh single-source comparison before native PPO starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.training.synchronous import SOURCE_ORDER, build_manifest


def prepare_run(
    root: Path, run_dir: Path, source: str, *, num_envs: int = 1024, iterations: int = 20000
) -> dict:
    """Validate and fingerprint the exact config/assets without constructing physics."""
    if source not in SOURCE_ORDER or min(num_envs, iterations) < 1:
        raise ValueError("Expected one training source and positive training budgets")
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Fresh comparison requires a new run directory: {run_dir}")
    with initialize_config_dir(config_dir=str(root / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose(
            "config",
            [
                f"task=g1_flip_tracking/{source}_comparison",
                f"algo.num_envs={num_envs}",
                f"algo.max_iterations={iterations}",
                "training.device=cuda:0",
                f"training.log_dir={run_dir}",
            ],
        )
    override = BackendAdapter(cfg, root_dir=root, algo_name="ppo").build_task_env_cfg_override()
    registry.ensure_registries()
    env_cfg = registry.materialize_env_config("G1FlipTracking")
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()
    manifest = build_manifest(cfg, root, [{"source": source, "env": override}])
    manifest.update(
        source=source,
        expected_iterations=iterations,
        num_envs=num_envs,
        env=override,
        config=OmegaConf.to_container(cfg, resolve=True),
        selection="fixed final checkpoint, selected before any MuJoCo evaluation",
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "single_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=SOURCE_ORDER)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=20000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    manifest = prepare_run(
        root, args.run_dir, args.source, num_envs=args.num_envs, iterations=args.iterations
    )
    print(json.dumps({"source": args.source, "digest": manifest["digest"]}))


if __name__ == "__main__":
    main()
