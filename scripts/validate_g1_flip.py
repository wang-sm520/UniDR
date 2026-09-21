"""Reload a trained baseline and evaluate on its training backend only."""

import argparse
import json
from contextlib import closing
from hashlib import sha256
from pathlib import Path

import torch
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
from uni_rl.algos.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime

from unilab.base.config_adapter import BackendAdapter
from unilab.base.env_factory import registry_env_factory
from unilab.tasks.motion_tracking.g1.validation import validate_flip_episode
from unilab.training import algo_config_dict
from unilab.utils.sim2sim import (
    extract_contract_snapshot,
    policy_load_dim_guard,
    resolve_sim2sim_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    manifest_path = checkpoint.parent / "run_config.json"
    output = args.output.resolve()
    if output.suffix != ".json" or any(
        output == source or (output.exists() and output.samefile(source))
        for source in (checkpoint, manifest_path)
    ):
        raise ValueError("Evaluation output must be a separate JSON file, not a training artifact")
    manifest = json.loads(manifest_path.read_text())
    snapshot = manifest.get("contract_snapshot")
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError("A complete training contract snapshot is required")
    cfg = OmegaConf.create(manifest["config"])
    backend = str(cfg.training.sim_backend)
    if backend not in {"motrix", "genesis", "isaacsim", "isaacgym"}:
        raise ValueError("This entrypoint excludes the MuJoCo holdout")
    if cfg.training.task_name != "G1FlipTracking" or args.num_envs < 1:
        raise ValueError("Expected G1FlipTracking and a positive evaluation environment count")
    if not extract_contract_snapshot(cfg).keys() <= snapshot.keys():
        raise ValueError("A complete training contract snapshot is required")
    resolve_sim2sim_config(checkpoint.parent, cfg, algo_name="ppo", strict=True)
    cfg.training.play_only = True
    cfg.training.play_env_num = args.num_envs
    cfg.training.device = args.device
    root = Path(__file__).resolve().parents[1]
    override = BackendAdapter(cfg, root_dir=root, algo_name="ppo").build_play_env_cfg_override()
    rl_cfg = algo_config_dict(cfg)
    runtime = resolve_rsl_rl_ppo_runtime(rl_cfg, default_wrapper_cls=RslRlVecEnvWrapper)
    before_hash = sha256(checkpoint.read_bytes()).hexdigest()
    with closing(registry_env_factory("G1FlipTracking", backend)(args.num_envs, override)) as env:
        wrapped = runtime.wrapper_cls(env, device=args.device)
        runner = (runtime.runner_cls or OnPolicyRunner)(
            wrapped, normalize_ppo_train_cfg(rl_cfg), log_dir=None, device=args.device
        )
        with policy_load_dim_guard(env_obs_dim=160, env_action_dim=29, algo_name="ppo"):
            runner.load(
                str(checkpoint),
                load_cfg={"actor": True, "critic": True, "iteration": True},
                map_location=args.device,
            )
        runner.alg.eval_mode()
        policy = runner.get_inference_policy(device=args.device)
        before = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
        with torch.inference_mode():
            result = validate_flip_episode(
                env, lambda: policy(wrapped.get_observations()).cpu().numpy(), seed=args.seed
            )
        if any(not torch.equal(v, runner.alg.actor.state_dict()[k]) for k, v in before.items()):
            raise RuntimeError("Evaluation changed actor or normalizer state")
        if runner.alg.storage.step != 0 or runner.alg.optimizer.state:
            raise RuntimeError("Evaluation accumulated training state")
    if sha256(checkpoint.read_bytes()).hexdigest() != before_hash:
        raise RuntimeError("Evaluation modified the checkpoint")
    result.update(
        backend=backend,
        checkpoint=str(checkpoint),
        checkpoint_sha256=before_hash,
        checkpoint_iteration=runner.current_learning_iteration,
        evaluation_seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
