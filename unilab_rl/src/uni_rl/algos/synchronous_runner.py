"""Central synchronous PPO lifecycle; checkpoint only completed native updates."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from uni_rl.algos.synchronous_ppo import CentralCollector


class SynchronousPPORunner(OnPolicyRunner):
    """Run 24-step windows and native adaptive PPO's five epochs/four minibatches.

    ``manifest_digest`` identifies the injected task/config/assets. Resume restores
    the learner, then resets the environment into a new recovery generation.
    """

    alg: Any
    env: Any
    logger: Any

    def __init__(
        self,
        env: Any,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        *,
        manifest_digest: str,
    ) -> None:
        self._closed = False
        self._checkpoint_ready = False
        self._writer_started = False
        try:
            if not manifest_digest or int(os.getenv("WORLD_SIZE", "1")) != 1:
                raise ValueError("a manifest digest and one central learner are required")
            if train_cfg["num_steps_per_env"] != 24 or train_cfg["save_interval"] != 500:
                raise ValueError("synchronous budget requires 24 steps and save_interval=500")
            self.source_slices = dict(env.env.source_slices)
            self.contract = {
                "manifest_digest": manifest_digest,
                "train_cfg": deepcopy(train_cfg),
                "sources": [(key, sl.start, sl.stop) for key, sl in self.source_slices.items()],
                "num_envs": env.num_envs,
            }
            super().__init__(env, deepcopy(train_cfg), log_dir, device)
            self.logger.writer = None
            if self.alg.schedule != "adaptive" or self.alg.desired_kl is None:
                raise ValueError("native adaptive PPO with desired_kl is required")
            self.collector = CentralCollector(self.alg, env, self.source_slices)
            self.next_iteration = self.policy_version = self.normalizer_version = 0
            self.optimizer_steps = self.total_transitions = self.generation = 0
            self.current_learning_iteration = -1
            self.latest_window: Any = None
            self.last_metrics: dict[str, Any] = {}
            self._returns: dict[str, deque[float]] = {
                key: deque(maxlen=100) for key in self.source_slices
            }
            self._lengths: dict[str, deque[float]] = {
                key: deque(maxlen=100) for key in self.source_slices
            }
            self._episode_rewards = torch.zeros(env.num_envs, device=device)
            self._episode_lengths = torch.zeros(env.num_envs, device=device)
        except BaseException:
            env.close()
            raise

    def _on_step(self, obs: Any, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        self.logger.process_env_step(rewards, dones, extras)
        self._episode_rewards += rewards
        self._episode_lengths += 1
        for key, sl in self.source_slices.items():
            mask = dones[sl].bool()
            self._returns[key].extend(self._episode_rewards[sl][mask].cpu().tolist())
            self._lengths[key].extend(self._episode_lengths[sl][mask].cpu().tolist())
        self._episode_rewards[dones.bool()] = 0
        self._episode_lengths[dones.bool()] = 0

    def _validate_optimizer(self, state: dict, expected: int, learning_rate: float) -> None:
        steps = {float(value["step"]) for value in state["state"].values()}
        if steps != {expected} or any(
            group["lr"] != learning_rate for group in state["param_groups"]
        ):
            raise ValueError("incomplete optimizer update or learning-rate mismatch")
        ids = [index for group in state["param_groups"] for index in group["params"]]
        params = [param for group in self.alg.optimizer.param_groups for param in group["params"]]
        if len(ids) != len(params) or len(ids) != len(set(ids)) or set(ids) != set(state["state"]):
            raise ValueError("incomplete optimizer parameter state")
        for index, param in zip(ids, params):
            for name in ("exp_avg", "exp_avg_sq"):
                value = state["state"][index][name]
                if value.shape != param.shape or not bool(torch.isfinite(value).all()):
                    raise ValueError("invalid optimizer moment")

    def _validate_normalizers(self, saved: dict, samples: int) -> None:
        for key in ("actor_state_dict", "critic_state_dict"):
            value = saved[key].get("obs_normalizer.count")
            if (
                value is None
                or value.dtype != torch.int64
                or value.numel() != 1
                or int(value) != samples
            ):
                raise ValueError("normalizer sample counter mismatch")
            if any(not bool(torch.isfinite(value).all()) for value in saved[key].values()):
                raise ValueError("non-finite model state")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if self._closed:
            raise RuntimeError("runner is closed")
        try:
            if num_learning_iterations <= 0:
                raise ValueError("positive iterations required")
            if init_at_random_ep_len:
                self.env.episode_length_buf = torch.randint_like(
                    self.env.episode_length_buf, high=int(self.env.max_episode_length)
                )
            self.alg.train_mode()
            self._writer_started = True
            self.logger.init_logging_writer()
            start_it, total_it = self.next_iteration, self.next_iteration + num_learning_iterations
            for it in range(start_it, total_it):
                self._checkpoint_ready = False
                start = time.monotonic()
                window = self.collector.collect(
                    it, self.policy_version, self.normalizer_version, on_step=self._on_step
                )
                self.latest_window = window
                self.alg.storage = window.prepare(self.alg)
                collect_time = time.monotonic() - start
                sample_budget = window.spec.transitions
                if sample_budget % 4:
                    raise ValueError("minibatch partition would drop rollout samples")
                generator = self.alg.storage.mini_batch_generator
                epoch_samples = [0] * 5
                batch_count = 0

                def audited_batches(num_mini_batches: int, num_epochs: int = 8):
                    nonlocal batch_count
                    for batch in generator(num_mini_batches, num_epochs):
                        if (
                            batch.actions is None
                            or batch_count >= 20
                            or len(batch.actions) != sample_budget // 4
                        ):
                            raise ValueError("native PPO minibatch budget mismatch")
                        epoch_samples[batch_count // 4] += len(batch.actions)
                        batch_count += 1
                        yield batch

                self.alg.storage.mini_batch_generator = audited_batches
                start = time.monotonic()
                losses = self.alg.update()
                learn_time = time.monotonic() - start
                expected_steps = (it + 1) * 20
                self._validate_optimizer(
                    self.alg.optimizer.state_dict(), expected_steps, self.alg.learning_rate
                )
                if epoch_samples != [sample_budget] * 5 or not all(
                    math.isfinite(float(value)) for value in losses.values()
                ):
                    raise ValueError("incomplete epochs or non-finite native PPO loss")
                self.current_learning_iteration = it
                self.next_iteration = self.policy_version = it + 1
                self.normalizer_version = (it + 1) * 24
                self.total_transitions = (it + 1) * sample_budget
                self.optimizer_steps = expected_steps
                self._validate_normalizers(self.alg.save(), self.total_transitions)
                self.logger.log(
                    it=it,
                    start_it=start_it,
                    total_it=total_it,
                    collect_time=collect_time,
                    learn_time=learn_time,
                    loss_dict=losses,
                    learning_rate=self.alg.learning_rate,
                    action_std=self.alg.get_policy().output_std,
                    rnd_weight=None,
                )
                self._log_window(window, losses, epoch_samples, collect_time, learn_time)
                self._checkpoint_ready = True
                if self.logger.log_dir is not None and (it % 500 == 0 or it == total_it - 1):
                    self.save(str(Path(self.logger.log_dir) / f"model_{it}.pt"))
        finally:
            self.close()

    def _log_window(
        self, window: Any, losses: dict, epoch_samples: list[int], collect: float, learn: float
    ) -> None:
        sources = {}
        for source in window.sources:
            key = source.source_id
            sources[key] = {
                "transitions": window.spec.quotas[key],
                "reward": float(source.raw.rewards.mean()),
                "terminated": int(source.terminated.sum()),
                "truncated": int(source.truncated.sum()),
                "episode_return": float(np.mean(self._returns[key]))
                if self._returns[key]
                else None,
                "episode_length": float(np.mean(self._lengths[key]))
                if self._lengths[key]
                else None,
            }
            if self.logger.writer is not None:
                for name, value in sources[key].items():
                    if value is not None:
                        self.logger.writer.add_scalar(
                            f"Source/{key}/{name}", value, self.current_learning_iteration
                        )
        self.last_metrics = {
            "iteration": self.current_learning_iteration,
            "generation": self.generation,
            "window_stamp": window.spec.stamp,
            "policy_version": self.policy_version,
            "normalizer_version": self.normalizer_version,
            "optimizer_steps": self.optimizer_steps,
            "total_transitions": self.total_transitions,
            "epoch_samples": epoch_samples,
            "sources": sources,
            "reward": sum(value["reward"] * value["transitions"] for value in sources.values())
            / window.spec.transitions,
            "episode_return": float(np.mean(self.logger.rewbuffer))
            if self.logger.rewbuffer
            else None,
            "episode_length": float(np.mean(self.logger.lenbuffer))
            if self.logger.lenbuffer
            else None,
            "losses": losses,
            "learning_rate": self.alg.learning_rate,
            "collect_seconds": collect,
            "learn_seconds": learn,
        }
        if self.logger.log_dir is not None:
            with (Path(self.logger.log_dir) / "synchronous_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(self.last_metrics, allow_nan=False) + "\n")

    def save(self, path: str, infos: dict | None = None) -> None:
        if not self._checkpoint_ready:
            raise RuntimeError("checkpoint requires a complete iteration")
        saved = self.alg.save()
        saved.update(iter=self.current_learning_iteration, infos=infos)
        saved["synchronous"] = {
            "format": 1,
            "complete": True,
            "contract": self.contract,
            **{
                key: getattr(self, key)
                for key in (
                    "next_iteration",
                    "policy_version",
                    "normalizer_version",
                    "optimizer_steps",
                    "total_transitions",
                    "generation",
                )
            },
            "learning_rate": self.alg.learning_rate,
            "logger": {
                "tot_time": self.logger.tot_time,
                "returns": dict(self._returns),
                "lengths": dict(self._lengths),
                "rewbuffer": self.logger.rewbuffer,
                "lenbuffer": self.logger.lenbuffer,
            },
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(saved, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        try:
            if self._closed or self._writer_started or load_cfg is not None or not strict:
                raise ValueError("resume requires a fresh runner and full strict learner restore")
            saved = torch.load(path, weights_only=False, map_location=map_location or self.device)
            meta = saved["synchronous"]
            if (
                meta["format"] != 1
                or meta["complete"] is not True
                or meta["contract"] != self.contract
            ):
                raise ValueError("incomplete checkpoint or config/assets/source topology mismatch")
            iteration = saved["iter"]
            if type(iteration) is not int or iteration < 0:
                raise ValueError("invalid completed iteration")
            count = iteration + 1
            expected = {
                "next_iteration": count,
                "policy_version": count,
                "normalizer_version": count * 24,
                "optimizer_steps": count * 20,
                "total_transitions": count * 24 * self.env.num_envs,
            }
            if any(
                type(meta[key]) is not int or meta[key] != value for key, value in expected.items()
            ):
                raise ValueError("checkpoint counter mismatch")
            if type(meta["generation"]) is not int or meta["generation"] < 0:
                raise ValueError("invalid recovery generation")
            lr = float(meta["learning_rate"])
            if not math.isfinite(lr) or lr <= 0:
                raise ValueError("invalid checkpoint learning rate")
            self._validate_optimizer(saved["optimizer_state_dict"], expected["optimizer_steps"], lr)
            self._validate_normalizers(saved, expected["total_transitions"])
            self.alg.load(saved, None, True)
            self.alg.learning_rate = lr
            for key, value in expected.items():
                setattr(self, key, value)
            self.current_learning_iteration = iteration
            self.generation = meta["generation"] + 1
            self.env.reset()
            self.collector = CentralCollector(self.alg, self.env, self.source_slices)
            self.collector.generation = self.generation
            self._returns, self._lengths = meta["logger"]["returns"], meta["logger"]["lengths"]
            self.logger.rewbuffer, self.logger.lenbuffer = (
                meta["logger"]["rewbuffer"],
                meta["logger"]["lenbuffer"],
            )
            self.logger.tot_time = meta["logger"]["tot_time"]
            self.logger.tot_timesteps = self.total_transitions
            rng = meta["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if rng["cuda"]:
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
            return saved["infos"] or {}
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._writer_started:
                try:
                    if self.logger.writer is not None:
                        try:
                            self.logger.writer.flush()
                        finally:
                            self.logger.writer.close()
                finally:
                    self.logger.stop_logging_writer()
        finally:
            self.env.close()
