"""Spawn-only source lifecycle; no simulator or algorithm dependencies."""

from __future__ import annotations

import os
import random
import tempfile
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from uni_rl.env_contract import EnvProtocol, SupportsEpisodeLengthBufferProtocol
from uni_rl.ipc._multi_source_resources import (
    exclude_transport_attachment,
    install_source_tracker,
)
from uni_rl.ipc._multi_source_shared import (
    SharedArrays,
    array,
    describe,
    info_metadata,
    receive_message,
    send_message,
)
from uni_rl.utils.nan_guard import NanGuard, NanGuardCfg

if TYPE_CHECKING:
    from uni_rl.ipc.multi_source_env import EnvSourceSpec, MultiSourceOptions


def _statistics_snapshot(statistics: dict[str, int | float]) -> dict[str, int | float]:
    """Sample current Linux worker RSS, excluding descendant processes."""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        statistics["rss_bytes"] = resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, ValueError):
        statistics["rss_bytes"] = 0
    return dict(statistics)


def _accumulate_environment_timings(
    statistics: dict[str, int | float], info: dict[str, Any]
) -> None:
    """Retain numeric env-owned substep diagnostics, including internal resets."""
    timings = info.get("timing", {})
    if not isinstance(timings, dict):
        return
    for name, value in timings.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            if not isinstance(name, str) or not np.isfinite(value):
                raise ValueError(
                    "Environment timing names and values must be finite numeric metadata"
                )
            statistics[f"timing/{name}/last"] = float(value)
            total_key = f"timing/{name}/total"
            statistics[total_key] = statistics.get(total_key, 0.0) + float(value)


def run_source(
    source: EnvSourceSpec,
    options: MultiSourceOptions,
    connection: Any,
    publication_lock: Any,
    tracker_connection: Any,
) -> None:
    """Own one source and publish only complete, validated transitions."""
    env: EnvProtocol | None = None
    shared: SharedArrays | None = None
    operation = "startup"
    sequence_id = 0
    statistics: dict[str, int | float] = {
        "num_envs": source.num_envs,
        "step_calls": 0,
        "transitions": 0,
        "reset_calls": 0,
        "step_seconds": 0.0,
        "reset_seconds": 0.0,
        "last_step_seconds": 0.0,
        "last_reset_seconds": 0.0,
        "pid": os.getpid(),
        "rss_bytes": 0,
    }
    try:
        install_source_tracker(tracker_connection)
        if os.name == "posix":
            os.setsid()
        send_message(connection, {"kind": "started", "operation": operation, "sequence_id": 0})
        if source.seed is not None:
            random.seed(source.seed)
            np.random.seed(source.seed)
            try:
                import torch
            except ImportError:
                pass
            else:
                torch.manual_seed(source.seed)
        env = source.factory(source.num_envs, source.env_cfg_override)
        state = env.init_state()
        descriptor = describe(env, state, source.num_envs)
        send_message(
            connection,
            {
                "kind": "descriptor",
                "operation": operation,
                "sequence_id": 0,
                "descriptor": descriptor,
            },
        )
        message = receive_message(connection)
        if message.get("operation") != "attach" or message.get("sequence_id") != 0:
            raise ValueError("expected startup SHM attachment")
        shared = SharedArrays(descriptor, source.num_envs, publication_lock, message["shm_name"])
        exclude_transport_attachment(shared.memory)
        info = shared.write_state(state, 0, "init_state")
        shared.publish(1, 0, options.operation_timeout_s)
        send_message(
            connection,
            {
                "kind": "ack",
                "operation": operation,
                "sequence_id": 0,
                "info": info,
                "statistics": _statistics_snapshot(statistics),
            },
        )
        while True:
            message = receive_message(connection)
            operation = message.get("operation", "unknown")
            incoming_sequence = message.get("sequence_id")
            if type(incoming_sequence) is not int or incoming_sequence != sequence_id + 1:
                raise ValueError(
                    f"stale or out-of-order command: {incoming_sequence}, previous {sequence_id}"
                )
            sequence_id = incoming_sequence
            slot = sequence_id % 2
            if operation == "close":
                closing_env, env = env, None
                closing_env.close()
                send_message(
                    connection, {"kind": "ack", "operation": operation, "sequence_id": sequence_id}
                )
                return
            shared.check_publication(0, sequence_id, options.operation_timeout_s)
            reset_info: dict[str, Any] = {}
            if operation == "step":
                started_at = time.monotonic()
                state = env.step(shared.arrays["actions"][slot])
                elapsed = time.monotonic() - started_at
                statistics["step_calls"] += 1
                statistics["transitions"] += source.num_envs
                statistics["step_seconds"] += elapsed
                statistics["last_step_seconds"] = elapsed
                _accumulate_environment_timings(statistics, state.info)
            elif operation == "reset":
                count = message["count"]
                if type(count) is not int or not 0 <= count <= source.num_envs:
                    raise ValueError("invalid reset count")
                if count:
                    indices = shared.arrays["indices"][slot, :count].copy()
                    if (
                        np.any(indices < 0)
                        or np.any(indices >= source.num_envs)
                        or len(np.unique(indices)) != count
                    ):
                        raise ValueError("invalid local reset indices")
                    started_at = time.monotonic()
                    observations, raw_info = env.reset(indices)
                    elapsed = time.monotonic() - started_at
                    statistics["reset_calls"] += 1
                    statistics["reset_seconds"] += elapsed
                    statistics["last_reset_seconds"] = elapsed
                    shared.write_observations(observations, "reset", slot, count)
                    reset_info = info_metadata(raw_info)
                    if env.state is None:
                        raise ValueError("reset must retain the initialized env.state")
                    state = env.state
                    for group, _, _ in descriptor.groups:
                        if not np.array_equal(state.obs[group][indices], observations[group]):
                            raise ValueError("reset observations disagree with env.state")
                    if "steps" in raw_info:
                        reset_steps = array(
                            raw_info["steps"], (count,), "reset/steps", descriptor.steps_dtype
                        )
                        if not np.array_equal(reset_steps, state.info["steps"][indices]):
                            raise ValueError("reset steps disagree with env.state")
                    if "final_observation" in raw_info or "_final_observation" in raw_info:
                        raise ValueError(
                            "reset info final-observation arrays are unsupported; expose them in env.state"
                        )
            elif operation == "set_episode_length_buf":
                episode_env = env
                if not isinstance(episode_env, SupportsEpisodeLengthBufferProtocol):
                    raise ValueError("source does not support set_episode_length_buf")
                values = shared.arrays["lengths"][slot].copy()
                episode_env.set_episode_length_buf(values)
                if env.state is None or not np.array_equal(env.state.info["steps"], values):
                    raise ValueError("set_episode_length_buf must update state.info['steps']")
                state = env.state
            elif operation == "set_nan_guard":
                guard_config = message["guard_config"]
                if guard_config is None:
                    env.set_nan_guard(None)
                else:
                    cfg = NanGuardCfg(**guard_config)
                    output_root = (
                        Path(cfg.output_dir)
                        if cfg.output_dir
                        else Path(tempfile.gettempdir()) / "uni_rl" / "nan_dumps"
                    )
                    cfg.output_dir = str(output_root / "source" / source.name)
                    env.set_nan_guard(NanGuard(cfg, source.num_envs, descriptor.physics_playback))
            else:
                raise ValueError(f"unsupported source operation: {operation}")
            info = shared.write_state(state, sequence_id, operation)
            shared.publish(1, sequence_id, options.operation_timeout_s)
            send_message(
                connection,
                {
                    "kind": "ack",
                    "operation": operation,
                    "sequence_id": sequence_id,
                    "info": info,
                    "reset_info": reset_info,
                    "statistics": _statistics_snapshot(statistics),
                },
            )
    except BaseException:
        with suppress(BaseException):
            send_message(
                connection,
                {
                    "kind": "error",
                    "operation": operation,
                    "sequence_id": sequence_id,
                    "remote_traceback": traceback.format_exc()[-48000:],
                },
            )
    finally:
        if env is not None:
            with suppress(BaseException):
                env.close()
        if shared is not None:
            with suppress(BaseException):
                shared.close()
        connection.close()
