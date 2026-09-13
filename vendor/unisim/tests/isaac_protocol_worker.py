"""Pipe/SHM test worker running native worker methods against NumPy SDK doubles."""

from __future__ import annotations

import argparse
import importlib.util
import sys

from isaac_worker_fakes import make_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--mode", default="normal")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("worker_protocol", args.protocol)
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    worker = make_worker(args.kind)
    worker.protocol = protocol
    if args.mode == "reject_mass":
        if args.kind == "isaacgym":
            worker.gym.reject = "mass"
        else:
            worker.robot.root_physx_view.reject = True
    try:
        while True:
            message = protocol.recv_message(sys.stdin.buffer)
            payload = message["payload"]
            command = message["cmd"]
            if command == protocol.CMD_SHUTDOWN:
                break
            try:
                if command == protocol.CMD_INIT:
                    protocol.validate_init(payload)
                    meta = {
                        **worker.get_meta(),
                        "dof_names": worker.contract_joint_names,
                        "body_names": worker.contract_body_names,
                        "gravity": [0, 0, -9.81],
                        "env_origins": [[0, 0, 0], [10, 0, 0], [20, 0, 0]],
                        "collision_filtering_applied": True,
                        "graphics_enabled": False,
                        "render_mode": "none",
                        "render_width": 1280,
                        "render_height": 720,
                    }
                    if args.mode == "old":
                        del meta["protocol_version"]
                    elif args.mode == "unsupported":
                        meta["supported_reset_terms"] = ["kp", "kd"]
                    elif args.mode == "no_contact":
                        meta["contact_reporter"] = None
                    elif args.mode == "bad_nominal":
                        meta["nominal_body_mass"] = [1, -1, 1]
                    elif args.mode == "wrong_reporter":
                        meta["contact_reporter"] = (
                            protocol.CONTACT_REPORTER_ISAACSIM
                            if args.kind == "isaacgym"
                            else protocol.CONTACT_REPORTER_ISAACGYM
                        )
                    protocol.send_message(sys.stdout.buffer, protocol.CMD_META, meta)
                    continue
                if command == protocol.CMD_ATTACH:
                    worker.attach_slots(payload)
                    result = None
                elif command == protocol.CMD_SET_STATE:
                    result = worker.set_state(payload)
                elif command == protocol.CMD_STEP:
                    result = worker.step(payload)
                else:
                    raise ValueError("unexpected command: " + command)
                protocol.send_message(sys.stdout.buffer, protocol.CMD_READY, result)
            except Exception as exc:
                protocol.send_message(
                    sys.stdout.buffer, protocol.CMD_ERROR, protocol.serialize_exception(exc)
                )
    finally:
        for handle in worker._shm_handles:
            handle.close()


if __name__ == "__main__":
    main()
