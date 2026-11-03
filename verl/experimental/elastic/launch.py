"""Run/submit/cancel jobs for the independent task-1 elastic scheduler."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .policy import Policy, Task, integer, positive
from .runtime import Controller, RayBackend
from .storage import read_json, write_json

logger = logging.getLogger(__name__)


def _packet(controller: Controller, packet: dict, defaults: dict) -> None:
    action = packet["action"]
    if action == "submit":
        task = Task.parse(packet["job"], command=defaults.get("command"), overrides=defaults.get("overrides"))
        submitted = packet.get("submitted_unix")
        submitted_seconds = submitted - defaults["started_unix"] if submitted is not None else None
        controller.submit(task, submitted_seconds)
    elif action == "cancel":
        controller.cancel(packet["job_name"])
    elif action == "capacity":
        policy = controller.policy
        new = (integer(packet["rollout_gpus"], "rollout_gpus"), integer(packet["train_gpus"], "train_gpus"))
        if sum(new) > defaults["hardware_gpus"]:
            raise ValueError("New admission capacity exceeds this controller's physical GPU budget")
        for name, task in policy.tasks.items():
            if name not in controller.results and (
                task.rollout.request * policy.rollout_width > new[0] or task.train.request * policy.train_width > new[1]
            ):
                raise ValueError(f"Capacity update cannot meet {name}'s request")
        old = (policy.rollout_gpus, policy.train_gpus)
        policy.rollout_gpus, policy.train_gpus = new
        try:
            decision = policy.rebalance(time.monotonic() - controller.started)
            if decision is None:
                raise ValueError("Capacity update cannot preserve active request/SLO constraints")
            if decision.affected:
                controller.apply(decision)
        except BaseException:
            policy.rollout_gpus, policy.train_gpus = old
            raise
        controller.event("capacity_updated", rollout_gpus=new[0], train_gpus=new[1])
    else:
        raise ValueError(f"Unknown elastic event: {action}")


def run(manifest: dict, output: Path, ray_address: str, *, serve=False, max_seconds=3600, base_port=20969):
    import ray

    from verl.experimental.weave.launch import _stop, _wait_ready

    if (output / "controller.json").exists():
        raise ValueError("Use a new output directory; live controller recovery is not implemented")
    output.mkdir(parents=True, exist_ok=True)
    policy = Policy(
        rollout_gpus=manifest["rollout_gpus"],
        train_gpus=manifest["train_gpus"],
        rollout_width=manifest.get("rollout_gpus_per_replica", 1),
        train_width=manifest.get("train_gpus_per_replica", 1),
        max_group_size=manifest.get("max_group_size", 5),
        simulate_steps=manifest.get("simulate_steps", 30),
    )
    if policy.train_width != 1:
        raise NotImplementedError("Elastic training restart currently supports pure DP with one GPU per DP unit")
    pending_arrivals = []
    for raw in manifest.get("jobs", []):
        task = Task.parse(raw, command=manifest.get("command"), overrides=manifest.get("overrides"))
        arrival = raw.get("arrival_seconds", 0)
        if isinstance(arrival, bool) or not isinstance(arrival, (float, int)) or arrival < 0:
            raise ValueError("arrival_seconds must be finite and nonnegative")
        if arrival:
            positive(arrival, "arrival_seconds")
        pending_arrivals.append((arrival, task))
    pending_arrivals.sort(key=lambda value: value[0])
    started_monotonic = time.monotonic()
    started_unix = time.time()
    ray.init(address=ray_address, namespace=f"elastic_controller_{uuid.uuid4().hex}")
    owner = None
    backend = None
    owner_log = None
    try:
        nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0)]
        if len(nodes) != 1:
            raise ValueError(
                "Elastic task-1 runtime currently supports one GPU node; cross-node restart is not implemented"
            )
        required = policy.rollout_gpus + policy.train_gpus
        hardware = int(ray.available_resources().get("GPU", 0))
        if hardware < required:
            raise ValueError(f"Elastic run requires {required} free GPUs, available: {hardware}")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
            os.environ[key] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        owner_path = output / "transfer-queue-owner.log"
        owner_log = owner_path.open("w")
        owner = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "verl.experimental.weave.transfer_queue_owner",
                "--ray-address",
                ray_address,
            ],
            stdout=owner_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _wait_ready(
            owner, lambda: "TRANSFER_QUEUE_OWNER_READY" in owner_path.read_text(), "elastic TransferQueue owner"
        )
        backend = RayBackend(
            output,
            ray_address,
            policy,
            started_unix,
            base_port=base_port,
            timeout=manifest.get("reconfigure_timeout", 1800),
            resident_config=manifest.get("resident_scheduler"),
        )
        controller = Controller(policy, backend, output, started=started_monotonic)
        defaults = {**manifest, "hardware_gpus": required, "started_unix": started_unix}
        write_json(output / "manifest.json", manifest)
        write_json(
            output / "controller.json",
            {
                "pid": os.getpid(),
                "started_unix": started_unix,
                "ray_address": ray_address,
                "rollout_gpus": policy.rollout_gpus,
                "train_gpus": policy.train_gpus,
            },
        )
        print("ELASTIC_CONTROLLER_READY", output, flush=True)
        inbox = output / "inbox"
        inbox.mkdir(exist_ok=True)
        processed = inbox / "processed"
        processed.mkdir(exist_ok=True)
        start = started_monotonic
        poll = positive(manifest.get("poll_interval", 0.2), "poll_interval")
        hold = positive(manifest.get("min_hold_seconds", 30), "min_hold_seconds")
        while True:
            if owner.poll() is not None:
                raise RuntimeError("Elastic TransferQueue owner exited unexpectedly")
            elapsed = time.monotonic() - start
            if elapsed > max_seconds:
                raise TimeoutError("Elastic run exceeded its time budget")
            while pending_arrivals and pending_arrivals[0][0] <= elapsed:
                arrival, task = pending_arrivals.pop(0)
                controller.submit(task, arrival)
            for path in sorted(inbox.glob("*.json")):
                packet = read_json(path)
                try:
                    _packet(controller, packet, defaults)
                except (ValueError, KeyError, TypeError) as exc:
                    logger.error("Rejected elastic event %s: %s", path.name, exc)
                    controller.event("event_rejected", request=path.stem, error=str(exc))
                    write_json(output / "responses" / path.name, {"accepted": False, "error": str(exc)})
                else:
                    write_json(output / "responses" / path.name, {"accepted": True})
                path.rename(processed / path.name)
            controller.refresh()
            controller.schedule(min_hold_seconds=hold)
            write_json(output / "state.json", controller.snapshot())
            if not serve and not pending_arrivals and not policy.pending and not policy.groups:
                break
            time.sleep(poll)
        seconds = time.monotonic() - start
        write_json(
            output / "result.json",
            {
                **controller.snapshot(),
                "wall_seconds": seconds,
                "cluster_gpu_hours": required * seconds / 3600,
                "cost_basis": "fixed initial two-pool GPU budget for this bounded run",
            },
        )
        return 0
    except BaseException as exc:
        logger.exception("Elastic execution failed")
        write_json(output / "failure.json", {"error": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if backend is not None:
            for name in list(backend.groups):
                backend.stop(name)
        if owner is not None:
            _stop(owner)
        if owner_log is not None:
            owner_log.close()
        ray.shutdown()


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--manifest", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--ray-address", default="auto")
    runner.add_argument("--serve", action="store_true")
    runner.add_argument("--max-seconds", type=float, default=3600)
    runner.add_argument("--base-port", type=int, default=20969)
    submit = commands.add_parser("submit")
    submit.add_argument("--output", type=Path, required=True)
    submit.add_argument("--job", type=Path, required=True)
    cancel = commands.add_parser("cancel")
    cancel.add_argument("--output", type=Path, required=True)
    cancel.add_argument("--job-name", required=True)
    capacity = commands.add_parser("capacity")
    capacity.add_argument("--output", type=Path, required=True)
    capacity.add_argument("--rollout-gpus", type=int, required=True)
    capacity.add_argument("--train-gpus", type=int, required=True)
    args = parser.parse_args()
    if args.action == "run":
        raise SystemExit(
            run(
                read_json(args.manifest),
                args.output.resolve(),
                args.ray_address,
                serve=args.serve,
                max_seconds=positive(args.max_seconds, "max_seconds"),
                base_port=args.base_port,
            )
        )
    if not (args.output / "controller.json").exists():
        raise ValueError("No initialized elastic controller at this output path")
    packet = {"action": args.action, "submitted_unix": time.time()}
    if args.action == "submit":
        packet["job"] = read_json(args.job)
    elif args.action == "cancel":
        packet["job_name"] = args.job_name
    else:
        packet.update(rollout_gpus=args.rollout_gpus, train_gpus=args.train_gpus)
    identifier = uuid.uuid4().hex
    write_json(args.output / "inbox" / f"{identifier}.json", packet)
    print(f"Queued request {identifier}; inspect {args.output / 'responses' / (identifier + '.json')}")


if __name__ == "__main__":
    main()
