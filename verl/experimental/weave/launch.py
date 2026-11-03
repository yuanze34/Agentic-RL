"""Execute a global WEAVE plan on an existing Ray cluster."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _placement_override(placements: list[dict[str, Any]]) -> str:
    return (
        "["
        + ",".join(
            "{" + ",".join(f"{key}:{json.dumps(value)}" for key, value in placement.items()) + "}"
            for placement in placements
        )
        + "]"
    )


def job_command(
    job: dict[str, Any],
    group: dict[str, Any],
    *,
    slot_width: int,
    rollout_placements: list[dict[str, str]],
    train_placement: dict[str, Any],
    port: int,
    migration: bool = True,
    train_placements: list[dict[str, Any]] | None = None,
    scheduler_host: str = "127.0.0.1",
    ray_address: str = "auto",
    run_id: str | None = None,
) -> list[str]:
    if not job.get("command"):
        raise ValueError(f"{job['job_name']} has no command to launch")
    replicas = [rollout_placements[slot] for slot in job["rollout_slots"]]
    placements = _placement_override(replicas)
    name = job["job_name"]
    train_gpus_per_node = min(len(job["train"]), group.get("gpus_per_node", group["num_train_gpus"]))
    settings = {
        "trainer.v1.trainer_mode": "weave_separate",
        "trainer.nnodes": len(job["train"]) // train_gpus_per_node,
        "trainer.n_gpus_per_node": train_gpus_per_node,
        "trainer.val_before_train": False,
        "trainer.test_freq": -1,
        "trainer.experiment_name": f"weave_{run_id}_{name}" if run_id else f"weave_{name}",
        "trainer.v1.separate_async.parameter_sync_step": 1,
        "trainer.v1.separate_async.num_warmup_batches": 0,
        "trainer.v1.sampler.max_off_policy_threshold": 1,
        "trainer.v1.weave.job_name": name,
        "trainer.v1.weave.port": port,
        "trainer.v1.weave.host": scheduler_host,
        "trainer.v1.weave.gen_device_affinity": "WEAVE_GEN",
        "trainer.v1.weave.train_device_affinity": "WEAVE_TRAIN",
        "trainer.v1.weave.gen_gpu_list": list(range(len(job["gen"]))),
        "trainer.v1.weave.train_gpu_list": list(range(len(job["train"]))),
        "trainer.v1.weave.gpus_per_gen_worker": slot_width,
        "trainer.v1.weave.gen_replica_placements": placements,
        "trainer.v1.weave.train_placement_group": train_placement["name"],
        "trainer.v1.weave.train_placement_group_id": train_placement["id"],
        "trainer.v1.weave.train_pool_size": group["num_train_gpus"],
        "trainer.v1.weave.train_replica_placements": _placement_override(train_placements or []),
        "trainer.v1.weave.max_tenants_per_gpu": max(2, len(group["jobs"])),
        "trainer.v1.weave.init_wait_timeout": 1800,
        "trainer.v1.weave.migration.enabled": migration,
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.nnodes": 1,
        "actor_rollout_ref.rollout.n_gpus_per_node": len(job["gen"]),
        "actor_rollout_ref.rollout.tensor_model_parallel_size": slot_width,
        "actor_rollout_ref.rollout.data_parallel_size": 1,
        "actor_rollout_ref.rollout.pipeline_model_parallel_size": 1,
        "actor_rollout_ref.rollout.free_cache_engine": True,
        "actor_rollout_ref.rollout.checkpoint_engine.backend": "nccl",
        "+transfer_queue.external_owner": True,
        "transfer_queue.partition_id": name,
        "+ray_kwargs.ray_init.address": ray_address,
        "+ray_kwargs.ray_init.namespace": f"verl_weave_{name}",
    }
    return [
        *job["command"],
        *job.get("overrides", []),
        *[f"{key}={value if isinstance(value, str) else json.dumps(value)}" for key, value in settings.items()],
    ]


def _stop(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _wait_ready(process: subprocess.Popen, ready, description: str, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while not ready():
        if process.poll() is not None:
            raise RuntimeError(f"{description} exited with code {process.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {description}")
        time.sleep(0.1)


def launch(
    plan: dict[str, Any], output: Path, base_port: int, ray_address: str, scheduler_host: str = "127.0.0.1"
) -> int:
    import ray
    import redis
    from ray.util.placement_group import remove_placement_group

    from .placement import SharedPoolSpec, create_detached_pool

    if plan.get("version") != 1 or not plan["groups"]:
        raise ValueError("Unsupported or empty WEAVE plan")
    # Do not stop or reuse unrelated Redis services.
    for index in range(len(plan["groups"])):
        with socket.socket() as probe:
            probe.bind((scheduler_host, base_port + index))
    ray.init(address=ray_address, namespace=f"weave_launcher_{uuid.uuid4().hex}")
    try:
        required = plan["num_gen_gpus"] + plan["num_train_gpus"]
        largest_pool = max(
            plan["gpus_per_worker"],
            max(
                min(group["num_train_gpus"], group.get("gpus_per_node", group["num_train_gpus"]))
                for group in plan["groups"]
            ),
        )
        nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) >= largest_pool]
        if not nodes:
            raise ValueError(f"WEAVE needs a Ray node large enough for a {largest_pool}-GPU execution pool")
        if ray.available_resources().get("GPU", 0) < required:
            raise ValueError(f"WEAVE plan needs {required} currently available GPUs")
        output.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        with ExitStack() as stack:
            processes: list[subprocess.Popen] = []

            def start(command: list[str], log: Path) -> subprocess.Popen:
                handle = stack.enter_context(log.open("w"))
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                stack.callback(_stop, process)
                return process

            owner_log = output / "transfer-queue-owner.log"
            owner = start(
                [sys.executable, "-m", "verl.experimental.weave.transfer_queue_owner", "--ray-address", ray_address],
                owner_log,
            )
            _wait_ready(owner, lambda: "TRANSFER_QUEUE_OWNER_READY" in owner_log.read_text(), "TransferQueue owner")
            jobs = []
            for index, group in enumerate(plan["groups"]):
                port = base_port + index
                directory = output / group["group_id"]
                directory.mkdir(exist_ok=True)
                mapping = directory / "mapping.txt"
                mapping.write_text("".join(f"{job['job_name']} 0 0\n" for job in group["jobs"]))
                placements = directory / "placements.json"
                placements.write_text(
                    json.dumps({job["job_name"]: {"gen": job["gen"], "train": job["train"]} for job in group["jobs"]})
                )
                # Redis is a separately owned process so cleanup waits for it too.
                redis_process = start(
                    ["redis-server", "--bind", scheduler_host, "--port", str(port), "--save", "", "--appendonly", "no"],
                    directory / "redis.log",
                )
                storage = redis.Redis(host=scheduler_host, port=port, decode_responses=True)
                stack.callback(storage.close)

                def redis_ready(storage=storage):
                    try:
                        return storage.ping()
                    except redis.ConnectionError:
                        return False

                _wait_ready(redis_process, redis_ready, "Redis")
                arrival_arguments = []
                if "benchmark_arrivals" in plan:
                    schedule = directory / "arrival-schedule.json"
                    schedule.write_text(
                        json.dumps(
                            {job["job_name"]: plan["benchmark_arrivals"][job["job_name"]] for job in group["jobs"]}
                        )
                    )
                    arrival_arguments = ["--arrival-schedule", str(schedule)]
                scheduler = start(
                    [
                        sys.executable,
                        "-m",
                        "verl.experimental.weave.scheduler",
                        "--port",
                        str(port),
                        "--host",
                        scheduler_host,
                        "--num-jobs",
                        str(len(group["jobs"])),
                        "--num-gen-gpus",
                        str(group["num_gen_gpus"]),
                        "--num-train-gpus",
                        str(group["num_train_gpus"]),
                        "--gen-affinity",
                        "WEAVE_GEN",
                        "--train-affinity",
                        "WEAVE_TRAIN",
                        "--mapping",
                        str(mapping),
                        "--placements",
                        str(placements),
                        "--ratio",
                        "0.5",
                        "--ready-token",
                        token,
                        *arrival_arguments,
                    ],
                    directory / "scheduler.log",
                )
                _wait_ready(
                    scheduler, lambda storage=storage: storage.get("weave:scheduler:ready") == token, "WEAVE scheduler"
                )
                tenants = max(2, len(group["jobs"]))
                train_placements = []
                train_node_size = group.get("gpus_per_node", group["num_train_gpus"])
                for node_index, start_gpu in enumerate(range(0, group["num_train_gpus"], train_node_size)):
                    train_name = f"weave_{token}_{index}_train_{node_index}"
                    size = min(train_node_size, group["num_train_gpus"] - start_gpu)
                    train = create_detached_pool(SharedPoolSpec(train_name, size, tenants))
                    stack.callback(remove_placement_group, train)
                    train_placements.append({"name": train_name, "id": train.id.hex(), "num_gpus": size})
                rollout = []
                for slot in range(group["num_rollout_slots"]):
                    name = f"weave_{token}_{index}_rollout_{slot}"
                    pool = create_detached_pool(SharedPoolSpec(name, plan["gpus_per_worker"], tenants))
                    stack.callback(remove_placement_group, pool)
                    rollout.append({"name": name, "id": pool.id.hex()})
                for job in group["jobs"]:
                    command = job_command(
                        job,
                        group,
                        slot_width=plan["gpus_per_worker"],
                        rollout_placements=rollout,
                        train_placement=train_placements[0],
                        train_placements=train_placements,
                        port=port,
                        scheduler_host=scheduler_host,
                        ray_address=ray_address,
                        run_id=token,
                    )
                    (directory / f"{job['job_name']}.command.json").write_text(json.dumps(command, indent=2))
                    process = start(command, directory / f"{job['job_name']}.log")
                    jobs.append((job["job_name"], process))

            pending = dict(jobs)
            failures = {}
            completion_unix = {}
            while pending:
                for process in processes:
                    if process not in pending.values() and process not in [job[1] for job in jobs]:
                        if process.poll() is not None:
                            raise RuntimeError(f"WEAVE control-plane process {process.pid} exited")
                for name, process in list(pending.items()):
                    if process.poll() is not None:
                        failures[name] = process.returncode
                        completion_unix[name] = time.time()
                        del pending[name]
                        if process.returncode:
                            raise RuntimeError(f"WEAVE job {name} failed with code {process.returncode}")
                time.sleep(0.1)
            (output / "exit-codes.json").write_text(json.dumps(failures, sort_keys=True) + "\n")
            (output / "completion-unix.json").write_text(json.dumps(completion_unix, sort_keys=True) + "\n")
        return 0
    finally:
        ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-port", type=int, default=9969)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument(
        "--scheduler-host",
        default="127.0.0.1",
        help="Launcher interface reachable from all trainer nodes (trusted cluster only)",
    )
    args = parser.parse_args()

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(
        launch(
            json.loads(args.plan.read_text()),
            args.output_dir.resolve(),
            args.base_port,
            args.ray_address,
            args.scheduler_host,
        )
    )


if __name__ == "__main__":
    main()
