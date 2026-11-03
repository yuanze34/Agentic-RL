"""WEAVE-compatible local FCFS scheduler.

Scheduling decisions are intentionally equivalent to the reference
``Weave/worker_scheduler/scheduler.py``.  Framework integration belongs in the
client/trainer adapter, not in this control plane.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .protocol import TENANT_EVENTS_CHANNEL, Event, EventLevel, EventParser, EventType, Phase
from .resource_manager import JobStatus, ResourceManager

logger = logging.getLogger(__name__)


class SharedStorage(Protocol):
    def get(self, name: str) -> Any: ...

    def set(self, name: str, value: Any) -> Any: ...

    def publish(self, channel: str, message: str) -> Any: ...

    def pubsub(self) -> Any: ...


def _decode(value: Any | None) -> str | None:
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def get_init_job_status(job_name: str, shared_storage: SharedStorage) -> JobStatus:
    gpus_per_worker = _decode(shared_storage.get(f"{job_name}:gpus_per_gen_worker"))
    train_gpu_list = _decode(shared_storage.get(f"{job_name}:train_gpu_list"))
    gen_gpu_list = _decode(shared_storage.get(f"{job_name}:gen_gpu_list"))
    return JobStatus(
        job_name=job_name,
        phase=Phase.INIT,
        gpus_per_gen_worker=int(gpus_per_worker) if gpus_per_worker is not None else 1,
        gen_gpu_list=[int(item) for item in gen_gpu_list.split(",")] if gen_gpu_list else [0],
        train_gpu_list=[int(item) for item in train_gpu_list.split(",")] if train_gpu_list else [0],
    )


class FCFSPolicy:
    """The reference WEAVE FCFS policy, including its init head-of-line barrier."""

    def __init__(self, shared_storage: SharedStorage, min_concurrency_ratio: float = 0.5):
        if not 0 < min_concurrency_ratio <= 1:
            raise ValueError("min_concurrency_ratio must be in (0, 1]")
        self.min_concurrency_ratio = min_concurrency_ratio
        self.shared_storage = shared_storage

    def __call__(
        self,
        event_queue: list[Event],
        resource_manager: ResourceManager,
        num_uninited_jobs: int,
    ) -> list[Event] | None:
        if not event_queue:
            return None

        # WEAVE gives every init event absolute priority and does not skip a
        # blocked init to run a later phase.
        for index, candidate in enumerate(event_queue):
            assert candidate.event_type == EventType.READY
            if candidate.level != EventLevel.JOB:
                continue
            if resource_manager.num_available_devices != resource_manager.cluster_size:
                return None
            resource_manager.register_job(get_init_job_status(candidate.job_name, self.shared_storage))
            resource_manager.allocate_all(candidate.job_name)
            event_queue.pop(index)
            return [Event(candidate.job_name, EventType.STATUS, value="initializing")]

        if num_uninited_jobs != 0:
            return None

        for index, candidate in enumerate(event_queue):
            assert candidate.event_type == EventType.READY
            assert candidate.level == EventLevel.PHASE
            status = resource_manager.job_mapping[candidate.job_name]

            if candidate.phase == Phase.UPDATE:
                # The sender still owns its training allocation. Weight transfer
                # must not wake rollout tenants currently owned by another job.
                if len(resource_manager.get_free_gpus(candidate.job_name, "gen")) < len(status.gen_gpu_list):
                    continue
                event_queue.pop(index)
                while resource_manager.allocate_worker(candidate.job_name, "gen") is not None:
                    pass
                return [Event(candidate.job_name, EventType.STATUS, phase=Phase.UPDATE, value="running")]

            if candidate.phase == Phase.GENERATE:
                free_gpus = resource_manager.get_free_gpus(candidate.job_name, "gen")
                if len(free_gpus) < self.min_concurrency_ratio * len(status.gen_gpu_list):
                    continue
                event_queue.pop(index)
                resource_manager.update_phase(candidate.job_name, Phase.GENERATE)
                events = [Event(candidate.job_name, EventType.STATUS, phase=Phase.GENERATE, value="running")]
                gpu_ids = resource_manager.allocate_worker(candidate.job_name, "gen")
                while gpu_ids is not None:
                    worker_id = gpu_ids[0] // status.gpus_per_gen_worker
                    logger.info("Allocated %s generate worker %s on %s", candidate.job_name, worker_id, gpu_ids)
                    events.append(
                        Event(
                            candidate.job_name,
                            EventType.STATUS,
                            phase=Phase.GENERATE,
                            worker_id=str(worker_id),
                            value="running",
                        )
                    )
                    gpu_ids = resource_manager.allocate_worker(candidate.job_name, "gen")
                return events

            if candidate.phase == Phase.TRAIN:
                free_gpus = resource_manager.get_free_gpus(candidate.job_name, "train")
                if len(free_gpus) < len(status.train_gpu_list):
                    continue
                event_queue.pop(index)
                resource_manager.update_phase(candidate.job_name, Phase.TRAIN)
                assert resource_manager.allocate_worker(candidate.job_name, "train") is not None
                return [Event(candidate.job_name, EventType.STATUS, phase=Phase.TRAIN, value="running")]

            raise AssertionError(f"Unexpected phase: {candidate.phase}")
        return None


class ArrivalPolicy:
    """Benchmark-only logical arrival gate around the unchanged FCFS policy."""

    def __init__(self, policy: FCFSPolicy, arrivals: dict[str, float], audit_path: Path):
        if not arrivals or any(
            not isinstance(name, str)
            or not name
            or isinstance(delay, bool)
            or not isinstance(delay, (int, float))
            or not math.isfinite(delay)
            or delay < 0
            for name, delay in arrivals.items()
        ):
            raise ValueError("Arrival schedule must map job names to finite nonnegative seconds")
        self.policy = policy
        self.arrivals = arrivals
        self.audit_path = audit_path
        self.started = None
        self.started_unix = None
        self.admitted = set()

    def __call__(self, event_queue, resource_manager, num_uninited_jobs):
        if num_uninited_jobs:
            return self.policy(event_queue, resource_manager, num_uninited_jobs)
        if self.started is None:
            self.started = time.monotonic()
            self.started_unix = time.time()
        elapsed = time.monotonic() - self.started
        blocked = []
        eligible = []
        for event in event_queue:
            if event.job_name not in self.arrivals:
                raise ValueError(f"Missing logical arrival for {event.job_name}")
            if event.phase == Phase.GENERATE and elapsed < self.arrivals[event.job_name]:
                blocked.append(event)
            else:
                eligible.append(event)
        result = self.policy(eligible, resource_manager, num_uninited_jobs)
        event_queue[:] = eligible + blocked
        if result is not None:
            for event in result:
                if (
                    event.phase == Phase.GENERATE
                    and event.level == EventLevel.PHASE
                    and event.job_name not in self.admitted
                ):
                    self.admitted.add(event.job_name)
                    with self.audit_path.open("a") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "job": event.job_name,
                                    "ready_unix": self.started_unix,
                                    "logical_arrival_seconds": self.arrivals[event.job_name],
                                    "admitted_unix": time.time(),
                                }
                            )
                            + "\n"
                        )
        return result


@dataclass(frozen=True)
class SchedulerConfig:
    num_jobs: int
    gen_device_affinity: str
    gen_device_ids: list[int]
    train_device_affinity: str
    train_device_ids: list[int]
    mapping_file: str
    min_concurrency_ratio: float = 1.0
    is_colo: bool = False
    ready_token: str = "1"
    placements: dict[str, dict[str, list[int]]] | None = None


class Scheduler:
    def __init__(self, shared_storage: SharedStorage, config: SchedulerConfig):
        self.shared_storage = shared_storage
        self.resource_manager = ResourceManager(
            config.gen_device_affinity,
            config.gen_device_ids,
            config.train_device_affinity,
            config.train_device_ids,
            config.mapping_file,
            is_colo=config.is_colo,
            placements=config.placements,
        )
        self.ready_queue: list[Event] = []
        self.lock = threading.Lock()
        self.select_policy = FCFSPolicy(shared_storage, config.min_concurrency_ratio)
        self.num_uninited_jobs = config.num_jobs
        self._resolved_init_jobs: set[str] = set()
        self.ready_token = config.ready_token
        self.shared_storage.set("weave:init_barrier", "pending")

    def _drop_ready_events(self, job_name: str) -> None:
        self.ready_queue = [event for event in self.ready_queue if event.job_name != job_name]

    def _clear_generate_workers(self, job_name: str) -> None:
        self.shared_storage.set(f"{job_name}:generate:allocated_workers", "")

    def _resolve_init_barrier(self, job_name: str) -> None:
        if job_name not in self._resolved_init_jobs:
            self._resolved_init_jobs.add(job_name)
            self.num_uninited_jobs = max(0, self.num_uninited_jobs - 1)
            if self.num_uninited_jobs == 0:
                self.shared_storage.set("weave:init_barrier", "open")

    def handle_event(self, event: Event) -> None:
        if event.event_type in (EventType.COMPLETED, EventType.CRASHED):
            self._drop_ready_events(event.job_name)
            self._clear_generate_workers(event.job_name)
            self._resolve_init_barrier(event.job_name)
            if event.job_name in self.resource_manager.job_mapping:
                self.resource_manager.cleanup_by_name(event.job_name)
            if event.event_type == EventType.CRASHED:
                if event.job_name in self.resource_manager.job_mapping:
                    self.resource_manager.update_phase(event.job_name, Phase.GENERATE)
                self.shared_storage.set(f"{event.job_name}:status", "crashed")
                self.shared_storage.set(f"{event.job_name}:generate:status", "crashed")
                self.shared_storage.set(f"{event.job_name}:train:status", "crashed")
                if self.shared_storage.get(f"{event.job_name}:update:status") is not None:
                    self.shared_storage.set(f"{event.job_name}:update:status", "crashed")
            return

        if event.event_type == EventType.RELEASE_GPU:
            assert event.value is not None
            device_affinity, *gpu_ids = str(event.value).split(",")
            self.resource_manager.release(event.job_name, device_affinity, [int(item) for item in gpu_ids])
            return

        if event.event_type == EventType.CREATED:
            assert event.level == EventLevel.JOB
            ready = deepcopy(event)
            ready.event_type = EventType.READY
            self.ready_queue.append(ready)
            return

        if event.event_type == EventType.READY and event.phase == Phase.UPDATE:
            if self.resource_manager.job_mapping[event.job_name].phase != Phase.TRAIN:
                raise ValueError("WEAVE weight update requires an active training phase")
            self.ready_queue.append(event)
            return

        if event.event_type == EventType.RECOVERED:
            assert event.level == EventLevel.JOB
            self._drop_ready_events(event.job_name)
            if event.job_name not in self.resource_manager.job_mapping:
                self.resource_manager.register_job(get_init_job_status(event.job_name, self.shared_storage))
            else:
                self.resource_manager.cleanup_by_name(event.job_name)
                self.resource_manager.update_phase(event.job_name, Phase.GENERATE)
            self.shared_storage.set(f"{event.job_name}:status", "running")
            self.shared_storage.set(f"{event.job_name}:generate:status", "pending")
            self.shared_storage.set(f"{event.job_name}:train:status", "pending")
            self._resolve_init_barrier(event.job_name)
            self.ready_queue.append(Event(event.job_name, EventType.READY, phase=Phase.GENERATE))
            return

        if event.event_type != EventType.DONE or event.level == EventLevel.WORKER:
            return

        assert event.level == EventLevel.PHASE
        if event.phase == Phase.INIT:
            self.resource_manager.update_phase(event.job_name, Phase.GENERATE)
            self.ready_queue.append(Event(event.job_name, EventType.READY, phase=Phase.GENERATE))
            self._resolve_init_barrier(event.job_name)
        elif event.phase == Phase.GENERATE:
            self._clear_generate_workers(event.job_name)
            self.resource_manager.cleanup_by_name(event.job_name)
            self.resource_manager.update_phase(event.job_name, Phase.TRAIN)
            self.ready_queue.append(Event(event.job_name, EventType.READY, phase=Phase.TRAIN))
        elif event.phase == Phase.TRAIN:
            self.resource_manager.cleanup_by_name(event.job_name)
            self.resource_manager.update_phase(event.job_name, Phase.GENERATE)
            self.ready_queue.append(Event(event.job_name, EventType.READY, phase=Phase.GENERATE))
        elif event.phase == Phase.UPDATE:
            self.shared_storage.set(f"{event.job_name}:update:status", "pending")
        else:
            raise AssertionError(f"Unexpected finish event: {event}")

    def schedule_once(self) -> list[Event] | None:
        with self.lock:
            events = self.select_policy(self.ready_queue, self.resource_manager, self.num_uninited_jobs)
            if events is not None:
                generate_workers = [
                    event.worker_id
                    for event in events
                    if event.phase == Phase.GENERATE
                    and event.level == EventLevel.WORKER
                    and event.event_type == EventType.STATUS
                ]
                if generate_workers:
                    self.shared_storage.set(
                        f"{events[0].job_name}:generate:allocated_workers",
                        ",".join(worker for worker in generate_workers if worker is not None),
                    )
                for event in events:
                    event.execute(self.shared_storage)
            return events

    def listen_events(self, stop_event: threading.Event) -> None:
        channel = self.shared_storage.pubsub()
        channel.subscribe(TENANT_EVENTS_CHANNEL)
        # Redis Pub/Sub does not retain messages. Consume the server's
        # subscription acknowledgement before publishing readiness so a
        # launcher cannot race the first job:created event.
        while not stop_event.is_set():
            confirmation = channel.get_message(timeout=1.0)
            if confirmation is not None and confirmation["type"] == "subscribe":
                break
        if stop_event.is_set():
            return
        self.shared_storage.set("weave:scheduler:ready", self.ready_token)
        for message in channel.listen():
            if stop_event.is_set():
                return
            if message["type"] != "message":
                continue
            raw_event = _decode(message["data"])
            if raw_event == "stop_server":
                stop_event.set()
                return
            event = EventParser.parse(raw_event or "")
            if event is None:
                logger.warning("Ignoring malformed WEAVE event: %r", raw_event)
                continue
            with self.lock:
                self.handle_event(event)

    def run(self, poll_interval: float = 0.02) -> None:
        stop_event = threading.Event()
        self.shared_storage.set("weave:scheduler:ready", "0")
        listener = threading.Thread(target=self.listen_events, args=(stop_event,), name="weave-listener")
        listener.start()
        try:
            while not stop_event.is_set():
                if self.schedule_once() is None:
                    time.sleep(poll_interval)
        except KeyboardInterrupt:
            self.shared_storage.publish(TENANT_EVENTS_CHANNEL, "stop_server")
        finally:
            stop_event.set()
            self.shared_storage.set("weave:scheduler:ready", "0")
            listener.join(timeout=5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the WEAVE-compatible local scheduler")
    parser.add_argument("--host", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SCHEDULER_PORT", "9969")))
    parser.add_argument("--start-redis", action="store_true")
    parser.add_argument("--num-jobs", type=int, default=int(os.environ.get("N_JOB", "1")))
    parser.add_argument("--num-gen-gpus", type=int, default=int(os.environ.get("NG", "1")))
    parser.add_argument("--num-train-gpus", type=int, default=int(os.environ.get("NT", "1")))
    parser.add_argument("--gen-affinity", default=os.environ.get("GDA", "NVIDIA H20"))
    parser.add_argument("--train-affinity", default=os.environ.get("TDA", "NVIDIA H800"))
    parser.add_argument("--mapping", default=os.environ.get("MAPFN", "mapping.txt"))
    parser.add_argument("--placements", help="Optional JSON job -> {gen, train} physical GPU mappings")
    parser.add_argument("--ratio", type=float, default=float(os.environ.get("RATIO", "1.0")))
    parser.add_argument("--colo", action="store_true", default=os.environ.get("COLO", "0") == "1")
    parser.add_argument("--ready-token", default="1")
    parser.add_argument("--arrival-schedule", type=Path, help="Opt-in preinitialized benchmark logical arrivals")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("Install the WEAVE control-plane dependency with `pip install redis`.") from exc

    redis_process = None
    if args.start_redis:
        redis_process = subprocess.Popen(["redis-server", "--bind", args.host, "--port", str(args.port), "--save", ""])
    storage = redis.Redis(host=args.host, port=args.port, db=0)
    while True:
        try:
            storage.ping()
            break
        except redis.ConnectionError as exc:
            if redis_process is not None and redis_process.poll() is not None:
                raise RuntimeError(
                    f"WEAVE Redis failed to start on {args.host}:{args.port} with exit code {redis_process.returncode}"
                ) from exc
            time.sleep(0.05)

    scheduler = Scheduler(
        storage,
        SchedulerConfig(
            num_jobs=args.num_jobs,
            gen_device_affinity=args.gen_affinity,
            gen_device_ids=list(range(args.num_gen_gpus)),
            train_device_affinity=args.train_affinity,
            train_device_ids=list(range(args.num_train_gpus)),
            mapping_file=args.mapping,
            min_concurrency_ratio=args.ratio,
            is_colo=args.colo,
            ready_token=args.ready_token,
            placements=json.loads(Path(args.placements).read_text()) if args.placements else None,
        ),
    )
    if args.arrival_schedule is not None:
        scheduler.select_policy = ArrivalPolicy(
            scheduler.select_policy,
            json.loads(args.arrival_schedule.read_text()),
            args.arrival_schedule.with_name("arrival-audit.jsonl"),
        )
    try:
        scheduler.run()
    finally:
        if redis_process is not None:
            redis_process.terminate()


if __name__ == "__main__":
    main()
