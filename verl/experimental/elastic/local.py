"""Deadline/aging-aware phase selection using the existing resource protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import redis

from verl.experimental.elastic.resident_policy import ResidentPolicy
from verl.experimental.elastic.storage import read_json
from verl.experimental.weave.protocol import EventLevel, Phase
from verl.experimental.weave.scheduler import FCFSPolicy, Scheduler, SchedulerConfig


class PhasePolicy(FCFSPolicy):
    def __init__(
        self,
        storage,
        metadata: dict,
        max_wait_seconds: float = 30,
        metadata_path: Path | None = None,
        resident_policy: ResidentPolicy | None = None,
    ):
        super().__init__(storage, 1.0)
        self.metadata = metadata
        self.max_wait_seconds = max_wait_seconds
        self.arrivals: dict[tuple[str, Phase | None], float] = {}
        self.metadata_path = metadata_path
        self.resident_capacities = {}
        self.resident_policy = resident_policy

    def _resident_control(self, resources):
        pending = []
        for name in resources.job_mapping:
            packet = self.shared_storage.get(f"{name}:resident:request")
            if packet is not None:
                request = json.loads(packet)
                if self.shared_storage.get(f"{name}:resident:ack") != request["id"]:
                    pending.append(name)
        if len(pending) > 1:
            raise RuntimeError(
                "Concurrent resident resize requests are not supported; serialize requests to avoid hold-and-wait"
            )
        for name, status in resources.job_mapping.items():
            allocation = self.shared_storage.get(f"{name}:resident:allocation")
            if allocation is None:
                continue
            capacities = self.resident_capacities.setdefault(
                name, {"gen": set(status.gen_gpu_list), "train": set(status.train_gpu_list)}
            )
            target = json.loads(allocation)
            for pool in ("gen", "train"):
                slots = target[pool]
                if (
                    not slots
                    or len(set(slots)) != len(slots)
                    or any(type(slot) is not int or slot not in capacities[pool] for slot in slots)
                ):
                    raise ValueError(f"Invalid resident {pool} slots for {name}: {slots}")
            if not status.allocated_gpus and status.phase == Phase.GENERATE:
                status.gen_gpu_list = target["gen"]
                status.train_gpu_list = target["train"]
            request_raw = self.shared_storage.get(f"{name}:resident:request")
            if request_raw is None:
                continue
            request = json.loads(request_raw)
            if self.shared_storage.get(f"{name}:resident:ack") == request["id"]:
                continue
            if status.phase != Phase.TRAIN:
                raise ValueError("Resident resize lease requests require TRAIN admission")
            # The prior UPDATE release must settle before reserving any new rollout slot.
            if any(owner == name for owner in resources.gen_resource_mapping.values()):
                continue
            wanted = {}
            for pool in ("gen", "train"):
                slots = request[pool]
                if (
                    not slots
                    or len(set(slots)) != len(slots)
                    or any(type(slot) is not int or slot not in capacities[pool] for slot in slots)
                ):
                    raise ValueError(f"Invalid resident resize {pool} slots for {name}")
                wanted[pool] = [resources.global_device(name, pool, slot) for slot in slots]
            if any(
                mapping[device] not in (None, name)
                for pool, mapping in (
                    ("gen", resources.gen_resource_mapping),
                    ("train", resources.train_resource_mapping),
                )
                for device in wanted[pool]
            ):
                continue
            for pool, mapping, available in (
                ("gen", resources.gen_resource_mapping, resources.gen_available_devices),
                ("train", resources.train_resource_mapping, resources.train_available_devices),
            ):
                for device in wanted[pool]:
                    if mapping[device] is None:
                        mapping[device] = name
                        available.remove(device)
                        status.allocated_gpus.append(device)
            self.shared_storage.set(f"{name}:resident:ack", request["id"])

    def __call__(self, event_queue, resource_manager, num_uninited_jobs):
        if self.resident_policy is not None:
            self.resident_policy.tick(num_uninited_jobs)
        self._resident_control(resource_manager)
        if self.metadata_path is not None and event_queue:
            self.metadata = read_json(self.metadata_path)
        now = time.time()
        present = {(event.job_name, event.phase) for event in event_queue}
        self.arrivals = {key: value for key, value in self.arrivals.items() if key in present}
        for key in present:
            self.arrivals.setdefault(key, now)

        def priority(event):
            if event.level == EventLevel.JOB:
                return (0, 0, 0, 0)
            if event.phase == Phase.UPDATE:
                return (1, 0, 0, 0)
            task = self.metadata[event.job_name]
            arrived = self.arrivals[(event.job_name, event.phase)]
            if now - arrived >= self.max_wait_seconds:
                return (2, arrived, 0, 0)
            duration = task["rollout_seconds"] if event.phase == Phase.GENERATE else task["train_seconds"]
            deadline = task["deadline_unix"]
            slack = deadline - now if deadline is not None else float("inf")
            return (3, slack, duration + task["switch_seconds"], arrived)

        event_queue.sort(key=priority)
        blocked = [
            event
            for event in event_queue
            if event.phase == Phase.GENERATE
            and self.shared_storage.get(f"{event.job_name}:resident:admitted") in ("0", b"0")
        ]
        eligible = [event for event in event_queue if event not in blocked]
        result = super().__call__(eligible, resource_manager, num_uninited_jobs)
        event_queue[:] = eligible + blocked
        if result is not None and self.resident_capacities and self.metadata_path is not None:
            with self.metadata_path.with_name("resident-ownership.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "events": [
                                {"job": event.job_name, "phase": event.phase.value if event.phase else None}
                                for event in result
                            ],
                            "resources": resource_manager.snapshot(),
                        }
                    )
                    + "\n"
                )
        remaining = {(event.job_name, event.phase) for event in event_queue}
        self.arrivals = {key: value for key, value in self.arrivals.items() if key in remaining}
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--gen-gpus", type=int, required=True)
    parser.add_argument("--train-gpus", type=int, required=True)
    parser.add_argument("--resident-policy", type=Path)
    args = parser.parse_args()
    metadata = read_json(args.metadata)
    storage = redis.Redis(host=args.host, port=args.port, decode_responses=True)
    scheduler = Scheduler(
        storage,
        SchedulerConfig(
            num_jobs=len(metadata),
            gen_device_affinity="WEAVE_GEN",
            train_device_affinity="WEAVE_TRAIN",
            gen_device_ids=list(range(args.gen_gpus)),
            train_device_ids=list(range(args.train_gpus)),
            mapping_file=str(args.mapping),
            placements=read_json(args.placements),
        ),
    )
    resident_policy = (
        ResidentPolicy(
            read_json(args.resident_policy),
            storage,
            audit_path=args.metadata.with_name("resident-decisions.jsonl"),
        )
        if args.resident_policy is not None
        else None
    )
    scheduler.select_policy = PhasePolicy(
        storage, metadata, metadata_path=args.metadata, resident_policy=resident_policy
    )
    scheduler.run()


if __name__ == "__main__":
    main()
