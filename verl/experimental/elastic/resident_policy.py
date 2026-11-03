"""Cost-gated DP changes for preinitialized resident tasks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

from .policy import Task, nonnegative


@dataclass(frozen=True)
class Forecast:
    completion: dict[str, float]

    @property
    def makespan(self):
        return max(self.completion.values(), default=0.0)

    @property
    def total_completion(self):
        return sum(self.completion.values())


def forecast(tasks, remaining, slots, *, expand=False, overhead=0.0):
    """Simulate finite G/T/sync phases, including both-pool weight-sync leases."""
    names = sorted(remaining)
    if any(remaining[name] < 0 for name in names):
        raise ValueError("Remaining steps cannot be negative")
    allocation = {name: {pool: tuple(ids) for pool, ids in slots[name].items()} for name in names}
    todo = dict(remaining)
    phase = {name: "gen" for name in names}
    ready = {name: overhead for name in names}
    running = {}
    owners = {"gen": {}, "train": {}}
    completed = {name: overhead for name in names if not todo[name]}
    now = overhead
    while len(completed) < len(names):
        candidates = sorted(
            (name for name in names if name not in completed and name not in running),
            key=lambda name: (phase[name] != "sync", ready[name], name),
        )
        for name in candidates:
            pools = ("gen",) if phase[name] in ("gen", "sync") else ("train",)
            if any(owners[pool].get(slot) not in (None, name) for pool in pools for slot in allocation[name][pool]):
                continue
            for pool in pools:
                for slot in allocation[name][pool]:
                    owners[pool][slot] = name
            task = tasks[name]
            duration = (
                task.switch_seconds
                if phase[name] == "sync"
                else getattr(task, "rollout" if phase[name] == "gen" else "train").seconds[
                    len(allocation[name][phase[name]])
                ]
            )
            running[name] = now + duration
        if not running:
            raise RuntimeError("Phase forecast deadlocked")
        now = min(running.values())
        finished = [name for name, end in running.items() if end == now]
        completed.update({name: now for name in finished if phase[name] == "sync" and todo[name] == 1})
        for name in finished:
            del running[name]
            old_phase = phase[name]
            if old_phase == "resize":
                phase[name] = "gen"
                ready[name] = now
                continue
            if old_phase != "train":
                pools = ("gen", "train") if old_phase == "sync" else ("gen",)
                for pool in pools:
                    owners[pool] = {slot: owner for slot, owner in owners[pool].items() if owner != name}
            ready[name] = now
            if old_phase == "gen":
                phase[name] = "train"
            elif old_phase == "train":
                phase[name] = "sync"
            else:
                todo[name] -= 1
                phase[name] = "gen"
                if not todo[name]:
                    completed[name] = now
                    continue
                if expand and all(peer == name or peer in completed for peer in names):
                    task = tasks[name]
                    target = {"gen": tuple(range(task.rollout.limit)), "train": tuple(range(task.train.limit))}
                    current_seconds = task.step_seconds((len(allocation[name]["gen"]), len(allocation[name]["train"])))
                    target_seconds = task.step_seconds((len(target["gen"]), len(target["train"])))
                    if (
                        target != allocation[name]
                        and todo[name] * (current_seconds - target_seconds) > task.reconfigure_seconds
                    ):
                        allocation[name] = target
                        # Reconfiguration occupies the safe boundary, before the next G phase.
                        ready[name] = now + task.reconfigure_seconds
                        running[name] = ready[name]
                        phase[name] = "resize"
    return Forecast(completed)


class ResidentPolicy:
    """One serialized resize at a time; no speculative lease acquisition."""

    @staticmethod
    def validate(config, physical_tasks=None):
        if config["mode"] not in ("benefit", "phase", "wait"):
            raise ValueError("Resident scheduling mode must be benefit, phase or wait")
        tasks = {raw["job_name"]: Task.parse(raw, command=["resident"]) for raw in config["jobs"]}
        if len(config["jobs"]) < 2 or len(tasks) != len(config["jobs"]):
            raise ValueError("Cost-gated resident scheduling requires at least two distinct predeclared tasks")
        if physical_tasks is not None:
            if set(tasks) != set(physical_tasks) or any(
                task.total_steps != physical_tasks[name].total_steps for name, task in tasks.items()
            ):
                raise ValueError("Resident policy profiles must match all physical tasks and step counts")
            for name, task in tasks.items():
                physical = physical_tasks[name]
                if task.rollout.limit > physical.rollout.limit or task.train.limit > physical.train.limit:
                    raise ValueError(f"{name}: logical DP limit exceeds its prepared physical capacity")
        arrivals = {
            raw["job_name"]: nonnegative(raw.get("arrival_seconds", 0), "resident arrival_seconds")
            for raw in config["jobs"]
        }
        if min(arrivals.values()) != 0:
            raise ValueError("At least one resident task must arrive at the ready epoch")
        margin = nonnegative(config.get("margin_seconds", 10), "resident margin_seconds")
        return tasks, arrivals, margin

    def __init__(self, config, storage, *, audit_path: Path | None = None):
        self.tasks, self.arrivals, self.margin = self.validate(config)
        self.mode = config["mode"]
        self.storage = storage
        self.audit_path = audit_path
        self.started = None
        self.admitted = set()
        self.handled = set()
        self.inflight = None
        for name in self.tasks:
            storage.set(f"{name}:resident:admitted", "0")

    def _read(self, name, suffix):
        raw = self.storage.get(f"{name}:resident:{suffix}")
        return json.loads(raw) if raw is not None else None

    def _completed(self, name):
        return self.storage.get(f"{name}:status") in ("completed", b"completed")

    def _admit(self, name):
        if name not in self.admitted:
            self.admitted.add(name)
            self.storage.set(f"{name}:resident:admitted", "1")
            self._audit({"action": "admitted", "task": name})

    def _audit(self, record):
        if self.audit_path is not None:
            with self.audit_path.open("a") as stream:
                stream.write(json.dumps({"time": time.time(), **record}) + "\n")

    def _reply(self, name, packet, target=None, **details):
        decision = {
            "revision": packet["revision"],
            "completed_step": packet["completed_step"],
            "id": f"benefit-{packet['completed_step']}",
            "action": "resize" if target is not None else "continue",
            **details,
        }
        if target is not None:
            decision.update(
                train_slots=list(target["train"]),
                rollout_slots=list(target["gen"]),
                train_dp=len(target["train"]),
                rollout_dp=len(target["gen"]),
            )
            self.inflight = (name, target)
        self.storage.set(f"{name}:resident:decision", json.dumps(decision))
        self.handled.add((name, packet["completed_step"]))
        self._audit({"task": name, **decision})

    def tick(self, num_uninited_jobs):
        if num_uninited_jobs:
            return
        if self.started is None:
            self.started = time.monotonic()
        elapsed = time.monotonic() - self.started
        visible = [name for name in self.tasks if self.arrivals[name] <= elapsed and not self._completed(name)]
        if self.inflight is not None:
            name, target = self.inflight
            if self._read(name, "allocation") == {pool: list(ids) for pool, ids in target.items()}:
                self.inflight = None
                for peer in visible:
                    self._admit(peer)
        active = sorted(name for name in self.admitted if not self._completed(name))
        for name in visible:
            if name not in self.admitted and (not active or self.mode == "phase"):
                self._admit(name)
                active.append(name)
        packets = {name: self._read(name, "boundary") for name in active}
        for name, packet in packets.items():
            if packet is None or (name, packet["completed_step"]) in self.handled:
                continue
            task = self.tasks[name]
            remaining = task.total_steps - packet["completed_step"]
            if remaining < 0:
                raise ValueError("Resident progress exceeds the declared workload")
            if not remaining or self.mode != "benefit" or self.inflight is not None:
                self._reply(name, packet)
                continue
            peers = [peer for peer in visible if peer != name]
            current = self._read(name, "allocation")
            if not peers:
                target = {"gen": list(range(task.rollout.limit)), "train": list(range(task.train.limit))}
                old = task.step_seconds((len(current["gen"]), len(current["train"])))
                new = task.step_seconds((len(target["gen"]), len(target["train"])))
                gain = remaining * (old - new) - task.reconfigure_seconds
                self._reply(
                    name, packet, target if gain > self.margin else None, reason="expand", predicted_saving=gain
                )
                continue
            allocations = {name: current}
            steps = {name: remaining}
            for peer in peers:
                other_packet = self._read(peer, "boundary")
                peer_remaining = self.tasks[peer].total_steps - (other_packet["completed_step"] if other_packet else 0)
                if peer_remaining < 0:
                    raise ValueError("Resident peer progress exceeds its declared workload")
                if peer_remaining:
                    allocations[peer] = self._read(peer, "allocation")
                    if allocations[peer] is None:
                        raise RuntimeError("Resident peer has not published its initial allocation")
                    steps[peer] = peer_remaining
            if len(steps) == 1:
                self._reply(name, packet)
                for peer in peers:
                    self._admit(peer)
                continue
            baseline = forecast(self.tasks, steps, allocations)
            candidates = []
            memberships = [
                [
                    list(slots)
                    for count in component.seconds
                    if count <= len(current[pool])
                    for slots in combinations(sorted(current[pool]), count)
                ]
                for pool, component in (("gen", task.rollout), ("train", task.train))
            ]
            for gen_slots, train_slots in product(*memberships):
                target = {"gen": gen_slots, "train": train_slots}
                if target == current:
                    continue
                changed = forecast(
                    self.tasks,
                    steps,
                    {**allocations, name: target},
                    expand=True,
                    overhead=task.reconfigure_seconds,
                )
                deadline_ok = all(
                    self.tasks[job].deadline is None or elapsed + seconds <= self.tasks[job].deadline
                    for job, seconds in changed.completion.items()
                )
                slo_ok = all(
                    seconds
                    <= self.tasks[job].slo
                    * steps[job]
                    * (
                        min(self.tasks[job].rollout.seconds.values())
                        + min(self.tasks[job].train.seconds.values())
                        + self.tasks[job].switch_seconds
                    )
                    for job, seconds in changed.completion.items()
                )
                gain = baseline.makespan - changed.makespan
                accept = (
                    gain > self.margin
                    and changed.total_completion <= baseline.total_completion
                    and deadline_ok
                    and slo_ok
                    and packet["completed_step"] > 0
                )
                candidates.append((accept, gain, -changed.total_completion, target, changed, deadline_ok, slo_ok))
            if not candidates:
                self._reply(name, packet)
                for peer in peers:
                    self._admit(peer)
                continue
            accept, gain, _, target, changed, deadline_ok, slo_ok = max(candidates, key=lambda row: row[:3])
            self._reply(
                name,
                packet,
                target if accept else None,
                reason="shrink-admit",
                predicted_saving=gain,
                baseline_completion=baseline.completion,
                candidate_completion=changed.completion,
                deadline_ok=deadline_ok,
                slo_ok=slo_ok,
            )
            if not accept:
                for peer in peers:
                    self._admit(peer)
