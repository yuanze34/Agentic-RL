"""Online control plane with checkpointed, affected-group-only restarts."""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .policy import Decision, Group, Policy, Task
from .storage import read_json, write_json

logger = logging.getLogger(__name__)


def resident_enabled(overrides: tuple[str, ...]) -> bool:
    values = [
        arg.partition("=")[2].lower() for arg in overrides if arg.startswith("trainer.v1.elastic.resident.enabled=")
    ]
    if values and values[-1] not in ("true", "false"):
        raise ValueError("Resident enabled must be a boolean Hydra override")
    return bool(values and values[-1] == "true")


def check_port_available(port: int) -> None:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


class Backend(Protocol):
    def start(self, group: Group, tasks: dict[str, Task], checkpoints: dict[str, str]) -> None: ...
    def pause(self, name: str) -> dict[str, dict]: ...
    def stop(self, name: str) -> None: ...
    def states(self, name: str) -> dict[str, dict]: ...
    def finished(self, group: str, task: str) -> bool: ...
    def wait_ready(self, name: str) -> None: ...
    def update_profiles(self, group: Group, tasks: dict[str, Task]) -> None: ...


@dataclass
class GroupRun:
    revision: str
    directory: Path
    jobs: dict[str, subprocess.Popen] = field(default_factory=dict)
    services: list[subprocess.Popen] = field(default_factory=list)
    pools: list[Any] = field(default_factory=list)
    logs: list[Any] = field(default_factory=list)


class RayBackend:
    """Own only this run's processes and PGs on an existing single-node cluster."""

    def __init__(
        self,
        output: Path,
        ray_address: str,
        policy: Policy,
        start_unix: float,
        *,
        base_port: int = 20969,
        timeout: float = 1800,
        resident_config: dict | None = None,
    ):
        self.output = output
        self.ray_address = ray_address
        self.policy = policy
        self.start_unix = start_unix
        self.base_port = base_port
        self.timeout = timeout
        self.resident_config = resident_config
        self.groups: dict[str, GroupRun] = {}

    def _spawn(self, run: GroupRun, command: list[str], log_path: Path) -> subprocess.Popen:
        stream = log_path.open("w")
        run.logs.append(stream)
        return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)

    def update_profiles(self, group: Group, tasks: dict[str, Task]) -> None:
        path = self.groups[group.name].directory / "metadata.json"
        metadata = read_json(path) if path.exists() else {}
        for name, allocation in group.allocations.items():
            task = tasks[name]
            metadata[name] = {
                **metadata.get(name, {}),
                "deadline_unix": self.start_unix + task.deadline if task.deadline is not None else None,
                "rollout_seconds": task.rollout.seconds[allocation[0]],
                "train_seconds": task.train.seconds[allocation[1]],
                "reconfigure_seconds": task.reconfigure_seconds,
                "switch_seconds": task.switch_seconds,
            }
        write_json(path, metadata)

    def start(self, group: Group, tasks: dict[str, Task], checkpoints: dict[str, str]) -> None:
        import redis

        from verl.experimental.weave.launch import _wait_ready, job_command
        from verl.experimental.weave.placement import SharedPoolSpec, create_detached_pool

        if group.name in self.groups:
            raise ValueError(f"Group already running: {group.name}")
        port = self.base_port + int(group.name.removeprefix("Elastic-"))
        check_port_available(port)
        revision = uuid.uuid4().hex
        directory = self.output / "runs" / group.name / revision
        directory.mkdir(parents=True)
        run = GroupRun(revision, directory)
        plan = group.plan["groups"][0]
        resident = any(resident_enabled(tasks[name].overrides) for name in group.allocations)
        if resident:
            if not all(resident_enabled(tasks[name].overrides) for name in group.allocations):
                raise ValueError("Resident and cold tasks cannot share an elastic group")
            if any(
                component.request != component.limit
                for name in group.allocations
                for component in (tasks[name].rollout, tasks[name].train)
            ):
                raise ValueError(
                    "Resident mode currently reserves a fixed request=limit physical lease; "
                    "active DP is controlled by resident resize commands"
                )
        self.groups[group.name] = run
        tenants = max(2, len(group.allocations))
        try:
            placements = []
            for slot in range(plan["num_rollout_slots"]):
                name = f"elastic_{revision}_rollout_{slot}"
                pool = create_detached_pool(SharedPoolSpec(name, self.policy.rollout_width, tenants))
                run.pools.append(pool)
                placements.append({"name": name, "id": pool.id.hex()})
            train_name = f"elastic_{revision}_train"
            train_pool = create_detached_pool(SharedPoolSpec(train_name, plan["num_train_gpus"], tenants))
            run.pools.append(train_pool)
            training = {"name": train_name, "id": train_pool.id.hex(), "num_gpus": plan["num_train_gpus"]}
            mapping_path = directory / "mapping.txt"
            mapping_path.write_text("".join(f"{name} 0 0\n" for name in group.allocations))
            placement_path = directory / "placements.json"
            write_json(
                placement_path, {job["job_name"]: {"gen": job["gen"], "train": job["train"]} for job in plan["jobs"]}
            )
            metadata_path = directory / "metadata.json"
            self.update_profiles(group, tasks)
            policy_arguments = []
            if self.resident_config is not None:
                from .resident_policy import ResidentPolicy

                if not resident:
                    raise ValueError("Cost-aware resident policy cannot control cold tasks")
                ResidentPolicy.validate(self.resident_config, {name: tasks[name] for name in group.allocations})
                resident_path = directory / "resident-policy.json"
                write_json(resident_path, self.resident_config)
                policy_arguments = ["--resident-policy", str(resident_path)]
            server = self._spawn(
                run,
                [
                    "redis-server",
                    "--bind",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--save",
                    "",
                    "--appendonly",
                    "no",
                ],
                directory / "redis.log",
            )
            run.services.append(server)
            storage = redis.Redis(host="127.0.0.1", port=port, decode_responses=True)
            try:

                def reachable():
                    try:
                        return storage.ping()
                    except redis.ConnectionError:
                        return False

                _wait_ready(server, reachable, "elastic Redis")
                scheduler = self._spawn(
                    run,
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "verl.experimental.elastic.local",
                        "--metadata",
                        str(metadata_path),
                        "--mapping",
                        str(mapping_path),
                        "--placements",
                        str(placement_path),
                        "--port",
                        str(port),
                        "--gen-gpus",
                        str(plan["num_gen_gpus"]),
                        "--train-gpus",
                        str(plan["num_train_gpus"]),
                        *policy_arguments,
                    ],
                    directory / "scheduler.log",
                )
                run.services.append(scheduler)
                _wait_ready(scheduler, lambda: storage.get("weave:scheduler:ready") == "1", "elastic phase scheduler")
            finally:
                storage.close()
            for specification in plan["jobs"]:
                name = specification["job_name"]
                task = tasks[name]
                enriched = {**specification, "command": list(task.command), "overrides": list(task.overrides)}
                command = job_command(
                    enriched,
                    plan,
                    slot_width=self.policy.rollout_width,
                    rollout_placements=placements,
                    train_placement=training,
                    train_placements=[training],
                    port=port,
                    ray_address=self.ray_address,
                    migration=False,
                    run_id=revision,
                )
                command = [
                    f"trainer.v1.trainer_mode={'elastic_resident_separate' if resident else 'elastic_separate'}"
                    if arg == "trainer.v1.trainer_mode=weave_separate"
                    else arg
                    for arg in command
                ]
                command = [
                    f"+ray_kwargs.ray_init.namespace=verl_elastic_{revision}_{name}"
                    if arg.startswith("+ray_kwargs.ray_init.namespace=")
                    else arg
                    for arg in command
                ]
                control = self.output / "control" / name
                control.mkdir(parents=True, exist_ok=True)
                command.extend(
                    [
                        f"trainer.v1.elastic.control_dir={control}",
                        f'trainer.v1.elastic.revision="{revision}"',
                        f"trainer.total_training_steps={task.total_steps}",
                        f"trainer.default_local_dir={self.output / 'checkpoints' / name}",
                        "trainer.del_local_ckpt_after_load=False",
                        "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]",
                        "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]",
                        "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1",
                        f"trainer.resume_mode={'resume_path' if name in checkpoints else 'disable'}",
                    ]
                )
                if self.resident_config is not None:
                    command.append("trainer.v1.elastic.resident.cost_aware=True")
                if name in checkpoints:
                    command.append(f"trainer.resume_from_path={checkpoints[name]}")
                write_json(directory / f"{name}.command.json", command)
                run.jobs[name] = self._spawn(run, command, directory / f"{name}.log")
            write_json(directory / "plan.json", group.plan)
        except BaseException:
            logger.exception("Failed to start elastic group %s", group.name)
            self.stop(group.name)
            raise

    def states(self, name: str) -> dict[str, dict]:
        run = self.groups[name]
        states = {}
        for task, process in run.jobs.items():
            path = self.output / "control" / task / "state.json"
            if path.exists():
                state = read_json(path)
                if state["revision"] == run.revision:
                    states[task] = state
            if process.poll() not in (None, 0):
                raise RuntimeError(f"Elastic task {task} exited {process.returncode}; see {run.directory}")
            if process.poll() == 0 and task not in states:
                raise RuntimeError(f"Elastic task {task} exited without a status acknowledgement")
        for service in run.services:
            if service.poll() is not None:
                raise RuntimeError(f"Elastic control service exited {service.returncode}; see {run.directory}")
        return states

    def finished(self, group: str, task: str) -> bool:
        return self.groups[group].jobs[task].poll() == 0

    def wait_ready(self, name: str) -> None:
        deadline = time.monotonic() + self.timeout
        while True:
            states = self.states(name)
            if len(states) == len(self.groups[name].jobs) and all(
                state["status"] in ("ready", "running", "completed") for state in states.values()
            ):
                return
            if any(state["status"] == "failed" for state in states.values()):
                raise RuntimeError(f"Elastic group {name} failed initialization")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Elastic group {name} did not become ready")
            time.sleep(0.1)

    def pause(self, name: str) -> dict[str, dict]:
        run = self.groups[name]
        for task in run.jobs:
            write_json(
                self.output / "control" / task / "command.json",
                {
                    "revision": run.revision,
                    "action": "pause",
                },
            )
        deadline = time.monotonic() + self.timeout
        while True:
            states = self.states(name)
            if len(states) == len(run.jobs) and all(
                state["status"] in ("paused", "completed") and self.finished(name, task)
                for task, state in states.items()
            ):
                if any(state["status"] == "paused" and not state["checkpoint"] for state in states.values()):
                    raise RuntimeError(f"Elastic group {name} acknowledged pause without a checkpoint")
                return states
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Elastic group {name} did not checkpoint and pause")
            time.sleep(0.1)

    def stop(self, name: str) -> None:
        from ray.util.placement_group import remove_placement_group

        from verl.experimental.weave.launch import _stop

        run = self.groups.pop(name)
        try:
            for process in [*run.jobs.values(), *reversed(run.services)]:
                _stop(process)
        finally:
            for pool in run.pools:
                remove_placement_group(pool)
            for stream in run.logs:
                stream.close()


class Controller:
    def __init__(self, policy: Policy, backend: Backend, output: Path, *, started: float | None = None):
        self.policy = policy
        self.backend = backend
        self.output = output
        self.started = time.monotonic() if started is None else started
        self.checkpoints: dict[str, str] = {}
        self.results: dict[str, dict] = {}
        self.submitted: dict[str, float] = {}
        self.last_reconfiguration = float("-inf")
        self.event_path = output / "events.jsonl"

    def event(self, action: str, **data) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("a") as stream:
            import json

            stream.write(json.dumps({"seconds": time.monotonic() - self.started, "action": action, **data}) + "\n")
        logger.info("ELASTIC %s %s", action, data)

    def submit(self, task: Task, submitted_seconds: float | None = None) -> None:
        if self.policy.groups and (
            resident_enabled(task.overrides)
            or any(
                resident_enabled(self.policy.tasks[name].overrides)
                for group in self.policy.groups.values()
                for name in group.allocations
            )
        ):
            raise ValueError("Resident tasks must currently be submitted together before the shared group starts")
        self.policy.submit(task)
        self.submitted[task.name] = (
            submitted_seconds if submitted_seconds is not None else time.monotonic() - self.started
        )
        self.event("submitted", task=task.name)

    def apply(self, decision: Decision) -> None:
        old = dict(self.policy.groups)
        if any(
            resident_enabled(self.policy.tasks[task].overrides)
            for name in decision.affected
            if name in old
            for task in old[name].allocations
        ):
            raise ValueError("Running resident groups resize in place; cold controller reconfiguration is unsupported")
        target = dict(decision.groups)
        completed = set()
        started = []
        stopped = []
        quiesced = []
        self.event(
            "reconfigure_begin",
            reason=decision.reason,
            affected=list(decision.affected),
            penalty_seconds=decision.penalty_seconds,
        )
        try:
            for name in decision.affected:
                if name not in old:
                    continue
                states = self.backend.pause(name)
                quiesced.append(name)
                for task, state in states.items():
                    self.policy.progress(task, state["completed_steps"])
                    if state["status"] == "completed":
                        if state["completed_steps"] != self.policy.tasks[task].total_steps:
                            raise RuntimeError(f"Task {task} completed without its full workload")
                        completed.add(task)
                        self._complete(task)
                    else:
                        self.checkpoints[task] = state["checkpoint"]
                        self.event(
                            "checkpointed", task=task, step=state["completed_steps"], checkpoint=state["checkpoint"]
                        )
            for name in decision.affected:
                if name in old:
                    self.backend.stop(name)
                    stopped.append(name)
                    self.event("physical_group_released", group=name)
            for name in decision.affected:
                if name not in target:
                    continue
                allocation = {task: value for task, value in target[name].allocations.items() if task not in completed}
                if not allocation:
                    del target[name]
                    continue
                group = self.policy.group(name, allocation)
                if group is None:
                    raise RuntimeError(f"Reconfigured group {name} no longer satisfies its profile SLOs")
                target[name] = group
                self.backend.start(group, self.policy.tasks, self.checkpoints)
                started.append(name)
            for name in started:
                self.backend.wait_ready(name)
            self.policy.commit(Decision(target, decision.affected, decision.reason, decision.penalty_seconds))
        except BaseException as error:
            logger.exception("Elastic reconfiguration failed; restoring affected groups")
            self.event("reconfigure_failed", reason=decision.reason)
            for name in started:
                self.backend.stop(name)
            for name in quiesced:
                if name not in stopped:
                    self.backend.stop(name)
                    stopped.append(name)
            for name in stopped:
                allocation = {task: value for task, value in old[name].allocations.items() if task not in completed}
                if allocation:
                    group = self.policy.group(name, allocation)
                    if group is None:
                        raise RuntimeError(f"Cannot rebuild rollback group {name}") from error
                    self.backend.start(group, self.policy.tasks, self.checkpoints)
                    self.backend.wait_ready(name)
                    old[name] = group
                else:
                    del old[name]
            self.policy.groups = old
            self.event("rollback_complete", affected=stopped)
            raise
        self.last_reconfiguration = time.monotonic()
        self.event(
            "reconfigure_committed",
            reason=decision.reason,
            groups={
                name: {"allocations": group.allocations, "resources": group.resources} for name, group in target.items()
            },
        )

    def _complete(self, task: str) -> None:
        if task not in self.results:
            self.results[task] = {
                "status": "completed",
                "completed_steps": self.policy.tasks[task].total_steps,
                "completion_seconds": time.monotonic() - self.started,
                "submitted_seconds": self.submitted[task],
            }
            self.results[task]["end_to_end_seconds"] = (
                self.results[task]["completion_seconds"] - self.results[task]["submitted_seconds"]
            )
            self.event("completed", task=task)

    def refresh(self) -> None:
        for group in list(self.policy.groups.values()):
            states = self.backend.states(group.name)
            for task, state in states.items():
                if task not in group.allocations:
                    continue
                durations = (
                    state.get("durations") if state["completed_steps"] > self.policy.completed_steps[task] else None
                )
                resident = resident_enabled(self.policy.tasks[task].overrides)
                self.policy.progress(task, state["completed_steps"], None if resident else durations)
                if durations and not resident:
                    self.backend.update_profiles(group, self.policy.tasks)
                if state["status"] == "failed":
                    raise RuntimeError(f"Elastic task {task} reported failure")
                if state["status"] == "completed" and self.backend.finished(group.name, task):
                    if state["completed_steps"] != self.policy.tasks[task].total_steps:
                        raise RuntimeError(f"Elastic task {task} completed with missing training steps")
                    self._complete(task)
                    self.policy.retire(task)
            current = self.policy.groups[group.name]
            if not current.allocations:
                self.backend.stop(group.name)
                del self.policy.groups[group.name]
                self.event("physical_group_released", group=group.name)

    def schedule(self, *, min_hold_seconds: float = 30) -> None:
        now = time.monotonic() - self.started
        if self.policy.pending and not self.policy.groups:
            tasks = [self.policy.tasks[name] for name in self.policy.pending]
            if any(resident_enabled(task.overrides) for task in tasks):
                if not all(resident_enabled(task.overrides) for task in tasks):
                    raise ValueError("Batch resident admission cannot mix cold and resident tasks")
                name = f"Elastic-{self.policy.next_group}"
                group = self.policy.group(name, {task.name: task.initial for task in tasks})
                if group is None or not self.policy.fits({name: group}):
                    raise ValueError("The initial resident task batch must fit a single shared physical group")
                self.apply(Decision({name: group}, (name,), "resident:batch-admit", 0))
                return
        if self.policy.groups and all(
            resident_enabled(self.policy.tasks[name].overrides)
            for group in self.policy.groups.values()
            for name in group.allocations
        ):
            # Logical DP decisions belong to the in-place phase scheduler, not the cold backend.
            return
        for task in self.policy.pending.copy():
            decision = self.policy.admit(task, now)
            if decision is not None:
                self.apply(decision)
                return
        if time.monotonic() - self.last_reconfiguration >= min_hold_seconds:
            decision = self.policy.expand(now) or self.policy.redistribute(now)
            if decision is not None:
                self.apply(decision)

    def cancel(self, task: str) -> None:
        if task not in self.policy.tasks or task in self.results:
            raise ValueError(f"Task is unknown or already finished: {task}")
        if task in self.policy.pending:
            self.policy.retire(task)
        else:
            group = next(group for group in self.policy.groups.values() if task in group.allocations)
            allocations = {key: value for key, value in group.allocations.items() if key != task}
            target = dict(self.policy.groups)
            if allocations:
                new_group = self.policy.group(group.name, allocations)
                if new_group is None:
                    raise RuntimeError(f"Cannot reconfigure cancelled task's group {group.name}")
                target[group.name] = new_group
            else:
                del target[group.name]
            self.apply(Decision(target, (group.name,), f"cancel:{task}", 0))
        if task not in self.results:
            self.results[task] = {"status": "cancelled", "completed_steps": self.policy.completed_steps[task]}
        self.event("cancelled", task=task)

    def snapshot(self) -> dict:
        return {
            "groups": {
                name: {"allocations": group.allocations, "resources": group.resources}
                for name, group in self.policy.groups.items()
            },
            "pending": self.policy.pending,
            "completed_steps": self.policy.completed_steps,
            "results": self.results,
            "checkpoints": self.checkpoints,
        }
