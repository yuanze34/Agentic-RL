"""Profile-driven admission and local DP resource reconfiguration."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any

from verl.experimental.weave.planner import build_plan


def positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class Component:
    request: int
    limit: int
    seconds: dict[int, float]

    @classmethod
    def parse(cls, raw: dict, name: str) -> Component:
        request = integer(raw["request"], f"{name}.request")
        limit = integer(raw["limit"], f"{name}.limit")
        if limit < request:
            raise ValueError(f"{name}.limit cannot be below request")
        durations = {}
        for key, value in raw["seconds"].items():
            count = int(key)
            if str(count) != str(key) or not request <= count <= limit:
                raise ValueError(f"{name}.seconds contains an invalid DP count: {key}")
            durations[count] = positive(value, f"{name}.seconds[{count}]")
        if request not in durations or limit not in durations:
            raise ValueError(f"{name} requires measured durations at request and limit")
        return cls(request, limit, durations)

    def capped(self, current: int, cap: int) -> int:
        return max(count for count in self.seconds if count <= min(current, cap))


@dataclass(frozen=True)
class Task:
    name: str
    rollout: Component
    train: Component
    total_steps: int
    slo: float
    command: tuple[str, ...]
    overrides: tuple[str, ...] = ()
    initial: tuple[int, int] = (1, 1)
    deadline: float | None = None
    reconfigure_seconds: float = 0
    switch_seconds: float = 0

    @classmethod
    def parse(cls, raw: dict, *, command: list[str] | None = None, overrides: list[str] | None = None) -> Task:
        name = raw["job_name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise ValueError("job_name must be a safe, nonempty identifier")
        rollout = Component.parse(raw["rollout"], f"{name}.rollout")
        train = Component.parse(raw["train"], f"{name}.train")
        initial = tuple(raw.get("initial_replicas", [rollout.request, train.request]))
        if (
            len(initial) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in initial)
            or initial[0] not in rollout.seconds
            or initial[1] not in train.seconds
        ):
            raise ValueError(f"{name}.initial_replicas must be profiled DP counts")
        slo = positive(raw["slo"], f"{name}.slo")
        if slo <= 1:
            raise ValueError("slo must exceed 1")
        argv = raw.get("command", command)
        local_settings = raw.get("overrides", [])
        if not isinstance(local_settings, list) or not isinstance(overrides or [], list):
            raise ValueError(f"{name}.overrides must be argv lists")
        settings = [*(overrides or []), *local_settings]
        if not argv or not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise ValueError(f"{name}.command must be a nonempty argv list")
        if not all(isinstance(item, str) for item in settings):
            raise ValueError(f"{name}.overrides must be strings")
        deadline = raw.get("deadline")
        if deadline is not None:
            deadline = positive(deadline, f"{name}.deadline")
        overhead = nonnegative(raw["reconfigure_seconds"], "reconfigure_seconds")
        switch = nonnegative(raw["switch_seconds"], "switch_seconds")
        return cls(
            name,
            rollout,
            train,
            integer(raw["total_steps"], f"{name}.total_steps"),
            slo,
            tuple(argv),
            tuple(settings),
            initial,
            deadline,
            float(overhead),
            switch,
        )

    def step_seconds(self, allocation: tuple[int, int]) -> float:
        return self.rollout.seconds[allocation[0]] + self.train.seconds[allocation[1]]


@dataclass(frozen=True)
class Group:
    name: str
    allocations: dict[str, tuple[int, int]]
    plan: dict

    @property
    def resources(self) -> tuple[int, int]:
        return self.plan["num_gen_gpus"], self.plan["num_train_gpus"]


@dataclass(frozen=True)
class Decision:
    groups: dict[str, Group]
    affected: tuple[str, ...]
    reason: str
    penalty_seconds: float


@dataclass
class Policy:
    rollout_gpus: int
    train_gpus: int
    rollout_width: int = 1
    train_width: int = 1
    max_group_size: int = 5
    simulate_steps: int = 30
    tasks: dict[str, Task] = field(default_factory=dict)
    groups: dict[str, Group] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)
    completed_steps: dict[str, int] = field(default_factory=dict)
    next_group: int = 0

    def __post_init__(self):
        for name in ("rollout_gpus", "train_gpus", "rollout_width", "train_width", "max_group_size", "simulate_steps"):
            integer(getattr(self, name), name)

    def submit(self, task: Task) -> None:
        if task.name in self.tasks:
            raise ValueError(f"Duplicate task: {task.name}")
        if task.rollout.request * self.rollout_width > self.rollout_gpus:
            raise ValueError(f"{task.name}: rollout request exceeds cluster capacity")
        if task.train.request * self.train_width > self.train_gpus:
            raise ValueError(f"{task.name}: train request exceeds cluster capacity")
        self.tasks[task.name] = task
        self.completed_steps[task.name] = 0
        self.pending.append(task.name)

    def group(self, name: str, allocations: dict[str, tuple[int, int]]) -> Group | None:
        if len(allocations) > self.max_group_size:
            return None
        specs = []
        for job, allocation in allocations.items():
            task = self.tasks[job]
            r, t = allocation
            if r not in task.rollout.seconds or t not in task.train.seconds:
                raise ValueError(f"Unprofiled allocation for {job}: {allocation}")
            specs.append(
                {
                    "job_name": job,
                    "rollout_seconds": task.rollout.seconds[r],
                    "train_seconds": task.train.seconds[t],
                    "slo": task.slo,
                    "rollout_workers": r,
                    "train_gpus": t * self.train_width,
                }
            )
        plan = build_plan(
            {
                "jobs": specs,
                "gpus_per_worker": self.rollout_width,
                "gpus_per_node": max(self.train_gpus, self.rollout_width),
                "max_group_size": self.max_group_size,
                "simulate_steps": self.simulate_steps,
            }
        )
        if len(plan["groups"]) != 1:
            return None
        plan["groups"][0]["group_id"] = name
        return Group(name, dict(allocations), plan)

    def fits(self, groups: dict[str, Group]) -> bool:
        return (
            sum(group.resources[0] for group in groups.values()) <= self.rollout_gpus
            and sum(group.resources[1] for group in groups.values()) <= self.train_gpus
        )

    def penalty(self, old: Group | None, new: Group, now: float) -> float | None:
        penalty = 0.0
        for name, allocation in new.allocations.items():
            task = self.tasks[name]
            remaining = max(0, task.total_steps - self.completed_steps[name])
            duration = task.step_seconds(allocation)
            overhead = task.reconfigure_seconds
            if task.deadline is not None and now + remaining * duration * task.slo + overhead > task.deadline:
                return None
            if old is not None and name in old.allocations:
                before = task.step_seconds(old.allocations[name])
                penalty += remaining * max(0, duration - before) + overhead
        return penalty

    def variants(self, group: Group, now: float) -> list[tuple[Group, float]]:
        variants = [(group, 0.0)]
        rebuilt = self.group(group.name, group.allocations)
        if (
            rebuilt is not None
            and any(after < before for after, before in zip(rebuilt.resources, group.resources, strict=True))
            and all(after <= before for after, before in zip(rebuilt.resources, group.resources, strict=True))
        ):
            cost = self.penalty(group, rebuilt, now)
            if cost is not None:
                variants.append((rebuilt, cost))
        seen = {tuple(group.allocations.items())}
        max_r = max(value[0] for value in group.allocations.values())
        max_t = max(value[1] for value in group.allocations.values())
        min_r = max(self.tasks[name].rollout.request for name in group.allocations)
        min_t = max(self.tasks[name].train.request for name in group.allocations)
        for r, t in product(range(min_r, max_r + 1), range(min_t, max_t + 1)):
            allocation = {
                name: (self.tasks[name].rollout.capped(old[0], r), self.tasks[name].train.capped(old[1], t))
                for name, old in group.allocations.items()
            }
            key = tuple(allocation.items())
            if key in seen:
                continue
            seen.add(key)
            candidate = self.group(group.name, allocation)
            if candidate is None:
                continue
            if any(after > before for after, before in zip(candidate.resources, group.resources, strict=True)):
                continue
            cost = self.penalty(group, candidate, now)
            if cost is not None:
                variants.append((candidate, cost))
        return variants

    def admit(self, name: str, now: float) -> Decision | None:
        if name not in self.pending:
            raise ValueError(f"{name} is not pending")
        task = self.tasks[name]
        minimum = (task.rollout.request, task.train.request)
        candidates = []
        # Try the requested initial scale first, but do not reclaim resources
        # merely to grant a new task more than its guaranteed minimum.
        for scale in dict.fromkeys((task.initial, minimum)):
            for target in [None, *self.groups]:
                if scale != minimum:
                    options = [[(group, 0.0)] for group in self.groups.values()]
                else:
                    options = [self.variants(group, now) for group in self.groups.values()]
                # DP over the two pool capacities keeps only the cheapest plan
                # per resource total; unaffected groups retain their identity.
                states = {(0, 0): ({}, 0.0)}
                for group_name, variants in zip(self.groups, options, strict=True):
                    next_states = {}
                    for groups, cost in states.values():
                        for variant, extra in variants:
                            if group_name == target:
                                variant = self.group(group_name, {**variant.allocations, name: scale})
                                if variant is None:
                                    continue
                                extra = self.penalty(self.groups[group_name], variant, now)
                                if extra is None:
                                    continue
                            selected = {**groups, group_name: variant}
                            resource = tuple(sum(g.resources[i] for g in selected.values()) for i in (0, 1))
                            if resource[0] > self.rollout_gpus or resource[1] > self.train_gpus:
                                continue
                            score = cost + extra
                            previous = next_states.get(resource)
                            if previous is None or score < previous[1]:
                                next_states[resource] = (selected, score)
                    states = next_states
                for groups, cost in states.values():
                    if target is None:
                        new_name = f"Elastic-{self.next_group}"
                        new_group = self.group(new_name, {name: scale})
                        if new_group is None or self.penalty(None, new_group, now) is None:
                            continue
                        groups = {**groups, new_name: new_group}
                    if not self.fits(groups):
                        continue
                    affected = tuple(key for key, value in groups.items() if self.groups.get(key) != value)
                    resources = sum(sum(group.resources) for group in groups.values())
                    candidates.append((cost, resources, len(affected), groups, affected))
            if candidates:
                break
        if not candidates:
            return None
        cost, _, _, groups, affected = min(candidates, key=lambda value: value[:3])
        return Decision(groups, affected, f"admit:{name}", cost)

    def expand(self, now: float) -> Decision | None:
        if self.pending:
            return None
        candidates = []
        for group in self.groups.values():
            for name, current in group.allocations.items():
                task = self.tasks[name]
                for component, profile in enumerate((task.rollout, task.train)):
                    larger = [count for count in profile.seconds if count > current[component]]
                    if not larger:
                        continue
                    new_scale = list(current)
                    new_scale[component] = min(larger)
                    allocation = (new_scale[0], new_scale[1])
                    new_group = self.group(group.name, {**group.allocations, name: allocation})
                    if new_group is None or self.penalty(group, new_group, now) is None:
                        continue
                    groups = {**self.groups, group.name: new_group}
                    remaining = task.total_steps - self.completed_steps[name]
                    benefit = remaining * (task.step_seconds(current) - task.step_seconds(allocation))
                    overhead = sum(self.tasks[job].reconfigure_seconds for job in group.allocations)
                    if self.fits(groups) and benefit > overhead:
                        urgency = benefit
                        if task.deadline is not None:
                            urgency /= max(1.0, task.deadline - now)
                        candidates.append((urgency, Decision(groups, (group.name,), f"expand:{name}", overhead)))
        return max(candidates, key=lambda value: value[0])[1] if candidates else None

    def rebalance(self, now: float) -> Decision | None:
        states = {(0, 0): ({}, 0.0)}
        for old in self.groups.values():
            next_states = {}
            for groups, cost in states.values():
                for variant, extra in self.variants(old, now):
                    selected = {**groups, old.name: variant}
                    if not self.fits(selected):
                        continue
                    resources = tuple(sum(group.resources[i] for group in selected.values()) for i in (0, 1))
                    previous = next_states.get(resources)
                    if previous is None or cost + extra < previous[1]:
                        next_states[resources] = (selected, cost + extra)
            states = next_states
        if not states:
            return None
        groups, cost = min(states.values(), key=lambda value: value[1])
        affected = tuple(name for name, group in groups.items() if self.groups[name] != group)
        return Decision(groups, affected, "capacity", cost)

    def redistribute(self, now: float) -> Decision | None:
        """Trade DP replicas between running tasks when no free capacity exists."""
        if self.pending:
            return None

        def remaining_cost(group: Group) -> float:
            return sum(
                (self.tasks[name].total_steps - self.completed_steps[name]) * self.tasks[name].step_seconds(allocation)
                for name, allocation in group.allocations.items()
            )

        before = sum(remaining_cost(group) for group in self.groups.values())
        options = {name: self.variants(group, now) for name, group in self.groups.items()}
        best = None
        for target in self.groups.values():
            for task_name, current in target.allocations.items():
                task = self.tasks[task_name]
                for index, component in enumerate((task.rollout, task.train)):
                    larger = sorted(count for count in component.seconds if count > current[index])
                    if not larger:
                        continue
                    scale = list(current)
                    scale[index] = larger[0]
                    allocation = (scale[0], scale[1])
                    states = {(0, 0): ({}, 0.0)}
                    for name, old in self.groups.items():
                        next_states = {}
                        for selected, score in states.values():
                            for variant, _ in options[name]:
                                if name == target.name:
                                    variant = self.group(name, {**variant.allocations, task_name: allocation})
                                    if variant is None or self.penalty(old, variant, now) is None:
                                        continue
                                groups = {**selected, name: variant}
                                if not self.fits(groups):
                                    continue
                                overhead = (
                                    sum(self.tasks[job].reconfigure_seconds for job in variant.allocations)
                                    if variant != old
                                    else 0
                                )
                                cost = score + remaining_cost(variant) + overhead
                                resources = tuple(sum(group.resources[i] for group in groups.values()) for i in (0, 1))
                                previous = next_states.get(resources)
                                if previous is None or cost < previous[1]:
                                    next_states[resources] = (groups, cost)
                        states = next_states
                    for groups, score in states.values():
                        benefit = before - score
                        if benefit > 0 and (best is None or benefit > best[0]):
                            affected = tuple(name for name, group in groups.items() if self.groups[name] != group)
                            overhead = sum(
                                self.tasks[job].reconfigure_seconds
                                for name in affected
                                for job in groups[name].allocations
                            )
                            best = (benefit, Decision(groups, affected, f"redistribute:{task_name}", overhead))
        return best[1] if best is not None else None

    def commit(self, decision: Decision) -> None:
        if not self.fits(decision.groups):
            raise ValueError("Cannot commit an over-capacity configuration")
        names = [name for group in decision.groups.values() for name in group.allocations]
        if len(names) != len(set(names)):
            raise ValueError("A task cannot belong to multiple groups")
        for group in decision.groups.values():
            for name, (r, t) in group.allocations.items():
                if r not in self.tasks[name].rollout.seconds or t not in self.tasks[name].train.seconds:
                    raise ValueError(f"Cannot commit unprofiled or out-of-quota allocation for {name}")
        self.groups = dict(decision.groups)
        self.pending = [name for name in self.pending if name not in names]
        while f"Elastic-{self.next_group}" in self.groups:
            self.next_group += 1

    def progress(self, name: str, step: int, durations: dict[str, float] | None = None) -> None:
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or not self.completed_steps[name] <= step <= self.tasks[name].total_steps
        ):
            raise ValueError(f"Invalid or regressing step for {name}: {step}")
        self.completed_steps[name] = step
        if durations:
            group = next((group for group in self.groups.values() if name in group.allocations), None)
            if group is not None:
                task = self.tasks[name]
                profiles = []
                for index, component in enumerate(("rollout", "train")):
                    profile = getattr(task, component)
                    values = dict(profile.seconds)
                    if component in durations:
                        measured = positive(durations[component], f"{name}.{component}.duration")
                        count = group.allocations[name][index]
                        values[count] = 0.8 * values[count] + 0.2 * measured
                    profiles.append(replace(profile, seconds=values))
                self.tasks[name] = replace(task, rollout=profiles[0], train=profiles[1])
                if "switch" in durations:
                    measured = nonnegative(durations["switch"], f"{name}.switch.duration")
                    self.tasks[name] = replace(
                        self.tasks[name], switch_seconds=0.8 * task.switch_seconds + 0.2 * measured
                    )

    def retire(self, name: str) -> str | None:
        if name not in self.tasks:
            raise ValueError(f"Unknown task: {name}")
        if name in self.pending:
            self.pending.remove(name)
        for group in list(self.groups.values()):
            if name not in group.allocations:
                continue
            allocations = {key: value for key, value in group.allocations.items() if key != name}
            # Keep existing placements until the affected group is rebuilt;
            # completion alone does not magically release its physical PGs.
            self.groups[group.name] = Group(group.name, allocations, group.plan)
            return group.name
        return None
