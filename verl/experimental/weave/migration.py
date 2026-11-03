"""WEAVE's one-shot, capacity-bounded request aggregation.

The static policy follows Weave/roll/distributed/scheduler/migration_scheduler.py
(Apache-2.0): check once per second, wait five seconds, then consolidate when
the remaining requests fit in the destination workers. Migration recomputes KV
from the prompt and generated tokens; it does not transfer KV blocks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class MigrationConfig:
    enabled: bool = False
    policy: str = "static"
    destination_workers: tuple[int, ...] = (-1,)
    max_batch_size: int = 16
    check_interval: float = 1.0
    min_elapsed: float = 5.0
    drain_timeout: float = 120.0

    def __post_init__(self):
        object.__setattr__(self, "destination_workers", tuple(self.destination_workers))
        if self.policy not in ("static", "naive"):
            raise ValueError("WEAVE migration policy must be 'static' or 'naive'")
        if not self.destination_workers or len(set(self.destination_workers)) != len(self.destination_workers):
            raise ValueError("WEAVE migration destinations must be nonempty and unique")
        if not all(isinstance(worker, int) for worker in self.destination_workers):
            raise ValueError("WEAVE migration destination workers must be integer IDs")
        if self.max_batch_size <= 0 or self.check_interval <= 0 or self.min_elapsed < 0 or self.drain_timeout <= 0:
            raise ValueError("Invalid WEAVE migration capacity or timing")


class AggregationPolicy:
    def __init__(self, config: MigrationConfig, *, now: float | None = None):
        self.config = config
        self.start_time = time.monotonic() if now is None else now
        self.last_check = self.start_time
        self.migrated = False

    def plan(
        self,
        requests: dict[str, int],
        ready_workers: list[int],
        *,
        now: float | None = None,
    ) -> dict[tuple[int, int], list[str]] | None:
        """None means no decision; an empty dict can still release idle sources."""
        now = time.monotonic() if now is None else now
        if self.migrated or now - self.last_check <= self.config.check_interval:
            return None
        self.last_check = now
        if now - self.start_time < self.config.min_elapsed or not requests:
            return None
        destinations = [
            ready_workers[index] if index < 0 and -index <= len(ready_workers) else index
            for index in self.config.destination_workers
        ]
        if len(set(destinations)) != len(destinations):
            raise ValueError("WEAVE migration destinations resolve to the same worker")
        if any(worker not in ready_workers for worker in destinations):
            self.migrated = True
            return None
        if self.config.policy == "static" and len(requests) > len(destinations) * self.config.max_batch_size:
            return None

        per_worker = {worker: [] for worker in ready_workers}
        for request, worker in requests.items():
            per_worker[worker].append(request)
        sources = [worker for worker in ready_workers if worker not in destinations and per_worker[worker]]
        plan = {}
        for destination in destinations:
            available = (
                len(requests)
                if self.config.policy == "naive"
                else self.config.max_batch_size - len(per_worker[destination])
            )
            while available > 0 and sources:
                source = sources.pop()
                moved = per_worker[source][:available]
                plan[source, destination] = moved
                available -= len(moved)
                per_worker[source] = per_worker[source][len(moved) :]
                if per_worker[source]:
                    sources.append(source)
        if any(per_worker[worker] for worker in ready_workers if worker not in destinations):
            return None
        self.migrated = True
        return plan


def resolved_destinations(config: MigrationConfig, workers: list[int]) -> list[int]:
    return [workers[index] if index < 0 and -index <= len(workers) else index for index in config.destination_workers]
