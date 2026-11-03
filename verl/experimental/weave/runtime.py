"""Trainer-side request aggregation and physical worker retirement."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict

import ray

from .migration import MigrationConfig

logger = logging.getLogger(__name__)


class MigrationController:
    def __init__(self, client, manager, checkpoint_manager, workers: list[int], config: MigrationConfig):
        self.client = client
        self.manager = manager
        self.checkpoint_manager = checkpoint_manager
        self.config = config
        self.remaining_workers = workers.copy()
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        replicas = manager.get_replicas()
        ray.get(
            manager.global_load_balancer.begin_phase.remote(
                {worker: replicas[worker].server_address for worker in workers}, asdict(config)
            )
        )

    def start(self) -> None:
        if self.config.enabled and len(self.remaining_workers) > 1:
            self._thread = threading.Thread(target=self._run, name="weave-migration", daemon=True)
            self._thread.start()

    def finish(self) -> list[int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.config.drain_timeout + 5)
            if self._thread.is_alive():
                raise TimeoutError("WEAVE migration controller did not stop")
        if self.error is not None:
            raise RuntimeError("WEAVE request migration failed") from self.error
        return self.remaining_workers.copy()

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.config.check_interval):
                plan = ray.get(self.manager.global_load_balancer.claim_migration.remote())
                if plan is not None:
                    self._retire(plan)
                    return
        except Exception as exc:
            self.error = exc
            logger.exception("WEAVE request migration failed")
            self.client.crash()
            ray.get(self.manager.global_load_balancer.fail_generation.remote(str(exc)), timeout=5)

    def _retire(self, plan: dict) -> None:
        workers = plan["workers"]
        replicas = self.manager.get_replicas()
        requests = []
        for worker in workers:
            for server in replicas[worker].servers:
                requests.append(server.migrate_requests.remote(plan["requests"][worker]))
        ray.get(requests, timeout=self.config.drain_timeout)
        deadline = time.monotonic() + self.config.drain_timeout
        while ray.get(self.manager.global_load_balancer.outstanding_on_workers.remote(workers)):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"WEAVE workers {workers} did not drain before migration timeout")
            time.sleep(0.01)
        for worker in workers:
            ray.get(
                [server.wait_for_requests_to_drain.remote() for server in replicas[worker].servers],
                timeout=self.config.drain_timeout,
            )
        self.checkpoint_manager.suspend_tenant_replicas([replicas[worker] for worker in workers])
        self.client.release_generate_workers(workers)
        self.remaining_workers = [worker for worker in self.remaining_workers if worker not in workers]
        logger.info("WEAVE consolidated requests and released rollout workers %s", workers)
