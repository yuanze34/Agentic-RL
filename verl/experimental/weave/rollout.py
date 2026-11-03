"""Ray load-balancer and client integration for WEAVE request recomputation."""

from __future__ import annotations

from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, GlobalRequestLoadBalancer

from .migration import AggregationPolicy, MigrationConfig, resolved_destinations


class WeaveLoadBalancer(GlobalRequestLoadBalancer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._generation_requests: dict[str, tuple[str, str]] = {}
        self._worker_servers: dict[int, str] = {}
        self._migration_config = MigrationConfig()
        self._aggregation: AggregationPolicy | None = None
        self._failure: str | None = None

    def begin_phase(self, workers: dict[int, str], migration: dict) -> None:
        if self._generation_requests:
            raise RuntimeError("Cannot begin WEAVE generation with unfinished requests")
        self._worker_servers = workers
        self._migration_config = MigrationConfig(**migration)
        self._aggregation = AggregationPolicy(self._migration_config)
        self._failure = None

    def acquire_generation(self, session_id: str, generation_id: str):
        self._raise_failure()
        if generation_id in self._generation_requests:
            raise ValueError(f"Duplicate WEAVE generation request {generation_id}")
        server_id, handle = self.acquire_server(session_id)
        self._generation_requests[generation_id] = (server_id, session_id)
        return server_id, handle

    def finish_generation(self, server_id: str, generation_id: str) -> None:
        if self._generation_requests.pop(generation_id, None) is None:
            raise ValueError(f"Unknown WEAVE generation request {generation_id}")
        self.release_server(server_id)
        self._raise_failure()

    def claim_migration(self, now: float | None = None) -> dict | None:
        self._raise_failure()
        if self._aggregation is None or not self._migration_config.enabled:
            return None
        server_workers = {server: worker for worker, server in self._worker_servers.items()}
        requests = {request: server_workers[server] for request, (server, _) in self._generation_requests.items()}
        workers = list(self._worker_servers)
        plan = self._aggregation.plan(requests, workers, now=now)
        if plan is None:
            return None
        destinations = resolved_destinations(self._migration_config, workers)
        sources = [worker for worker in workers if worker not in destinations]
        # A claimed plan atomically stops new dispatches to every source.
        self.remove_servers([self._worker_servers[worker] for worker in sources])
        for (source, destination), request_ids in plan.items():
            for request_id in request_ids:
                _, session = self._generation_requests[request_id]
                self._request_id_to_server[session] = self._worker_servers[destination]
        return {
            "workers": sources,
            "requests": {
                worker: [request for request, owner in requests.items() if owner == worker] for worker in sources
            },
        }

    def outstanding_on_workers(self, workers: list[int]) -> int:
        servers = {self._worker_servers[worker] for worker in workers}
        return sum(server in servers for server, _ in self._generation_requests.values())

    def fail_generation(self, message: str) -> None:
        self._failure = message

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise RuntimeError(f"WEAVE migration failed: {self._failure}")


class WeaveLLMServerClient(FullyAsyncLLMServerClient):
    async def _acquire_generation_server(self, request_id: str, generation_id: str):
        return await self._load_balancer.acquire_generation.remote(request_id, generation_id)

    async def _release_generation_server(self, server_id: str, generation_id: str) -> None:
        await self._load_balancer.finish_generation.remote(server_id, generation_id)
