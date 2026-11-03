"""verl-side client for the WEAVE scheduler protocol."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .protocol import TENANT_EVENTS_CHANNEL, Phase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeaveClientConfig:
    host: str
    port: int
    job_name: str
    gen_device_affinity: str
    train_device_affinity: str
    gen_gpu_list: list[int]
    train_gpu_list: list[int]
    gpus_per_gen_worker: int = 1
    poll_interval: float = 0.02
    wait_timeout: float | None = None
    init_wait_timeout: float | None = None

    @classmethod
    def from_dict(cls, config: Any) -> WeaveClientConfig:
        return cls(
            host=str(config.host),
            port=int(config.port),
            job_name=str(config.job_name),
            gen_device_affinity=str(config.gen_device_affinity),
            train_device_affinity=str(config.train_device_affinity),
            gen_gpu_list=[int(device) for device in config.gen_gpu_list],
            train_gpu_list=[int(device) for device in config.train_gpu_list],
            gpus_per_gen_worker=int(config.gpus_per_gen_worker),
            poll_interval=float(config.poll_interval),
            wait_timeout=None if config.wait_timeout is None else float(config.wait_timeout),
            init_wait_timeout=(None if config.init_wait_timeout is None else float(config.init_wait_timeout)),
        )


class WeaveSchedulerClient:
    def __init__(self, config: WeaveClientConfig, storage: Any | None = None):
        self.config = config
        if storage is None:
            try:
                import redis
            except ImportError as exc:
                raise RuntimeError("WEAVE mode requires the `redis` Python package.") from exc
            storage = redis.Redis(host=config.host, port=config.port, db=0, decode_responses=True)
        self.storage = storage
        self.step_counter = {Phase.GENERATE: 0, Phase.TRAIN: 0}

    def _publish(self, message: str) -> None:
        self.storage.publish(TENANT_EVENTS_CHANNEL, message)

    def _set(self, key: str, value: str | int, *, nx: bool = False) -> None:
        self.storage.set(key, value, nx=nx)

    def wait_key(self, key: str, expected: str, timeout: float | None = None) -> str:
        start = time.monotonic()
        timeout = self.config.wait_timeout if timeout is None else timeout
        while True:
            value = self.storage.get(key)
            if isinstance(value, bytes):
                value = value.decode()
            if value is not None and expected in str(value):
                return str(value)
            if value is not None and "crashed" in str(value):
                raise RuntimeError(f"WEAVE job {self.config.job_name} crashed while waiting for {key}={expected}")
            if timeout is not None and time.monotonic() - start >= timeout:
                raise TimeoutError(f"Timed out waiting for {key} to contain {expected!r}; last value={value!r}")
            time.sleep(self.config.poll_interval)

    def register_and_wait_for_init(self) -> None:
        job = self.config.job_name
        self._set(f"{job}:status", "created")
        self._set(f"{job}:gpus_per_gen_worker", self.config.gpus_per_gen_worker)
        self._set(f"{job}:gen_gpu_list", ",".join(map(str, self.config.gen_gpu_list)))
        self._set(f"{job}:train_gpu_list", ",".join(map(str, self.config.train_gpu_list)))
        self._publish(f"{job}:created")
        init_timeout = self.config.init_wait_timeout
        self.wait_key(f"{job}:status", "initializing", timeout=init_timeout)

    def finish_init(self) -> None:
        job = self.config.job_name
        if self.config.gen_device_affinity == self.config.train_device_affinity:
            gpu_ids = sorted(set(self.config.gen_gpu_list + self.config.train_gpu_list))
            self.release_gpus(Phase.INIT, self.config.gen_device_affinity, gpu_ids)
        else:
            self.release_gpus(Phase.INIT, self.config.gen_device_affinity, self.config.gen_gpu_list)
            self.release_gpus(Phase.INIT, self.config.train_device_affinity, self.config.train_gpu_list)
        self._publish(f"{job}:init:done")
        self._set(f"{job}:status", "running")

    def wait_for_phase(self, phase: Phase) -> None:
        self.wait_key("weave:init_barrier", "open", timeout=self.config.init_wait_timeout)
        key = f"{self.config.job_name}:{phase.value}:status"
        self._set(key, "pending", nx=True)
        self.wait_key(key, "running")
        logger.info("WEAVE admitted %s phase for %s", phase.value, self.config.job_name)

    def get_allocated_generate_workers(self) -> list[int]:
        key = f"{self.config.job_name}:generate:allocated_workers"
        value = self.storage.get(key)
        if isinstance(value, bytes):
            value = value.decode()
        if not value:
            raise RuntimeError(f"WEAVE admitted generate without publishing {key}")
        return [int(worker_id) for worker_id in str(value).split(",")]

    def finish_phase(self, phase: Phase) -> None:
        step = self.step_counter[phase]
        self._set(f"{self.config.job_name}:{phase.value}:status", "pending")
        self._publish(f"{self.config.job_name}:{phase.value}:done[{step}]")
        self.step_counter[phase] += 1

    def release_gpus(self, phase: Phase, device_affinity: str, gpu_ids: list[int]) -> None:
        if not gpu_ids:
            return
        payload = ",".join([device_affinity, *map(str, gpu_ids)])
        self._publish(f"{self.config.job_name}:{phase.value}:release_gpu[{payload}]")

    def release_generate_workers(self, worker_ids: list[int] | None = None) -> None:
        width = self.config.gpus_per_gen_worker
        allowed_workers = None if worker_ids is None else set(worker_ids)
        for start in range(0, len(self.config.gen_gpu_list), width):
            gpu_ids = self.config.gen_gpu_list[start : start + width]
            worker_id = gpu_ids[0] // width
            if allowed_workers is not None and worker_id not in allowed_workers:
                continue
            self.release_gpus(Phase.GENERATE, self.config.gen_device_affinity, gpu_ids)
            self._publish(f"{self.config.job_name}:generate:{worker_id}:done")

    def complete(self) -> None:
        self._set(f"{self.config.job_name}:status", "completed")
        self._publish(f"{self.config.job_name}:completed")

    def wait_for_weight_update(self) -> None:
        key = f"{self.config.job_name}:update:status"
        self._set(key, "pending")
        self._publish(f"{self.config.job_name}:update:ready")
        self.wait_key(key, "running")

    def finish_weight_update(self) -> None:
        self.release_gpus(Phase.UPDATE, self.config.gen_device_affinity, self.config.gen_gpu_list)
        self._publish(f"{self.config.job_name}:update:done")

    def crash(self) -> None:
        job = self.config.job_name
        self._set(f"{job}:status", "crashed")
        self._publish(f"{job}:crashed")
