"""Task-independent, two-engine rollout slots with CPU-prepared full weights."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)
NAMESPACE = "verl_dual_engine"


@dataclass(frozen=True)
class ModelVersion:
    task: str
    step: int

    def __post_init__(self):
        if not self.task or type(self.step) is not int or self.step < 0:
            raise ValueError("A model version needs a nonempty task and nonnegative integer step")


class DualEngineSlot:
    """Two engines regardless of task count; serial control, concurrent generation."""

    def __init__(self, engines: list[Any], signature: str):
        if len(engines) != 2 or engines[0] is engines[1]:
            raise ValueError("A rollout slot requires exactly two distinct engines")
        self.engines = engines
        self.signature = signature
        self.active: int | None = None
        self.current: ModelVersion | None = None
        self.prepared: tuple[int, ModelVersion] | None = None
        self.images: dict[str, tuple[int, dict]] = {}
        self.leased = False
        self.inflight = 0
        self.failed: str | None = None
        self.history: list[dict] = []
        self._control = asyncio.Lock()

    def _healthy(self):
        if self.failed is not None:
            raise RuntimeError(f"Dual Engine slot is unavailable: {self.failed}")

    async def _prepare(self, version: ModelVersion) -> tuple[int, ModelVersion]:
        index = 0 if self.active is None else 1 - self.active
        if self.prepared == (index, version):
            return self.prepared
        step, weights = self.images[version.task]
        if step != version.step:
            raise ValueError("Requested weight version is no longer available")
        self.prepared = None
        await self.engines[index].prepare(version, weights)
        self.prepared = (index, version)
        return self.prepared

    async def publish(self, task: str, step: int, weights: dict, signature: str):
        version = ModelVersion(task, step)
        async with self._control:
            self._healthy()
            if signature != self.signature:
                raise ValueError("Tasks sharing a dual Engine slot must have compatible model and rollout layouts")
            if not weights:
                raise ValueError("A full weight image must not be empty")
            if task in self.images and step <= self.images[task][0]:
                raise ValueError("Weight publication must advance the task model version")
            self.images[task] = (step, weights)
            started = time.monotonic()
            await self._prepare(version)
            return {"prepare_seconds": time.monotonic() - started, "task": task, "step": step}

    async def activate(self, task: str, step: int):
        version = ModelVersion(task, step)
        async with self._control:
            self._healthy()
            if self.leased:
                if self.current == version:
                    return
                raise RuntimeError("Cannot switch a rollout slot while another task owns its generation lease")
            if self.inflight:
                raise RuntimeError("Cannot switch a rollout slot with unfinished requests")
            if task not in self.images or self.images[task][0] != step:
                raise ValueError("The requested task/version has no published weight image")
            started = time.monotonic()
            warm = self.prepared is not None and self.prepared[1] == version
            index, _ = await self._prepare(version)
            old = self.active
            self.current = None
            try:
                # Never wake destination weights before releasing the source.
                sleep_metrics = await self.engines[old].sleep() if old is not None else None
                activate_metrics = await self.engines[index].activate(version)
            except BaseException as exc:
                self.failed = f"switch to {task}@{step} failed: {exc}"
                logger.exception("Dual Engine switch failed; generation is disabled")
                raise
            self.active = index
            self.current = version
            self.prepared = None
            self.leased = True
            self.history.append(
                {
                    "task": task,
                    "step": step,
                    "engine": index,
                    "prewarmed": warm,
                    "switch_seconds": time.monotonic() - started,
                    "sleep_metrics": sleep_metrics,
                    "activate_metrics": activate_metrics,
                }
            )

    async def release(self, task: str):
        async with self._control:
            self._healthy()
            if not self.leased or self.current is None or self.current.task != task:
                raise RuntimeError("Only the current generation owner may release a dual Engine slot")
            if self.inflight:
                raise RuntimeError("Cannot release a rollout slot with unfinished requests")
            # Keep the owner active until its successor switches in. This permits
            # CPU preparation on the other engine while no second model is awake.
            self.leased = False

    async def generate(self, task: str, step: int, **kwargs):
        self._healthy()
        if not self.leased or self.current != ModelVersion(task, step) or self.active is None:
            raise RuntimeError("Generation task/version does not match the slot lease")
        index = self.active
        self.inflight += 1
        try:
            output = await self.engines[index].generate(**kwargs)
            output.extra_fields.update(task=task, global_steps=step, dual_engine=index)
            return output
        finally:
            self.inflight -= 1

    async def status(self):
        async with self._control:
            return {
                "engine_count": len(self.engines),
                "active": self.active,
                "task": self.current.task if self.current else None,
                "step": self.current.step if self.current else None,
                "leased": self.leased,
                "inflight": self.inflight,
                "failed": self.failed,
                "published_tasks": sorted(self.images),
                "history": list(self.history),
                "engines": [await engine.status() for engine in self.engines],
            }


class RayEngine:
    def __init__(self, server):
        self.server = server

    async def prepare(self, version, weights):
        return await self.server.prepare_image.remote(version.task, version.step, weights)

    async def activate(self, version):
        return await self.server.activate_image.remote(version.task, version.step)

    async def sleep(self):
        return await self.server.sleep_image.remote()

    async def generate(self, **kwargs):
        return await self.server.generate.remote(**kwargs)

    async def status(self):
        return await self.server.image_status.remote()


def layout_signature(config) -> str:
    import json

    from omegaconf import OmegaConf

    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config import HFModelConfig, RolloutConfig

    rollout = omega_conf_to_dataclass(config.actor_rollout_ref.rollout, dataclass_type=RolloutConfig)
    if rollout.name != "vllm" or rollout.nnodes != 1:
        raise ValueError("Dual Engine rollout currently requires single-node vLLM")
    if any(
        value != 1
        for value in (
            rollout.tensor_model_parallel_size,
            rollout.data_parallel_size,
            rollout.pipeline_model_parallel_size,
        )
    ):
        raise ValueError("Dual Engine rollout currently requires one GPU per slot (TP=DP=PP=1)")
    model = omega_conf_to_dataclass(config.actor_rollout_ref.model, dataclass_type=HFModelConfig)
    if model.lora_rank > 0 or rollout.quantization is not None or rollout.disaggregation.enabled:
        raise ValueError("Dual Engine rollout supports full, unquantized, non-disaggregated models only")
    if not rollout.enforce_eager or not rollout.enable_sleep_mode or not rollout.free_cache_engine:
        raise ValueError("Dual Engine rollout requires eager execution and sleep/free_cache_engine enabled")
    hf = model.hf_config.to_dict()
    for key in ("_name_or_path", "transformers_version"):
        hf.pop(key, None)
    runtime = OmegaConf.to_container(rollout, resolve=True) if OmegaConf.is_config(rollout) else asdict(rollout)
    # Task transport, tracing and pool size do not define an Engine's layout.
    for key in ("checkpoint_engine", "name_suffix", "n_gpus_per_node", "trace"):
        runtime.pop(key, None)
    return json.dumps({"model": hf, "rollout": runtime}, sort_keys=True, default=str)


def slot_name(placement_id: str) -> str:
    return f"dual_engine_{placement_id}"


def close_shared_slot(placement_id: str):
    import ray

    try:
        slot = ray.get_actor(slot_name(placement_id), namespace=NAMESPACE)
    except ValueError:
        return
    ray.get(slot.close.remote())
    ray.kill(slot, no_restart=True)


class SharedDualEngineSlot:
    """Named Ray owner shared by trainers, not by their model lifetimes."""

    def __init__(self):
        self.slot: DualEngineSlot | None = None
        self._init_lock = asyncio.Lock()
        self.servers = []

    async def initialize(self, config, placement_id: str):
        import ray
        from ray._raylet import PlacementGroupID
        from ray.util.placement_group import PlacementGroup
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from verl.experimental.elastic.dual_engine_vllm import DualEngineServer

        signature = layout_signature(config)
        async with self._init_lock:
            if self.slot is not None:
                if signature != self.slot.signature:
                    raise ValueError("Incompatible tasks cannot share an existing dual Engine slot")
                return
            pg = PlacementGroup(PlacementGroupID.from_hex(placement_id))
            if pg.bundle_count != 1:
                raise ValueError("Dual Engine slots require a single GPU placement bundle")
            try:
                for index in range(2):
                    server = (
                        ray.remote(DualEngineServer)
                        .options(
                            num_cpus=1,
                            num_gpus=0.5,
                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                placement_group=pg, placement_group_bundle_index=0
                            ),
                            max_concurrency=128,
                        )
                        .remote(config, index)
                    )
                    self.servers.append(server)
                    await server.launch_server.remote()
                    await server.sleep_image.remote()
                self.slot = DualEngineSlot([RayEngine(server) for server in self.servers], signature)
            except Exception:
                logger.exception("Failed to initialize shared dual Engine slot")
                await self.close()
                raise

    async def publish(self, *args):
        if self.slot is None:
            raise RuntimeError("Dual Engine slot is not initialized")
        return await self.slot.publish(*args)

    async def activate(self, *args):
        if self.slot is None:
            raise RuntimeError("Dual Engine slot is not initialized")
        return await self.slot.activate(*args)

    async def release(self, *args):
        if self.slot is None:
            raise RuntimeError("Dual Engine slot is not initialized")
        return await self.slot.release(*args)

    async def generate(self, *args, **kwargs):
        if self.slot is None:
            raise RuntimeError("Dual Engine slot is not initialized")
        return await self.slot.generate(*args, **kwargs)

    async def status(self):
        if self.slot is None:
            raise RuntimeError("Dual Engine slot is not initialized")
        return await self.slot.status()

    async def close(self):
        import ray

        for server in self.servers:
            ray.kill(server, no_restart=True)
        self.servers = []
        self.slot = None
