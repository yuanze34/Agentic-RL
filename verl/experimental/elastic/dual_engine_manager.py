"""Trainer-facing adapters for shared rollout slots and CPU weight publication."""

from __future__ import annotations

import time
from dataclasses import dataclass

import ray
import torch

from verl.experimental.elastic.dual_engine import NAMESPACE, SharedDualEngineSlot, layout_signature, slot_name
from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.experimental.weave.rollout import WeaveLoadBalancer
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.rollout.llm_server import LLMServerClient


class DualEngineActorWorker(DetachActorWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def export_rollout_cpu_weights(self):
        # All FSDP ranks must participate in full-state export, even though
        # only rank zero returns the CPU image to the task controller.
        engine = self.actor.engine
        tensors, peft_config = engine.get_per_tensor_param()
        if peft_config is not None:
            raise ValueError("Dual Engine publication requires full-model weights")
        rank = torch.distributed.get_rank()
        image = {}
        for name, tensor in tensors:
            if rank == 0:
                image[name] = tensor.detach().to(device="cpu", copy=True)
        torch.cuda.synchronize()
        return image if rank == 0 else None


class TaskEndpoint:
    """A CPU-only task proxy; it never creates an Engine."""

    def __init__(self, slot: ray.actor.ActorHandle, task: str):
        self.slot = slot
        self.task = task
        self.step: int | None = None

    def set_step(self, step: int):
        self.step = step

    async def generate(self, **kwargs):
        if self.step is None:
            raise RuntimeError("Generation requested before a task weight version was published")
        return await self.slot.generate.remote(self.task, self.step, **kwargs)


@dataclass
class SharedReplica:
    replica_rank: int
    server_address: str
    server_handle: ray.actor.ActorHandle
    slot: ray.actor.ActorHandle

    async def sleep(self):
        # Initial setup has no task lease or device-resident task weights.
        return


class DualEngineManager:
    def __init__(self, config):
        self.config = config
        self.task = config.trainer.v1.weave.job_name
        self.signature = layout_signature(config)
        self.step = None
        placements = config.trainer.v1.weave.gen_replica_placements
        if not placements or len(placements) != config.actor_rollout_ref.rollout.n_gpus_per_node:
            raise ValueError("Dual Engine rollout requires explicit shared placement metadata for every slot")
        self.rollout_replicas = []
        owner_cls = ray.remote(SharedDualEngineSlot)
        for index, placement in enumerate(placements):
            slot = owner_cls.options(
                name=slot_name(placement["id"]),
                namespace=NAMESPACE,
                get_if_exists=True,
                lifetime="detached",
                num_cpus=0,
                max_concurrency=128,
            ).remote()
            ray.get(slot.initialize.remote(config, placement["id"]))
            endpoint = ray.remote(TaskEndpoint).options(num_cpus=0).remote(slot, self.task)
            self.rollout_replicas.append(SharedReplica(index, f"dual://{self.task}/{placement['id']}", endpoint, slot))
        self.server_addresses = [replica.server_address for replica in self.rollout_replicas]
        self.server_handles = [replica.server_handle for replica in self.rollout_replicas]
        self.global_load_balancer = ray.remote(WeaveLoadBalancer).options(num_cpus=0).remote(servers={})

    def get_replicas(self):
        return self.rollout_replicas

    def get_client(self, client_cls=LLMServerClient):
        return client_cls(config=self.config, load_balancer_handle=self.global_load_balancer)

    def publish(self, weights, step):
        started = time.monotonic()
        ray.get(
            [replica.slot.publish.remote(self.task, step, weights, self.signature) for replica in self.rollout_replicas]
        )
        ray.get([replica.server_handle.set_step.remote(step) for replica in self.rollout_replicas])
        self.step = step
        return {"dual_engine/prepare_seconds": time.monotonic() - started}

    def activate(self, worker_ids):
        if self.step is None:
            raise RuntimeError("Cannot activate a task before weight publication")
        ray.get([self.rollout_replicas[index].slot.activate.remote(self.task, self.step) for index in worker_ids])

    def release(self, worker_ids):
        ray.get([self.rollout_replicas[index].slot.release.remote(self.task) for index in worker_ids])


class DualEngineCheckpointManager:
    def __init__(self, actor_wg, manager):
        self.actor_wg = actor_wg
        self.manager = manager

    def update_weights(self, global_steps):
        started = time.monotonic()
        images = ray.get(self.actor_wg.export_rollout_cpu_weights())
        nonempty = [image for image in images if image is not None]
        if len(nonempty) != 1 or not nonempty[0]:
            raise RuntimeError("FSDP export must return exactly one complete CPU image")
        metrics = self.manager.publish(ray.put(nonempty[0]), global_steps)
        metrics["dual_engine/publish_seconds"] = time.monotonic() - started
        return metrics
