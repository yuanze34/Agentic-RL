"""Resident GPU workers with replaceable, canonical-checkpointed FSDP groups."""

from __future__ import annotations

import gc
import hashlib
import io
import logging
import os
import weakref
from copy import deepcopy

import ray
import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

from verl.experimental.elastic.checkpoint import _schema
from verl.experimental.elastic.engine_workers import ElasticDetachActorWorker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup

logger = logging.getLogger(__name__)


def initialize_adam_state_for_cpu_restore(optimizer):
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        raise TypeError("Resident CPU-offloaded restore supports only Adam/AdamW")
    if optimizer.state:
        return
    # DCP otherwise runs a dummy GPU Adam step, allocating grads/moments/foreach temporaries.
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            step_device = parameter.device if group.get("capturable") or group.get("fused") else "cpu"
            state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)
            state["exp_avg"] = torch.zeros_like(parameter, device="cpu")
            state["exp_avg_sq"] = torch.zeros_like(parameter, device="cpu")
            if group.get("amsgrad"):
                state["max_exp_avg_sq"] = torch.zeros_like(parameter, device="cpu")


def release_retired_module_tensors(module):
    """Sever tensor views in an irrevocably retired FSDP module graph."""
    for child in module.modules():
        # FSDP's shared-weight views are plain attributes, not registered parameters;
        # Module.cpu() does not visit them and they can pin the complete flat storage.
        for name, value in list(vars(child).items()):
            if isinstance(value, torch.Tensor):
                setattr(child, name, None)
        child._parameters.clear()
        child._buffers.clear()


class ResidentDetachActorWorker(ElasticDetachActorWorker):
    def __init__(self):
        Worker.__init__(self)
        self.actor = None
        self.ref = None
        self.checkpoint_engine = None
        self._resident_prepared = False
        self._visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        torch.cuda.init()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def resident_prepare(self, config, role, rank, world_size, master_addr, master_port):
        if self._resident_prepared or torch.distributed.is_initialized():
            raise RuntimeError("Resident rank must be released before preparation")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != self._visible_devices:
            raise RuntimeError("Resident CUDA device binding changed")
        os.environ.update(
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            LOCAL_WORLD_SIZE=str(world_size),
            MASTER_ADDR=str(master_addr),
            MASTER_PORT=str(master_port),
        )
        ElasticDetachActorWorker.__init__(self, deepcopy(config), role)
        self._is_rollout = False
        self.cpu_saved_models = {}
        self._resident_prepared = True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        if self.config.actor.strategy != "fsdp" or not self.actor.engine._is_offload_optimizer:
            raise ValueError("Resident restore currently requires FSDP1 with optimizer CPU offload")
        initialize_adam_state_for_cpu_restore(self.actor.engine.optimizer)
        return super().load_checkpoint(local_path, hdfs_path, del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def resident_snapshot(self, step):
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

        engine = self.actor.engine
        manager = engine.checkpoint_manager
        torch.cuda.synchronize()
        manager._local_phase(manager._validate_optimizer_initialized)
        scheduler = engine.lr_scheduler.state_dict()
        manager._agree((step, manager._scheduler_digest(scheduler)))
        rng = manager._rng_state()
        rngs = [None] * self.world_size
        torch.distributed.all_gather_object(rngs, rng)
        if engine._is_offload_param:
            load_fsdp_model_to_gpu(engine.module)
        model, optimizer = get_state_dict(
            engine.module,
            engine.optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True, strict=True),
        )
        if engine._is_offload_param:
            offload_fsdp_model_to_cpu(engine.module)
        torch.cuda.synchronize()
        result = None

        def capture():
            nonlocal result
            if self.rank != 0:
                return
            manager._validate_optimizer(optimizer)
            payload = {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "global_step": step,
            }
            encoded = io.BytesIO()
            torch.save(rngs[0], encoded)
            metadata = {
                "world_size": self.world_size,
                "global_step": step,
                "model_schema": _schema(model),
                "files": {"rng_rank_0.pt": hashlib.sha256(encoded.getvalue()).hexdigest()},
            }
            # One archive preserves flat-storage aliasing; Ray's per-tensor pickle path can expand views.
            archive = io.BytesIO()
            torch.save({"metadata": metadata, "payload": payload, "rngs": rngs}, archive)
            result = archive.getvalue()

        manager._local_phase(capture)
        return {"state": ray.put(result)} if self.rank == 0 else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def resident_restore(self, snapshot):
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu, offload_fsdp_optimizer

        engine = self.actor.engine
        snapshot = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=False)
        initialize_adam_state_for_cpu_restore(engine.optimizer)
        if engine._is_offload_param:
            load_fsdp_model_to_gpu(engine.module)
        metadata = snapshot["metadata"]
        engine.checkpoint_manager.restore_state(
            metadata,
            snapshot["payload"],
            snapshot["rngs"][self.rank if self.rank < metadata["world_size"] else 0],
        )
        torch.distributed.barrier()
        if engine._is_offload_param:
            offload_fsdp_model_to_cpu(engine.module)
        if engine._is_offload_optimizer:
            offload_fsdp_optimizer(engine.optimizer)
        torch.cuda.synchronize()
        # Restoring state can cache gigabytes needed by the next tenant on this slot.
        del snapshot
        gc.collect()
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def resident_release(self):
        if not self._resident_prepared:
            raise RuntimeError("Cannot release an inactive resident rank")
        transport = self.checkpoint_engine
        if transport is not None:
            if not transport.rebuild_group:
                raise RuntimeError("Resident transport requires finalized per-sync NCCL groups")
            if getattr(transport, "rank", None) is not None:
                raise RuntimeError("Resident weight transport is still active")
        torch.cuda.synchronize()
        # Checkpointed boundaries have no gradient accumulation to retain across a group replacement.
        self.actor.engine.optimizer.zero_grad(set_to_none=True)
        for worker in (self.actor, self.ref):
            if worker is not None:
                release_retired_module_tensors(worker.engine.module)
        retired = [weakref.ref(worker.engine.module) for worker in (self.actor, self.ref) if worker is not None]
        del worker
        if transport is not None:
            if hasattr(transport, "socket"):
                context = transport.socket.context
                transport.socket.close(linger=0)
                context.term()
        self.actor = None
        self.ref = None
        self.checkpoint_engine = None
        self.cpu_saved_models = {}
        self._strategy_handlers = None
        gc.collect()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        gc.collect()
        workspace_bytes_before = torch.cuda.memory_allocated()
        # Retired FSDP streams leave BLAS workspaces cached even after their modules are collected.
        torch._C._cuda_clearCublasWorkspaces()
        workspace_bytes_after = torch.cuda.memory_allocated()
        torch.cuda.empty_cache()
        # Reference CPUOffload caches pinned shards even after the old group is retired.
        torch._C._host_emptyCache()
        alive = sum(reference() is not None for reference in retired)
        logger.info(
            "RESIDENT released retired_models=%d cuda_allocated=%d cuda_reserved=%d",
            alive,
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
        )
        if alive:
            raise RuntimeError("Retired FSDP modules still have live owners")
        self._resident_prepared = False
        return {
            "retired_models": alive,
            "workspace_bytes_released": workspace_bytes_before - workspace_bytes_after,
            "cuda_allocated_bytes": workspace_bytes_after,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def resident_info(self):
        initialized = torch.distributed.is_initialized()
        steps = []
        if self.actor is not None:
            steps = sorted(
                {int(state["step"].item()) for state in self.actor.engine.optimizer.state.values() if "step" in state}
            )
        storage = {}
        for role, worker in (("actor", self.actor), ("ref", self.ref)):
            if worker is None:
                continue
            tensors = []
            for module in worker.engine.module.modules():
                tensors.extend(
                    (f"{role}.view.{name}", value)
                    for name, value in vars(module).items()
                    if isinstance(value, torch.Tensor)
                )
            for handle in worker.engine.module._all_handles:
                parameter = handle.flat_param
                tensors.append((f"{role}.flat", parameter))
                tensors.extend(
                    (f"{role}.flat.{name}", value)
                    for name, value in vars(parameter).items()
                    if isinstance(value, torch.Tensor)
                )
                if parameter.grad is not None:
                    tensors.append((f"{role}.grad", parameter.grad))
            optimizer = worker.engine.optimizer
            if optimizer is not None:
                for state in optimizer.state.values():
                    tensors.extend(
                        (f"{role}.optimizer.{name}", value)
                        for name, value in state.items()
                        if isinstance(value, torch.Tensor)
                    )
            for name, tensor in tensors:
                if tensor.is_cuda:
                    backing = tensor.untyped_storage()
                    key = str(backing.data_ptr())
                    entry = storage.setdefault(key, {"bytes": backing.nbytes(), "owners": []})
                    if name not in entry["owners"]:
                        entry["owners"].append(name)
        return {
            "pid": os.getpid(),
            "actor_id": ray.get_runtime_context().get_actor_id(),
            "device": torch.cuda.current_device(),
            "visible_devices": self._visible_devices,
            "rank": torch.distributed.get_rank() if initialized else None,
            "world_size": torch.distributed.get_world_size() if initialized else 0,
            "optimizer_steps": steps,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_tensor_storage": list(storage.values()),
        }


class ResidentTrainingPool:
    """A fixed physical lease; only the active FSDP rank set changes."""

    def __init__(self, resource_pool, *, name_prefix, wg_kwargs):
        self.ray_cls = RayClassWithInitArgs(ray.remote(ResidentDetachActorWorker))
        self.pool = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=self.ray_cls,
            name_prefix=name_prefix,
            **wg_kwargs,
        )
        self.capacity = self.pool.world_size
        self.active_size = 0
        self.active_indices = []

    def prepare(self, config, role, size, indices=None):
        if type(size) is not int or not 1 <= size <= self.capacity:
            raise ValueError(f"Training DP {size} is outside resident capacity {self.capacity}")
        if self.active_size:
            raise RuntimeError("Release the old FSDP group before preparing its replacement")
        indices = list(range(size)) if indices is None else list(indices)
        if (
            len(indices) != size
            or len(set(indices)) != size
            or any(type(index) is not int or not 0 <= index < self.capacity for index in indices)
        ):
            raise ValueError("Resident rank indices must be distinct physical slots within the pool")
        handles = [self.pool.workers[index] for index in indices]
        address, port = ray.get(handles[0].get_available_master_addr_port.remote())
        ray.get(
            [
                worker.resident_prepare.remote(config, role, rank, size, address, port)
                for rank, worker in enumerate(handles)
            ]
        )
        self.active_size = size
        self.active_indices = indices
        return RayWorkerGroup.from_detached(
            ray_cls_with_init=self.ray_cls,
            name_prefix=self.pool.name_prefix,
            worker_names=[self.pool.worker_names[index] for index in indices],
            worker_handles=handles,
        )

    def release(self):
        if not self.active_size:
            raise RuntimeError("No active FSDP group to release")
        result = ray.get([self.pool.workers[index].resident_release.remote() for index in self.active_indices])
        self.active_size = 0
        self.active_indices = []
        return result

    def snapshot(self, step):
        results = ray.get([self.pool.workers[index].resident_snapshot.remote(step) for index in self.active_indices])
        return results[0]["state"]

    def restore(self, snapshot):
        ray.get([self.pool.workers[index].resident_restore.remote(snapshot) for index in self.active_indices])

    def identities(self):
        return ray.get([worker.resident_info.remote() for worker in self.pool.workers])
