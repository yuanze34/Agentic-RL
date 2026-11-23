"""vLLM full-model CPU packing and level-2 Engine rotation."""

from __future__ import annotations

import os
import time
from multiprocessing.shared_memory import SharedMemory

import ray
import torch

from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


def _cpu_image_metadata(weights: dict[str, torch.Tensor]):
    if not weights:
        raise ValueError("A full-model CPU image cannot be empty")
    metadata = []
    offset = 0
    for name, tensor in weights.items():
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError(f"Weight {name!r} must be a CPU tensor")
        if tensor.layout != torch.strided or tensor.is_quantized or tensor.numel() == 0:
            raise ValueError(f"Weight {name!r} must be a nonempty dense unquantized tensor")
        alignment = tensor.element_size()
        offset = (offset + alignment - 1) // alignment * alignment
        metadata.append((name, str(tensor.dtype).removeprefix("torch."), list(tensor.shape), offset, tensor.numel()))
        offset += tensor.nbytes
    return offset, metadata


class DualEngineWorkerExtension(vLLMColocateWorkerExtension):
    def prepare_cpu_image_shm(self, task: str, step: int, name: str, metadata: list):
        memory = SharedMemory(name=name)
        weights = {}
        try:
            for key, dtype, shape, offset, count in metadata:
                weights[key] = torch.frombuffer(
                    memory.buf, dtype=getattr(torch, dtype), count=count, offset=offset
                ).reshape(shape)
            return self.prepare_cpu_image(task, step, weights)
        finally:
            weights.clear()
            memory.close()

    def prepare_cpu_image(self, task: str, step: int, weights: dict):
        model = self.model_runner.model
        params = dict(model.named_parameters())
        required = set(params)
        for packed, sources in getattr(model, "packed_modules_mapping", {}).items():
            for name in tuple(required):
                if f".{packed}." in name:
                    required.remove(name)
                    required.update(name.replace(f".{packed}.", f".{source}.") for source in sources)
        missing_inputs = required - weights.keys()
        if missing_inputs:
            raise ValueError(f"Incomplete HF weight image: {sorted(missing_inputs)}")
        if any(tensor.device.type != "cpu" for tensor in weights.values()):
            raise ValueError("Standby preparation accepts CPU weights only")
        original = {name: param.data for name, param in params.items()}
        self._dual_image = None
        try:
            # Run the actual vLLM fused/TP weight loaders against CPU storage,
            # without touching the sleeping Engine's unmapped CUDA pointers.
            storage = {
                name: torch.empty_like(tensor, device="cpu", pin_memory=tensor.device.type == "cuda")
                for name, tensor in original.items()
            }
            for name, param in params.items():
                param.data = storage[name]
            loaded = model.load_weights(weights.items())
            missing = set(params) - loaded
            if missing:
                raise ValueError(f"Incomplete full-model image; missing parameters: {sorted(missing)}")
            # Some loaders rebind parameter storage to the input image. Retain
            # our owned pinned buffers, never views into transient shared memory.
            with torch.no_grad():
                for name, param in params.items():
                    storage[name].copy_(param.detach())
            self._dual_image = (task, step, storage)
        finally:
            for name, param in params.items():
                param.data = original[name]
        return {"parameters": len(params), "cpu_bytes": sum(p.nbytes for p in self._dual_image[2].values())}

    def load_cpu_image(self, task: str, step: int):
        image = getattr(self, "_dual_image", None)
        if image is None or image[:2] != (task, step):
            raise ValueError("Prepared image does not match the requested task/version")
        params = dict(self.model_runner.model.named_parameters())
        started = time.monotonic()
        with torch.no_grad():
            for name, tensor in image[2].items():
                params[name].copy_(tensor, non_blocking=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        self._dual_image = None
        return {
            "copy_seconds": elapsed,
            "parameters": len(params),
            "bytes": sum(tensor.nbytes for tensor in image[2].values()),
            "pinned_bytes": sum(tensor.nbytes for tensor in image[2].values() if tensor.is_pinned()),
        }

    def dual_image_status(self):
        from vllm.device_allocator import get_mem_allocator_instance

        allocator = get_mem_allocator_instance()
        records = allocator.pointer_to_data.values()
        image = getattr(self, "_dual_image", None)
        tensors = image[2].values() if image is not None else ()
        return {
            "pid": os.getpid(),
            "allocated_bytes": torch.cuda.memory_allocated(),
            "virtual_pool_bytes": allocator.get_current_usage(),
            "cpu_backup_bytes": sum(
                record.cpu_backup_tensor.nbytes
                for record in allocator.pointer_to_data.values()
                if record.cpu_backup_tensor is not None
            ),
            "cpu_image": image is not None,
            "cpu_image_bytes": sum(tensor.nbytes for tensor in tensors),
            "cpu_image_pinned_bytes": sum(tensor.nbytes for tensor in tensors if tensor.is_pinned()),
            "allocation_tags": sorted({record.tag for record in records}),
        }


class DualEngineServer(vLLMHttpServer):
    def __init__(self, config, index: int):
        super().__init__(
            config=config.actor_rollout_ref.rollout,
            model_config=config.actor_rollout_ref.model,
            rollout_mode=RolloutMode.STANDALONE,
            workers=[],
            replica_rank=index,
            node_rank=0,
            gpus_per_node=1,
            nnodes=1,
            cuda_visible_devices=",".join(str(int(device)) for device in ray.get_gpu_ids()),
        )
        self._image_sleeping = False
        self._prepared_version = None

    def _get_worker_extension_cls(self):
        return "verl.experimental.elastic.dual_engine_vllm.DualEngineWorkerExtension"

    async def sleep_image(self):
        started = time.monotonic()
        await self.engine.sleep(level=2, mode="wait")
        slept = time.monotonic()
        await self.engine.reset_encoder_cache()
        self._image_sleeping = True
        return {
            "sleep_seconds": slept - started,
            "encoder_reset_seconds": time.monotonic() - slept,
        }

    async def prepare_image(self, task, step, weights):
        if not self._image_sleeping:
            raise RuntimeError("CPU image preparation requires a sleeping standby Engine")
        self._prepared_version = None
        # vLLM utility RPC kwargs are untyped msgpack and cannot reconstruct
        # arbitrary Tensor dictionaries. Transfer only CPU shared-memory metadata.
        size, metadata = _cpu_image_metadata(weights)
        memory = SharedMemory(create=True, size=size)
        try:
            # Reserve tmpfs pages before copying: an oversized mmap write would
            # otherwise terminate the actor with SIGBUS instead of raising ENOSPC.
            with open(f"/dev/shm/{memory.name}", "r+b", buffering=0) as allocation:
                os.posix_fallocate(allocation.fileno(), 0, size)
            for name, _, _, offset, _ in metadata:
                tensor = weights[name]
                view = torch.frombuffer(memory.buf, dtype=torch.uint8, count=tensor.nbytes, offset=offset)
                view.copy_(tensor.detach().contiguous().reshape(-1).view(torch.uint8))
                del view
            results = await self.engine.collective_rpc(
                "prepare_cpu_image_shm",
                kwargs={"task": task, "step": step, "name": memory.name, "metadata": metadata},
            )
        finally:
            memory.close()
            memory.unlink()
        self._prepared_version = (task, step)
        return results

    async def activate_image(self, task, step):
        if not self._image_sleeping or self._prepared_version != (task, step):
            raise RuntimeError("Engine activation requires a matching prepared CPU image")
        started = time.monotonic()
        await self.engine.wake_up(tags=["weights", "kv_cache"])
        woke = time.monotonic()
        workers = await self.engine.collective_rpc("load_cpu_image", kwargs={"task": task, "step": step})
        loaded = time.monotonic()
        await self.engine.reset_prefix_cache(reset_connector=True)
        reset = time.monotonic()
        await self.engine.resume_generation()
        self.global_steps = step
        self._prepared_version = None
        self._image_sleeping = False
        return {
            "wake_seconds": woke - started,
            "load_rpc_seconds": loaded - woke,
            "cache_reset_seconds": reset - loaded,
            "resume_seconds": time.monotonic() - reset,
            "workers": workers,
        }

    async def image_status(self):
        return {
            "sleeping": self._image_sleeping,
            "workers": await self.engine.collective_rpc("dual_image_status"),
        }
