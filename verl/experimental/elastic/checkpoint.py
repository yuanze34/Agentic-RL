# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Portable, trusted-local-filesystem checkpoints for quiescent, pure-DP workers.

PyTorch >= 2.8 is required. Its DCP state-dict APIs convert both FSDP1 flat
parameters and FSDP2 DTensors to canonical full model/optimizer dictionaries.
Every loading rank reads the full CPU checkpoint: rank-zero broadcasting does
not support FSDP1's legacy ShardedTensor. Budget host memory accordingly.

Only complete model/Adam/AdamW/extra checkpoints are supported, after at least one
optimizer update. A checkpoint is immutable, and COMMITTED.json is published
only after every rank has durably written its RNG state. SHA256 validates bytes,
not authenticity; never load checkpoints from an untrusted source. Cross-PyTorch
minor-version loading is deliberately rejected because DCP APIs are experimental.

Existing ranks recover their exact RNG streams. Added ranks receive seeds derived
from the saved rank-zero RNG digest, global step and new rank. Changing DP size
changes sample/collective ordering and cannot promise bitwise random continuity.
The caller must stop updates and hold membership fixed throughout save/load.
"""

import hashlib
import io
import json
import os
import pickle
import random
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from packaging.version import Version
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)

from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, writer: Callable) -> None:
    staging = path.with_name(path.name + ".writing")
    with staging.open("xb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _qualified_name(obj: object) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _schema(state: dict) -> dict:
    result = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or type(value) is not torch.Tensor:
            raise ValueError(f"Unsupported non-tensor or non-full model state: {name}")
        result[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return result


class PortableCheckpointManager(BaseCheckpointManager):
    """BaseCheckpointManager-compatible full-state training restart manager.

    ``previous_global_step`` is restored on load (the standard hook returns None).
    ``processing_class`` is retained but HF export/adapter-only/remote checkpoints
    are intentionally unsupported. No existing FSDP shard checkpoints are accepted.
    All methods are collective on the default process group.
    """

    FORMAT = "verl.elastic.full-state"
    FORMAT_VERSION = 1
    COMMIT_FILE = "COMMITTED.json"

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler=None,
        processing_class=None,
        checkpoint_config=None,
    ):
        if Version(torch.__version__.split("+")[0]) < Version("2.8"):
            raise NotImplementedError("Portable checkpoints require PyTorch >= 2.8 DCP state_dict APIs")
        if not dist.is_initialized():
            raise RuntimeError("Portable checkpoints require an initialized default process group")
        if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW):
            raise NotImplementedError("Portable checkpoints currently support standard torch Adam and AdamW only")
        super().__init__(model, optimizer, lr_scheduler, processing_class, checkpoint_config)
        required = {"model", "optimizer", "extra"}
        for contents in (self.checkpoint_save_contents, self.checkpoint_load_contents):
            if set(contents) != required or len(contents) != len(required):
                raise NotImplementedError("Portable training checkpoints require exactly model, optimizer, extra")
        if self.should_save_lora_only:
            raise NotImplementedError("Adapter-only portable checkpoints are unsupported")
        self._torch_version = ".".join(torch.__version__.split("+")[0].split(".")[:2])

    def _local_phase(self, action: Callable[[], Any]) -> Any:
        """Propagate local validation/I/O failures before the next collective."""
        error = None
        result = None
        try:
            result = action()
        except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [None] * self.world_size
        dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError("Portable checkpoint failed: " + "; ".join(x for x in errors if x))
        return result

    def _agree(self, value: object) -> None:
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        if any(item != values[0] for item in values):
            raise ValueError("All ranks must agree on checkpoint path, step, configuration and scheduler state")

    @staticmethod
    def _path(local_path: str, hdfs_path: str | None) -> Path:
        if hdfs_path is not None or local_path is None:
            raise NotImplementedError("Portable checkpoints require local_path and do not support remote storage")
        path = os.fspath(local_path)
        if "://" in path or path.startswith(("hdfs:", "s3:", "gs:", "file:")):
            raise NotImplementedError("Portable checkpoints accept filesystem paths, not URI paths")
        return Path(path).absolute()

    @staticmethod
    def _rng_state() -> dict:
        return {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        }

    @staticmethod
    def _restore_rng(state: dict) -> None:
        cuda = state["cuda"]
        if (cuda is not None) != torch.cuda.is_initialized():
            raise ValueError("Checkpoint and worker RNG device types differ (CPU versus initialized CUDA)")
        if cuda is not None and len(cuda) != torch.cuda.device_count():
            raise ValueError("Checkpoint and worker CUDA RNG device counts differ")
        torch.set_rng_state(state["cpu"])
        np.random.set_state(state["numpy"])
        random.setstate(state["random"])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)

    @staticmethod
    def _scheduler_digest(state: dict | None) -> str:
        stream = io.BytesIO()
        torch.save(state, stream)
        return hashlib.sha256(stream.getvalue()).hexdigest()

    def _validate_optimizer_initialized(self):
        if not self.optimizer.state:
            raise ValueError("Save requires an initialized optimizer after at least one real update")

    def save_checkpoint(
        self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep: int = None
    ):
        path = self._path(local_path, hdfs_path)
        if type(global_step) is not int or global_step < 0:
            raise ValueError("global_step must be a nonnegative integer")
        if max_ckpt_to_keep is not None and (type(max_ckpt_to_keep) is not int or max_ckpt_to_keep < 1):
            raise ValueError("max_ckpt_to_keep must be None or a positive integer")
        rng = self._rng_state()
        scheduler = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
        self._agree((str(path), global_step, max_ckpt_to_keep, self._scheduler_digest(scheduler)))

        def prepare():
            self._validate_optimizer_initialized()
            if self.rank == 0:
                path.mkdir(parents=True, exist_ok=True)
                if any(path.iterdir()):
                    raise ValueError(
                        "Checkpoint directory must be empty; committed/incomplete saves cannot be overwritten"
                    )
                with (path / "IN_PROGRESS").open("xb") as stream:
                    stream.write(b"immutable portable checkpoint\n")
                    stream.flush()
                    os.fsync(stream.fileno())

        self._local_phase(prepare)
        model_state, optim_state = get_state_dict(
            self.model, self.optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True, strict=True)
        )

        def write_states():
            _atomic_write(path / f"rng_rank_{self.rank}.pt", lambda stream: torch.save(rng, stream))
            if self.rank == 0:
                _schema(model_state)
                self._validate_optimizer(optim_state)
                payload = {
                    "model": model_state,
                    "optimizer": optim_state,
                    "scheduler": scheduler,
                    "global_step": global_step,
                }
                _atomic_write(path / "training.pt", lambda stream: torch.save(payload, stream))

        self._local_phase(write_states)

        def commit():
            if self.rank != 0:
                return
            files = ["training.pt", *(f"rng_rank_{rank}.pt" for rank in range(self.world_size))]
            metadata = {
                "format": self.FORMAT,
                "version": self.FORMAT_VERSION,
                "torch_version": self._torch_version,
                "world_size": self.world_size,
                "global_step": global_step,
                "contents": ["model", "optimizer", "extra"],
                "optimizer_class": _qualified_name(self.optimizer),
                "scheduler_class": _qualified_name(self.lr_scheduler) if self.lr_scheduler is not None else None,
                "model_schema": _schema(model_state),
                "files": {name: _digest(path / name) for name in files},
            }
            encoded = json.dumps(metadata, sort_keys=True).encode()
            _atomic_write(path / self.COMMIT_FILE, lambda stream: stream.write(encoded))

        self._local_phase(commit)
        self.previous_global_step = global_step
        if self.rank == 0:
            self._local_phase(lambda: self.register_checkpoint(str(path), max_ckpt_to_keep))
        else:
            self._local_phase(lambda: None)

    @staticmethod
    def _validate_optimizer(state: dict) -> None:
        if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
            raise ValueError("Invalid canonical optimizer checkpoint")
        groups = state["param_groups"]
        if not isinstance(groups, list) or not groups or not state["state"]:
            raise ValueError("Missing optimizer groups or initialized moments")
        names = [name for group in groups for name in group["params"]]
        if len(names) != len(set(names)) or not all(isinstance(name, str) for name in names):
            raise ValueError("Invalid canonical optimizer parameter names")
        if set(state["state"]) != set(names):
            raise ValueError("Every optimizer parameter must have initialized moments (unused parameters unsupported)")
        for name, moment in state["state"].items():
            if not {"step", "exp_avg", "exp_avg_sq"}.issubset(moment):
                raise ValueError(f"Missing Adam moments for {name}")

    def _read(self, path: Path) -> tuple[dict, dict, dict]:
        marker = path / self.COMMIT_FILE
        if not marker.is_file():
            raise ValueError("Not a committed portable checkpoint (legacy shards/incomplete saves are unsupported)")
        metadata = json.loads(marker.read_text())
        expected = {
            "format": self.FORMAT,
            "version": self.FORMAT_VERSION,
            "torch_version": self._torch_version,
            "contents": ["model", "optimizer", "extra"],
            "optimizer_class": _qualified_name(self.optimizer),
            "scheduler_class": _qualified_name(self.lr_scheduler) if self.lr_scheduler is not None else None,
        }
        if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError("Unsupported checkpoint metadata, optimizer, scheduler or PyTorch minor version")
        world_size = metadata.get("world_size")
        step = metadata.get("global_step")
        if type(world_size) is not int or world_size < 1 or type(step) is not int or step < 0:
            raise ValueError("Invalid checkpoint world_size/global_step metadata")
        names = {"training.pt", *(f"rng_rank_{rank}.pt" for rank in range(world_size))}
        if not isinstance(metadata.get("files"), dict) or set(metadata["files"]) != names:
            raise ValueError("Incomplete checkpoint file manifest")
        for name, digest in metadata["files"].items():
            if _digest(path / name) != digest:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
        payload = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or set(payload) != {"model", "optimizer", "scheduler", "global_step"}:
            raise ValueError("Incomplete training state")
        if payload["global_step"] != step or _schema(payload["model"]) != metadata.get("model_schema"):
            raise ValueError("Training state does not match committed metadata")
        if (payload["scheduler"] is None) != (self.lr_scheduler is None):
            raise ValueError("LR scheduler presence differs from checkpoint")
        self._validate_optimizer(payload["optimizer"])
        saved_rank = self.rank if self.rank < world_size else 0
        rng = torch.load(path / f"rng_rank_{saved_rank}.pt", map_location="cpu", weights_only=False)
        if not isinstance(rng, dict) or set(rng) != {"cpu", "numpy", "random", "cuda"}:
            raise ValueError("Invalid RNG checkpoint")
        return metadata, payload, rng

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False):
        if local_path is None and hdfs_path is None:
            return
        path = self._path(local_path, hdfs_path)
        self._agree((str(path), bool(del_local_after_load)))
        metadata, payload, rng = self._local_phase(lambda: self._read(path))
        self._agree(metadata)
        self.restore_state(metadata, payload, rng)
        if del_local_after_load:
            self._local_phase(lambda: shutil.rmtree(path) if self.rank == 0 else None)

    def restore_state(self, metadata: dict, payload: dict, rng: dict) -> None:
        """Restore an already validated canonical state on the current DP group."""
        current, current_optim = get_state_dict(
            self.model, self.optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True, strict=True)
        )

        def validate_model():
            if self.rank == 0 and _schema(current) != metadata["model_schema"]:
                raise ValueError("Current model names, shapes or dtypes differ from checkpoint")
            if self.rank == 0:
                saved_groups = [group["params"] for group in payload["optimizer"]["param_groups"]]
                current_groups = [group["params"] for group in current_optim["param_groups"]]
                if saved_groups != current_groups:
                    raise ValueError("Current canonical optimizer parameter group layout differs from checkpoint")
            cuda = rng["cuda"]
            if (cuda is not None) != torch.cuda.is_initialized():
                raise ValueError("Checkpoint and worker RNG device types differ")
            if cuda is not None and len(cuda) != torch.cuda.device_count():
                raise ValueError("Checkpoint and worker CUDA RNG device counts differ")

        self._local_phase(validate_model)
        current.clear()
        current_optim.clear()
        incompatible = set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=payload["model"],
            optim_state_dict=payload["optimizer"],
            options=StateDictOptions(full_state_dict=True, cpu_offload=True, strict=True),
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Model state was not restored strictly: {incompatible}")

        def restore_extra():
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(payload["scheduler"])
            if self.rank < metadata["world_size"]:
                self._restore_rng(rng)
            else:
                material = f"{metadata['files']['rng_rank_0.pt']}:{metadata['global_step']}:{self.rank}"
                seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
                random.seed(seed)
                np.random.seed(seed % 2**32)
                torch.random.default_generator.manual_seed(seed)
                if torch.cuda.is_initialized():
                    torch.cuda.manual_seed_all(seed)

        self._local_phase(restore_extra)
        self.previous_global_step = metadata["global_step"]
