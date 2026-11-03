"""WEAVE-compatible rollout/training GPU accounting."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .protocol import Phase


@dataclass
class JobStatus:
    job_name: str
    phase: Phase
    gpus_per_gen_worker: int
    gen_gpu_list: list[int]
    train_gpu_list: list[int]
    allocated_gpus: list[int] = field(default_factory=list)


def parse_job_offset(path: str | os.PathLike[str]) -> dict[str, tuple[int, int]]:
    """Parse the original ``job gen_offset train_offset`` mapping format."""

    offsets: dict[str, tuple[int, int]] = {}
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        job, gen_offset, train_offset = line.split()
        offsets[job] = (int(gen_offset), int(train_offset))
    return offsets


class ResourceManager:
    """Track WEAVE's two physical GPU pools and per-job allocations."""

    def __init__(
        self,
        gen_device_affinity: str,
        gen_device_ids: list[int],
        train_device_affinity: str,
        train_device_ids: list[int],
        offset_mapping_fn: str | os.PathLike[str],
        *,
        is_colo: bool | None = None,
        placements: dict[str, dict[str, list[int]]] | None = None,
    ) -> None:
        self.gen_device_affinity = gen_device_affinity
        self.train_device_affinity = train_device_affinity
        self.gen_available_devices = sorted(gen_device_ids.copy())
        self.train_available_devices = sorted(train_device_ids.copy())
        self.gen_resource_mapping: dict[int, str | None] = {device: None for device in gen_device_ids}
        self.train_resource_mapping: dict[int, str | None] = {device: None for device in train_device_ids}
        self.job_mapping: dict[str, JobStatus] = {}
        self.gen_cluster_size = len(gen_device_ids)
        self.train_cluster_size = len(train_device_ids)
        self.offset_mapping = parse_job_offset(offset_mapping_fn)
        self.placements = placements or {}
        self.is_colo = (os.environ.get("COLO", "0") == "1") if is_colo is None else is_colo

    @property
    def num_available_devices(self) -> int:
        return len(self.gen_available_devices) + len(self.train_available_devices)

    @property
    def cluster_size(self) -> int:
        return self.gen_cluster_size + self.train_cluster_size

    def register_job(self, job_status: JobStatus) -> None:
        assert len(job_status.gen_gpu_list) <= self.gen_cluster_size
        assert len(job_status.train_gpu_list) <= self.train_cluster_size
        assert job_status.job_name in self.offset_mapping
        for pool, local_devices, cluster in (
            ("gen", job_status.gen_gpu_list, self.gen_resource_mapping),
            ("train", job_status.train_gpu_list, self.train_resource_mapping),
        ):
            devices = [self.global_device(job_status.job_name, pool, device) for device in local_devices]
            if len(set(devices)) != len(devices) or any(device not in cluster for device in devices):
                raise ValueError(f"Invalid {pool} placement for {job_status.job_name}: {devices}")
        self.job_mapping[job_status.job_name] = job_status

    def global_device(self, job: str, pool: str, device: int) -> int:
        if job in self.placements:
            return self.placements[job][pool][device]
        return device + self.offset_mapping[job][0 if pool == "gen" else 1]

    def update_phase(self, job_name: str, phase: Phase) -> None:
        self.job_mapping[job_name].phase = phase

    def get_free_gpus(self, job_name: str, pool: Literal["gen", "train"]) -> list[int]:
        status = self.job_mapping[job_name]
        local_devices = status.gen_gpu_list if pool == "gen" else status.train_gpu_list
        available = self.gen_available_devices if pool == "gen" else self.train_available_devices
        return [device for device in local_devices if self.global_device(job_name, pool, device) in available]

    def allocate_worker(self, job_name: str, pool: Literal["gen", "train"]) -> list[int] | None:
        assert pool in ("gen", "train")
        status = self.job_mapping[job_name]
        if pool == "gen":
            gpus_per_worker = status.gpus_per_gen_worker
            assert self.gen_cluster_size % gpus_per_worker == 0
            start = min(status.gen_gpu_list)
            stop = max(status.gen_gpu_list) + 1
            for first_device in range(start, stop, gpus_per_worker):
                local_devices = list(range(first_device, first_device + gpus_per_worker))
                if not all(device in status.gen_gpu_list for device in local_devices):
                    continue
                devices = [self.global_device(job_name, pool, device) for device in local_devices]
                if not all(device in self.gen_available_devices for device in devices):
                    continue
                for device in devices:
                    self.gen_available_devices.remove(device)
                    assert self.gen_resource_mapping[device] is None
                    self.gen_resource_mapping[device] = job_name
                status.allocated_gpus.extend(devices)
                return local_devices
            return None

        required = len(status.train_gpu_list)
        free_devices = [device for device, owner in self.train_resource_mapping.items() if owner is None]
        assert len(free_devices) == len(self.train_available_devices)
        local_devices = self.get_free_gpus(job_name, "train")
        if len(local_devices) < required:
            return None
        devices = [self.global_device(job_name, pool, device) for device in local_devices]
        for device in devices:
            self.train_available_devices.remove(device)
            self.train_resource_mapping[device] = job_name
        self.train_available_devices.sort()
        assert status.allocated_gpus == []
        status.allocated_gpus = devices.copy()
        return local_devices

    def allocate_all(self, job_name: str) -> None:
        status = self.job_mapping[job_name]
        gen_devices = [self.global_device(job_name, "gen", device) for device in status.gen_gpu_list]
        train_devices = [self.global_device(job_name, "train", device) for device in status.train_gpu_list]
        for device in gen_devices:
            assert self.gen_resource_mapping[device] is None
            self.gen_resource_mapping[device] = job_name
            self.gen_available_devices.remove(device)
        for device in train_devices:
            assert self.train_resource_mapping[device] is None
            self.train_resource_mapping[device] = job_name
            self.train_available_devices.remove(device)
        status.allocated_gpus = list(set(gen_devices + train_devices)) if self.is_colo else gen_devices + train_devices

    def release(self, job_name: str, device_affinity: str, gpu_list: list[int]) -> None:
        assert device_affinity in (self.gen_device_affinity, self.train_device_affinity)
        status = self.job_mapping[job_name]

        if self.gen_device_affinity == self.train_device_affinity:
            for device in gpu_list:
                released = False
                if device in self.gen_resource_mapping and self.gen_resource_mapping[device] == job_name:
                    self.gen_resource_mapping[device] = None
                    self.gen_available_devices.append(device)
                    released = True
                if device in self.train_resource_mapping and self.train_resource_mapping[device] == job_name:
                    self.train_resource_mapping[device] = None
                    self.train_available_devices.append(device)
                    released = True
                if not released:
                    raise ValueError(f"GPU {device} is not allocated to {job_name}")
                status.allocated_gpus.remove(device)
            if self.is_colo:
                self.gen_available_devices = list(set(self.gen_available_devices))
                self.train_available_devices = list(set(self.train_available_devices))
        elif device_affinity == self.gen_device_affinity:
            for local_device in gpu_list:
                global_device = self.global_device(job_name, "gen", local_device)
                assert self.gen_resource_mapping[global_device] == job_name
                self.gen_resource_mapping[global_device] = None
                self.gen_available_devices.append(global_device)
                status.allocated_gpus.remove(global_device)
        else:
            for local_device in gpu_list:
                global_device = self.global_device(job_name, "train", local_device)
                assert self.train_resource_mapping[global_device] == job_name
                self.train_resource_mapping[global_device] = None
                self.train_available_devices.append(global_device)
                status.allocated_gpus.remove(global_device)

        self.gen_available_devices.sort()
        self.train_available_devices.sort()

    def cleanup_by_name(self, job_name: str) -> int:
        released = 0
        for device, owner in self.gen_resource_mapping.items():
            if owner == job_name:
                self.gen_resource_mapping[device] = None
                self.gen_available_devices.append(device)
                released += 1
        for device, owner in self.train_resource_mapping.items():
            if owner == job_name:
                self.train_resource_mapping[device] = None
                self.train_available_devices.append(device)
                released += 1
        self.job_mapping[job_name].allocated_gpus = []
        self.gen_available_devices.sort()
        self.train_available_devices.sort()
        return released

    def snapshot(self) -> dict:
        """Return deterministic state for parity tests and diagnostics."""

        return {
            "gen_available_devices": self.gen_available_devices.copy(),
            "train_available_devices": self.train_available_devices.copy(),
            "gen_resource_mapping": self.gen_resource_mapping.copy(),
            "train_resource_mapping": self.train_resource_mapping.copy(),
            "jobs": {
                name: {
                    "phase": status.phase.value,
                    "allocated_gpus": status.allocated_gpus.copy(),
                }
                for name, status in sorted(self.job_mapping.items())
            },
        }
