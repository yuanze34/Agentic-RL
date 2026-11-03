"""Bridge WEAVE global placements to the local scheduler mapping contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence


class PlacedJob(Protocol):
    job_id: str
    rollout_nodes: Sequence[str] | None
    train_nodes: Sequence[str] | None


@dataclass(frozen=True)
class JobPlacement:
    job_name: str
    gen_offset: int
    train_offset: int
    gen_gpu_list: list[int]
    train_gpu_list: list[int]


def _node_sort_key(node: str) -> tuple[int, int | str]:
    digits = "".join(character for character in node if character.isdigit())
    return (0, int(digits)) if digits else (1, node)


def _pool_index(nodes: Iterable[str]) -> dict[str, int]:
    return {node: index for index, node in enumerate(sorted(set(nodes), key=_node_sort_key))}


def _contiguous_local_ids(global_ids: list[int], job_name: str, pool: str) -> tuple[int, list[int]]:
    if not global_ids:
        raise ValueError(f"{job_name} has no {pool} placement")
    ordered = sorted(global_ids)
    expected = list(range(ordered[0], ordered[-1] + 1))
    if ordered != expected:
        raise ValueError(
            f"{job_name} {pool} placement {ordered} is not contiguous and cannot be represented "
            "by WEAVE's offset mapping"
        )
    return ordered[0], list(range(len(ordered)))


def build_job_placements(jobs: Iterable[PlacedJob]) -> list[JobPlacement]:
    """Convert placed jobs without changing WEAVE's offset representation."""

    jobs = list(jobs)
    rollout_index = _pool_index(node for job in jobs for node in (job.rollout_nodes or ()))
    train_index = _pool_index(node for job in jobs for node in (job.train_nodes or ()))
    placements: list[JobPlacement] = []
    for job in jobs:
        gen_offset, gen_gpu_list = _contiguous_local_ids(
            [rollout_index[node] for node in (job.rollout_nodes or ())], job.job_id, "rollout"
        )
        train_offset, train_gpu_list = _contiguous_local_ids(
            [train_index[node] for node in (job.train_nodes or ())], job.job_id, "train"
        )
        placements.append(
            JobPlacement(
                job_name=job.job_id,
                gen_offset=gen_offset,
                train_offset=train_offset,
                gen_gpu_list=gen_gpu_list,
                train_gpu_list=train_gpu_list,
            )
        )
    return placements


def write_mapping_file(placements: Iterable[JobPlacement], path: str | Path) -> None:
    """Write the exact ``job gen_offset train_offset`` local-scheduler format."""

    lines = ["# JobID gen_offset train_offset"]
    lines.extend(
        f"{placement.job_name} {placement.gen_offset} {placement.train_offset}"
        for placement in sorted(placements, key=lambda placement: placement.job_name)
    )
    Path(path).write_text("\n".join(lines) + "\n")
