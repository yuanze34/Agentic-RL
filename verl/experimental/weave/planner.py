"""Profile-driven WEAVE grouping and physical rollout-slot planning."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from .global_scheduler import Job, WeaveScheduler, per_time_cost


def build_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    jobs = manifest["jobs"]
    if not jobs:
        raise ValueError("A WEAVE manifest needs at least one job")
    slot_width = manifest.get("gpus_per_worker", 1)
    if not isinstance(slot_width, int) or slot_width < 1:
        raise ValueError("gpus_per_worker must be a positive integer")
    gpus_per_node = manifest.get("gpus_per_node", 8)
    if not isinstance(gpus_per_node, int) or gpus_per_node < slot_width:
        raise ValueError("gpus_per_node must fit a complete rollout TP*DP*PP worker")
    max_group_size = manifest.get("max_group_size", 5)
    simulate_steps = manifest.get("simulate_steps", 100)
    if max_group_size < 1 or simulate_steps < 1:
        raise ValueError("max_group_size and simulate_steps must be positive")
    scheduler = WeaveScheduler(
        per_time_cost,
        max_group_size=max_group_size,
        simulate_steps=simulate_steps,
        rollout_cost=manifest.get("rollout_cost", 1 / 3),
        train_cost=manifest.get("train_cost", 1.0),
    )
    profiles = {}
    for key in ("command", "overrides"):
        if not isinstance(manifest.get(key, []), list) or not all(
            isinstance(item, str) for item in manifest.get(key, [])
        ):
            raise ValueError(f"Manifest {key} must be a list of strings")
    for raw_spec in jobs:
        spec = {"command": manifest.get("command", []), **raw_spec}
        name = spec["job_name"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or name in profiles:
            raise ValueError(f"Invalid or duplicate WEAVE job name: {name!r}")
        for key in ("rollout_seconds", "train_seconds", "slo"):
            if not isinstance(spec[key], (int, float)) or not math.isfinite(spec[key]) or spec[key] <= 0:
                raise ValueError(f"{name}.{key} must be a finite positive number")
        if spec["slo"] <= 1:
            raise ValueError(f"{name}.slo must exceed 1 (upstream WEAVE uses a strict slowdown bound)")
        for key in ("rollout_workers", "train_gpus"):
            if not isinstance(spec[key], int) or spec[key] < 1:
                raise ValueError(f"{name}.{key} must be a positive integer")
        if spec["train_gpus"] > gpus_per_node and spec["train_gpus"] % gpus_per_node:
            raise ValueError(f"{name}.train_gpus must use complete nodes for multi-node training")
        for key in ("command", "overrides"):
            if not isinstance(spec.get(key, []), list) or not all(isinstance(item, str) for item in spec.get(key, [])):
                raise ValueError(f"{name}.{key} must be a list of strings")
        spec["overrides"] = [*manifest.get("overrides", []), *spec.get("overrides", [])]
        scheduler.add_job(
            Job(
                name,
                spec["rollout_seconds"],
                spec["train_seconds"],
                spec["slo"],
                rollout_width=spec["rollout_workers"],
                train_width=spec["train_gpus"],
            )
        )
        profiles[name] = spec

    groups = []
    for group in scheduler.job_groups.values():
        rollout_nodes = sorted(group.all_rollout_nodes, key=int)
        train_nodes = sorted(group.all_train_nodes, key=lambda node: int(node.removeprefix("TN")))
        rollout_index = {node: index for index, node in enumerate(rollout_nodes)}
        train_index = {node: index for index, node in enumerate(train_nodes)}
        planned_jobs = []
        for job in group.jobs:
            slots = [rollout_index[node] for node in sorted(job.rollout_nodes, key=int)]
            train = [train_index[node] for node in job.train_nodes]
            if train != list(range(len(train))):
                raise ValueError("WEAVE training placement must be a prefix of the group's training pool")
            planned_jobs.append(
                {
                    **profiles[job.job_id],
                    "rollout_slots": slots,
                    "gen": [slot * slot_width + lane for slot in slots for lane in range(slot_width)],
                    "train": train,
                }
            )
        groups.append(
            {
                "group_id": group.group_id,
                "num_gen_gpus": len(rollout_nodes) * slot_width,
                "num_train_gpus": len(train_nodes),
                "num_rollout_slots": len(rollout_nodes),
                "gpus_per_node": gpus_per_node,
                "jobs": planned_jobs,
            }
        )
    total_gen = sum(group["num_gen_gpus"] for group in groups)
    total_train = sum(group["num_train_gpus"] for group in groups)
    for key, required in (("num_gen_gpus", total_gen), ("num_train_gpus", total_train)):
        if key in manifest and required > manifest[key]:
            raise ValueError(f"WEAVE plan needs {required} {key}, capacity is {manifest[key]}")
    return {
        "version": 1,
        "gpus_per_worker": slot_width,
        "num_gen_gpus": total_gen,
        "num_train_gpus": total_train,
        "groups": groups,
        "case_statistics": scheduler.case_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the original WEAVE global grouping algorithm")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(json.loads(args.manifest.read_text()))
    args.output.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    print(
        f"WEAVE planned {len(plan['groups'])} groups: "
        f"{plan['num_gen_gpus']} rollout GPUs, {plan['num_train_gpus']} train GPUs"
    )


if __name__ == "__main__":
    main()
