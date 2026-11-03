"""Shared Ray placement groups for real WEAVE multi-tenant GPU pools."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verl.single_controller.ray.base import RayResourcePool, ResourcePoolManager, SubRayResourcePool


@dataclass(frozen=True)
class SharedPoolSpec:
    name: str
    num_gpus: int
    max_tenants_per_gpu: int
    id_hex: str | None = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("shared placement group name must not be empty")
        if self.num_gpus <= 0:
            raise ValueError("shared placement group must contain at least one GPU")
        if self.max_tenants_per_gpu <= 0:
            raise ValueError("max_tenants_per_gpu must be positive")


class SharedRayResourcePool(RayResourcePool):
    """Attach worker groups from different Ray jobs to one detached PG."""

    def __init__(self, spec: SharedPoolSpec):
        spec.validate()
        super().__init__(
            process_on_nodes=[spec.num_gpus],
            use_gpu=True,
            name_prefix=spec.name,
            max_colocate_count=spec.max_tenants_per_gpu,
            detached=True,
        )
        self.shared_spec = spec

    def get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name="cuda"):
        del strategy, name, device_name
        if self.pgs is not None:
            return self.pgs

        import ray

        if self.shared_spec.id_hex:
            from ray._raylet import PlacementGroupID
            from ray.util.placement_group import PlacementGroup

            placement_group = PlacementGroup(PlacementGroupID.from_hex(self.shared_spec.id_hex))
        else:
            from ray.util.placement_group import get_placement_group

            try:
                placement_group = get_placement_group(self.shared_spec.name)
            except ValueError as exc:
                raise RuntimeError(
                    f"WEAVE placement group {self.shared_spec.name!r} does not exist; "
                    "start the shared placement bootstrap before launching jobs"
                ) from exc
        ray.get(placement_group.ready())
        if placement_group.bundle_count != self.shared_spec.num_gpus:
            raise ValueError(
                f"WEAVE placement group {self.shared_spec.name!r} has "
                f"{placement_group.bundle_count} bundles, expected {self.shared_spec.num_gpus}"
            )
        self.pgs = [placement_group]
        return self.pgs


class WeaveResourcePoolManager(ResourcePoolManager):
    """ResourcePoolManager that reuses scheduler-owned detached GPU pools."""

    def __init__(
        self,
        resource_pool_spec: dict[str, list[int]],
        mapping: dict[int, str] | None,
        shared_pool_names: dict[str, str],
        max_tenants_per_gpu: int,
        shared_pool_ids: dict[str, str] | None = None,
        shared_pool_sizes: dict[str, int] | None = None,
        shared_pool_placements: dict[str, list[dict[str, Any]]] | None = None,
    ):
        super().__init__(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
            max_colocate_count=max_tenants_per_gpu,
        )
        self.shared_pool_names = shared_pool_names
        self.max_tenants_per_gpu = max_tenants_per_gpu
        self.shared_pool_ids = shared_pool_ids or {}
        self.shared_pool_sizes = shared_pool_sizes or {}
        self.shared_pool_placements = shared_pool_placements or {}

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            shared_name = self.shared_pool_names.get(resource_pool_name)
            if shared_name is None:
                raise ValueError(f"Missing shared placement group for resource pool {resource_pool_name!r}")
            placements = self.shared_pool_placements.get(resource_pool_name)
            if placements:
                if len(placements) < len(process_on_nodes):
                    raise ValueError("Not enough shared WEAVE training-node placements")
                pool = RayResourcePool(
                    process_on_nodes=process_on_nodes,
                    use_gpu=True,
                    name_prefix=shared_name,
                    max_colocate_count=self.max_tenants_per_gpu,
                    detached=True,
                )
                pool.pgs = []
                for required, placement in zip(process_on_nodes, placements, strict=False):
                    if required > placement["num_gpus"]:
                        raise ValueError("WEAVE node placement is smaller than the requested training shard")
                    node_pool = SharedRayResourcePool(
                        SharedPoolSpec(
                            name=placement["name"],
                            num_gpus=placement["num_gpus"],
                            max_tenants_per_gpu=self.max_tenants_per_gpu,
                            id_hex=placement["id"],
                        )
                    )
                    pool.pgs.extend(node_pool.get_placement_groups())
                self.resource_pool_dict[resource_pool_name] = pool
                continue
            if len(process_on_nodes) != 1:
                raise ValueError("Multi-node WEAVE training requires train_replica_placements from the global launcher")
            pool_size = self.shared_pool_sizes.get(resource_pool_name, process_on_nodes[0])
            if pool_size < process_on_nodes[0]:
                raise ValueError("WEAVE training placement group is smaller than the job")
            pool = SharedRayResourcePool(
                SharedPoolSpec(
                    name=shared_name,
                    num_gpus=pool_size,
                    max_tenants_per_gpu=self.max_tenants_per_gpu,
                    id_hex=self.shared_pool_ids.get(resource_pool_name),
                )
            )
            if pool_size == process_on_nodes[0]:
                self.resource_pool_dict[resource_pool_name] = pool
            else:
                self.resource_pool_dict[resource_pool_name] = SubRayResourcePool(
                    placement_groups=pool.get_placement_groups(),
                    start_bundle_index=0,
                    subgroup_world_size=process_on_nodes[0],
                    process_on_nodes=process_on_nodes,
                    use_gpu=True,
                    name_prefix=shared_name,
                    max_colocate_count=self.max_tenants_per_gpu,
                    detached=True,
                )

        # Resolving each PG validates existence and bundle counts. The normal
        # available-resource check cannot be used because detached PGs already
        # reserve their GPUs from Ray's free-resource accounting.
        for resource_pool in self.resource_pool_dict.values():
            resource_pool.get_placement_groups()


def create_detached_pool(spec: SharedPoolSpec) -> Any:
    """Create one scheduler-owned placement group, or validate an existing one."""

    spec.validate()
    import ray
    from ray.util.placement_group import get_placement_group, placement_group

    try:
        existing = get_placement_group(spec.name)
    except ValueError:
        existing = None

    if existing is not None:
        ray.get(existing.ready(), timeout=120)
        if existing.bundle_count != spec.num_gpus:
            raise ValueError(
                f"Existing placement group {spec.name!r} has {existing.bundle_count} bundles, expected {spec.num_gpus}"
            )
        return existing

    bundles = [{"CPU": spec.max_tenants_per_gpu, "GPU": 1} for _ in range(spec.num_gpus)]
    pool = placement_group(
        bundles=bundles,
        strategy="STRICT_PACK",
        name=spec.name,
        lifetime="detached",
    )
    try:
        ray.get(pool.ready(), timeout=120)
    except ray.exceptions.GetTimeoutError as exc:
        from ray.util.placement_group import remove_placement_group

        remove_placement_group(pool)
        raise TimeoutError(f"WEAVE placement group {spec.name!r} could not acquire its GPU bundles") from exc
    return pool


def create_weave_pools(
    gen_pool: SharedPoolSpec,
    train_pool: SharedPoolSpec,
) -> tuple[Any, Any]:
    """Create physically disjoint rollout and training PGs on one Ray node."""

    if gen_pool.name == train_pool.name:
        raise ValueError("rollout and training placement groups must have different names")
    return create_detached_pool(gen_pool), create_detached_pool(train_pool)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create detached WEAVE rollout/train Ray placement groups")
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--gen-pool", default="weave_gen_pool")
    parser.add_argument("--train-pool", default="weave_train_pool")
    parser.add_argument("--num-gen-gpus", type=int, required=True)
    parser.add_argument("--num-train-gpus", type=int, required=True)
    parser.add_argument("--max-tenants-per-gpu", type=int, default=2)
    parser.add_argument("--output-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import ray

    ray.init(address=args.ray_address)
    gen_pool, train_pool = create_weave_pools(
        SharedPoolSpec(args.gen_pool, args.num_gen_gpus, args.max_tenants_per_gpu),
        SharedPoolSpec(args.train_pool, args.num_train_gpus, args.max_tenants_per_gpu),
    )
    placement_info = {
        "gen": {"name": args.gen_pool, "id": gen_pool.id.hex(), "num_gpus": args.num_gen_gpus},
        "train": {"name": args.train_pool, "id": train_pool.id.hex(), "num_gpus": args.num_train_gpus},
        "max_tenants_per_gpu": args.max_tenants_per_gpu,
    }
    if args.output_file is not None:
        args.output_file.write_text(json.dumps(placement_info, sort_keys=True) + "\n")
    print(
        f"WEAVE placement ready: gen={args.gen_pool}[{args.num_gen_gpus}], "
        f"train={args.train_pool}[{args.num_train_gpus}], "
        f"max_tenants={args.max_tenants_per_gpu}"
    )


if __name__ == "__main__":
    main()
