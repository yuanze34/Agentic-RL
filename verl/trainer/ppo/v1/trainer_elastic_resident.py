"""In-place DP resizing within resident shared physical resource pools."""

from __future__ import annotations

import json
import logging
import random
import time
from copy import deepcopy
from dataclasses import replace

import numpy as np
import torch
from omegaconf import ListConfig, open_dict

from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.experimental.elastic.resident import ResidentTrainingPool
from verl.experimental.elastic.storage import read_json, write_json
from verl.experimental.weave.protocol import Phase
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.trainer.ppo.v1.trainer_elastic import PPOTrainerElasticSeparate
from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode, PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.trainer_weave import PPOTrainerWeaveSeparate
from verl.utils import omega_conf_to_dataclass

logger = logging.getLogger(__name__)


def validate_resize(train_dp, rollout_dp, train_capacity, rollout_capacity):
    for name, size, capacity in (("train", train_dp, train_capacity), ("rollout", rollout_dp, rollout_capacity)):
        if type(size) is not int or not 1 <= size <= capacity:
            raise ValueError(f"{name} DP {size!r} must be an integer in [1, {capacity}]")


def validate_slots(slots, size, capacity):
    indices = list(range(size)) if not slots else list(slots)
    if (
        len(indices) != size
        or len(set(indices)) != size
        or any(type(index) is not int or not 0 <= index < capacity for index in indices)
    ):
        raise ValueError("Active DP slots must be distinct physical pool indices")
    return indices


def completion_dependencies(value, job_name):
    if value is None:
        return []
    names = [value] if isinstance(value, str) else value
    if (
        not isinstance(names, (list, tuple, ListConfig))
        or not names
        or any(not isinstance(name, str) or not name or name == job_name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Resident expand_after must name distinct other tasks, as a string or nonempty list")
    return list(names)


@register_trainer("elastic_resident_separate")
class PPOTrainerElasticResident(PPOTrainerElasticSeparate):
    """Retain the driver and vLLM engines; replace only the active training group."""

    def __init__(self, config):
        resident = config.trainer.v1.elastic.resident
        self.train_capacity = config.trainer.n_gpus_per_node
        self.rollout_capacity = config.actor_rollout_ref.rollout.n_gpus_per_node
        if config.trainer.nnodes != 1 or config.actor_rollout_ref.rollout.nnodes != 1:
            raise ValueError("Resident resizing currently requires a single node")
        rollout = config.actor_rollout_ref.rollout
        if any(
            size != 1
            for size in (
                rollout.tensor_model_parallel_size,
                rollout.data_parallel_size,
                rollout.pipeline_model_parallel_size,
            )
        ):
            raise ValueError("Resident rollout resizing currently requires one GPU per replica")
        if config.distillation.enabled:
            raise ValueError("Resident resizing does not yet support distillation workers")
        if config.actor_rollout_ref.actor.strategy != "fsdp":
            raise ValueError("Resident resizing currently requires FSDP1")
        if config.trainer.v1.weave.migration.enabled:
            raise ValueError("Resident resizing requires request migration disabled")
        validate_resize(resident.train_dp, resident.rollout_dp, self.train_capacity, self.rollout_capacity)
        if resident.get("state_transfer", "disk") not in ("disk", "memory"):
            raise ValueError("Resident state_transfer must be disk or memory")
        self.train_slots = validate_slots(resident.get("train_slots", []), resident.train_dp, self.train_capacity)
        self.rollout_slots = validate_slots(
            resident.get("rollout_slots", []), resident.rollout_dp, self.rollout_capacity
        )
        self.shared_slots = resident.get("shared_slots", False)
        self.state_transfer = resident.get("state_transfer", "disk")
        self.expand_after = completion_dependencies(resident.get("expand_after"), config.trainer.v1.weave.job_name)
        self.cost_aware = resident.get("cost_aware", False)
        self.debug_memory = resident.get("debug_memory", False)
        if self.cost_aware and (not self.shared_slots or self.expand_after or resident.resize_schedule):
            raise ValueError("Cost-aware scheduling requires shared slots without fixed resize/expand rules")
        self.active_rollout_dp = resident.rollout_dp
        self._schedule = [dict(item) for item in resident.resize_schedule]
        previous = 0
        for entry in self._schedule:
            step = entry["completed_step"]
            if type(step) is not int or not previous < step < config.trainer.total_training_steps:
                raise ValueError("Resident resize steps must be strictly increasing and precede the final step")
            validate_resize(entry["train_dp"], entry["rollout_dp"], self.train_capacity, self.rollout_capacity)
            previous = step
        self._processed_commands = set()
        self._resize_history = []
        self._resident_pool = None
        super().__init__(config)
        # The phase protocol owns the full physical lease, not just the currently active DP ranks.
        with open_dict(config.trainer):
            config.trainer.n_gpus_per_node = resident.train_dp
        with open_dict(config.actor_rollout_ref.rollout):
            config.actor_rollout_ref.rollout.n_gpus_per_node = resident.rollout_dp

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        self.resource_pool_manager.resource_pool_spec["global_pool"] = [self.train_capacity]

    def _create_worker_groups(self, resource_pool, class_dict, wg_kwargs):
        if len(class_dict) != 1:
            raise ValueError("Resident training currently supports the combined actor/reference role only")
        self._resident_role = next(iter(class_dict))
        # Capture after dataloader setup sets optimizer total steps, before init_model mutates ref config.
        self._resident_actor_config = deepcopy(self.config.actor_rollout_ref)
        self._resident_pool = ResidentTrainingPool(
            resource_pool,
            name_prefix=f"resident_{self.revision}_{self.config.trainer.v1.weave.job_name}",
            wg_kwargs=wg_kwargs,
        )
        group = self._resident_pool.prepare(
            self._resident_actor_config,
            self._resident_role,
            self.config.trainer.n_gpus_per_node,
            self.train_slots,
        )
        return {self._resident_role: group}

    def _create_llm_server_manager(self, resource_pool, name_suffix):
        config = deepcopy(self.config)
        with open_dict(config.actor_rollout_ref.rollout):
            config.actor_rollout_ref.rollout.n_gpus_per_node = self.rollout_capacity
        original = self.config
        self.config = config
        try:
            return PPOTrainerWeaveSeparate._create_standalone_server_manager(self, start_rank=0)
        finally:
            self.config = original

    def _get_initial_checkpoint_engine_config(self):
        return omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)

    def _setup(self):
        PPOTrainer._setup(self)
        self.standalone_server_manager = self.llm_server_manager
        self.standalone_checkpoint_manager = self.checkpoint_manager
        self.current_mode = HybridEngineMode.TRAINER

    def on_init_end(self):
        try:
            self.checkpoint_manager.update_weights(self.global_steps)
            self._remove_standalone_replicas(self.standalone_server_manager.get_replicas())
            self.checkpoint_manager.suspend_tenant_replicas()
            if self.shared_slots:
                self._publish_allocation()
            self.weave_client.finish_init()
            if self.shared_slots:
                self._set_client_slots()
                self._set_active_transport()
            self._state("ready", step=self.global_steps)
            self._write_resident_state()
        except BaseException:
            self.weave_client.crash()
            raise

    def _selected_workers(self, worker_ids):
        selected = [slot for slot in self.rollout_slots if slot in worker_ids]
        if len(selected) != self.active_rollout_dp:
            raise RuntimeError("The resident physical lease is smaller than the active rollout DP")
        return selected

    def _active_replicas(self):
        replicas = self.llm_server_manager.get_replicas()
        return [replicas[index] for index in self.rollout_slots]

    def _set_active_transport(self):
        self.checkpoint_manager = CheckpointEngineManager(
            self._get_initial_checkpoint_engine_config(), self.actor_rollout_wg, self._active_replicas()
        )
        self.standalone_checkpoint_manager = self.checkpoint_manager

    def _publish_allocation(self):
        self.weave_client.storage.set(
            f"{self.weave_client.config.job_name}:resident:allocation",
            json.dumps({"gen": self.rollout_slots, "train": self.train_slots}),
        )

    def _set_client_slots(self):
        self.weave_client.config = replace(
            self.weave_client.config, gen_gpu_list=list(self.rollout_slots), train_gpu_list=list(self.train_slots)
        )

    def _acquire_resize_lease(self, command_id, train_slots, rollout_slots):
        client = self.weave_client
        client.storage.set(
            f"{client.config.job_name}:resident:request",
            json.dumps(
                {
                    "id": command_id,
                    "train": train_slots,
                    "gen": rollout_slots,
                }
            ),
        )
        client.wait_key(
            f"{client.config.job_name}:resident:ack", command_id, timeout=self.config.trainer.v1.weave.init_wait_timeout
        )

    def _weave_activate_generate(self, worker_ids):
        selected = self._selected_workers(worker_ids)
        super()._weave_activate_generate(selected)
        logger.info(
            "RESIDENT generation step=%d rollout_dp=%d workers=%s", self.global_steps, self.active_rollout_dp, selected
        )

    def _weave_deactivate_generate(self, worker_ids):
        super()._weave_deactivate_generate(self._selected_workers(worker_ids))

    def _weave_start_migration(self):
        # All slots belong to this one lease; no intra-step migration to inactive replicas.
        if self._weave_migration_config.enabled:
            raise ValueError("Resident DP resizing requires request migration disabled")

    def _get_n_gpus_for_throughput(self):
        return self.train_capacity + self.rollout_capacity

    def _write_resident_state(self):
        write_json(
            self.control_dir / "resident.json",
            {
                "schema": 1,
                "revision": self.revision,
                "completed_steps": min(self.global_steps, self.total_training_steps),
                "train_dp": self.actor_rollout_wg.world_size,
                "rollout_dp": self.active_rollout_dp,
                "train_slots": self.train_slots,
                "rollout_slots": self.rollout_slots,
                "leased_gpus": self.train_capacity + self.rollout_capacity,
                "training_workers": self._resident_pool.identities(),
                "rollout_actors": [worker._actor_id.hex() for worker in self.llm_server_manager.server_handles],
                "rollout_addresses": self.llm_server_manager.server_addresses,
                "resizes": self._resize_history,
            },
        )

    def _resize(self, train_dp, rollout_dp, command_id, *, train_slots=None, rollout_slots=None):
        validate_resize(train_dp, rollout_dp, self.train_capacity, self.rollout_capacity)
        train_slots = validate_slots(train_slots, train_dp, self.train_capacity)
        rollout_slots = validate_slots(rollout_slots, rollout_dp, self.rollout_capacity)
        started = time.monotonic()
        if self.shared_slots:
            self._acquire_resize_lease(command_id, train_slots, rollout_slots)
        before = self._resident_pool.identities()
        old_steps = [before[index]["optimizer_steps"] for index in self.train_slots]
        if not old_steps or not old_steps[0] or any(steps != old_steps[0] for steps in old_steps):
            raise RuntimeError("Resident resize requires initialized, consistent optimizer state")
        checkpoint = None
        timings = {}
        release_memory = []
        snapshot = None
        if train_slots != self.train_slots:
            driver_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
            stage_started = time.monotonic()
            if self.state_transfer == "memory":
                snapshot = self._resident_pool.snapshot(self.global_steps)
            else:
                checkpoint = self._save_checkpoint()
            timings["checkpoint"] = time.monotonic() - stage_started
            self._state("resizing", step=self.global_steps, checkpoint=checkpoint)
            stage_started = time.monotonic()
            release_memory = self._resident_pool.release()
            timings["release_group"] = time.monotonic() - stage_started
            stage_started = time.monotonic()
            group = self._resident_pool.prepare(self._resident_actor_config, self._resident_role, train_dp, train_slots)
            group.init_model()
            timings["initialize_group"] = time.monotonic() - stage_started
            stage_started = time.monotonic()
            if snapshot is not None:
                self._resident_pool.restore(snapshot)
            else:
                group.load_checkpoint(f"{checkpoint}/actor", del_local_after_load=False)
            timings["restore"] = time.monotonic() - stage_started
            self.actor_rollout_wg = group
            if self.use_reference_policy:
                self.ref_policy_wg = group
            torch.set_rng_state(driver_rng[0])
            np.random.set_state(driver_rng[1])
            random.setstate(driver_rng[2])
        after = self._resident_pool.identities()
        if [info["pid"] for info in before] != [info["pid"] for info in after]:
            raise RuntimeError("A resident worker restarted during resizing")
        if any(
            after[index]["world_size"] != train_dp or after[index]["optimizer_steps"] != old_steps[0]
            for index in train_slots
        ):
            raise RuntimeError("Resident training DP or restored optimizer state does not match")
        added_rollout = set(rollout_slots) - set(self.rollout_slots)
        self.train_slots = train_slots
        self.rollout_slots = rollout_slots
        self.active_rollout_dp = rollout_dp
        self._set_active_transport()
        if added_rollout:
            self.checkpoint_manager.update_weights(
                self.global_steps,
                wake_weights_only=True,
                resume_after_update=False,
            )
            self.checkpoint_manager.wake_up_replicas(tags=["kv_cache"])
            self.checkpoint_manager.suspend_tenant_replicas()
        if self.shared_slots:
            self._publish_allocation()
            self.weave_client.release_gpus(Phase.UPDATE, self.weave_client.config.gen_device_affinity, rollout_slots)
            self._set_client_slots()
        with open_dict(self.config.trainer):
            self.config.trainer.n_gpus_per_node = train_dp
        with open_dict(self.config.actor_rollout_ref.rollout):
            self.config.actor_rollout_ref.rollout.n_gpus_per_node = rollout_dp
        self._resize_history.append(
            {
                "command_id": command_id,
                "completed_step": self.global_steps,
                "train_dp": train_dp,
                "rollout_dp": rollout_dp,
                "train_slots": train_slots,
                "rollout_slots": rollout_slots,
                "state_transfer": self.state_transfer,
                "seconds": time.monotonic() - started,
                "timings": timings,
                "release_memory": release_memory,
                "checkpoint": checkpoint,
                "before": before,
                "after": after,
            }
        )
        self._processed_commands.add(command_id)
        self._write_resident_state()
        self._state("running", step=self.global_steps)
        logger.info(
            "RESIDENT resized step=%d train_dp=%d rollout_dp=%d seconds=%.3f",
            self.global_steps,
            train_dp,
            rollout_dp,
            self._resize_history[-1]["seconds"],
        )

    def _weave_suspend_after_weight_sync(self):
        if getattr(self, "debug_memory", False):
            logger.info("RESIDENT memory step=%d workers=%s", self.global_steps, self._resident_pool.identities())
        command = None
        path = self.control_dir / "command.json"
        if path.exists():
            packet = read_json(path)
            if packet["revision"] == self.revision:
                if packet["action"] != "resize":
                    raise ValueError("Resident mode accepts resize, not cold-backend pause commands")
                if packet["id"] not in self._processed_commands:
                    command = packet
        scheduled = [entry for entry in self._schedule if entry["completed_step"] == self.global_steps]
        if self.expand_after and "expand-after" not in self._processed_commands:
            completed = all(
                self.weave_client.storage.get(f"{name}:status") in ("completed", b"completed")
                for name in self.expand_after
            )
            if completed and self.global_steps < self.total_training_steps:
                if command is not None or scheduled:
                    raise ValueError("Completion-triggered expansion conflicts with another resize command")
                command = {"id": "expand-after", "train_dp": self.train_capacity, "rollout_dp": self.rollout_capacity}
        if command is not None and scheduled:
            raise ValueError("An external resize conflicts with a scheduled resize at this boundary")
        if scheduled:
            command = {**scheduled[0], "id": f"schedule-{self.global_steps}"}
        if getattr(self, "cost_aware", False):
            if command is not None:
                raise ValueError("Cost-aware scheduling conflicts with an external resize command")
            client = self.weave_client
            client.storage.set(
                f"{client.config.job_name}:resident:boundary",
                json.dumps({"revision": self.revision, "completed_step": self.global_steps}),
            )
            decision_key = f"{client.config.job_name}:resident:decision"
            deadline = time.monotonic() + self.config.trainer.v1.weave.init_wait_timeout
            while True:
                raw = client.storage.get(decision_key)
                packet = json.loads(raw) if raw is not None else None
                if (
                    packet is not None
                    and packet["revision"] == self.revision
                    and packet["completed_step"] == self.global_steps
                ):
                    if packet["action"] not in ("continue", "resize"):
                        raise ValueError("Invalid cost-aware resident decision")
                    command = packet if packet["action"] == "resize" else None
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Resident cost-aware scheduler did not acknowledge the safe boundary")
                time.sleep(0.02)
        if command is not None:
            if self.global_steps >= self.total_training_steps:
                raise ValueError("Cannot resize after the final training step")
            slots = {key: command[key] for key in ("train_slots", "rollout_slots") if key in command}
            self._resize(command["train_dp"], command["rollout_dp"], command["id"], **slots)

    def fit(self, agent_loop_manager):
        try:
            result = PPOTrainerSeparateAsync.fit(self, agent_loop_manager)
            self._write_resident_state()
            self._resident_pool.release()
        except BaseException:
            logger.exception("Resident training or in-place resizing failed")
            self.weave_client.crash()
            self._state("failed", step=max(0, self.global_steps - 1))
            raise
        self.weave_client.complete()
        self._state("completed", step=self.total_training_steps)
        return result
