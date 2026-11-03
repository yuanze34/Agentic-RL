"""WEAVE-gated V1 trainers.

These adapters preserve verl's computation while making phase admission an
exclusive decision of the external WEAVE scheduler.
"""

from __future__ import annotations

import logging

from omegaconf import open_dict

from verl.experimental.weave.client import WeaveClientConfig, WeaveSchedulerClient
from verl.experimental.weave.migration import MigrationConfig
from verl.experimental.weave.placement import WeaveResourcePoolManager
from verl.experimental.weave.protocol import Phase
from verl.experimental.weave.rollout import WeaveLLMServerClient, WeaveLoadBalancer
from verl.experimental.weave.runtime import MigrationController
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode, PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.workers.rollout.llm_server import LLMServerManager

logger = logging.getLogger(__name__)


class _WeaveTrainerMixin:
    """Translate verl lifecycle hooks into the reference WEAVE protocol."""

    def __init__(self, config):
        weave_config = config.trainer.v1.weave
        self.weave_client = WeaveSchedulerClient(WeaveClientConfig.from_dict(weave_config))
        self._weave_generation_active = False
        self._weave_active_gen_workers: list[int] = []
        self.weave_client.register_and_wait_for_init()
        try:
            super().__init__(config)
        except BaseException:
            self.weave_client.crash()
            raise

    def on_init_end(self):
        try:
            super().on_init_end()
            self._weave_suspend_after_init()
            self.weave_client.finish_init()
        except BaseException:
            self.weave_client.crash()
            raise

    def init(self):
        try:
            return super().init()
        except BaseException:
            self.weave_client.crash()
            raise

    def _add_batch_to_generate(self):
        self._weave_check_health()
        if not self._weave_generation_active:
            self.weave_client.wait_for_phase(Phase.GENERATE)
            self._weave_active_gen_workers = self.weave_client.get_allocated_generate_workers()
            self._weave_activate_generate(self._weave_active_gen_workers)
            self._weave_start_migration()
            self._weave_generation_active = True
        return super()._add_batch_to_generate()

    def on_sample_end(self):
        try:
            super().on_sample_end()
            self._weave_finish_migration()
            self._weave_deactivate_generate(self._weave_active_gen_workers)
            self.weave_client.release_generate_workers(self._weave_active_gen_workers)
            self.weave_client.finish_phase(Phase.GENERATE)
            self._weave_generation_active = False
            self._weave_active_gen_workers = []
            self.weave_client.wait_for_phase(Phase.TRAIN)
        except BaseException:
            self.weave_client.crash()
            raise

    def on_step_end(self):
        try:
            super().on_step_end()
            self._weave_suspend_after_weight_sync()
            self.weave_client.finish_phase(Phase.TRAIN)
        except BaseException:
            self.weave_client.crash()
            raise

    def fit(self, agent_loop_manager):
        try:
            result = super().fit(agent_loop_manager)
        except BaseException:
            self.weave_client.crash()
            self._weave_finish_migration()
            raise
        self.weave_client.complete()
        return result

    def _weave_suspend_after_init(self):
        pass

    def _weave_activate_generate(self, worker_ids: list[int]):
        pass

    def _weave_deactivate_generate(self, worker_ids: list[int]):
        pass

    def _weave_suspend_after_weight_sync(self):
        pass

    def _weave_start_migration(self):
        pass

    def _weave_finish_migration(self):
        pass

    def _weave_check_health(self):
        pass

    def _build_replay_buffer(self):
        replay_buffer = super()._build_replay_buffer()
        replay_buffer.health_check = self._weave_check_health
        return replay_buffer


@register_trainer("weave_sync")
class PPOTrainerWeaveSync(_WeaveTrainerMixin, PPOTrainerSync):
    """Single-pool WEAVE adapter used for colocated/single-GPU validation."""


@register_trainer("weave_separate")
class PPOTrainerWeaveSeparate(_WeaveTrainerMixin, PPOTrainerSeparateAsync):
    """Disaggregated WEAVE adapter for distinct rollout and training pools."""

    def __init__(self, config):
        weave_mode = config.trainer.v1.separate_async
        sampler = config.trainer.v1.sampler
        weave = config.trainer.v1.weave
        if weave_mode.parameter_sync_step != 1:
            raise ValueError("WEAVE requires trainer.v1.separate_async.parameter_sync_step=1")
        if weave_mode.num_warmup_batches != 0:
            raise ValueError("WEAVE requires trainer.v1.separate_async.num_warmup_batches=0")
        # ReplayBufferAsync requires a positive threshold. With one fresh
        # rollout followed by one train step, 1 admits only current-step data.
        if sampler.max_off_policy_threshold != 1:
            raise ValueError("WEAVE requires trainer.v1.sampler.max_off_policy_threshold=1")
        if config.trainer.nnodes != 1 and not weave.train_replica_placements:
            raise ValueError("Multi-node WEAVE training requires train_replica_placements")
        if config.actor_rollout_ref.rollout.nnodes != 1 and not weave.gen_replica_placements:
            raise ValueError("Multi-node WEAVE rollout requires gen_replica_placements")
        if len(weave.train_gpu_list) != config.trainer.n_gpus_per_node * config.trainer.nnodes:
            raise ValueError("WEAVE train_gpu_list must match the trainer world size")
        if len(weave.gen_gpu_list) != (
            config.actor_rollout_ref.rollout.n_gpus_per_node * config.actor_rollout_ref.rollout.nnodes
        ):
            raise ValueError("WEAVE gen_gpu_list must match the rollout world size")
        if weave.max_tenants_per_gpu < 2:
            raise ValueError("Multi-job WEAVE requires max_tenants_per_gpu >= 2")
        if config.reward.reward_model.enable_resource_pool:
            raise ValueError("A separate reward-model pool is not supported by WEAVE shared placement yet")
        if config.trainer.val_before_train or config.trainer.test_freq > 0:
            raise ValueError("WEAVE multi-job mode currently requires val_before_train=False and test_freq<=0")
        if config.actor_rollout_ref.actor.strategy != "fsdp":
            raise ValueError("The initial WEAVE shared placement implementation supports FSDP training")
        if config.actor_rollout_ref.rollout.name != "vllm":
            raise ValueError("WEAVE shared residency currently requires the vLLM rollout backend")
        rollout = config.actor_rollout_ref.rollout
        width = rollout.tensor_model_parallel_size * rollout.data_parallel_size * rollout.pipeline_model_parallel_size
        if weave.gpus_per_gen_worker != width:
            raise ValueError("WEAVE gpus_per_gen_worker must equal the rollout replica TP*DP*PP size")
        if rollout.disaggregation.enabled:
            raise ValueError("WEAVE request migration does not support prefill/decode disaggregation")
        self._weave_migration: MigrationController | None = None
        self._weave_migration_config = MigrationConfig(**dict(weave.migration))
        with open_dict(config.actor_rollout_ref.rollout):
            config.actor_rollout_ref.rollout.name_suffix = f"_{weave.job_name}"
        with open_dict(config.actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs):
            engine_kwargs = config.actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs
            if "nccl" not in engine_kwargs:
                engine_kwargs["nccl"] = {}
            with open_dict(engine_kwargs["nccl"]):
                engine_kwargs["nccl"]["group_name"] = f"verl_weave_{weave.job_name}"
        if not config.actor_rollout_ref.rollout.free_cache_engine:
            raise ValueError("WEAVE shared residency requires rollout.free_cache_engine=True")
        if not config.actor_rollout_ref.actor.fsdp_config.param_offload:
            raise ValueError("WEAVE multi-tenant train GPUs require actor.fsdp_config.param_offload=True")
        if not config.actor_rollout_ref.actor.fsdp_config.optimizer_offload:
            raise ValueError("WEAVE multi-tenant train GPUs require actor.fsdp_config.optimizer_offload=True")
        super().__init__(config)

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        original = self.resource_pool_manager
        unsupported_pools = set(original.resource_pool_spec) - {"global_pool"}
        if unsupported_pools:
            raise ValueError(f"WEAVE shared placement does not support extra pools: {sorted(unsupported_pools)}")
        self.resource_pool_manager = WeaveResourcePoolManager(
            resource_pool_spec=original.resource_pool_spec,
            mapping=original.mapping,
            shared_pool_names={"global_pool": str(self.config.trainer.v1.weave.train_placement_group)},
            max_tenants_per_gpu=int(self.config.trainer.v1.weave.max_tenants_per_gpu),
            shared_pool_ids={"global_pool": str(self.config.trainer.v1.weave.train_placement_group_id)}
            if self.config.trainer.v1.weave.train_placement_group_id is not None
            else None,
            shared_pool_sizes=(
                {"global_pool": int(self.config.trainer.v1.weave.train_pool_size)}
                if self.config.trainer.v1.weave.train_pool_size is not None
                else None
            ),
            shared_pool_placements=(
                {"global_pool": list(self.config.trainer.v1.weave.train_replica_placements)}
                if self.config.trainer.v1.weave.train_replica_placements
                else None
            ),
        )

    def _create_standalone_server_manager(self, start_rank: int) -> LLMServerManager:
        weave = self.config.trainer.v1.weave
        return LLMServerManager.create(
            config=self.config,
            start_rank=start_rank,
            shared_placement_group=str(weave.gen_placement_group),
            shared_placement_group_id=(
                str(weave.gen_placement_group_id) if weave.gen_placement_group_id is not None else None
            ),
            max_tenants_per_gpu=int(weave.max_tenants_per_gpu),
            replica_name_suffix=str(weave.job_name),
            load_balancer_cls=WeaveLoadBalancer,
            shared_replica_placements=list(weave.gen_replica_placements),
        )

    def _standalone_replicas_for_workers(self, worker_ids: list[int]):
        replicas = self.standalone_server_manager.get_replicas()
        if any(worker_id < 0 or worker_id >= len(replicas) for worker_id in worker_ids):
            raise ValueError(f"WEAVE allocated rollout workers {worker_ids}, but verl has {len(replicas)} replicas")
        return [replicas[worker_id] for worker_id in worker_ids]

    def _remove_standalone_replicas(self, replicas) -> None:
        if not replicas:
            return
        import ray

        ray.get(
            self.standalone_server_manager.global_load_balancer.remove_servers.remote(
                [replica.server_address for replica in replicas]
            )
        )

    def _add_standalone_replicas(self, replicas) -> None:
        if not replicas:
            return
        import ray

        servers = {replica.server_address: replica.server_handle for replica in replicas}
        ray.get(self.standalone_server_manager.global_load_balancer.add_servers.remote(servers))

    def _weave_suspend_after_init(self):
        standalone_replicas = self.standalone_server_manager.get_replicas()
        logger.info("WEAVE suspending %d rollout replicas after init", len(standalone_replicas))
        self.remove_replicas_from_balancer()
        self._remove_standalone_replicas(standalone_replicas)
        self.checkpoint_manager.sleep_replicas()
        self.standalone_checkpoint_manager.suspend_tenant_replicas()
        self.current_mode = HybridEngineMode.TRAINER
        logger.info("WEAVE rollout replicas suspended after init")

    def _update_standalone_weights(self):
        self.weave_client.wait_for_weight_update()
        result = self.standalone_checkpoint_manager.update_weights(
            self.global_steps,
            wake_weights_only=True,
            resume_after_update=False,
        )
        self.standalone_checkpoint_manager.wake_up_replicas(tags=["kv_cache"])
        self.standalone_checkpoint_manager.suspend_tenant_replicas()
        self.weave_client.finish_weight_update()
        return result

    def _weave_activate_generate(self, worker_ids: list[int]):
        replicas = self._standalone_replicas_for_workers(worker_ids)
        logger.info("WEAVE resuming rollout workers %s", worker_ids)
        self.standalone_checkpoint_manager.resume_tenant_replicas(replicas)
        logger.info("WEAVE resumed rollout workers %s", worker_ids)
        self._add_standalone_replicas(replicas)
        logger.info("WEAVE added rollout workers %s to the load balancer", worker_ids)

    def _weave_deactivate_generate(self, worker_ids: list[int]):
        replicas = self._standalone_replicas_for_workers(worker_ids)
        self._remove_standalone_replicas(replicas)
        self.standalone_checkpoint_manager.suspend_tenant_replicas(replicas)

    def _weave_suspend_after_weight_sync(self):
        pass

    def get_llm_client(self):
        return self.standalone_server_manager.get_client(client_cls=WeaveLLMServerClient)

    def _weave_start_migration(self):
        self._weave_migration = MigrationController(
            self.weave_client,
            self.standalone_server_manager,
            self.standalone_checkpoint_manager,
            self._weave_active_gen_workers,
            self._weave_migration_config,
        )
        self._weave_migration.start()

    def _weave_finish_migration(self):
        if self._weave_migration is not None:
            try:
                self._weave_active_gen_workers = self._weave_migration.finish()
            finally:
                self._weave_migration = None

    def _weave_check_health(self):
        if self._weave_migration is not None and self._weave_migration.error is not None:
            raise RuntimeError("WEAVE request migration failed") from self._weave_migration.error
