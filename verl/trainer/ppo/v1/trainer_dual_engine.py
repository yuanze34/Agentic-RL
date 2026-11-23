"""Fixed-capacity, task-independent dual Engine rollout for WEAVE phases."""

import ray

from verl.experimental.elastic.dual_engine_manager import (
    DualEngineActorWorker,
    DualEngineCheckpointManager,
    DualEngineManager,
)
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode
from verl.trainer.ppo.v1.trainer_weave import PPOTrainerWeaveSeparate


@register_trainer("dual_engine_separate")
class PPOTrainerDualEngine(PPOTrainerWeaveSeparate):
    def __init__(self, config):
        if config.trainer.v1.weave.migration.enabled:
            raise ValueError("Dual Engine rollout currently requires request migration disabled")
        if config.actor_rollout_ref.actor.strategy != "fsdp":
            raise ValueError("Dual Engine trainer currently supports FSDP1 weight export")
        if config.actor_rollout_ref.model.lora_rank > 0 or config.distillation.enabled:
            raise ValueError("Dual Engine trainer currently requires full-model training without distillation")
        if config.critic.enable is True or config.algorithm.adv_estimator == "gae":
            raise ValueError("Dual Engine trainer currently supports actor-only GRPO")
        if config.trainer.val_before_train or config.trainer.test_freq > 0:
            raise ValueError("Dual Engine validation needs a generation lease; disable periodic validation for now")
        if config.trainer.v1.separate_async.parameter_sync_step != 1:
            raise ValueError("Dual Engine trainer currently requires parameter_sync_step=1")
        super().__init__(config)

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        for role in (Role.ActorRolloutRef, Role.ActorRollout):
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = ray.remote(DualEngineActorWorker)

    def _create_llm_server_manager(self, resource_pool, name_suffix):
        return DualEngineManager(self.config)

    def _setup(self):
        # Avoid SeparateAsync._setup, which would create per-task hybrid and
        # standalone engines in addition to the shared slot pool.
        PPOTrainer._setup(self)
        self.standalone_server_manager = self.llm_server_manager
        self.checkpoint_manager = DualEngineCheckpointManager(self.actor_rollout_wg, self.llm_server_manager)
        self.standalone_checkpoint_manager = self.checkpoint_manager
        self.current_mode = HybridEngineMode.TRAINER

    def on_init_end(self):
        try:
            self.checkpoint_manager.update_weights(self.global_steps)
            self.weave_client.finish_init()
        except BaseException:
            self.weave_client.crash()
            raise

    def _update_standalone_weights(self):
        # CPU publication does not acquire UPDATE's generation-slot lease:
        # the standby can be prepared while another task is generating.
        return self.standalone_checkpoint_manager.update_weights(self.global_steps)

    def _weave_activate_generate(self, worker_ids):
        self.standalone_server_manager.activate(worker_ids)
        self._add_standalone_replicas(self._standalone_replicas_for_workers(worker_ids))

    def _weave_deactivate_generate(self, worker_ids):
        self._remove_standalone_replicas(self._standalone_replicas_for_workers(worker_ids))
        self.standalone_server_manager.release(worker_ids)

    def _weave_start_migration(self):
        return

    def _weave_finish_migration(self):
        return
