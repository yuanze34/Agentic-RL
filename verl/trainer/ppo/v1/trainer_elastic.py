"""Safe-boundary restartable training for online DP reconfiguration."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import numpy as np
import ray
import torch
from omegaconf import open_dict

from verl.experimental.elastic.storage import read_json, write_json
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.trainer_weave import PPOTrainerWeaveSeparate

logger = logging.getLogger(__name__)


class _ElasticPaused(Exception):
    pass


def configure_transport(config, revision: str) -> None:
    kwargs = config.actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs
    with open_dict(kwargs.nccl):
        kwargs.nccl.group_name = f"verl_elastic_{config.trainer.v1.weave.job_name}_{revision}"
        # Ray NCCL creates detached rendezvous actors. Destroy them through
        # the normal finalize hook instead of retaining stale epoch state.
        kwargs.nccl.rebuild_group = True


@register_trainer("elastic_separate")
class PPOTrainerElasticSeparate(PPOTrainerWeaveSeparate):
    """Reuse phase/resource primitives without changing the WEAVE baseline."""

    def __init__(self, config):
        elastic = config.trainer.v1.elastic
        if not elastic.control_dir or not elastic.revision:
            raise ValueError("elastic_separate requires a controller directory and revision")
        if config.critic.enable is True or config.algorithm.adv_estimator == "gae":
            raise ValueError("Elastic DP restart currently supports actor-only GRPO, not a critic")
        if config.trainer.default_hdfs_dir is not None:
            raise ValueError("Elastic restart requires a shared local filesystem")
        if config.trainer.save_freq > 0:
            raise ValueError("Elastic mode currently uses controller checkpoints only; set trainer.save_freq=-1")
        if config.skip.rollout_tq.enable or config.skip.rollout.enable:
            raise ValueError("Elastic safe-boundary restart does not support rollout skip")
        if config.algorithm.filter_groups.enable:
            raise ValueError("Elastic restart currently requires an unfiltered, single rollout batch per step")
        if config.algorithm.use_kl_in_reward:
            raise ValueError("Elastic restart currently requires use_kl_in_reward=False")
        if config.actor_rollout_ref.model.lora_rank > 0:
            raise ValueError("Elastic restart currently supports full-model training, not LoRA")
        self.control_dir = Path(elastic.control_dir)
        self.revision = elastic.revision
        self._pause_checkpoint: str | None = None
        self._generation_started: float | None = None
        self._training_started: float | None = None
        self._durations: dict[str, float] = {}
        super().__init__(config)
        configure_transport(config, self.revision)

    def _init_resource_pool_mgr(self):
        from verl.experimental.elastic.engine_workers import ElasticDetachActorWorker

        super()._init_resource_pool_mgr()
        for role in (Role.ActorRolloutRef, Role.ActorRollout):
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = ray.remote(ElasticDetachActorWorker)

    def _state(self, status: str, *, step: int, checkpoint: str | None = None) -> None:
        write_json(
            self.control_dir / "state.json",
            {
                "revision": self.revision,
                "status": status,
                "completed_steps": step,
                "checkpoint": checkpoint,
                "durations": self._durations,
                "updated_at": time.time(),
            },
        )

    def on_init_end(self):
        super().on_init_end()
        self._state("ready", step=self.global_steps)

    def _weave_activate_generate(self, worker_ids):
        start = time.monotonic()
        super()._weave_activate_generate(worker_ids)
        self._durations["switch"] = time.monotonic() - start
        self._generation_started = time.monotonic()

    def on_sample_end(self):
        if self._generation_started is not None:
            self._durations["rollout"] = time.monotonic() - self._generation_started
        super().on_sample_end()
        self._training_started = time.monotonic()

    def _update_standalone_weights(self):
        if self._training_started is not None:
            self._durations["train"] = time.monotonic() - self._training_started
        return super()._update_standalone_weights()

    def _weave_suspend_after_weight_sync(self):
        # The training allocation is still held here. Saving at the next
        # on_step_begin would race another tenant on the training GPU.
        command_path = self.control_dir / "command.json"
        if command_path.exists():
            command = read_json(command_path)
            if command["revision"] == self.revision and command["action"] == "pause":
                if self.global_steps < self.total_training_steps:
                    self._pause_checkpoint = self._save_checkpoint()

    def on_step_begin(self):
        if self._pause_checkpoint is not None:
            raise _ElasticPaused()
        super().on_step_begin()

    def _compute_metrics(self, *args, **kwargs):
        super()._compute_metrics(*args, **kwargs)
        self._state("running", step=self.global_steps, checkpoint=self._pause_checkpoint)

    def fit(self, agent_loop_manager):
        try:
            # Bypass the baseline mixin's crash-on-exception wrapper only for
            # our explicit successful checkpoint/pause protocol.
            result = PPOTrainerSeparateAsync.fit(self, agent_loop_manager)
        except _ElasticPaused:
            self._shutdown_dump_executor()
            self.weave_client.complete()
            self._state("paused", step=self.global_steps - 1, checkpoint=self._pause_checkpoint)
            logger.info("ELASTIC paused at step %s: %s", self.global_steps - 1, self._pause_checkpoint)
            return None
        except BaseException:
            self.weave_client.crash()
            self._state("failed", step=max(0, self.global_steps - 1))
            raise
        self.weave_client.complete()
        self._state("completed", step=self.total_training_steps)
        return result

    def _save_checkpoint(self) -> str:
        step = self.global_steps
        directory = Path(self.config.trainer.default_local_dir) / f"global_step_{step}"
        directory.mkdir(parents=True, exist_ok=True)
        self.actor_rollout_wg.save_checkpoint(str(directory / "actor"), global_step=step)
        data_path = directory / "data.pt"
        temporary = directory / "data.pt.tmp"
        torch.save(
            {
                "dataloader": self.train_dataloader.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            },
            temporary,
        )
        temporary.replace(data_path)
        write_json(
            directory / "elastic.json",
            {
                "schema": 1,
                "completed_steps": step,
                "total_steps": self.total_training_steps,
                "train_batch_size": self.config.data.train_batch_size,
                "model_path": self.config.actor_rollout_ref.model.path,
            },
        )
        # This checkpoint never snapshots the global TransferQueue: other
        # groups must not have their queue overwritten on a local restart.
        return str(directory)

    def _load_checkpoint(self):
        self.global_steps = 0
        mode = self.config.trainer.resume_mode
        if mode == "disable":
            return
        if mode != "resume_path" or not self.config.trainer.resume_from_path:
            raise ValueError("Elastic restart requires disable or an explicit committed resume_path")
        directory = Path(self.config.trainer.resume_from_path)
        state = read_json(directory / "elastic.json")
        expected = {
            "schema": 1,
            "total_steps": self.total_training_steps,
            "train_batch_size": self.config.data.train_batch_size,
            "model_path": self.config.actor_rollout_ref.model.path,
        }
        if any(state[key] != value for key, value in expected.items()):
            raise ValueError(f"Elastic checkpoint workload mismatch: {directory}")
        step = state["completed_steps"]
        if not isinstance(step, int) or not 0 <= step < self.total_training_steps:
            raise ValueError(f"Invalid elastic checkpoint step: {step}")
        data = torch.load(directory / "data.pt", map_location="cpu", weights_only=False)
        self.actor_rollout_wg.load_checkpoint(str(directory / "actor"), del_local_after_load=False)
        self.train_dataloader.load_state_dict(data["dataloader"])
        torch.set_rng_state(data["torch_rng"])
        np.random.set_state(data["numpy_rng"])
        random.setstate(data["python_rng"])
        self.global_steps = step
        logger.info("ELASTIC restored model/optimizer/data at step %s from %s", step, directory)

    def _reissue_inflight_prompts(self, partition_id=None):
        import transfer_queue as tq

        partition = partition_id or self.train_partition_id
        data = tq.kv_list(partition)
        items = data.get(partition, {}) if data else {}
        if any(tag.get("is_prompt", False) and tag.get("status") in ("pending", "running") for tag in items.values()):
            raise RuntimeError("Elastic boundary restart found unfinished prompts; refusing to discard agent state")
        return 0
