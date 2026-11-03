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

from verl.experimental.elastic.checkpoint import PortableCheckpointManager
from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.single_controller.base.decorator import Dispatch, register


class ElasticDetachActorWorker(DetachActorWorker):
    """Detached actor with full-state, DP-world-size-independent checkpoint hooks."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.actor.strategy not in ("fsdp", "fsdp2"):
            raise NotImplementedError("Elastic detached actors support only fsdp/fsdp2, not Megatron or VeOmni")
        if "actor" not in self.role:
            raise NotImplementedError("ElasticDetachActorWorker requires an actor role")
        super().init_model()
        engine = self.actor.engine
        config = engine.engine_config
        if config.ulysses_sequence_parallel_size != 1 or config.fsdp_size != -1:
            raise NotImplementedError("Elastic portable checkpoints require pure DP and fsdp_size=-1 (no hybrid/SP)")
        if config.forward_only:
            raise NotImplementedError("Elastic portable checkpoints require a training engine")
        old = engine.checkpoint_manager
        engine.checkpoint_manager = PortableCheckpointManager(
            model=old.model,
            optimizer=old.optimizer,
            lr_scheduler=old.lr_scheduler,
            processing_class=old.processing_class,
            checkpoint_config=old.checkpoint_config,
        )
