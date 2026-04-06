# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extended Trainer for Cosmos Policy with epoch tracking.

This trainer extends the base ImaginaireTrainer to add:
- Epoch tracking and sampler epoch setting for proper distributed sampling
"""

import signal

import torch
import torch.utils.data

from cosmos_policy._src.imaginaire.model import ImaginaireModel
from cosmos_policy._src.imaginaire.trainer import ImaginaireTrainer
from cosmos_policy._src.imaginaire.utils import distributed, log, misc
from cosmos_policy._src.imaginaire.utils.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling


class CosmosPolicyTrainer(ImaginaireTrainer):
    """
    Extended Trainer for Cosmos Policy.

    Adds special handling for:
    - Epoch tracking to properly set dataloader sampler epochs (needed for distributed training)
    - Simplified initial validation check (removes run_validation_on_start requirement)
    """

    def __init__(self, config):
        super().__init__(config)

    def train(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:
        """The training function.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_train (torch.utils.data.DataLoader): The training data loader.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
        """
        # Leaving this for backward compability for now, but we can think about moving this to model.on_train_start for all models.
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)  # type: ignore
        model.on_train_start(self.config.trainer.memory_format)

        # Initialize the optimizer, scheduler, and grad_scaler.
        self.callbacks.on_optimizer_init_start()
        optimizer, scheduler = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()
        # Load the model checkpoint and get the starting iteration number.
        iteration = self.checkpointer.load(model, optimizer, scheduler, grad_scaler)
        grad_accum_iter = 0
        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        if self.config.trainer.distributed_parallelism == "ddp":
            # Create a DDP model wrapper.
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")

        log.info("Starting training...")
        # print("starting training")
        self.callbacks.on_train_start(model, iteration=iteration)
        # Initial validation.
        if self.config.trainer.run_validation and iteration == 0 and self.config.trainer.run_validation_on_start:
            self.validate(model, dataloader_val, iteration=iteration)
        _end_training = False
        epoch = 0
        local_rank = distributed.get_rank()
        while True:
            dataloader_train.sampler.set_epoch(epoch)
            dataloader_train_iter = iter(dataloader_train)

            while True:
                # 取数据
                data_batch = next(dataloader_train_iter)

                # 结束条件
                if iteration >= self.config.trainer.max_iter:
                    _end_training = True
                    break

                # 放到GPU
                data_batch = misc.to(data_batch, device="cuda")
                for k, v in data_batch.items():
                    if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                        data_batch[k] = v.to(dtype=torch.bfloat16)

                # 确保train模式
                if not model.training:
                    model_ddp.train()

                # 训练一步
                output_batch, loss, grad_accum_iter = self.training_step(
                    model_ddp,
                    optimizer,
                    scheduler,
                    grad_scaler,
                    data_batch,
                    iteration=iteration,
                    grad_accum_iter=grad_accum_iter,
                )

                # 梯度累积
                if grad_accum_iter != 0:
                    continue

                # 完成一次iteration
                iteration += 1

                # 打印loss
                print(f"rank:{local_rank} iter: {iteration}, loss: {loss.item()}")

            epoch += 1
            if _end_training:
                break
        log.success("Done with training.")
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()
