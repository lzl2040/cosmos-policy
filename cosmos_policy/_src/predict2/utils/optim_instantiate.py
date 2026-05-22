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


import hydra
import torch
from omegaconf import ListConfig
from torch import nn

from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy._src.predict2.utils.fused_adam_dtensor import FusedAdam


from typing import List
import torch
import torch.nn as nn


def get_regular_param_group(models: List[nn.Module]):
    """
    Separate parameters into decay and no_decay groups
    from multiple models.
    """

    param_dict = {}

    for model in models:
        for pn, p in model.named_parameters():
            if p.requires_grad:
                # 防止不同模型中参数名重复
                param_dict[f"{id(model)}.{pn}"] = p
    print(param_dict.keys())
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

    return decay_params, nodecay_params


def get_base_optimizer(
    models: List[nn.Module],
    lr: float,
    weight_decay: float,
    optim_type: str = "adamw",
    **kwargs,
) -> torch.optim.Optimizer:

    net_decay_param, net_nodecay_param = get_regular_param_group(models)

    num_decay_params = sum(p.numel() for p in net_decay_param)
    num_nodecay_params = sum(p.numel() for p in net_nodecay_param)
    net_param_total = num_decay_params + num_nodecay_params

    log.critical(f"total num parameters : {net_param_total:,}")

    param_group = [
        {
            "params": net_decay_param,
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": net_nodecay_param,
            "lr": lr,
            "weight_decay": 0.0,
        },
    ]

    if optim_type == "adamw":
        opt_cls = torch.optim.AdamW
    elif optim_type == "fusedadam":
        opt_cls = FusedAdam
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")

    for k, v in kwargs.items():
        if isinstance(v, ListConfig):
            kwargs[k] = list(v)

    return opt_cls(param_group, **kwargs)


def get_base_scheduler(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    scheduler_config: dict,
):
    net_scheduler = hydra.utils.instantiate(scheduler_config)
    net_scheduler.model = model

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            net_scheduler.schedule,
        ],
    )
