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
Cosmos Policy Diffusion Model - extends Text2WorldModel with policy-specific functionality.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import attrs
import torch
import torch.nn as nn
from einops import rearrange

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_policy._src.imaginaire.modules.res_sampler import COMMON_SOLVER_OPTIONS
from cosmos_policy._src.imaginaire.utils import misc
from cosmos_policy._src.imaginaire.utils.context_parallel import broadcast_split_tensor, cat_outputs_cp
from cosmos_policy._src.predict2.models.text2world_model import (
    DiffusionModel as BaseDiffusionModel,
)
from cosmos_policy._src.predict2.models.text2world_model import (
    Text2WorldModelConfig as BaseText2WorldModelConfig,
)
from cosmos_policy._src.imaginaire.utils.denoise_prediction import DenoisePrediction
from cosmos_policy.conditioner import Text2WorldCondition
from cosmos_policy.modules.cosmos_sampler import CosmosPolicySampler
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE
from cosmos_policy.experiments.robot.cosmos_utils import extract_action_chunk_from_latent_sequence
from cosmos_policy.models.ace_vision import VisionEncoder
from cosmos_policy.models.ace_action import ActionChunkEncoder, ACEConfig
from cosmos_policy.models.ace import ACE

def replace_latent_with_action_chunk(
    x0: torch.Tensor, action_chunk: torch.Tensor, action_indices: torch.Tensor
) -> torch.Tensor:
    """
    Replaces the image latent (at the specified action index) in clean input image latents x0 with the action chunk.

    Example:
    Let's say x0 has shape (B=32, C'=16, T', H'=28, W'=28) and action_chunk has shape (B=32, chunk_size=8, action_dim=7).
    Then, this function will overwrite the (C'=16, H'=28, W'=28) volume at x0[:,:,action_indices,:,:] with the action chunk,
    repeating it as many times as needed to fill the entire volume.

    Args:
        x0 (torch.Tensor): Clean image latents.
        action_chunk (torch.Tensor): Ground-truth action chunk.
        action_indices (torch.Tensor): Batch indices of the image latents to replace.

    Returns:
        torch.Tensor: Modified image latents.
    """
    # Get latent to be replaced
    batch_indices = torch.arange(x0.shape[0], device=x0.device)
    action_image_latent = x0[batch_indices, :, action_indices, :, :]

    # Create a new tensor with the same shape as action_image_latent, filled with zeros
    result = torch.zeros_like(action_image_latent)

    # Get shapes
    batch_size, latent_channels, latent_h, latent_w = action_image_latent.shape

    # Flatten action_chunk (preserving batch dimension)
    flat_action = action_chunk.reshape(batch_size, -1)
    num_action_elements = flat_action.shape[1]

    # Calculate total elements in the target tensor (per batch)
    latent_elements = latent_channels * latent_h * latent_w

    # Check that there is enough room in the target tensor for all the action elements
    assert num_action_elements <= latent_elements, (
        f"Not enough room in the latent tensor for the full action chunk: {num_action_elements} action elements > {latent_elements} latent elements!"
    )

    # Calculate how many times we need to repeat the action tensor
    # The expression below is a concise way of doing ceiling division to get the correct number of repeats
    num_repeats = (latent_elements + num_action_elements - 1) // num_action_elements

    # Repeat the action tensor along dimension 1
    repeated_action = flat_action.repeat(1, num_repeats)

    # Take only what we need to fill the result tensor
    repeated_action = repeated_action[:, :latent_elements]

    # Reshape the target tensor to put all channel and spatial dimensions together
    flat_result = result.reshape(batch_size, -1)

    # Place the action chunk values into the beginning of the flattened result
    flat_result[:, :] = repeated_action

    # Reshape back to original shape
    result = flat_result.reshape(batch_size, latent_channels, latent_h, latent_w)

    # Get final latents tensor
    new_x0 = x0
    new_x0[batch_indices, :, action_indices, :, :] = result

    return new_x0


def replace_latent_with_proprio(x0: torch.Tensor, proprio: torch.Tensor, proprio_indices: torch.Tensor) -> torch.Tensor:
    """
    Replaces the image latent (at the specified proprio index) in clean input image latents x0 with the proprio.

    Example:
    Let's say x0 has shape (B=32, C'=16, T', H'=28, W'=28) and proprio has shape (B=32, proprio_dim=9).
    Then, this function will overwrite the (C'=16, H'=28, W'=28) volume at x0[:,:,proprio_indices,:,:] with the proprio,
    repeating it as many times as needed to fill the entire volume.

    Args:
        x0 (torch.Tensor): Clean image latents.
        proprio (torch.Tensor): Ground-truth proprio.
        proprio_indices (torch.Tensor): Batch indices of the image latents to replace.

    Returns:
        torch.Tensor: Modified image latents.
    """
    # Get latent to be replaced
    batch_indices = torch.arange(x0.shape[0], device=x0.device)
    proprio_image_latent = x0[batch_indices, :, proprio_indices, :, :]

    # Create a new tensor with the same shape as proprio_image_latent, filled with zeros
    result = torch.zeros_like(proprio_image_latent)

    # Get shapes
    batch_size, latent_channels, latent_h, latent_w = proprio_image_latent.shape

    # Get number of proprio elements
    num_proprio_elements = proprio.shape[1]

    # Calculate total elements in the target tensor (per batch)
    latent_elements = latent_channels * latent_h * latent_w

    # Check that there is enough room in the target tensor for all the proprio elements
    assert num_proprio_elements <= latent_elements, (
        f"Not enough room in the latent tensor for the full proprio: {num_proprio_elements} proprio elements > {latent_elements} latent elements!"
    )

    # Calculate how many times we need to repeat the proprio tensor
    # The expression below is a concise way of doing ceiling division to get the correct number of repeats
    num_repeats = (latent_elements + num_proprio_elements - 1) // num_proprio_elements

    # Repeat the proprio tensor along dimension 1
    repeated_proprio = proprio.repeat(1, num_repeats)

    # Take only what we need to fill the result tensor
    repeated_proprio = repeated_proprio[:, :latent_elements]

    # Reshape the target tensor to put all channel and spatial dimensions together
    flat_result = result.reshape(batch_size, -1)

    # Place the proprio values into the beginning of the flattened result
    flat_result[:, :] = repeated_proprio

    # Reshape latent back to original shape
    result = flat_result.reshape(batch_size, latent_channels, latent_h, latent_w)

    # Get final latents tensor
    new_x0 = x0
    new_x0[batch_indices, :, proprio_indices, :, :] = result

    return new_x0


@attrs.define(slots=False)
class CosmosPolicyModelConfig(BaseText2WorldModelConfig):
    """
    Extended config for Cosmos Policy diffusion model.
    Uses Cosmos Policy's HybridEDMSDE instead of the original EDMSDE.
    Also adds policy-specific parameters for loss masking and action prediction.
    """

    sde: LazyDict = L(HybridEDMSDE)(
        # Note: Most of these values get overridden later in the experiment configs
        p_mean=0.0,
        p_std=1.0,
        sigma_max=80,
        sigma_min=0.0002,
        hybrid_sigma_distribution=True,
        uniform_lower=1.0,
        uniform_upper=85.0,
    )

    # Whether to use loss masking to separate action, future state, and value prediction
    # - Policy prediction: only take loss on action predictions
    # - World model prediction: only take loss on future state predictions
    # - Value function prediction: only take loss on value predictions
    mask_loss_for_action_future_state_prediction: bool = False
    # Whether to use loss masking on value prediction during policy predictions (so we only take loss on action + future state predictions)
    mask_value_prediction_loss_for_policy_prediction: bool = False
    # Whether to mask out some inputs (current state and action) during future state value prediction
    mask_current_state_action_for_value_prediction: bool = False
    # Whether to mask out some inputs (future state) during Q(s,a) prediction
    mask_future_state_for_qvalue_prediction: bool = False

    # Action loss multiplier (if greater than 1, upweights loss on predicting actions relative to other losses)
    # (Must be an integer - or will be cast to an integer later!)
    action_loss_multiplier: int = 1

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert not (
            self.mask_loss_for_action_future_state_prediction and self.mask_value_prediction_loss_for_policy_prediction
        ), (
            "Cannot enable both mask_loss_for_action_future_state_prediction and mask_value_prediction_loss_for_policy_prediction!"
        )


class CosmosPolicyDiffusionModel(BaseDiffusionModel):
    """
    Cosmos Policy Diffusion Model - extends Text2WorldModel with policy-specific functionality.

    Adds support for:
    - Action chunk prediction and injection
    - Proprioception (proprio) prediction and injection
    - Value function prediction
    - Loss masking for different prediction types (action, future state, value)
    - Multi-component loss tracking
    """

    def __init__(self, config: CosmosPolicyModelConfig):
        super().__init__(config)
        self.config: CosmosPolicyModelConfig = config

        # Cosmos Policy SDE and Sampler
        self.sde = lazy_instantiate(config.sde)
        self.sampler = CosmosPolicySampler()
        
        # add action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768, eps=1e-6),
            nn.SiLU(),
            nn.Linear(768, 128), # 128 = 32 * group_size
        )
        # self.action_proj = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.LayerNorm(768, eps=1e-6),
        #     nn.SiLU(),
        #     nn.Linear(768, 2048)
        # ).to(self.precision)
        self.action_group_size = 4

    def denoise(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        sigma: torch.Tensor,
        condition: Text2WorldCondition,
        action_latent: Optional[torch.Tensor] = None,
        num_action_tokens: int = 4,
    ) -> Tuple[DenoisePrediction, Optional[torch.Tensor]]:
        """
        Performs denoising on the input noise data, noise level, and condition.
        
        Extended to support action latent concatenation.

        Args:
            xt_B_C_T_H_W (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (Text2WorldCondition): conditional information, generated from self.conditioner
            action_latent (Optional[torch.Tensor]): Optional action latent tensor [B, num_action_tokens, hidden_dim].
                If provided, will be concatenated with image tokens in the denoising network.
            num_action_tokens (int): Number of action tokens. Default: 4.

        Returns:
            Tuple[DenoisePrediction, Optional[torch.Tensor]]: 
                - DenoisePrediction: The denoised prediction, includes clean data prediction (x0), noise prediction (eps_pred).
                - action_output: Optional action output tensor [B, num_action_tokens, hidden_dim] from the network.
        """
        if sigma.ndim == 1:
            sigma_B_T = rearrange(sigma, "b -> b 1")
        elif sigma.ndim == 2:
            sigma_B_T = sigma
        else:
            raise ValueError(f"sigma shape {sigma.shape} is not supported")
        sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")
        # get precondition for the network
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling(sigma=sigma_B_1_T_1_1)
        # Get action scaling coefficients (same as image latent scaling)
        action_c_skip_B_T, action_c_out_B_T, action_c_in_B_T, action_c_noise_B_T = self.scaling(sigma=sigma_B_1_T_1_1.squeeze(dim=[1, 3, 4]))

        # forward pass through the network with optional action latent
        net_output = self.net(
            x_B_C_T_H_W=(xt_B_C_T_H_W * c_in_B_1_T_1_1).to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
            action_latent=action_latent,
            num_action_tokens=num_action_tokens,
        )
        
        # Handle different return types from network
        action_output = None
        if isinstance(net_output, tuple):
            # Network returns (image_output, intermediate_features, action_output)
            net_output_B_C_T_H_W = net_output[0].float()
            action_output = net_output[2] if len(net_output) > 2 else None
        else:
            net_output_B_C_T_H_W = net_output.float()

        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W
        
        # Apply scaling to action_output (same as image latent)
        # action_x0_pred = c_skip * action_latent + c_out * network_output
        if action_output is not None and action_latent is not None:
            # action_c_skip_B_T: [B, T] -> [B, 1] for action tokens (single noise level per sample)
            # action_c_out_B_T: [B, T] -> [B, 1] for action tokens
            action_c_skip = action_c_skip_B_T[:, :1]  # [B, 1]
            action_c_out = action_c_out_B_T[:, :1]    # [B, 1]
            # action_output: [B, num_action_tokens, hidden_dim]
            # action_latent: [B, num_action_tokens, hidden_dim]
            action_output = action_c_skip.unsqueeze(-1) * action_latent + action_c_out.unsqueeze(-1) * action_output

        # get noise prediction based on sde
        eps_pred_B_C_T_H_W = (xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / sigma_B_1_T_1_1

        return DenoisePrediction(x0_pred_B_C_T_H_W, eps_pred_B_C_T_H_W, None), action_output

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the Cosmos Policy diffusion model.

        Extended from base to pass policy-specific data (actions, proprio, values, masks).

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: The computed loss for the training step as a PyTorch Tensor.
        """
        self._update_train_stats(data_batch)

        # Obtain text embeddings online
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        # there, state will be filled in condition.gt_frames
        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)

        # Sample pertubation noise levels and N(0, 1) noises
        sigma_B_T, epsilon_B_C_T_H_W = self.draw_training_sigma_and_epsilon(x0_B_C_T_H_W.size(), condition)
        # print(sigma_B_T.shape) # 6 1

        # Broadcast and split the input data and condition for model parallelism
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = self.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )
        output_batch, kendall_loss, _, _, kendall_loss_action_mse_loss = self.compute_loss_with_epsilon_and_sigma(
            x0_B_C_T_H_W,
            condition,
            epsilon_B_C_T_H_W,
            sigma_B_T,
            sample_rate=data_batch["sample_rate"],
            action_chunk=data_batch["actions"],
            action_indices=data_batch["action_latent_idx"],
            proprio=data_batch["proprio"],
            current_proprio_indices=data_batch["current_proprio_latent_idx"],
            future_proprio=data_batch["future_proprio"],
            future_proprio_indices=data_batch["future_proprio_latent_idx"],
            future_wrist_image_indices=data_batch["future_wrist_image_latent_idx"],
            future_wrist_image2_indices=(
                data_batch["future_wrist_image2_latent_idx"] if "future_wrist_image2_latent_idx" in data_batch else None
            ),
            future_image_indices=data_batch["future_image_latent_idx"],
            future_image2_indices=(
                data_batch["future_image2_latent_idx"] if "future_image2_latent_idx" in data_batch else None
            ),
            rollout_data_mask=data_batch["rollout_data_mask"],
            world_model_sample_mask=data_batch["world_model_sample_mask"],
            value_function_sample_mask=data_batch["value_function_sample_mask"],
            value_function_return=data_batch["value_function_return"],
            value_indices=data_batch["value_latent_idx"],
            gripper_index=data_batch["gripper_index"],
        )

        # print(self.loss_reduce)
        if self.loss_reduce == "mean":
            kendall_loss = kendall_loss.mean() * self.loss_scale
        elif self.loss_reduce == "sum":
            kendall_loss = kendall_loss.sum(dim=1).mean() * self.loss_scale
        else:
            raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")

        return output_batch, kendall_loss + 0.2 * kendall_loss_action_mse_loss

    def compute_loss_with_epsilon_and_sigma(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: Text2WorldCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
        sample_rate: torch.Tensor,
        action_chunk: torch.Tensor,
        action_indices: torch.Tensor,
        proprio: torch.Tensor,
        current_proprio_indices: torch.Tensor,
        future_proprio: torch.Tensor,
        future_proprio_indices: torch.Tensor,
        future_wrist_image_indices: torch.Tensor,
        future_wrist_image2_indices: Optional[torch.Tensor],
        future_image_indices: torch.Tensor,
        future_image2_indices: Optional[torch.Tensor],
        rollout_data_mask: torch.Tensor,
        world_model_sample_mask: torch.Tensor,
        value_function_sample_mask: torch.Tensor,
        value_function_return: torch.Tensor,
        value_indices: torch.Tensor,
        gripper_index: torch.Tensor,
    ):
        """
        NOTE (user): Modified to add action chunk prediction and future image prediction + action chunk loss logging.

        Compute loss given epsilon and sigma with policy-specific functionality.

        This method extends the base implementation to support:
        1. Latent injection for actions, proprioception, and values
        2. Loss masking for different prediction types (action, future state, value)
        3. Detailed per-component loss tracking

        Args:
            x0_B_C_T_H_W: image/video latent
            condition: text condition
            epsilon_B_C_T_H_W: noise
            sigma_B_T: noise level
            action_chunk: ground truth action chunk
            action_indices: indices for action latent frames
            proprio: current proprioception
            current_proprio_indices: indices for current proprio latent frames
            future_proprio: future proprioception
            future_proprio_indices: indices for future proprio latent frames
            future_wrist_image_indices: indices for future wrist image latent frames
            future_wrist_image2_indices: indices for future wrist image #2 latent frames
            future_image_indices: indices for future primary image latent frames
            future_image2_indices: indices for future secondary image latent frames
            rollout_data_mask: mask for rollout vs demo data
            world_model_sample_mask: mask for world model samples
            value_function_sample_mask: mask for value function samples
            value_function_return: ground truth value function returns
            value_indices: indices for value latent frames

        Returns:
            tuple: A tuple containing four elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor 1: kendall loss,
                - Tensor 2: MSE loss,
                - Tensor 3: EDM loss
        """
        # NOTE (user): Action chunk, proprio, value injection
        # x0_B_C_T_H_W: (B, C', T', H', W')
        # x0_B_C_T_H_W is the VAE-encoded latent representing several input images (which may be in a different order than below):
        # - conditional frames (e.g., wrist camera image, primary image, optional history)
        # - future image(s) to predict
        # NOTE: Action is now handled separately as action latent, NOT injected into image latent
        # NOTE: Proprio is removed - not used as input or prediction target
        condition.orig_x0_B_C_T_H_W = x0_B_C_T_H_W.clone()  # Keep a backup of the original gt_frames
        batch_indices = torch.arange(x0_B_C_T_H_W.shape[0], device=x0_B_C_T_H_W.device)
        C_latent, H_latent, W_latent = x0_B_C_T_H_W.shape[1], x0_B_C_T_H_W.shape[3], x0_B_C_T_H_W.shape[4]
        
        # Prepare action embeddings (encode action chunk)
        # action_chunk: [B, chunk_size, action_dim]
        with torch.no_grad():
            action_embeddings = self.ace.action_encoder(action_chunk, sample_rate)  # [B, chunk_size // group_size, hidden_dim]
        
        # action_embeddings = self.action_proj(action_embeddings)  # [B, chunk_size // group_size, hidden_dim_dit]
        
        # Action
        x0_B_C_T_H_W = replace_latent_with_action_chunk(
            x0_B_C_T_H_W,
            action_embeddings,
            action_indices=action_indices,
        )
        
        condition.orig_gt_frames = condition.gt_frames.clone()  # Keep a backup of the original gt_frames
        condition.gt_frames = replace_latent_with_action_chunk(
            condition.gt_frames, action_embeddings, action_indices=action_indices
        )
        
        # Get the mean and stand deviation of the marginal probability distribution.
        mean_B_C_T_H_W, std_B_T = self.sde.marginal_prob(x0_B_C_T_H_W, sigma_B_T)
        # Generate noisy observations
        xt_B_C_T_H_W = mean_B_C_T_H_W + epsilon_B_C_T_H_W * rearrange(std_B_T, "b t -> b 1 t 1 1")
        # make prediction - pass action latent to denoise network
        model_pred = self.denoise(
            xt_B_C_T_H_W, 
            sigma_B_T, 
            condition,
        )
        # loss weights for different noise levels
        weights_per_sigma_B_T = self.get_per_sigma_loss_weights(sigma=sigma_B_T)

        # Construct mask to support masking out loss (or scaling it by some multiplier) for different types of predictions
        B, T = x0_B_C_T_H_W.shape[0], x0_B_C_T_H_W.shape[2]
        final_mask_B_T = torch.ones((B, T), dtype=torch.long, device=sigma_B_T.device)  # All 1s mask initially

        # extra loss mask for each sample, for example, human faces, hands
        pred_mse_B_C_T_H_W = (x0_B_C_T_H_W - model_pred.x0) ** 2
        edm_loss_B_C_T_H_W = pred_mse_B_C_T_H_W * rearrange(weights_per_sigma_B_T, "b t -> b 1 t 1 1")
        
        kendall_loss = edm_loss_B_C_T_H_W # prediction loss
        
        # action reconstruction loss
        action_shape = action_embeddings.shape[1:]
        pred_action_embeds = extract_action_chunk_from_latent_sequence(model_pred.x0, action_shape, action_indices)
        pred_action = self.action_decoder(pred_action_embeds)  # [B, chunk_size // group_size, action_dim]
        pred_action = pred_action.view(pred_action.shape[0], pred_action.shape[1], self.action_group_size, -1) 
        pred_action = pred_action.view(pred_action.shape[0], -1, pred_action.shape[-1])  # [B, chunk_size, action_dim]
        kendall_loss_action_mse_loss = ((pred_action - action_chunk) ** 2).mean()
        kendall_loss_action_l1_loss = torch.abs(pred_action - action_chunk).mean()
        # print(f"Kendall loss action mse: {kendall_loss_action_mse_loss.item():.4f}, l1: {kendall_loss_action_l1_loss.item():.4f}") 
        
        # kendall_loss_action_mse_loss = torch.tensor(0.0, device=x0_B_C_T_H_W.device)
        # kendall_loss_action_l1_loss = torch.tensor(0.0, device=x0_B_C_T_H_W.device)
        kendall_loss_state_mse_loss = torch.tensor(0.0, device=x0_B_C_T_H_W.device)
        kendall_loss_state_l1_loss = torch.tensor(0.0, device=x0_B_C_T_H_W.device)

        # Get losses for future third-person image prediction
        if torch.all(future_image_indices != -1):  # -1 indicates future third-person image is not used
            batch_indices = torch.arange(x0_B_C_T_H_W.shape[0], device=x0_B_C_T_H_W.device)
            future_image_diff = (
                x0_B_C_T_H_W[batch_indices, :, future_image_indices, :, :]
                - model_pred.x0[batch_indices, :, future_image_indices, :, :]
            )
            future_image_diff_demo = future_image_diff[rollout_data_mask == 0]
            future_image_diff_world_model = future_image_diff[world_model_sample_mask == 1]

            demo_sample_future_image_mse_loss = (future_image_diff_demo**2).mean()
            demo_sample_future_image_l1_loss = torch.abs(future_image_diff_demo).mean()
            world_model_sample_future_image_mse_loss = (future_image_diff_world_model**2).mean()
            world_model_sample_future_image_l1_loss = torch.abs(future_image_diff_world_model).mean()
            all_samples_future_image_mse_loss = (future_image_diff**2).mean()
            all_samples_future_image_l1_loss = torch.abs(future_image_diff).mean()
        else:
            # If not generating future third-person images, set all future image losses to nan
            demo_sample_future_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            demo_sample_future_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)

        # Get losses for future wrist image prediction
        if torch.all(future_wrist_image_indices != -1):  # -1 indicates future wrist image is not used
            future_wrist_image_diff = (
                x0_B_C_T_H_W[batch_indices, :, future_wrist_image_indices, :, :]
                - model_pred.x0[batch_indices, :, future_wrist_image_indices, :, :]
            )
            future_wrist_image_diff_demo = future_wrist_image_diff[rollout_data_mask == 0]
            future_wrist_image_diff_world_model = future_wrist_image_diff[world_model_sample_mask == 1]
            
            demo_sample_future_wrist_image_mse_loss = (future_wrist_image_diff_demo**2).mean()
            demo_sample_future_wrist_image_l1_loss = torch.abs(future_wrist_image_diff_demo).mean()
            world_model_sample_future_wrist_image_mse_loss = (future_wrist_image_diff_world_model**2).mean()
            world_model_sample_future_wrist_image_l1_loss = torch.abs(future_wrist_image_diff_world_model).mean()
            all_samples_future_wrist_image_mse_loss = (future_wrist_image_diff**2).mean()
            all_samples_future_wrist_image_l1_loss = torch.abs(future_wrist_image_diff).mean()
        else:
            # If not generating future wrist images, set all future wrist image losses to nan
            demo_sample_future_wrist_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            demo_sample_future_wrist_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_wrist_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_wrist_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_wrist_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_wrist_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)

        # Get losses for future proprio prediction
        if torch.all(future_proprio_indices != -1):  # -1 indicates future proprio is not used
            future_proprio_diff = (
                x0_B_C_T_H_W[batch_indices, :, future_proprio_indices, :, :]
                - model_pred.x0[batch_indices, :, future_proprio_indices, :, :]
            )
            future_proprio_diff_demo = future_proprio_diff[rollout_data_mask == 0]
            future_proprio_diff_world_model = future_proprio_diff[world_model_sample_mask == 1]
            
            
            demo_sample_future_proprio_mse_loss = (future_proprio_diff_demo**2).mean()
            demo_sample_future_proprio_l1_loss = torch.abs(future_proprio_diff_demo).mean()
            world_model_sample_future_proprio_mse_loss = (future_proprio_diff_world_model**2).mean()
            world_model_sample_future_proprio_l1_loss = torch.abs(future_proprio_diff_world_model).mean()
            all_samples_future_proprio_mse_loss = (future_proprio_diff**2).mean()
            all_samples_future_proprio_l1_loss = torch.abs(future_proprio_diff).mean()
        else:
            # If not generating future proprio, set all future proprio losses to nan
            demo_sample_future_proprio_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            demo_sample_future_proprio_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_proprio_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_proprio_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_proprio_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_proprio_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)

        # demo_sample_action_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)  # Mask out demo sample action loss for now (not used in current experiments)
        # demo_sample_action_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
        # all_samples_action_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)  # Mask out all sample action loss for now (not used in current experiments)
        # all_samples_action_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
        
        value_function_sample_value_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)  # Mask out value function loss for now (not used in current experiments)
        value_function_sample_value_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)    
        world_model_sample_value_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)  # Mask out world model value loss for now (not used in current experiments)
        world_model_sample_value_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
        demo_sample_value_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)  # Mask out demo sample value loss for now (not used in current experiments)
        demo_sample_value_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
        
        # Get losses for action prediction
        action_diff = (
            x0_B_C_T_H_W[batch_indices, :, action_indices, :, :] - model_pred.x0[batch_indices, :, action_indices, :, :]
        )
        action_diff_demo = action_diff[rollout_data_mask == 0]
        action_diff_world_model = action_diff[world_model_sample_mask == 1]
        demo_sample_action_mse_loss = (action_diff_demo**2).mean()
        demo_sample_action_l1_loss = torch.abs(action_diff_demo).mean()
        all_samples_action_mse_loss = (action_diff**2).mean()
        all_samples_action_l1_loss = torch.abs(action_diff).mean()

        # Get losses for value function prediction
        # value_diff = (
        #     x0_B_C_T_H_W[batch_indices, :, value_indices, :, :] - model_pred.x0[batch_indices, :, value_indices, :, :]
        # )
        # value_diff_demo = value_diff[rollout_data_mask == 0]
        # value_diff_world_model = value_diff[world_model_sample_mask == 1]
        # value_diff_value_function = value_diff[value_function_sample_mask == 1]
        # demo_sample_value_mse_loss = (value_diff_demo**2).mean()
        # demo_sample_value_l1_loss = torch.abs(value_diff_demo).mean()
        # world_model_sample_value_mse_loss = (value_diff_world_model**2).mean()
        # world_model_sample_value_l1_loss = torch.abs(value_diff_world_model).mean()
        # value_function_sample_value_mse_loss = (value_diff_value_function**2).mean()
        # value_function_sample_value_l1_loss = torch.abs(value_diff_value_function).mean()
        

        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigma_B_T,
            "weights_per_sigma": weights_per_sigma_B_T,
            "condition": condition,
            "model_pred": model_pred,
            "mse_loss": pred_mse_B_C_T_H_W.mean(),
            # "mse_loss": mse_loss,
            "action_mse_loss": kendall_loss_action_mse_loss,
            "action_l1_loss": kendall_loss_action_l1_loss,
            "state_mse_loss": kendall_loss_state_mse_loss,
            "state_l1_loss": kendall_loss_state_l1_loss,
            "edm_loss": edm_loss_B_C_T_H_W.mean(),
            "edm_loss_per_frame": torch.mean(edm_loss_B_C_T_H_W, dim=[1, 3, 4]),
            # Demo sample losses
            "demo_sample_action_mse_loss": demo_sample_action_mse_loss,  # Main action loss for policy
            "demo_sample_action_l1_loss": demo_sample_action_l1_loss,  # Main action loss for policy
            "demo_sample_future_proprio_mse_loss": demo_sample_future_proprio_mse_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_proprio_l1_loss": demo_sample_future_proprio_l1_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_wrist_image_mse_loss": demo_sample_future_wrist_image_mse_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_wrist_image_l1_loss": demo_sample_future_wrist_image_l1_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_image_mse_loss": demo_sample_future_image_mse_loss,  # Auxiliary future state loss for policy
            "demo_sample_future_image_l1_loss": demo_sample_future_image_l1_loss,  # Auxiliary future state loss for policy
            "demo_sample_value_mse_loss": demo_sample_value_mse_loss,  # Auxiliary value loss for policy
            "demo_sample_value_l1_loss": demo_sample_value_l1_loss,  # Auxiliary value loss for policy
            # World model sample losses
            "world_model_sample_future_proprio_mse_loss": world_model_sample_future_proprio_mse_loss,  # Main future state loss for world model
            "world_model_sample_future_proprio_l1_loss": world_model_sample_future_proprio_l1_loss,  # Main future state loss for world model
            "world_model_sample_future_wrist_image_mse_loss": world_model_sample_future_wrist_image_mse_loss,  # Main future state loss for world model
            "world_model_sample_future_wrist_image_l1_loss": world_model_sample_future_wrist_image_l1_loss,  # Main future state loss for world model
            "world_model_sample_future_image_mse_loss": world_model_sample_future_image_mse_loss,  # Main future state loss for world model
            "world_model_sample_future_image_l1_loss": world_model_sample_future_image_l1_loss,  # Main future state loss for world model
            "world_model_sample_value_mse_loss": world_model_sample_value_mse_loss,  # Auxiliary value loss for world model
            "world_model_sample_value_l1_loss": world_model_sample_value_l1_loss,  # Auxiliary value loss for world model
            # Value function sample losses
            "value_function_sample_value_mse_loss": value_function_sample_value_mse_loss,  # Main loss for value function
            "value_function_sample_value_l1_loss": value_function_sample_value_l1_loss,  # Main loss for value function
        }
        return output_batch, kendall_loss, pred_mse_B_C_T_H_W, edm_loss_B_C_T_H_W, kendall_loss_action_mse_loss

    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
        x_sigma_max: Optional[torch.Tensor] = None,
        sigma_max: float | None = None,
        use_variance_scale: bool = False,
        worker_id: int = 0,
        skip_vae_encoding: bool = False,
        previous_generated_latent: torch.Tensor = None,
        return_orig_clean_latent_frames: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch with Cosmos Policy extensions.

        Extended to support:
        - Variance scaling for increased diversity
        - Autoregressive generation (skip_vae_encoding, previous_generated_latent)
        - Returning original clean latent frames

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            solver_option (str): differential equation solver option, default to "2ab"
            use_variance_scale (bool): use variance scale to increase diversity in outputs
            worker_id (int): worker id for random seed
            skip_vae_encoding (bool): Skip VAE encoding if True
            previous_generated_latent (torch.Tensor): Previous generated sample
            return_orig_clean_latent_frames (bool): Whether to return the clean latent frames
        """
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            # print(f"Height:{_H} Width:{_W}")
            state_shape = [
                self.config.state_ch,
                # self.tokenizer.get_latent_num_frames(_T),
                # _H // self.tokenizer.spatial_compression_factor,
                # _W // self.tokenizer.spatial_compression_factor,
                9,
                28,
                28
            ]

        if return_orig_clean_latent_frames:
            x0_fn, orig_clean_latent_frames = self.get_x0_fn_from_batch(
                data_batch,
                guidance,
                is_negative_prompt=is_negative_prompt,
                skip_vae_encoding=skip_vae_encoding,
                previous_generated_latent=previous_generated_latent,
                return_orig_clean_latent_frames=True,
            )
        else:
            x0_fn = self.get_x0_fn_from_batch(
                data_batch,
                guidance,
                is_negative_prompt=is_negative_prompt,
                skip_vae_encoding=skip_vae_encoding,
                previous_generated_latent=previous_generated_latent,
            )

        # NOTE (user): Add random variance scaling to increase diversity in outputs
        if use_variance_scale:
            torch.manual_seed(seed)
            sigma_max_variance_scale = torch.rand(1).item() * 2.0 + 1.0  # uniform between 1.0 and 3.0
            sigma_min_variance_scale = torch.rand(1).item() * 0.9 + 0.1  # uniform between 0.1 and 1.0
        else:
            sigma_max_variance_scale = 1.0
            sigma_min_variance_scale = 1.0

        # NOTE: FlowUniPC scheduler support (inherited from base, keeping for compatibility)
        if self.config.use_flowunipc_scheduler:
            # Use parent implementation for FlowUniPC
            return super().generate_samples_from_batch(
                data_batch,
                guidance,
                seed,
                state_shape,
                n_sample,
                is_negative_prompt,
                num_steps,
                solver_option,
                x_sigma_max,
                sigma_max,
                **kwargs,
            )

        if x_sigma_max is None:
            # 生成随机噪声
            x_sigma_max = (
                misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    self.tensor_kwargs["device"],
                    seed,
                )
                * self.sde.sigma_max
                * sigma_max_variance_scale
            )
            #

        if self.net.is_context_parallel_enabled:
            x_sigma_max = broadcast_split_tensor(
                x_sigma_max, seq_dim=2, process_group=self.get_context_parallel_group()
            )

        if sigma_max is None:
            sigma_max = self.sde.sigma_max
        samples = self.sampler(
            x0_fn,
            x_sigma_max,
            num_steps=num_steps,
            sigma_max=sigma_max * sigma_max_variance_scale,
            sigma_min=self.sde.sigma_min * sigma_min_variance_scale,
            solver_option=solver_option,
        )
        if self.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())

        if return_orig_clean_latent_frames:
            return samples, orig_clean_latent_frames
        else:
            return samples
