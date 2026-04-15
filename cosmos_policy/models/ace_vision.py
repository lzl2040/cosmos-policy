import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from torch import Tensor, nn
from transformers import SiglipModel, AutoProcessor
import math

class SmallConvBottleneck(nn.Module):
    """
    Small bottleneck:
    [B, D, H, W] -> [B, C_out, H_out, W_out]
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        out_hw: Tuple[int, int],
    ):
        super().__init__()
        self.out_hw = out_hw

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=min(8, mid_channels), num_channels=mid_channels),
            nn.SiLU(),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=min(8, mid_channels), num_channels=mid_channels),
            nn.SiLU(),

            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, H, W]
        x = self.net(x)
        x = F.adaptive_avg_pool2d(x, self.out_hw)  # force target shape
        return x


class TokenFusionHead(nn.Module):
    """
    Fuse original cls token and pooled dense token.
    """
    def __init__(self, cls_dim: int, pooled_dim: int, output_dim: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.LayerNorm(cls_dim + pooled_dim),
            nn.Linear(cls_dim + pooled_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, cls_token: torch.Tensor, pooled_token: torch.Tensor) -> torch.Tensor:
        # cls_token:   [B, cls_dim]
        # pooled_token:[B, pooled_dim]
        x = torch.cat([cls_token, pooled_token], dim=-1)
        return self.fuse(x)


class VisionEncoder(nn.Module):
    """
    Vision encoder using SigLIP2 or similar CLIP-like model.

    Outputs:
        - final_token: fused token for action alignment
        - vae_feature: dense VAE-like feature map
        - cls_token: original cls token from SigLIP
        - pooled_token: avg pooled token from dense feature
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        output_dim: int = 768,
        vae_shape: Tuple[int, int, int] = (16, 28, 28),   # (C_out, H_out, W_out) # for image_size=224, wan 2.1 vae
        bottleneck_mid: int = 256,
        dtype=torch.bfloat16,
    ):
        super().__init__()

        self.model = SiglipModel.from_pretrained(
            model_name,
            dtype=torch.float32
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        # hidden size
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "hidden_size"):
                vision_hidden_size = self.model.config.hidden_size
            elif hasattr(self.model.config, "vision_config"):
                vision_hidden_size = self.model.config.vision_config.hidden_size
            else:
                vision_hidden_size = 768
        else:
            vision_hidden_size = 768

        self.hidden_size = vision_hidden_size
        self.dtype = dtype

        # target dense latent shape
        self.vae_channels, self.vae_h, self.vae_w = vae_shape

        # projection for cls token if needed
        if vision_hidden_size != output_dim:
            self.cls_projection = nn.Linear(vision_hidden_size, output_dim)
        else:
            self.cls_projection = nn.Identity()

        # projection for patch tokens before bottleneck
        self.patch_projection = nn.Identity()

        # bottleneck to produce VAE-like feature
        self.bottleneck = SmallConvBottleneck(
            in_channels=vision_hidden_size,
            mid_channels=bottleneck_mid,
            out_channels=self.vae_channels,
            out_hw=(self.vae_h, self.vae_w),
        )

        # fuse [original cls] + [pooled dense token]
        self.fusion_head = TokenFusionHead(
            cls_dim=output_dim,
            pooled_dim=self.vae_channels,
            output_dim=output_dim,
        )

    def _infer_patch_grid(self, num_patch_tokens: int) -> Tuple[int, int]:
        """
        Infer patch grid from N.
        Assumes square grid.
        """
        side = int(math.sqrt(num_patch_tokens))
        if side * side != num_patch_tokens:
            raise ValueError(
                f"Patch token count {num_patch_tokens} is not a perfect square, "
                "cannot infer 2D grid automatically."
            )
        return side, side

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: image tensor, usually [B, C, H, W] or list of PIL images

        Returns:
            {
                "final_token": [B, output_dim],
                "vae_feature": [B, C_out, H_out, W_out],
                "cls_token":   [B, output_dim],
                "pooled_token":[B, C_out],
                "patch_tokens":[B, N, D],
            }
        """
        device = next(self.parameters()).device

        B, C, T, H, W = images.shape
        images = images.view(B * T, C, H, W).to(dtype=torch.float32)  # treat time
        
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {
            k: v.to(device=device if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }

        # vision_outputs.last_hidden_state:
        # often [B, 1+N, D], where first token is cls
        vision_outputs = self.model.vision_model(**inputs)
        patch_tokens = vision_outputs.last_hidden_state  # [B, L, D]
        cls_token = vision_outputs.pooler_output
        # print(f"cls_token shape: {cls_token.shape}, patch_tokens shape: {patch_tokens.shape}")

        # split cls token and patch tokens
        # cls_token = hidden[:, 0]         # [B, D]
        # patch_tokens = hidden[:, 1:]     # [B, N, D]

        # original cls -> projection
        cls_token = self.cls_projection(cls_token)  # [B, output_dim]

        # patch tokens -> 2D map
        BT, N, D = patch_tokens.shape
        H_patch, W_patch = self._infer_patch_grid(N)

        patch_tokens_2d = self.patch_projection(patch_tokens)         # [B, N, D]
        patch_tokens_2d = patch_tokens_2d.view(BT, H_patch, W_patch, D)
        patch_tokens_2d = patch_tokens_2d.permute(0, 3, 1, 2).contiguous()  # [B, D, H_patch, W_patch]

        # bottleneck -> VAE-like feature
        vae_feature = self.bottleneck(patch_tokens_2d)  # [B, C_out, H_out, W_out]

        # average pool -> token
        pooled_token = F.adaptive_avg_pool2d(vae_feature, output_size=1).flatten(1)  # [B, C_out]

        # fuse pooled token with original cls token
        final_token = self.fusion_head(cls_token, pooled_token)  # [B, output_dim]
        
        final_token = final_token.view(B, T, -1)  # reshape back to [B, T, output_dim]
        vae_feature = vae_feature.view(B, T, self.vae_channels, self.vae_h, self.vae_w)  # [B, T, C_out, H_out, W_out]
        cls_token = cls_token.view(B, T, -1)  # [B, T, output_dim]
        pooled_token = pooled_token.view(B, T, -1)  # [B, T, C_out]
        patch_tokens = patch_tokens.view(B, T, N, D)

        return {
            "final_token": final_token,
            "vae_feature": vae_feature,
            "cls_token": cls_token,
            "pooled_token": pooled_token,
            "patch_tokens": patch_tokens,
        }