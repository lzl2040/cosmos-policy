import torch.nn as nn
from dataclasses import dataclass, field

@dataclass
class ACEConfig:
    """Configuration for Action Chunk Encoder.
    
    Args:
        action_dim: Dimension of action vectors (will be padded to action_dim_padded)
        action_dim_padded: Padded action dimension (default: 32)
        chunk_size: Number of actions in a chunk
        group_size: Number of actions per group for processing (default: 4)
        hidden_dim: Hidden dimension for the encoder (default: 768)
        num_attention_heads: Number of attention heads (default: 12)
        num_hidden_layers: Number of transformer layers (default: 12)
        intermediate_dim: Intermediate dimension in FFN (default: 3072)
        dropout: Dropout probability (default: 0.1)
        max_position_embeddings: Maximum position for RoPE (default: 512)
        output_dim: Output embedding dimension (optional, defaults to hidden_dim)
    """
    action_dim: int = 7
    max_action_dim: int = 32
    chunk_size: int = 16
    group_size: int = 4
    hidden_dim: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_dim: int = 3072
    dropout: float = 0.0
    max_position_embeddings: int = 512
    output_dim: int = None
    frozen_ace: bool = False
    
    def __post_init__(self):
        if self.output_dim is None:
            self.output_dim = self.hidden_dim
        # Validate that hidden_dim is divisible by num_attention_heads
        assert self.hidden_dim % self.num_attention_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_attention_heads ({self.num_attention_heads})"


"""Action Chunk Encoder (ACE) Model with Axial RoPE.

This module encodes action chunks into compact embeddings using:
1. Grouping actions (G actions per group)
2. Projecting to hidden dimension
3. Bidirectional self-attention with axial RoPE:
   - type-axis RoPE: distinguishes sample-rate token vs action tokens
   - time-axis RoPE: models temporal order of action tokens only
4. Output compact embedding

Sequence layout:
    [sample_rate_token, action_group_1, action_group_2, ..., action_group_N]

Axial RoPE design:
- type-axis:
    sample_rate_token -> type_id = 0
    action_tokens     -> type_id = 1
- time-axis:
    sample_rate_token -> no time rotation
    action_tokens     -> time_id = 0, 1, 2, ..., N-1
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


## decoder
class ACEDecoderAttention(nn.Module):
    """Self-attention for temporal action-token decoding."""

    def __init__(self, config: ACEConfig):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_dim // config.num_attention_heads

        assert self.hidden_dim % self.num_heads == 0

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)

        return self.o_proj(out)


class ACEDecoderMLP(nn.Module):
    def __init__(self, config: ACEConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim * 4)
        self.fc2 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ACEDecoderLayer(nn.Module):
    """Pre-norm temporal decoder layer."""

    def __init__(self, config: ACEConfig):
        super().__init__()

        self.attention = ACEDecoderAttention(config)
        self.mlp = ACEDecoderMLP(config)

        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask=attention_mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) cache.

    This module precomputes sin/cos tables for positions [0, max_position).
    """

    def __init__(self, dim: int, max_position: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base

        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def get_cos_sin_by_position_ids(
        self, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather cos/sin by explicit position ids.

        Args:
            position_ids: Tensor of shape (batch, seq_len) or (seq_len,)

        Returns:
            cos, sin: shape (batch, seq_len, dim) if input is 2D,
                      shape (1, seq_len, dim) if input is 1D
        """
        if position_ids.dim() == 1:
            cos = self.cos_cached[position_ids].unsqueeze(0)
            sin = self.sin_cached[position_ids].unsqueeze(0)
        elif position_ids.dim() == 2:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        else:
            raise ValueError(
                f"position_ids must have dim 1 or 2, got shape {position_ids.shape}"
            )
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half.

    [x1, x2] -> [-x2, x1]
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to x.

    Args:
        x:   (batch, seq_len, num_heads, dim)
        cos: (batch_or_1, seq_len, dim)
        sin: (batch_or_1, seq_len, dim)

    Returns:
        rotated x with same shape as input
    """
    cos = cos.unsqueeze(2)  # (batch_or_1, seq_len, 1, dim)
    sin = sin.unsqueeze(2)  # (batch_or_1, seq_len, 1, dim)
    return (x * cos) + (rotate_half(x) * sin)


class ACEAttention(nn.Module):
    """Multi-head self-attention with axial RoPE.

    Head dimension is split into:
    - type_dim: rotary over token type axis
    - time_dim: rotary over time axis
    """

    def __init__(self, config: ACEConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_dim // config.num_attention_heads
        self.hidden_dim = config.hidden_dim

        if config.hidden_dim % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_dim ({config.hidden_dim}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )

        # Split one head into type-axis part and time-axis part
        # Both RoPE dims must be even, so enforce even split.
        self.type_dim = self.head_dim // 2
        if self.type_dim % 2 != 0:
            self.type_dim -= 1
        self.time_dim = self.head_dim - self.type_dim
        if self.time_dim % 2 != 0:
            self.time_dim -= 1
            self.type_dim += 1

        if self.type_dim <= 0 or self.time_dim <= 0:
            raise ValueError(
                f"Invalid axial split: head_dim={self.head_dim}, "
                f"type_dim={self.type_dim}, time_dim={self.time_dim}"
            )
        if self.type_dim % 2 != 0 or self.time_dim % 2 != 0:
            raise ValueError(
                f"Both type_dim and time_dim must be even, got "
                f"type_dim={self.type_dim}, time_dim={self.time_dim}"
            )
        if self.type_dim + self.time_dim != self.head_dim:
            raise ValueError(
                f"type_dim + time_dim must equal head_dim, got "
                f"{self.type_dim} + {self.time_dim} != {self.head_dim}"
            )

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

    def _apply_axial_rope(
        self,
        x: torch.Tensor,
        cos_type: torch.Tensor,
        sin_type: torch.Tensor,
        cos_time: torch.Tensor,
        sin_time: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply type-axis and time-axis RoPE to q or k.

        Args:
            x:         (B, S, H, D)
            cos_type:  (B, S, type_dim)
            sin_type:  (B, S, type_dim)
            cos_time:  (B, S, time_dim)
            sin_time:  (B, S, time_dim)
            time_mask: (B, S), bool. True means this token uses time-axis RoPE.

        Returns:
            x_rotated: (B, S, H, D)
        """
        x_type = x[..., : self.type_dim]
        x_time = x[..., self.type_dim : self.type_dim + self.time_dim]

        # Always apply type-axis RoPE to all tokens
        x_type = apply_rotary_pos_emb(x_type, cos_type, sin_type)

        # Apply time-axis RoPE only to tokens where time_mask == True
        x_time_rot = apply_rotary_pos_emb(x_time, cos_time, sin_time)

        time_mask_expanded = time_mask.unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
        x_time = torch.where(time_mask_expanded, x_time_rot, x_time)

        return torch.cat([x_type, x_time], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_type: torch.Tensor,
        sin_type: torch.Tensor,
        cos_time: torch.Tensor,
        sin_time: torch.Tensor,
        time_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # (B, S, H, D)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply axial RoPE to Q and K
        q = self._apply_axial_rope(q, cos_type, sin_type, cos_time, sin_time, time_mask)
        k = self._apply_axial_rope(k, cos_type, sin_type, cos_time, sin_time, time_mask)

        # (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        # Back to (B, S, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        return self.o_proj(attn_output)


class ACEMLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: ACEConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.fc2 = nn.Linear(config.intermediate_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ACELayer(nn.Module):
    """Single transformer layer with self-attention and FFN."""

    def __init__(self, config: ACEConfig):
        super().__init__()
        self.attention = ACEAttention(config)
        self.mlp = ACEMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_type: torch.Tensor,
        sin_type: torch.Tensor,
        cos_time: torch.Tensor,
        sin_time: torch.Tensor,
        time_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(
            hidden_states=hidden_states,
            cos_type=cos_type,
            sin_type=sin_type,
            cos_time=cos_time,
            sin_time=sin_time,
            time_mask=time_mask,
            attention_mask=attention_mask,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class ActionReconstructionHead(nn.Module):
    """Temporal decoder for reconstructing action groups from action tokens."""

    def __init__(self, config: ACEConfig):
        super().__init__()

        self.group_dim = config.group_size * config.max_action_dim
        self.hidden_dim = config.hidden_dim

        # self.global_fuse = nn.Sequential(
        #     nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(config.hidden_dim),
        # )

        self.decoder_layers = nn.ModuleList(
            [
                ACEDecoderLayer(config)
                for _ in range(4)  # 先用 2 层，后面可以试 4 层
            ]
        )

        self.output_norm = nn.LayerNorm(config.hidden_dim)

        self.out_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, self.group_dim),
        )

    def forward(
        self,
        action_tokens: torch.Tensor,
        global_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # global_tokens = global_embedding.unsqueeze(1).expand_as(action_tokens)

        # x = torch.cat(
        #     [action_tokens, global_tokens],
        #     dim=-1,
        # )
        x = action_tokens

        # x = self.global_fuse(x)

        for layer in self.decoder_layers:
            x = layer(
                x,
                attention_mask=attention_mask,
            )

        x = self.output_norm(x)

        reconstructed_groups = self.out_mlp(x)

        return reconstructed_groups

class ActionChunkEncoder(nn.Module):
    """Action Chunk Encoder (ACE) with axial RoPE + temporal reconstruction decoder."""

    def __init__(self, config: ACEConfig):
        super().__init__()
        self.config = config
        self.frozen_ace = config.frozen_ace
        
        self.action_dim_padded = config.max_action_dim
        self.chunk_size = config.chunk_size
        self.group_size = config.group_size
        self.hidden_dim = config.hidden_dim

        assert config.chunk_size % config.group_size == 0, (
            f"chunk_size ({config.chunk_size}) must be divisible by "
            f"group_size ({config.group_size})"
        )

        self.num_groups = config.chunk_size // config.group_size
        self.group_dim = config.group_size * config.max_action_dim

        # Input projection
        self.input_proj = nn.Linear(self.group_dim, config.hidden_dim)

        # RoPE split
        head_dim = config.hidden_dim // config.num_attention_heads
        type_dim = head_dim // 2
        if type_dim % 2 != 0:
            type_dim -= 1

        time_dim = head_dim - type_dim
        if time_dim % 2 != 0:
            time_dim -= 1
            type_dim += 1

        if type_dim <= 0 or time_dim <= 0:
            raise ValueError(
                f"Invalid RoPE split derived from head_dim={head_dim}: "
                f"type_dim={type_dim}, time_dim={time_dim}"
            )

        self.type_dim = type_dim
        self.time_dim = time_dim

        # Axial RoPE
        self.type_rope = RotaryPositionEmbedding(
            dim=self.type_dim,
            max_position=2,
            base=10000.0,
        )

        self.time_rope = RotaryPositionEmbedding(
            dim=self.time_dim,
            max_position=max(config.max_position_embeddings, self.num_groups),
            base=10000.0,
        )

        # Encoder transformer layers
        self.layers = nn.ModuleList(
            [ACELayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Output
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)

        # Token embeddings
        self.sample_rate_embed = nn.Embedding(50, config.hidden_dim)
        self.token_type_embed = nn.Embedding(2, config.hidden_dim)
        self.tanh = nn.Tanh()

        # Reconstruction decoder
        self.action_decoder = ActionReconstructionHead(config)

    def _pad_actions(self, actions: torch.Tensor) -> torch.Tensor:
        current_dim = actions.shape[-1]

        if current_dim >= self.action_dim_padded:
            return actions[..., : self.action_dim_padded]

        padding = torch.zeros(
            *actions.shape[:-1],
            self.action_dim_padded - current_dim,
            device=actions.device,
            dtype=actions.dtype,
        )

        return torch.cat([actions, padding], dim=-1)

    def _group_actions(self, actions: torch.Tensor) -> torch.Tensor:
        batch_size = actions.shape[0]

        actions = actions.view(
            batch_size,
            self.num_groups,
            self.group_size,
            self.action_dim_padded,
        )

        actions = actions.view(
            batch_size,
            self.num_groups,
            self.group_dim,
        )

        return actions

    def _build_position_ids(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        seq_len = self.num_groups + 1

        type_ids = torch.zeros(
            batch_size,
            seq_len,
            dtype=torch.long,
            device=device,
        )
        type_ids[:, 1:] = 1

        time_ids = torch.zeros(
            batch_size,
            seq_len,
            dtype=torch.long,
            device=device,
        )

        action_time_ids = torch.arange(
            self.num_groups,
            dtype=torch.long,
            device=device,
        )

        time_ids[:, 1:] = action_time_ids.unsqueeze(0)

        time_mask = torch.zeros(
            batch_size,
            seq_len,
            dtype=torch.bool,
            device=device,
        )

        time_mask[:, 1:] = True

        return type_ids, time_ids, time_mask

    def decode_actions(self, action_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = action_tokens.shape[0]

        reconstructed_groups = self.action_decoder(
            action_tokens=action_tokens,
            global_embedding=None,
            attention_mask=None,
        )

        reconstructed_actions = reconstructed_groups.view(
            batch_size,
            self.num_groups,
            self.group_size,
            self.action_dim_padded,
        )

        reconstructed_actions = reconstructed_actions.reshape(
            batch_size,
            self.chunk_size,
            self.action_dim_padded,
        )

        return reconstructed_actions
    
    def forward(
        self,
        actions: torch.Tensor,
        sample_rate: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        if not torch.is_tensor(sample_rate):
            sample_rate = torch.tensor(
                sample_rate,
                device=actions.device,
            )

        if sample_rate.dim() == 0:
            sample_rate = sample_rate.unsqueeze(0).expand(actions.shape[0])

        elif (
            sample_rate.dim() == 1
            and sample_rate.shape[0] == 1
            and actions.shape[0] > 1
        ):
            sample_rate = sample_rate.expand(actions.shape[0])

        batch_size = actions.shape[0]

        # Save GT before grouping
        # gt_actions = actions[..., : self.action_dim]
        gt_actions = actions

        # Pad action dim
        if actions.shape[-1] != self.action_dim_padded:
            actions = self._pad_actions(actions)

        # Group actions
        grouped_actions = self._group_actions(actions)

        # Project action groups
        action_hidden = self.input_proj(grouped_actions)

        # Sample-rate token
        sample_token = self.sample_rate_embed(
            sample_rate.long()
        ).unsqueeze(1)

        # Build sequence
        hidden_states = torch.cat(
            [sample_token, action_hidden],
            dim=1,
        )

        # Token type embedding
        type_ids, time_ids, time_mask = self._build_position_ids(
            batch_size=batch_size,
            device=actions.device,
        )

        hidden_states = hidden_states + self.token_type_embed(type_ids)

        # RoPE
        cos_type, sin_type = self.type_rope.get_cos_sin_by_position_ids(type_ids)
        cos_time, sin_time = self.time_rope.get_cos_sin_by_position_ids(time_ids)

        # Encoder
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                cos_type=cos_type,
                sin_type=sin_type,
                cos_time=cos_time,
                sin_time=sin_time,
                time_mask=time_mask,
                attention_mask=attention_mask,
            )

        # Normalize hidden states as in your current version
        hidden_states = hidden_states / (
            hidden_states.abs().max(dim=-1, keepdim=True)[0] + 1e-8
        )

        # Global embedding
        embedding = hidden_states[:, 0, :]

        # Local action tokens
        action_tokens = hidden_states[:, 1:, :]
        return action_tokens