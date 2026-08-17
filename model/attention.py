"""
attention.py

Educational Llama-style attention.

Includes:

- Q/K/V projections
- RoPE
- Q/K normalization
- Grouped Query Attention
- causal masking
- scaled dot-product attention
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size,
        eps=1e-5
    ):

        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(hidden_size)
        )

        self.eps = eps

    def forward(self, x):

        # Keep normalization in float32
        # for numerical stability.

        x_float = x.float()

        variance = (
            x_float.pow(2)
            .mean(
                dim=-1,
                keepdim=True
            )
        )

        x_norm = (
            x_float
            * torch.rsqrt(
                variance + self.eps
            )
        )

        return (
            x_norm
            * self.weight
        ).to(x.dtype)


# ============================================================
# Q/K normalization
# ============================================================

class QKNorm(nn.Module):

    def __init__(self, eps=1e-6):

        super().__init__()

        self.eps = eps

    def forward(self, x):

        mean_square = (
            x.float()
            .pow(2)
            .mean(
                dim=-1,
                keepdim=True
            )
        )

        return (
            x
            * torch.rsqrt(
                mean_square + self.eps
            )
        )


# ============================================================
# Rotary Position Embedding
# ============================================================

class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim,
        max_position_embeddings,
        theta=10000.0
    ):

        super().__init__()

        assert head_dim % 2 == 0

        self.head_dim = head_dim

        inv_freq = (
            1.0
            / (
                theta
                ** (
                    torch.arange(
                        0,
                        head_dim,
                        2,
                        dtype=torch.float32
                    )
                    / head_dim
                )
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False
        )

        positions = torch.arange(
            max_position_embeddings,
            dtype=torch.float32
        )

        # torch.outer is exactly the operation
        # you asked about previously.
        freqs = torch.outer(
            positions,
            inv_freq
        )

        self.register_buffer(
            "cos",
            freqs.cos(),
            persistent=False
        )

        self.register_buffer(
            "sin",
            freqs.sin(),
            persistent=False
        )

    def forward(self, q, k):

        seq_len = q.shape[-2]

        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        # [seq, head_dim / 2]
        #
        # becomes
        #
        # [1, 1, seq, head_dim / 2]

        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        # -----------------------------------------------
        # Split even and odd dimensions
        # -----------------------------------------------

        q_even = q[..., 0::2]
        q_odd = q[..., 1::2]

        k_even = k[..., 0::2]
        k_odd = k[..., 1::2]

        # -----------------------------------------------
        # Rotation
        # -----------------------------------------------

        q_even_new = (
            q_even * cos
            - q_odd * sin
        )

        q_odd_new = (
            q_even * sin
            + q_odd * cos
        )

        k_even_new = (
            k_even * cos
            - k_odd * sin
        )

        k_odd_new = (
            k_even * sin
            + k_odd * cos
        )

        # -----------------------------------------------
        # Interleave dimensions again
        # -----------------------------------------------

        q_out = torch.stack(
            [
                q_even_new,
                q_odd_new
            ],
            dim=-1
        ).flatten(-2)

        k_out = torch.stack(
            [
                k_even_new,
                k_odd_new
            ],
            dim=-1
        ).flatten(-2)

        return q_out, k_out


# ============================================================
# GQA helper
# ============================================================

def repeat_kv(
    x,
    num_repeats
):

    """
    Input:

        [batch, kv_heads, seq, head_dim]

    Output:

        [batch, attention_heads, seq, head_dim]
    """

    if num_repeats == 1:
        return x

    batch, kv_heads, seq, head_dim = x.shape

    x = x[:, :, None, :, :]

    x = x.expand(
        batch,
        kv_heads,
        num_repeats,
        seq,
        head_dim
    )

    return x.reshape(
        batch,
        kv_heads * num_repeats,
        seq,
        head_dim
    )


# ============================================================
# Llama Attention
# ============================================================

class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        max_position_embeddings,
        rope_theta=10000.0,
        qk_norm_eps=1e-6,
        bias=False
    ):

        super().__init__()

        assert (
            hidden_size
            % num_attention_heads
            == 0
        )

        assert (
            num_attention_heads
            % num_key_value_heads
            == 0
        )

        self.hidden_size = hidden_size

        self.num_heads = (
            num_attention_heads
        )

        self.num_kv_heads = (
            num_key_value_heads
        )

        self.head_dim = (
            hidden_size
            // num_attention_heads
        )

        self.num_kv_groups = (
            num_attention_heads
            // num_key_value_heads
        )

        # ----------------------------------------------------
        # Q projection
        # ----------------------------------------------------

        self.q_proj = nn.Linear(
            hidden_size,
            num_attention_heads
            * self.head_dim,
            bias=bias
        )

        # ----------------------------------------------------
        # K projection
        # ----------------------------------------------------

        self.k_proj = nn.Linear(
            hidden_size,
            num_key_value_heads
            * self.head_dim,
            bias=bias
        )

        # ----------------------------------------------------
        # V projection
        # ----------------------------------------------------

        self.v_proj = nn.Linear(
            hidden_size,
            num_key_value_heads
            * self.head_dim,
            bias=bias
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        self.o_proj = nn.Linear(
            num_attention_heads
            * self.head_dim,
            hidden_size,
            bias=bias
        )

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings,
            rope_theta
        )

        self.q_norm = QKNorm(
            qk_norm_eps
        )

        self.k_norm = QKNorm(
            qk_norm_eps
        )

    def forward(self, x):

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # ====================================================
        # Q K V
        # ====================================================

        q = self.q_proj(x)

        k = self.k_proj(x)

        v = self.v_proj(x)

        # ====================================================
        # Reshape
        # ====================================================

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        k = k.view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        )

        v = v.view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        )

        # Move heads before sequence.

        q = q.transpose(1, 2)

        k = k.transpose(1, 2)

        v = v.transpose(1, 2)

        # ====================================================
        # RoPE
        # ====================================================

        q, k = self.rope(q, k)

        # ====================================================
        # Q/K normalization
        # ====================================================

        q = self.q_norm(q)

        k = self.k_norm(k)

        # ====================================================
        # Grouped Query Attention
        # ====================================================

        k = repeat_kv(
            k,
            self.num_kv_groups
        )

        v = repeat_kv(
            v,
            self.num_kv_groups
        )

        # ====================================================
        # Attention scores
        #
        # Q: [B,H,S,D]
        #
        # K: [B,H,S,D]
        #
        # K^T: [B,H,D,S]
        #
        # Q @ K^T:
        #
        # [B,H,S,S]
        # ====================================================

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        scores = (
            scores
            / math.sqrt(self.head_dim)
        )

        # ====================================================
        # Causal mask
        # ====================================================

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            causal_mask,
            float("-inf")
        )

        # ====================================================
        # Softmax
        # ====================================================

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # ====================================================
        # Weighted values
        # ====================================================

        output = torch.matmul(
            attention_weights,
            v
        )

        # [B,H,S,D]
        #
        # ->
        #
        # [B,S,H,D]

        output = output.transpose(
            1,
            2
        ).contiguous()

        # [B,S,H,D]
        #
        # ->
        #
        # [B,S,H*D]

        output = output.view(
            batch_size,
            seq_len,
            self.hidden_size
        )

        # ====================================================
        # Output projection
        # ====================================================

        output = self.o_proj(
            output
        )

        return output
