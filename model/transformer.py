"""
transformer.py

Complete small Llama-style Transformer.

This is an educational model, NOT the actual
Meta Llama 4 implementation/checkpoint.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from attention import (
    RMSNorm,
    LlamaAttention,
)

from feedforward import (
    SwiGLU,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class LlamaConfig:

    vocab_size: int

    hidden_size: int = 128

    num_hidden_layers: int = 4

    num_attention_heads: int = 8

    num_key_value_heads: int = 2

    intermediate_size: int = 352

    max_position_embeddings: int = 128

    rope_theta: float = 10000.0

    rms_norm_eps: float = 1e-5

    qk_norm_eps: float = 1e-6

    bias: bool = False


# ============================================================
# Transformer block
# ============================================================

class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()

        # ----------------------------------------------------
        # Attention normalization
        # ----------------------------------------------------

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps
        )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=(
                config.num_attention_heads
            ),
            num_key_value_heads=(
                config.num_key_value_heads
            ),
            max_position_embeddings=(
                config.max_position_embeddings
            ),
            rope_theta=config.rope_theta,
            qk_norm_eps=config.qk_norm_eps,
            bias=config.bias
        )

        # ----------------------------------------------------
        # Feed-forward normalization
        # ----------------------------------------------------

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps
        )

        # ----------------------------------------------------
        # Feed-forward
        # ----------------------------------------------------

        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=(
                config.intermediate_size
            ),
            bias=config.bias
        )

    def forward(self, x):

        # ====================================================
        # Attention
        # ====================================================

        residual = x

        x = self.input_layernorm(x)

        x = self.self_attn(x)

        # Residual connection

        x = residual + x

        # ====================================================
        # Feed Forward
        # ====================================================

        residual = x

        x = self.post_attention_layernorm(
            x
        )

        x = self.mlp(x)

        # Residual connection

        x = residual + x

        return x


# ============================================================
# Complete model
# ============================================================

class MiniLlama(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()

        self.config = config

        # ====================================================
        # Token embedding
        # ====================================================

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )

        # ====================================================
        # Transformer layers
        # ====================================================

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config)
                for _ in range(
                    config.num_hidden_layers
                )
            ]
        )

        # ====================================================
        # Final normalization
        # ====================================================

        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps
        )

        # ====================================================
        # LM Head
        # ====================================================

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        # ====================================================
        # Weight tying
        #
        # The input embedding and output projection
        # share the same weight matrix.
        # ====================================================

        self.lm_head.weight = (
            self.embed_tokens.weight
        )

    def forward(
        self,
        input_ids
    ):

        # ====================================================
        # Token IDs
        #
        # [batch, sequence]
        # ====================================================

        x = self.embed_tokens(
            input_ids
        )

        # ====================================================
        # Transformer layers
        # ====================================================

        for layer in self.layers:

            x = layer(x)

        # ====================================================
        # Final RMSNorm
        # ====================================================

        x = self.norm(x)

        # ====================================================
        # LM head
        #
        # [B,S,H]
        #
        # ->
        #
        # [B,S,V]
        # ====================================================

        logits = self.lm_head(
            x
        )

        return logits

    def parameter_count(self):

        return sum(
            p.numel()
            for p in self.parameters()
        )


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":

    config = LlamaConfig(
        vocab_size=1000
    )

    model = MiniLlama(
        config
    )

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 10)
    )

    logits = model(
        input_ids
    )

    print(
        "Input:",
        input_ids.shape
    )

    print(
        "Output:",
        logits.shape
    )

    print(
        "Parameters:",
        model.parameter_count()
    )