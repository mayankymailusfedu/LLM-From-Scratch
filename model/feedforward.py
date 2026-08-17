"""
feedforward.py

Educational Llama-style feed-forward network.

Uses:

    gate = SiLU(x W_gate)
    up   = x W_up

    hidden = gate * up

    output = hidden W_down
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        bias=False
    ):

        super().__init__()

        # ----------------------------------------------------
        # Gate projection
        # ----------------------------------------------------

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias
        )

        # ----------------------------------------------------
        # Up projection
        # ----------------------------------------------------

        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias
        )

        # ----------------------------------------------------
        # Down projection
        # ----------------------------------------------------

        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=bias
        )

    def forward(self, x):

        # ====================================================
        # Gate branch
        # ====================================================

        gate = self.gate_proj(x)

        gate = F.silu(gate)

        # ====================================================
        # Up branch
        # ====================================================

        up = self.up_proj(x)

        # ====================================================
        # Element-wise gating
        # ====================================================

        hidden = gate * up

        # ====================================================
        # Project back to hidden size
        # ====================================================

        output = self.down_proj(
            hidden
        )

        return output