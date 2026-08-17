"""
debug.py

Inspect the internal workings of our MiniLlama model.

This is an educational/debugging script.
"""

import torch
import torch.nn.functional as F

from tokenizer import BPETokenizer
from transformer import LlamaConfig, MiniLlama


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()


# ============================================================
# Same corpus used during training
# ============================================================

CORPUS = """
hello world
hello dog
hello cat

the dog is happy
the dog is friendly
the dog likes food
the dog likes running

the cat is happy
the cat is friendly
the cat likes food
the cat likes sleeping

dogs are animals
cats are animals

a dog is an animal
a cat is an animal

dogs have four legs
cats have four legs

a dog can run
a dog can eat
a dog can sleep

a cat can run
a cat can eat
a cat can sleep

what is a dog
a dog is an animal

what is a cat
a cat is an animal
"""


# ============================================================
# Model configuration
# ============================================================

def create_model(tokenizer):

    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,

        hidden_size=128,

        num_hidden_layers=4,

        num_attention_heads=8,

        num_key_value_heads=2,

        intermediate_size=352,

        max_position_embeddings=64
    )

    model = MiniLlama(
        config
    ).to(DEVICE)

    return model


# ============================================================
# Load trained model
# ============================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    tokenizer = BPETokenizer(
        CORPUS,
        num_merges=100
    )

    model = create_model(
        tokenizer
    )

    checkpoint = torch.load(
        "mini_llama.pt",
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Device:",
        DEVICE
    )

    print(
        "Vocabulary:",
        tokenizer.vocab_size
    )

    print(
        "Parameters:",
        f"{model.parameter_count():,}"
    )

    return model, tokenizer


# ============================================================
# Print tensor information
# ============================================================

def tensor_info(
    name,
    tensor,
    show_values=False
):

    print()
    print(f"{name}")
    print("-" * 70)

    print(
        "shape:",
        tuple(tensor.shape)
    )

    print(
        "dtype:",
        tensor.dtype
    )

    print(
        "device:",
        tensor.device
    )

    print(
        "min:",
        tensor.min().item()
    )

    print(
        "max:",
        tensor.max().item()
    )

    print(
        "mean:",
        tensor.float().mean().item()
    )

    print(
        "std:",
        tensor.float().std().item()
    )

    if show_values:

        print(
            "values:"
        )

        print(tensor)


# ============================================================
# Show tokenizer
# ============================================================

def debug_tokenizer(
    tokenizer,
    prompt
):

    print()
    print("=" * 70)
    print("1. TOKENIZATION")
    print("=" * 70)

    print(
        "Prompt:",
        repr(prompt)
    )

    token_ids = tokenizer.encode(
        prompt,
        add_bos=True,
        add_eos=False
    )

    print(
        "Token IDs:",
        token_ids
    )

    print()
    print("Token mapping:")

    for token_id in token_ids:

        token = tokenizer.id_to_token[
            token_id
        ]

        print(
            f"{token_id:4d} -> {repr(token)}"
        )

    return token_ids


# ============================================================
# Debug embedding
# ============================================================

def debug_embedding(
    model,
    input_ids
):

    print()
    print("=" * 70)
    print("2. TOKEN EMBEDDING")
    print("=" * 70)

    print(
        "Input IDs shape:",
        tuple(input_ids.shape)
    )

    embeddings = model.embed_tokens(
        input_ids
    )

    tensor_info(
        "Embeddings",
        embeddings,
        show_values=False
    )

    return embeddings


# ============================================================
# Debug RMSNorm
# ============================================================

def debug_rmsnorm(
    layer,
    x,
    name
):

    output = layer(x)

    tensor_info(
        name,
        output
    )

    return output


# ============================================================
# Debug Attention
# ============================================================

def debug_attention(
    attention,
    x
):

    print()
    print("=" * 70)
    print("4. ATTENTION")
    print("=" * 70)

    batch_size = x.shape[0]

    seq_len = x.shape[1]

    hidden_size = x.shape[2]

    num_heads = attention.num_heads

    num_kv_heads = attention.num_kv_heads

    head_dim = attention.head_dim

    print()
    print("Configuration")
    print("-" * 70)

    print(
        "hidden size:",
        hidden_size
    )

    print(
        "attention heads:",
        num_heads
    )

    print(
        "KV heads:",
        num_kv_heads
    )

    print(
        "head dimension:",
        head_dim
    )

    print(
        "KV groups:",
        attention.num_kv_groups
    )

    # ========================================================
    # Q
    # ========================================================

    q = attention.q_proj(x)

    tensor_info(
        "Q projection",
        q
    )

    q = q.view(
        batch_size,
        seq_len,
        num_heads,
        head_dim
    )

    q = q.transpose(
        1,
        2
    )

    tensor_info(
        "Q reshaped",
        q
    )

    # ========================================================
    # K
    # ========================================================

    k = attention.k_proj(x)

    tensor_info(
        "K projection",
        k
    )

    k = k.view(
        batch_size,
        seq_len,
        num_kv_heads,
        head_dim
    )

    k = k.transpose(
        1,
        2
    )

    tensor_info(
        "K reshaped",
        k
    )

    # ========================================================
    # V
    # ========================================================

    v = attention.v_proj(x)

    tensor_info(
        "V projection",
        v
    )

    v = v.view(
        batch_size,
        seq_len,
        num_kv_heads,
        head_dim
    )

    v = v.transpose(
        1,
        2
    )

    tensor_info(
        "V reshaped",
        v
    )

    # ========================================================
    # RoPE
    # ========================================================

    q_rope, k_rope = attention.rope(
        q,
        k
    )

    tensor_info(
        "Q after RoPE",
        q_rope
    )

    tensor_info(
        "K after RoPE",
        k_rope
    )

    # ========================================================
    # Q/K normalization
    # ========================================================

    q_norm = attention.q_norm(
        q_rope
    )

    k_norm = attention.k_norm(
        k_rope
    )

    tensor_info(
        "Q after QK Norm",
        q_norm
    )

    tensor_info(
        "K after QK Norm",
        k_norm
    )

    # ========================================================
    # GQA
    # ========================================================

    from attention import repeat_kv

    k_gqa = repeat_kv(
        k_norm,
        attention.num_kv_groups
    )

    v_gqa = repeat_kv(
        v,
        attention.num_kv_groups
    )

    tensor_info(
        "K after GQA",
        k_gqa
    )

    tensor_info(
        "V after GQA",
        v_gqa
    )

    # ========================================================
    # Attention scores
    # ========================================================

    scores = torch.matmul(
        q_norm,
        k_gqa.transpose(
            -2,
            -1
        )
    )

    scores = (
        scores
        / (head_dim ** 0.5)
    )

    tensor_info(
        "Attention scores BEFORE causal mask",
        scores
    )

    # ========================================================
    # Causal mask
    # ========================================================

    causal_mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            device=x.device,
            dtype=torch.bool
        ),
        diagonal=1
    )

    print()
    print("Causal mask")
    print("-" * 70)

    print(
        causal_mask[0]
        if causal_mask.ndim > 2
        else causal_mask
    )

    scores_masked = scores.masked_fill(
        causal_mask,
        float("-inf")
    )

    tensor_info(
        "Attention scores AFTER causal mask",
        scores_masked
    )

    # ========================================================
    # Softmax
    # ========================================================

    attention_weights = F.softmax(
        scores_masked,
        dim=-1
    )

    tensor_info(
        "Attention weights",
        attention_weights
    )

    # ========================================================
    # Weighted values
    # ========================================================

    attention_output = torch.matmul(
        attention_weights,
        v_gqa
    )

    tensor_info(
        "Attention weighted output",
        attention_output
    )

    # ========================================================
    # Merge heads
    # ========================================================

    attention_output = (
        attention_output
        .transpose(1, 2)
        .contiguous()
    )

    attention_output = attention_output.view(
        batch_size,
        seq_len,
        hidden_size
    )

    tensor_info(
        "Attention merged heads",
        attention_output
    )

    # ========================================================
    # Output projection
    # ========================================================

    output = attention.o_proj(
        attention_output
    )

    tensor_info(
        "Attention final output",
        output
    )

    return output


# ============================================================
# Debug Feed Forward
# ============================================================

def debug_feedforward(
    mlp,
    x
):

    print()
    print("=" * 70)
    print("5. FEED-FORWARD / SWIGLU")
    print("=" * 70)

    # ========================================================
    # Gate
    # ========================================================

    gate_linear = mlp.gate_proj(x)

    tensor_info(
        "Gate projection",
        gate_linear
    )

    # ========================================================
    # SiLU
    # ========================================================

    gate = F.silu(
        gate_linear
    )

    tensor_info(
        "Gate after SiLU",
        gate
    )

    # ========================================================
    # Up projection
    # ========================================================

    up = mlp.up_proj(x)

    tensor_info(
        "Up projection",
        up
    )

    # ========================================================
    # Element-wise multiplication
    # ========================================================

    hidden = gate * up

    tensor_info(
        "Gate * Up",
        hidden
    )

    # ========================================================
    # Down projection
    # ========================================================

    output = mlp.down_proj(
        hidden
    )

    tensor_info(
        "Down projection",
        output
    )

    return output


# ============================================================
# Debug one transformer layer
# ============================================================

def debug_layer(
    model,
    layer_number,
    x
):

    layer = model.layers[
        layer_number
    ]

    print()
    print("=" * 70)
    print(
        f"TRANSFORMER LAYER {layer_number}"
    )
    print("=" * 70)

    tensor_info(
        "Layer input",
        x
    )

    # ========================================================
    # Attention block
    # ========================================================

    residual = x

    x_norm = layer.input_layernorm(
        x
    )

    tensor_info(
        "After input RMSNorm",
        x_norm
    )

    attention_output = debug_attention(
        layer.self_attn,
        x_norm
    )

    x = residual + attention_output

    tensor_info(
        "After attention residual",
        x
    )

    # ========================================================
    # Feed-forward block
    # ========================================================

    residual = x

    x_norm = layer.post_attention_layernorm(
        x
    )

    tensor_info(
        "After FFN RMSNorm",
        x_norm
    )

    ffn_output = debug_feedforward(
        layer.mlp,
        x_norm
    )

    x = residual + ffn_output

    tensor_info(
        "After FFN residual",
        x
    )

    return x


# ============================================================
# Debug final LM head
# ============================================================

def debug_lm_head(
    model,
    tokenizer,
    x
):

    print()
    print("=" * 70)
    print("6. FINAL RMSNORM + LM HEAD")
    print("=" * 70)

    # ========================================================
    # Final RMSNorm
    # ========================================================

    x = model.norm(x)

    tensor_info(
        "Final RMSNorm",
        x
    )

    # ========================================================
    # LM Head
    # ========================================================

    logits = model.lm_head(
        x
    )

    tensor_info(
        "Logits",
        logits
    )

    # ========================================================
    # Final position
    # ========================================================

    next_token_logits = logits[
        :,
        -1,
        :
    ]

    tensor_info(
        "Final-position logits",
        next_token_logits
    )

    # ========================================================
    # Probabilities
    # ========================================================

    probabilities = F.softmax(
        next_token_logits,
        dim=-1
    )

    # ========================================================
    # Top predictions
    # ========================================================

    top_k = min(
        10,
        probabilities.shape[-1]
    )

    values, indices = torch.topk(
        probabilities,
        top_k
    )

    print()
    print("=" * 70)
    print("7. NEXT TOKEN PREDICTIONS")
    print("=" * 70)

    for rank in range(top_k):

        token_id = indices[
            0,
            rank
        ].item()

        probability = values[
            0,
            rank
        ].item()

        token = tokenizer.id_to_token[
            token_id
        ]

        print(
            f"{rank + 1:2d}. "
            f"ID={token_id:3d} "
            f"Probability={probability * 100:6.2f}% "
            f"Token={repr(token)}"
        )

    return logits


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("#" * 70)
    print("# MINI LLAMA INTERNAL DEBUG")
    print("#" * 70)

    # ========================================================
    # Load
    # ========================================================

    model, tokenizer = load_model()

    # ========================================================
    # Prompt
    # ========================================================

    prompt = "the dog"

    # ========================================================
    # Tokenizer
    # ========================================================

    token_ids = debug_tokenizer(
        tokenizer,
        prompt
    )

    # ========================================================
    # Tensor
    # ========================================================

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )

    tensor_info(
        "Input IDs",
        input_ids,
        show_values=True
    )

    # ========================================================
    # Embedding
    # ========================================================

    x = debug_embedding(
        model,
        input_ids
    )

    # ========================================================
    # Debug first transformer layer
    # ========================================================

    x = debug_layer(
        model,
        0,
        x
    )

    # ========================================================
    # Run remaining layers normally
    # ========================================================

    for layer_number in range(
        1,
        len(model.layers)
    ):

        x = model.layers[
            layer_number
        ](x)

    # ========================================================
    # Final LM Head
    # ========================================================

    debug_lm_head(
        model,
        tokenizer,
        x
    )

    print()
    print("#" * 70)
    print("# DEBUG COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
