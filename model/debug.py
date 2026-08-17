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

# Output
# ######################################################################
# # MINI LLAMA INTERNAL DEBUG
# ######################################################################

# ======================================================================
# LOADING MODEL
# ======================================================================
# Device: mps
# Vocabulary: 31
# Parameters: 709,632

# ======================================================================
# 1. TOKENIZATION
# ======================================================================
# Prompt: 'the dog'
# Token IDs: [0, 28, 12]

# Token mapping:
#    0 -> '<BOS>'
#   28 -> 'the</w>'
#   12 -> 'dog</w>'

# Input IDs
# ----------------------------------------------------------------------
# shape: (1, 3)
# dtype: torch.int64
# device: mps:0
# min: 0
# max: 28
# mean: 13.333333015441895
# std: 14.047538757324219
# values:
# tensor([[ 0, 28, 12]], device='mps:0')

# ======================================================================
# 2. TOKEN EMBEDDING
# ======================================================================
# Input IDs shape: (1, 3)

# Embeddings
# ----------------------------------------------------------------------
# shape: (1, 3, 128)
# dtype: torch.float32
# device: mps:0
# min: -2.6791934967041016
# max: 2.718240737915039
# mean: -0.04746243357658386
# std: 0.9651340246200562

# ======================================================================
# TRANSFORMER LAYER 0
# ======================================================================

# Layer input
# ----------------------------------------------------------------------
# shape: (1, 3, 128)
# dtype: torch.float32
# device: mps:0
# min: -2.6791934967041016
# max: 2.718240737915039
# mean: -0.04746243357658386
# std: 0.9651340246200562

# After input RMSNorm
# ----------------------------------------------------------------------
# shape: (1, 3, 128)
# dtype: torch.float32
# device: mps:0
# min: -2.6540658473968506
# max: 2.791532278060913
# mean: -0.04779749736189842
# std: 1.007008671760559

# ======================================================================
# 4. ATTENTION
# ======================================================================

# Configuration
# ----------------------------------------------------------------------
# hidden size: 128
# attention heads: 8
# KV heads: 2
# head dimension: 16
# KV groups: 4

# Q projection
# ----------------------------------------------------------------------
# shape: (1, 3, 128)
# dtype: torch.float32
# device: mps:0
# min: -1.6619439125061035
# max: 1.4912405014038086
# mean: -0.058726996183395386
# std: 0.5966809988021851

# Q reshaped
# ----------------------------------------------------------------------
# shape: (1, 8, 3, 16)
# dtype: torch.float32
# device: mps:0
# min: -1.6619439125061035
# max: 1.4912405014038086
# mean: -0.058726996183395386
# std: 0.5966809988021851

# K projection
# ----------------------------------------------------------------------
# shape: (1, 3, 32)
# dtype: torch.float32
# device: mps:0
# min: -1.4549529552459717
# max: 1.7871932983398438
# mean: -0.04186191037297249
# std: 0.5832851529121399

# K reshaped
# ----------------------------------------------------------------------
# shape: (1, 2, 3, 16)
# dtype: torch.float32
# device: mps:0
# min: -1.4549529552459717
# max: 1.7871932983398438
# mean: -0.04186190664768219
# std: 0.5832851529121399

# V projection
# ----------------------------------------------------------------------
# shape: (1, 3, 32)
# dtype: torch.float32
# device: mps:0
# min: -1.6115750074386597
# max: 2.580749750137329
# mean: -0.05088081583380699
# std: 0.7809058427810669

# V reshaped
# ----------------------------------------------------------------------
# shape: (1, 2, 3, 16)
# dtype: torch.float32
# device: mps:0
# min: -1.6115750074386597
# max: 2.580749750137329
# mean: -0.05088081583380699
# std: 0.7809058427810669

# Q after RoPE
# ----------------------------------------------------------------------
# shape: (1, 8, 3, 16)
# dtype: torch.float32
# device: mps:0
# min: -1.6619439125061035
# max: 1.7740354537963867
# mean: -0.03906180337071419
# std: 0.5982944369316101

# K after RoPE
# ----------------------------------------------------------------------
# shape: (1, 2, 3, 16)
# dtype: torch.float32
# device: mps:0
# min: -1.4559763669967651
# max: 1.7871932983398438
# mean: -0.045866817235946655
# std: 0.5829806923866272

# Q after QK Norm
# ----------------------------------------------------------------------
# shape: (1, 8, 3, 16)
# dtype: torch.float32
# device: mps:0
# min: -2.974116086959839
# max: 2.423210620880127
# mean: -0.07334887981414795
# std: 0.9986059069633484

# K after QK Norm
# ----------------------------------------------------------------------
# shape: (1, 2, 3, 16)
# dtype: torch.float32
# device: mps:0
# min: -2.511239528656006
# max: 2.6167187690734863
# mean: -0.10866229981184006
# std: 0.9992954134941101

# K after GQA
# ----------------------------------------------------------------------
# shape: (1, 8, 3, 16)
# dtype: torch.float32
# device: mps:0
# min: -2.511239528656006
# max: 2.6167187690734863
# mean: -0.10866229981184006
# std: 0.9953740239143372

# V after GQA
# ----------------------------------------------------------------------
# shape: (1, 8, 3, 16)
# dtype: torch.float32
# device: mps:0
# min: -1.6115750074386597
# max: 2.580749750137329
# mean: -0.05088081583380699
# std: 0.7778413891792297

# Attention scores BEFORE causal mask
# ----------------------------------------------------------------------
# shape: (1, 8, 3, 3)
# dtype: torch.float32
# device: mps:0
# min: -2.0673487186431885
# max: 2.969982147216797
# mean: -0.039048563688993454
# std: 1.119742751121521

# Causal mask
# ----------------------------------------------------------------------
# tensor([[False,  True,  True],
#         [False, False,  True],
#         [False, False, False]], device='mps:0')

# Attention scores AFTER causal mask
# ----------------------------------------------------------------------
# shape: (1, 8, 3, 3)
# dtype: torch.float32
# device: mps:0
# min: -inf
# max: 2.969982147216797
# mean: -inf
# std: nan

# Attention weights
# ----------------------------------------------------------------------
# shape: (1, 8, 3, 3)
# dtype: torch.float32
# device: mps:0
# min: 0.0
# max: 1.0
# mean: 0.3333333432674408
# std: 0.3723907768726349

# Attention weighted output
# ----------------------------------------------------------------------
# shape: (1, 8, 3, 16)
# dtype: torch.float32
# device: mps:0
# min: -1.2704976797103882
# max: 2.021559000015259
# mean: -0.022868165746331215
# std: 0.5625491142272949

# Attention merged heads
# ----------------------------------------------------------------------
# shape: (1, 3, 128)
# dtype: torch.float32
# device: mps:0
# min: -1.2704976797103882
# max: 2.021559000015259
# mean: -0.022868163883686066
# std: 0.5625491142272949

# Attention final output
# ----------------------------------------------------------------------
# shape: (1, 3, 128)
# dtype: torch.float32
# device: mps:0
# min: -0.9093199968338013
# max: 0.8970301151275635
# mean: -0.0068966299295425415
# std: 0.354360431432724

# After attention residual
# ----------------------------------------------------------------------
# shape: (1, 3, 128)
# dtype: torch.float32
# device: mps:0
# min: -2.803251028060913
# max: 3.3098764419555664
# mean: -0.0543590672314167
# std: 1.030838131904602

# After FFN RMSNorm
# ----------------------------------------------------------------------
# shape: (1, 3, 128)
# dtype: torch.float32
# device: mps:0
# min: -2.5781619548797607
# max: 3.041574239730835
# mean: -0.05249258875846863
# std: 1.0132862329483032

# ======================================================================
# 5. FEED-FORWARD / SWIGLU
# ======================================================================

# Gate projection
# ----------------------------------------------------------------------
# shape: (1, 3, 352)
# dtype: torch.float32
# device: mps:0
# min: -2.4445056915283203
# max: 3.035996198654175
# mean: 0.0875726118683815
# std: 0.7088752388954163

# Gate after SiLU
# ----------------------------------------------------------------------
# shape: (1, 3, 352)
# dtype: torch.float32
# device: mps:0
# min: -0.27846187353134155
# max: 2.896868944168091
# mean: 0.15706388652324677
# std: 0.4327039122581482

# Up projection
# ----------------------------------------------------------------------
# shape: (1, 3, 352)
# dtype: torch.float32
# device: mps:0
# min: -2.5374698638916016
# max: 3.202803611755371
# mean: -0.05126645043492317
# std: 0.7300488948822021

# Gate * Up
# ----------------------------------------------------------------------
# shape: (1, 3, 352)
# dtype: torch.float32
# device: mps:0
# min: -4.962349891662598
# max: 9.278101921081543
# mean: -0.0062238313257694244
# std: 0.5873585343360901

# Down projection
# ----------------------------------------------------------------------
# shape: (1, 3, 128)
# dtype: torch.float32
# device: mps:0
# min: -2.4476706981658936
# max: 2.23755145072937
# mean: -0.004880508873611689
# std: 0.6543762683868408

# After FFN residual
# ----------------------------------------------------------------------
# shape: (1, 3, 128)
# dtype: torch.float32
# device: mps:0
# min: -2.7300045490264893
# max: 3.3500819206237793
# mean: -0.05923957750201225
# std: 1.0566699504852295

# ======================================================================
# 6. FINAL RMSNORM + LM HEAD
# ======================================================================

# Final RMSNorm
# ----------------------------------------------------------------------
# shape: (1, 3, 128)
# dtype: torch.float32
# device: mps:0
# min: -2.7440595626831055
# max: 2.475200891494751
# mean: -0.1272677332162857
# std: 0.9835406541824341

# Logits
# ----------------------------------------------------------------------
# shape: (1, 3, 31)
# dtype: torch.float32
# device: mps:0
# min: -27.827495574951172
# max: 38.409507751464844
# mean: 7.4531354904174805
# std: 11.15538215637207

# Final-position logits
# ----------------------------------------------------------------------
# shape: (1, 31)
# dtype: torch.float32
# device: mps:0
# min: -23.003103256225586
# max: 38.409507751464844
# mean: 7.747537136077881
# std: 12.106205940246582

# ======================================================================
# 7. NEXT TOKEN PREDICTIONS
# ======================================================================
#  1. ID= 21 Probability= 99.92% Token='is</w>'
#  2. ID= 23 Probability=  0.07% Token='likes</w>'
#  3. ID= 20 Probability=  0.01% Token='hello</w>'
#  4. ID= 15 Probability=  0.00% Token='food</w>'
#  5. ID=  4 Probability=  0.00% Token='a</w>'
#  6. ID= 24 Probability=  0.00% Token='run</w>'
#  7. ID= 11 Probability=  0.00% Token='cats</w>'
#  8. ID=  1 Probability=  0.00% Token='<EOS>'
#  9. ID=  9 Probability=  0.00% Token='can</w>'
# 10. ID=  8 Probability=  0.00% Token='are</w>'

# ######################################################################
# # DEBUG COMPLETE
# ######################################################################