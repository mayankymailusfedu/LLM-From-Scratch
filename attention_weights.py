import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

# Configuration
hidden_size = 128 # Dimiensionality of the model's hidden states - size of embedding vector for the token
num_attention_heads = 16 # Total number of query heads
num_key_value_heads = 4 # Numebr of key value heads
head_dim = hidden_size // num_attention_heads # Dimension of each attention head
max_position_embeddings = 256 # Maximum sequence length that model expects
rope_theta = 10000.0 # Base for RoPE (Rotary position embeddings) frequency calculation (encode token position) - theta is angle of rotation 
rms_norm_eps = 1e-5 # Epsilon for RMSNorm
attention_bias = False # Whether to use bias in Q projections 
attention_dropout = 0.0 # Dropout probability for attention weights
use_qk_norm = True # Whether to apply L2 norm to Q and K before attention

# Sample Input
batch_size = 2
sequence_length = 10 # Context Window
hidden_states = torch.randn(batch_size, sequence_length, hidden_size)
# Creates position IDs of each token in the sequence, repeated for each batch
# torch.arange(0, sequence_length) generates a 1 dimension tensor with values from 0 to sequence_length-1
# unsqueeze(0) adds an extra dimension at the 0th position, making it (1, sequence_length)
# repeat(batch_size, 1) creates a tensor of shape (batch_size, sequence_length) 
position_ids = torch.arange(0, sequence_length).unsqueeze(0).repeat(batch_size, 1) # Shape: (batch_size, sequence_length)
attention_mask = torch.triu(torch.ones(sequence_length, sequence_length) * -torch.inf, diagonal=1) # Simple upper triangular mask
attention_mask = attention_mask.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, sequence_length, sequence_length)
attention_mask = attention_mask.expand(batch_size, 1, -1, -1) # Shape: (batch_size, 1, sequence_length, sequence_length)

print("Configuration")
print(f"    hidden_size: {hidden_size}")
print(f"    num_attention_heads: {num_attention_heads}")
print(f"    num_key_value_heads: {num_key_value_heads}")

print("\nSampling Input Shapes")
print(f"    hidden_states: {hidden_states.shape}")
print(f"    position_ids: {position_ids.shape}")
print(f"    attention_mask: {attention_mask.shape}")

# Using Group Query Attention (GQA) - there are fewer K and V than Q heads
# num_key_value_groups - how many Q heads share the same K and V head
# Define projection layer
q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias) # W^Q
k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias) # W^K
v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias) # W^V
o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias) # Output projection which utilize marox transpose ie., switch row and column

# Calculate projections
query_states = q_proj(hidden_states)
key_states = k_proj(hidden_states)
value_states = v_proj(hidden_states)

# Reshape q, K, V for multi-head attention so that each head different things and extract different info
query_states = query_states.view(batch_size, sequence_length, num_attention_heads, head_dim).transpose(1, 2)
key_states = key_states.view(batch_size, sequence_length, num_key_value_heads, head_dim).transpose(1, 2)
value_states = value_states.view(batch_size, sequence_length, num_key_value_heads, head_dim).transpose(1, 2)

print("\nProject Shapes:")
print(f"    query_states: {query_states}")
print(f"    key_states: {key_states}")
print(f"    value_states: {value_states}")

num_key_value_groups = num_attention_heads // num_key_value_heads
print(f"\nNum Key/Value Groups (Q heads per K/V head): {num_key_value_groups}")

# Simplified RoPE Calculation and Application
def simple_rope_calculation(dim, max_seq_len, base=10000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim)) # Inverse frequencies
    t = torch.arange(max_seq_len, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    # Calculating cosine and sine embeddings allows us to encode rotational transformation in a 2D plane
    freqs_cos = emb.cos() # Real part
    freq_sin = emb.sin() # Imaginary part
    # Complex representation allows for efficient rotation of vectors, which is key in Rotary Positional Embeddings (RoPE)
    freqs_cis = torch.complex(freqs_cos, freq_sin) # Shapes: (max_seq_len, dim)
    return freqs_cis

def apply_rotary_emb_torch(
        xq: torch.Tensor, # Query Tensor, Shape (batch, num_heads, seq_len, head_dim)
        xk: torch.Tensor, # Key Tensor, Shape (batch, num_heads, seq_len, head_dim)
        freqs_cis: torch.Tensor, # Precomputed complex rotations, Shape (max_seq_len, head_dim)
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Applies RoPE rotations to Q and K using torch complex numbers
    freqs_cis = freqs_cis.to(xq.device) # CPU/GPU
    freqs_cis = freqs_cis[position_ids] # Select correct rotation vectors for current sequence positions, Shape (batch, seq_len, head_dim), complex
    freqs_cis = freqs_cis[:, None, :, :] # Add a dimension for broadcasting across attention heads, Shape (batch, 1, , seq_len, head_sim), complex

    # Prepare Q and K for Complex Multiplication
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # Reshape Q to view adjacent pairs as complex numbers
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) # Reshape K to view adjacent pairs as complex numbers

    # Prepare freqs_cis for Complex Multiplication
    freqs_cis_broadcast = freqs_cis[..., :xq_.shape[-1]] # Slice the last dim

    # Apply the Rotation - Perform RoPE rotation using element-wise complex multiplication
    rotated_xq = xq_ * freqs_cis_broadcast
    rotated_xk = xk_ * freqs_cis_broadcast

    # Convert back to Real Representation - Convert the rotated complex vectors back to real vectors
    xq_out = torch.view_as_real(rotated_xq).flatten(3)
    xk_out = torch.view_as_real(rotated_xk).flatten(3)

    # Cast back to original input datatype (float)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# Calculate RoPE frequencies (precomputed usually) - RoPE is applied to head_dim not to hidden_states
freqs_cis = simple_rope_calculation(head_dim, max_position_embeddings, base=rope_theta, device=hidden_states.device)
print(f"\nCalculated freqs_cis shape: {freqs_cis.shape}") # (max_pos_emb, head_dim)

# Apply RoPE - RoPE is applied before repeating K/V GOA
query_states_rope, key_states_rope = apply_rotary_emb_torch(query_states, key_states, freqs_cis)

print("\nShapes after RoPE")
print(f"    query_states_rope: {query_states_rope.shape}")
print(f"    key_states_rope: {key_states_rope.shape}")

# Apply QK Normalization
class SimpleL2Norm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # Normalize along the last dimension (head_dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

if use_qk_norm:
    qk_norm = SimpleL2Norm()
    query_states_final = qk_norm(query_states_rope)
    key_states_final = qk_norm(key_states_rope)
    print("\nApplied QK Normalization")
else:
    query_states_final = query_states_rope
    key_states_final = key_states_rope
    print("\nSkipped QK Normalization")

print("\nShapes before attention score calculation")
print(f"    query_states_final: {query_states_final.shape}")
print(f"    key_states_final: {key_states_final.shape}")

# Group Query Attentio (GQA) - Key/Value Repeating
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # Repeats Key/Value heads for GQA - Input: (batch, num_key_value_heads, seq_len, head_dim), Output: (batch, num_attention_heads, seq_len, head_dim)
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Repeat K and V heads
key_states_repeated = repeat_kv(key_states_final, num_key_value_heads)
value_states_repeated = repeat_kv(value_states, num_key_value_heads)

print("\nShapes after repeating K/V for GQA")
print(f"    key_states_repeated: {key_states_repeated.shape}")
print(f"    value_states_repeated: {value_states_repeated.shape}")

# Calculate Attenstion Scores (Q @ K^T)
attn_weights = torch.matmul(query_states_final, key_states_repeated.transpose(2, 3))

# Scale 
scaling_factor = 1.0 / math.sqrt(head_dim)
attn_weights = attn_weights * scaling_factor

# Apply Mask - so that current token keys cannot lookup future token keys
if attention_mask is not None:
    print(f"\nApplying attention mask with shape: {attention_mask.shape}")
    causal_mask = attention_mask[:, :, :, :key_states_repeated.shape[-2]]
    attn_weights = attn_weights * causal_mask
else:
    print("\nNo attention mask applied")

# Softmax
attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)

# Dropout (skipped for inference example)
# attn_weights = nn.functional.dropout(attn_weights, p=attention_dropout, training=self.training)

# Calculate Output (Weights Sum of Values)
attn_output = torch.matmul(attn_weights, value_states_repeated)

print("\nAttention Calculation Shapes")
print(f"    attn_weights (raw scores): {attn_weights.shape}")
print(f"    attn_weights (after softmax): {attn_weights.shape}")
print(f"    attn_output: {attn_output.shape}")

# Reshape attention output
attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.view(batch_size, sequence_length, hidden_size)

# Apply output projections
final_attn_output = o_proj(attn_output)

print("\nFinal Output Shapes")
print(f"    attn_output: {attn_output.shape}")
print(f"    final_attn_output: {final_attn_output.shape}") # Should be (batch, seq_len, hidden_size)

# Simplfied Llama4TextAttention Forward Pass
class SimplifiedLlama4Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['num_attention_heads']
        self.num_key_value_heads = config['num_key_value_heads']
        self.head_dim = config['head_dim']
        self.num_key_value_groups = config['num_key_value_groups']
        self.max_position_embeddings = config['max_position_embeddings']
        self.rope_theta = config['rope_theta']
        self.attention_bias = config['attention_bias']
        self.use_qk_norm = config['use_qk_norm']

        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=self.attention_bias)

        self.freqs_cis = simple_rope_calculation(self.head_dim, self.max_position_embeddings, base=self.rope_theta)

        if self.use_qk_norm:
            self.qk_norm = SimpleL2Norm()

    def forward(self, hidden_states, attention_mask, position_ids):
        batch_size, sequence_length, _ = hidden_states.shape

        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(batch_size, sequence_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        current_freqs_cis = self.freqs_cis.to(hidden_states.device) # Get precomputed freqs
        query_states_rope, key_states_rope = apply_rotary_emb_torch(query_states, key_states, current_freqs_cis)

        # QK Norm
        if self.use_qk_norm:
            query_states_final = self.qk_norm(query_states_rope)
            key_states_final = self.qk_norm(key_states_rope)
        else:
            query_states_final = query_states_rope
            key_states_final = key_states_rope

        # Repeat K/V for GQA
        key_states_repeated = repeat_kv(key_states_final, self.num_key_value_groups)
        value_states_repeated = repeat_kv(value_states, self.num_key_value_groups)

        # Attention Calculation
        attn_weights = torch.matmul(query_states_final, key_states_repeated.transpose(2, 3))
        scaling_factor = 1.0 / math.sqrt(self.head_dim)
        attn_weights = attn_weights * scaling_factor

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states_repeated.shape[-2]] # prevents a token from looking at future tokens
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)

        # Dropout would be here in training

        attn_output = torch.matmul(attn_weights, value_states_repeated)
        
        # Reshape and output Projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, sequence_length, self.hidden_size)
        final_attn_output = self.o_proj(attn_output)

        # Return weights for inspection
        return final_attn_output, attn_weights

# Instantiate and run the simplified module
config_dict = {
    'hidden_size': hidden_size,
    'num_attention_heads': num_attention_heads,
    'num_key_value_heads': num_key_value_heads,
    'max_position_embeddings': max_position_embeddings,
    'rope_theta': rope_theta,
    'attention_bias': attention_bias,
    'use_qk_norm': use_qk_norm,
    'head_dim': head_dim,
    'num_key_value_groups': num_key_value_groups,
}

simplified_attn_module = SimplifiedLlama4Attention(config_dict)

# Run forward pass
final_output_simplified, final_weights_simplified = simplified_attn_module(hidden_states, attention_mask, position_ids)

print(f"\nOutput shape from simplified modele: {final_output_simplified.shape}")
print(f"Attention weights shape from simplified module: {final_weights_simplified.shape}")