"""
train.py

Train our small educational Llama-style model.

This is NOT training the real Llama 4.
It is a tiny model designed for learning how
LLM training works end-to-end.
"""

import torch
import torch.nn as nn

from tokenizer import BPETokenizer
from transformer import (
    LlamaConfig,
    MiniLlama,
)


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
# Training corpus
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
# Configuration
# ============================================================

NUM_MERGES = 100

HIDDEN_SIZE = 128

NUM_LAYERS = 4

NUM_ATTENTION_HEADS = 8

NUM_KV_HEADS = 2

INTERMEDIATE_SIZE = 352

MAX_SEQUENCE_LENGTH = 64

LEARNING_RATE = 3e-4

NUM_EPOCHS = 500

PRINT_EVERY = 25

CHECKPOINT_PATH = "mini_llama.pt"


# ============================================================
# Create tokenizer
# ============================================================

print("=" * 60)
print("Creating tokenizer")
print("=" * 60)

tokenizer = BPETokenizer(
    CORPUS,
    num_merges=NUM_MERGES
)

print(
    "Vocabulary size:",
    tokenizer.vocab_size
)


# ============================================================
# Convert corpus to token IDs
# ============================================================

all_tokens = tokenizer.encode(
    CORPUS,
    add_bos=True,
    add_eos=True
)

print(
    "Total tokens:",
    len(all_tokens)
)


# ============================================================
# Create training examples
# ============================================================

def create_training_data(
    token_ids,
    sequence_length
):

    inputs = []
    targets = []

    # --------------------------------------------------------
    # We predict the next token.
    #
    # Example:
    #
    # input:
    #
    # [the, dog, is]
    #
    # target:
    #
    # [dog, is, happy]
    # --------------------------------------------------------

    for i in range(
        0,
        len(token_ids) - sequence_length
    ):

        input_sequence = token_ids[
            i:i + sequence_length
        ]

        target_sequence = token_ids[
            i + 1:i + sequence_length + 1
        ]

        inputs.append(
            input_sequence
        )

        targets.append(
            target_sequence
        )

    return (
        torch.tensor(
            inputs,
            dtype=torch.long
        ),
        torch.tensor(
            targets,
            dtype=torch.long
        )
    )


input_ids, target_ids = create_training_data(
    all_tokens,
    MAX_SEQUENCE_LENGTH
)


print(
    "Input tensor:",
    input_ids.shape
)

print(
    "Target tensor:",
    target_ids.shape
)


# ============================================================
# Model configuration
# ============================================================

config = LlamaConfig(

    vocab_size=tokenizer.vocab_size,

    hidden_size=HIDDEN_SIZE,

    num_hidden_layers=NUM_LAYERS,

    num_attention_heads=NUM_ATTENTION_HEADS,

    num_key_value_heads=NUM_KV_HEADS,

    intermediate_size=INTERMEDIATE_SIZE,

    max_position_embeddings=MAX_SEQUENCE_LENGTH
)


# ============================================================
# Create model
# ============================================================

model = MiniLlama(
    config
).to(DEVICE)


print()
print("=" * 60)
print("Model")
print("=" * 60)

print(
    "Device:",
    DEVICE
)

print(
    "Parameters:",
    f"{model.parameter_count():,}"
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Loss function
# ============================================================

loss_function = nn.CrossEntropyLoss()


# ============================================================
# Training
# ============================================================

print()
print("=" * 60)
print("Training")
print("=" * 60)


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    model.train()

    # --------------------------------------------------------
    # Move data to GPU/MPS/CPU
    # --------------------------------------------------------

    x = input_ids.to(DEVICE)

    y = target_ids.to(DEVICE)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits = model(x)

    # logits shape:
    #
    # [batch, sequence, vocab_size]
    #
    # Example:
    #
    # [100, 64, 200]
    #
    # CrossEntropyLoss expects:
    #
    # predictions:
    # [N, C]
    #
    # targets:
    # [N]
    #
    # So flatten batch + sequence.

    logits_flat = logits.reshape(
        -1,
        config.vocab_size
    )

    targets_flat = y.reshape(
        -1
    )

    # --------------------------------------------------------
    # Calculate loss
    # --------------------------------------------------------

    loss = loss_function(
        logits_flat,
        targets_flat
    )

    # --------------------------------------------------------
    # Clear previous gradients
    # --------------------------------------------------------

    optimizer.zero_grad()

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    # --------------------------------------------------------
    # Update weights
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % PRINT_EVERY == 0
    ):

        print(
            f"Epoch {epoch:4d} "
            f"/ {NUM_EPOCHS} "
            f"| Loss: {loss.item():.4f}"
        )


# ============================================================
# Save checkpoint
# ============================================================

print()
print("=" * 60)
print("Saving model")
print("=" * 60)


checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "config":
        config.__dict__,

    "vocab":
        tokenizer.token_to_id,

    "merges":
        tokenizer.merges,

}


torch.save(
    checkpoint,
    CHECKPOINT_PATH
)


print(
    "Saved:",
    CHECKPOINT_PATH
)