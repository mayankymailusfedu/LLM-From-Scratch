"""
generate.py

Load the trained MiniLlama model and generate text.
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
# IMPORTANT:
# This must be the SAME corpus used by train.py
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
# Load model
# ============================================================

def load_model():

    print("Loading checkpoint...")

    checkpoint = torch.load(
        "mini_llama.pt",
        map_location=DEVICE,
        weights_only=False
    )

    print("Checkpoint loaded.")

    # --------------------------------------------------------
    # Recreate tokenizer
    # --------------------------------------------------------

    tokenizer = BPETokenizer(
        CORPUS,
        num_merges=100
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size
    )

    # --------------------------------------------------------
    # Recreate model
    # --------------------------------------------------------

    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=352,
        max_position_embeddings=64
    )

    model = MiniLlama(config)

    # --------------------------------------------------------
    # Load learned weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    print("Model loaded.")
    print(
        "Parameters:",
        f"{model.parameter_count():,}"
    )

    return model, tokenizer


# ============================================================
# Generate
# ============================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=20,
    temperature=0.7,
    top_k=10
):

    # --------------------------------------------------------
    # Text -> token IDs
    # --------------------------------------------------------

    token_ids = tokenizer.encode(
        prompt,
        add_bos=True,
        add_eos=False
    )

    print()
    print("Prompt:")
    print(prompt)

    print()
    print("Input token IDs:")
    print(token_ids)

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Generate tokens
    # --------------------------------------------------------

    for step in range(max_new_tokens):

        # Keep sequence within context window.

        input_ids_cond = input_ids[
            :,
            -model.config.max_position_embeddings:
        ]

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(
            input_ids_cond
        )

        # ----------------------------------------------------
        # Only look at the final position.
        #
        # [batch, sequence, vocab]
        #
        # becomes
        #
        # [batch, vocab]
        # ----------------------------------------------------

        next_token_logits = logits[:, -1, :]

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        next_token_logits = (
            next_token_logits / temperature
        )

        # ----------------------------------------------------
        # Top-K filtering
        # ----------------------------------------------------

        if top_k is not None:

            k = min(
                top_k,
                next_token_logits.shape[-1]
            )

            values, indices = torch.topk(
                next_token_logits,
                k
            )

            filtered_logits = torch.full_like(
                next_token_logits,
                float("-inf")
            )

            filtered_logits.scatter_(
                1,
                indices,
                values
            )

            next_token_logits = filtered_logits

        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        probabilities = F.softmax(
            next_token_logits,
            dim=-1
        )

        # ----------------------------------------------------
        # Sample next token
        # ----------------------------------------------------

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # ----------------------------------------------------
        # Append token
        # ----------------------------------------------------

        input_ids = torch.cat(
            [
                input_ids,
                next_token
            ],
            dim=1
        )

        print(
            f"Generated token {step + 1}:",
            next_token.item()
        )

        # ----------------------------------------------------
        # EOS?
        # ----------------------------------------------------

        if (
            next_token.item()
            == tokenizer.eos_id
        ):
            break

    # --------------------------------------------------------
    # IDs -> text
    # --------------------------------------------------------

    result = tokenizer.decode(
        input_ids[0].tolist()
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("MINI LLAMA GENERATION")
    print("=" * 60)

    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    model, tokenizer = load_model()

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="the dog",
        max_new_tokens=20,
        temperature=0.7,
        top_k=10
    )

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(result)


# ============================================================
# THIS IS VERY IMPORTANT
# ============================================================

if __name__ == "__main__":
    main()