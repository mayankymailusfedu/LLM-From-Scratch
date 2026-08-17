# Understand and Build an LLM from the Ground Up

An educational implementation of a small Transformer-based language model inspired by concepts used in modern Llama-family architectures.

The goal of this project is NOT to reproduce production Llama 4.

The goal is to understand how an LLM works internally:

- Tokenization
- BPE
- Token IDs
- Embeddings
- Tensors
- Transformer architecture
- RMSNorm
- Self-Attention
- Query / Key / Value
- Grouped Query Attention (GQA)
- Rotary Positional Embeddings (RoPE)
- Causal masking
- SwiGLU
- SiLU / Swish
- Residual connections
- Logits
- Softmax
- Probability distributions
- Cross-entropy loss
- Backpropagation
- Training
- Checkpoints
- Autoregressive generation
- Debugging the internal workings of an LLM


# 1. Project Goal

The purpose of this project is to build a very small language model from scratch so that we can understand what happens inside an LLM.

For example, after training on a small dataset such as:

    the dog is happy
    the dog is friendly
    the dog likes food
    the dog likes running

    the cat is happy
    the cat is friendly
    the cat likes food

we can give the model:

    the dog

and it may generate:

    the dog is friendly

The model is not actually "thinking" about dogs.

It has learned statistical patterns from the training data.


# 2. Important: This Is an Educational Mini Llama

This project uses concepts found in modern Llama-family architectures, including:

- RMSNorm
- RoPE
- GQA
- SwiGLU
- Transformer blocks
- Causal self-attention
- Autoregressive generation

However, this implementation is intentionally tiny.

For example, our model has approximately:

    700K parameters

while production LLMs contain billions of parameters.

The training dataset is also extremely small.

Therefore, this project should be understood as:

    An educational Mini Llama / Transformer implementation.

It is designed to help us understand the mechanics of an LLM.


# 3. Project Structure

    llama4/
    │
    ├── tokenizer.py
    ├── attention.py
    ├── feedforward.py
    ├── transformer.py
    ├── train.py
    ├── generate.py
    ├── debug.py
    │
    ├── mini_llama.pt
    │
    └── README.md


# 4. Responsibilities of Each File

## tokenizer.py

Responsible for converting text into tokens and token IDs.

    Text
      ↓
    Tokenizer
      ↓
    Tokens
      ↓
    Token IDs

For example:

    "the dog"

might become:

    ["the</w>", "dog</w>"]

and then:

    [28, 12]

The actual IDs depend on our vocabulary.


## attention.py

Contains the attention mechanism.

It is responsible for:

- Query projection
- Key projection
- Value projection
- RoPE
- Q/K normalization
- Grouped Query Attention
- Causal attention
- Attention output projection

Conceptually:

    Input
      ↓
    Q, K, V
      ↓
    RoPE
      ↓
    Q/K normalization
      ↓
    GQA
      ↓
    QKᵀ
      ↓
    Scaling
      ↓
    Causal Mask
      ↓
    Softmax
      ↓
    Attention Probabilities
      ↓
    Weighted V
      ↓
    Output Projection


## feedforward.py

Contains the feed-forward part of the Transformer using a SwiGLU-style architecture.

    Input
      │
      ├───────────────┐
      │               │
      ▼               ▼
    Gate            Up Projection
    Projection           │
      │                  │
      ▼                  │
    SiLU                 │
      │                  │
      └────────┬─────────┘
               │
               ▼
      Element-wise Multiply
               │
               ▼
        Down Projection
               │
               ▼
             Output


## transformer.py

Defines the complete Mini Llama model.

    Token IDs
        ↓
    Embedding
        ↓
    Transformer Layers
        ↓
    Final RMSNorm
        ↓
    LM Head
        ↓
    Logits

Each Transformer layer contains:

    Input
      ↓
    RMSNorm
      ↓
    Attention
      ↓
    Residual Connection
      ↓
    RMSNorm
      ↓
    Feed Forward / SwiGLU
      ↓
    Residual Connection
      ↓
    Output


## train.py

Responsible for training the model.

    Training Text
         ↓
    Tokenizer
         ↓
    Token IDs
         ↓
    Training Sequences
         ↓
    Mini Llama
         ↓
    Logits
         ↓
    Cross Entropy Loss
         ↓
    Backpropagation
         ↓
    Optimizer
         ↓
    Updated Weights

After training, the model weights are saved into:

    mini_llama.pt


## generate.py

Responsible for loading the trained model and generating text.

    Prompt
      ↓
    Tokenizer
      ↓
    Token IDs
      ↓
    Mini Llama
      ↓
    Logits
      ↓
    Softmax
      ↓
    Probability Distribution
      ↓
    Sample Next Token
      ↓
    Append Token
      ↓
    Run Model Again
      ↓
    Repeat


## debug.py

Designed to inspect the internal workings of the model instead of treating the Transformer as a black box.

Useful intermediate values include:

- Token IDs
- Embedding
- RMSNorm
- Q
- K
- V
- RoPE
- Q/K normalization
- GQA
- Attention scores
- Causal mask
- Attention probabilities
- Attention output
- SwiGLU
- Residual connections
- Final hidden state
- Logits
- Next-token probabilities


# 5. The Complete LLM Pipeline

## Training

    TRAINING
       │
       ▼
    Training Text
       │
       ▼
    Tokenizer
       │
       ▼
    Token IDs
       │
       ▼
    Mini Llama
       │
       ▼
    Logits
       │
       ▼
    Loss Function
       │
       ▼
    Backpropagation
       │
       ▼
    Weight Updates
       │
       ▼
    mini_llama.pt


## Generation

    PROMPT
       │
       ▼
    Tokenizer
       │
       ▼
    Token IDs
       │
       ▼
    Mini Llama
       │
       ▼
    Logits
       │
       ▼
    Softmax
       │
       ▼
    Probability Distribution
       │
       ▼
    Next Token
       │
       ▼
    Append Token
       │
       └─────────────┐
                     │
                     ▼
                  Run Again


# 6. What Is a Token?

A token is a unit of text processed by an LLM.

A token does not necessarily represent one complete word.

For example:

    "hello world"

could become:

    ["hello", " world"]

The exact result depends on the tokenizer.

The important concept is:

    Text
     ↓
    Tokens
     ↓
    Numbers

The Transformer works with numbers, not raw text.


# 7. What Is BPE?

BPE stands for:

    Byte Pair Encoding

BPE is a tokenization algorithm.

The basic idea is to start with smaller pieces and repeatedly merge frequently occurring pairs.

For example, imagine:

    low
    lower
    lowest

Initially:

    l o w
    l o w e r
    l o w e s t

Frequently occurring pieces can be merged into larger subword units.

This allows a vocabulary to represent many words without requiring a separate vocabulary entry for every possible word.

For example, a tokenizer may learn pieces such as:

    low
    er
    est

Instead of needing:

    lower
    lowest

as completely independent vocabulary entries.


# 8. Why Do We See </w>?

Our educational tokenizer uses:

    </w>

to indicate the end of a word.

For example:

    the</w>
    dog</w>

means:

    the
    dog

This is an implementation detail of our tokenizer.

Production tokenizers may use different conventions.


# 9. Token IDs

The tokenizer converts text into numbers.

For example:

    "the dog"

might become:

    [28, 12]

where:

    28 → the</w>
    12 → dog</w>

A token ID is simply an index into the vocabulary.


# 10. What Is an Embedding?

A token ID is just an integer.

The model uses an embedding table to convert it into a vector.

Suppose:

    Vocabulary size = 31
    Hidden size = 128

The embedding matrix has shape:

    [31, 128]

There is one 128-dimensional vector for every vocabulary token.

The model learns these vectors during training.

Therefore:

    Token ID
       ↓
    Embedding Lookup
       ↓
    Vector


# 11. What Is a Tensor?

A tensor is a multidimensional array of numbers.

For example:

    x = torch.tensor([1, 2, 3])

has shape:

    [3]

A matrix:

    x = torch.tensor([
        [1, 2],
        [3, 4]
    ])

has shape:

    [2, 2]

A neural network uses tensors everywhere.


# 12. Understanding Tensor Shapes

Model input is commonly:

    [B, S]

where:

    B = Batch Size
    S = Sequence Length

After embedding:

    [B, S, D]

where:

    D = Hidden Dimension

For example:

    [1, 3, 128]

means:

    Batch Size   = 1
    Sequence Len = 3
    Hidden Size  = 128

Always pay attention to tensor shapes when debugging neural networks.


# 13. What Is a Transformer?

A Transformer is a neural-network architecture based primarily on attention.

A simplified Transformer layer:

    Input
      │
      ▼
    RMSNorm
      │
      ▼
    Attention
      │
      ▼
    Residual Add
      │
      ▼
    RMSNorm
      │
      ▼
    Feed Forward
      │
      ▼
    Residual Add
      │
      ▼
    Output

Multiple Transformer layers are stacked together.


# 14. What Is RMSNorm?

RMSNorm stands for:

    Root Mean Square Normalization

Its purpose is to keep activations at a useful numerical scale.

A simplified equation is:

    RMS(x) = sqrt(mean(x²) + epsilon)

Then:

    normalized_x = x / RMS(x)

A learned scale parameter is then applied.


# 15. What Is Epsilon?

Epsilon is a very small positive number such as:

    epsilon = 1e-6

It prevents division by zero or extremely small denominators.

For example:

    sqrt(mean(x²) + epsilon)


# 16. What Are Q, K and V?

Attention uses three representations:

    Q = Query
    K = Key
    V = Value

They are produced from the input:

    Q = XWq
    K = XWk
    V = XWv

where:

    X  = input representation
    Wq = learned Query projection
    Wk = learned Key projection
    Wv = learned Value projection


# 17. Intuition for Query, Key and Value

Imagine a database.

A Query asks:

    What information am I looking for?

A Key says:

    What kind of information do I contain?

A Value says:

    Here is the actual information.

Attention compares Query and Keys to determine how much attention should be given to each Value.


# 18. Self-Attention

The basic scaled dot-product attention equation is:

    Attention(Q,K,V)
    =
    softmax(QKᵀ / sqrt(d)) V

Important steps:

    QKᵀ
      ↓
    Similarity Scores
      ↓
    Scale by sqrt(d)
      ↓
    Causal Mask
      ↓
    Softmax
      ↓
    Attention Probabilities
      ↓
    Multiply by V
      ↓
    Attention Output


# 19. Why Divide by sqrt(d)?

The dot product QKᵀ can become very large when vector dimension is large.

Large values can make softmax extremely sharp.

We therefore divide by:

    sqrt(head_dimension)

This is called:

    Scaled Dot-Product Attention


# 20. What Is a Causal Mask?

When generating text, a token must not see future tokens.

Suppose:

    I like dogs

When processing "like", the model can see "I" and "like", but it cannot see "dogs" because "dogs" is in the future.

Conceptually:

                  I      like      dogs

    I            YES      NO        NO

    like         YES      YES       NO

    dogs         YES      YES       YES

This is a:

    Causal Mask


# 21. Why Is the Causal Mask Necessary?

During training, the model predicts future tokens.

If it could see the answer token itself, it could cheat.

The causal mask forces the model to predict future tokens using only previous tokens.

    Past tokens
        ↓
      Allowed

    Future tokens
        ↓
      Blocked


# 22. What Is an Attention Head?

Attention is divided into multiple heads.

Suppose:

    Hidden Size = 128
    Number of Attention Heads = 8

Then:

    Head Dimension = 128 / 8 = 16

Each head operates on 16 dimensions.


# 23. Why Multiple Attention Heads?

Different heads can learn different relationships.

Multiple heads allow the model to process different attention patterns in parallel.

We should not assume every head has one fixed human-interpretable meaning.


# 24. What Is GQA?

GQA stands for:

    Grouped Query Attention

Suppose:

    Query Heads = 8
    Key/Value Heads = 2

Then:

    Q = 8 heads
    K = 2 heads
    V = 2 heads

The Query heads are grouped around shared Key/Value heads.

Conceptually:

    Q1 Q2 Q3 Q4 → K1 V1
    Q5 Q6 Q7 Q8 → K2 V2

The key idea is:

    Multiple Query heads share fewer Key/Value heads.

This reduces memory requirements for Key/Value representations.


# 25. GQA Tensor Shapes

Before expanding K and V:

    Q:
    [B, 8, S, 16]

    K:
    [B, 2, S, 16]

    V:
    [B, 2, S, 16]

After repeating K and V for Query groups:

    Q:
    [B, 8, S, 16]

    K:
    [B, 8, S, 16]

    V:
    [B, 8, S, 16]


# 26. What Is RoPE?

RoPE stands for:

    Rotary Positional Embedding

A Transformer needs information about token positions.

RoPE adds positional information by rotating parts of Query and Key vectors based on token position.

Conceptually:

    Q
     ↓
    Rotation based on position
     ↓
    Q'

and similarly for K.

The important idea is that positional information becomes part of the attention calculation.


# 27. What Is the Feed-Forward Network?

After attention, each token is passed through a feed-forward network.

Our implementation uses a SwiGLU-style network:

                     x
                     │
            ┌────────┴────────┐
            │                 │
            ▼                 ▼
       Gate Projection    Up Projection
            │                 │
           SiLU               │
            │                 │
            └────────┬────────┘
                     │
                     ▼
             Element-wise *
                     │
                     ▼
             Down Projection
                     │
                     ▼
                   Output


# 28. What Is SiLU?

SiLU stands for:

    Sigmoid Linear Unit

The equation is:

    SiLU(x) = x * sigmoid(x)

It is also commonly called:

    Swish


# 29. What Is SwiGLU?

SwiGLU combines SiLU, gating, and linear projections.

A simplified equation is:

    gate = SiLU(xW_gate)
    up = xW_up
    hidden = gate * up
    output = hidden W_down

The * means:

    Element-wise multiplication

It does NOT mean matrix multiplication.


# 30. What Is a Residual Connection?

A residual connection adds the original input back to the transformed output.

For example:

    x = x + Attention(x)

and:

    x = x + FeedForward(x)

Residual connections help information and gradients flow through deeper networks.


# 31. Complete Transformer Layer

                     Input
                       │
                       ▼
                    RMSNorm
                       │
                       ▼
                   Attention
                       │
                       ▼
              x + Attention(x)
                       │
                       ▼
                    RMSNorm
                       │
                       ▼
                    SwiGLU
                       │
                       ▼
              x + FeedForward(x)
                       │
                       ▼
                     Output

Multiple layers are stacked.


# 32. What Is a Logit?

After the final Transformer layer, the model produces a number for every possible token in the vocabulary.

These numbers are called:

    logits

If the vocabulary contains 31 tokens, there is one logit for each token.

Logits are NOT probabilities.

They can be positive or negative and do not need to sum to 1.


# 33. What Is Softmax?

Softmax converts logits into probabilities.

For example:

    Logits:

    [2.0, 1.0, 0.0]

might become approximately:

    [0.665, 0.245, 0.090]

These values sum to 1 and form a probability distribution.


# 34. Next-Token Prediction

Suppose our prompt is:

    the dog

The model might produce:

    Token       Probability

    is             61%
    likes          24%
    can             7%
    happy           3%
    cat             1%
    ...

The model is effectively saying:

    Given the tokens so far,
    "is" is the most likely next token.


# 35. Autoregressive Generation

The model predicts one token at a time.

    the dog
       ↓
      is

    the dog is
       ↓
    friendly

    the dog is friendly
       ↓
      the

This is called:

    Autoregressive Generation


# 36. Why Does Generation Run the Model Repeatedly?

The model predicts the next token.

The newly generated token becomes part of the input for the next prediction.

For example:

    the dog
       ↓
      is

    the dog is
       ↓
    friendly

    the dog is friendly
       ↓
      the

This repeats until the model generates EOS or reaches the configured maximum number of new tokens.


# 37. What Is EOS?

EOS means:

    End Of Sequence

Example:

    <BOS> the dog is happy <EOS>

If the model produces EOS, generation can stop.


# 38. What Is BOS?

BOS means:

    Beginning Of Sequence

Example:

    <BOS> the dog is happy <EOS>


# 39. What Is Training?

Training means adjusting model parameters so predictions become better.

For example:

    Input:
    the dog is

    Target:
    happy

The model predicts probabilities for the next token.

A loss function measures how wrong the prediction is.


# 40. Cross-Entropy Loss

Language models commonly use:

    Cross-Entropy Loss

If the correct token is "happy" and the model predicts:

    happy → 80%

the loss is relatively low.

If it predicts:

    happy → 1%

the loss is high.

Training tries to minimize this loss.


# 41. Backpropagation

After calculating loss:

    Loss
      ↓
    Backpropagation
      ↓
    Gradients
      ↓
    Optimizer
      ↓
    Updated Weights

The gradient tells each parameter how changing it would affect the loss.


# 42. What Is an Optimizer?

An optimizer updates model parameters during training.

Conceptually:

    Old Parameter
          +
    Gradient Information
          ↓
      Optimizer
          ↓
    New Parameter

This is repeated many times.


# 43. What Is a Parameter?

A parameter is a number learned by the model.

Training changes these values.

The collection of all learned parameters is commonly called:

    Model Weights

Our Mini Llama contains approximately:

    709,632 parameters


# 44. What Is mini_llama.pt?

After training, learned model parameters are saved into:

    mini_llama.pt

Think of it this way:

    transformer.py
          ↓
    Defines the architecture

    mini_llama.pt
          ↓
    Contains the learned weights

Together they form the trained model.


# 45. Why Do We Need mini_llama.pt During Generation?

Creating:

    model = MiniLlama(config)

creates the architecture with newly initialized weights.

Those weights have not learned our training data.

During inference we need to load the trained weights:

    model = MiniLlama(config)

    checkpoint = torch.load("mini_llama.pt")

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

Then the trained model can generate text.


# 46. Understanding generate.py

A simplified generation implementation is:

    input_ids = tokenizer.encode(prompt)

    logits = model(input_ids)

    next_token_logits = logits[:, -1, :]

    probabilities = torch.softmax(
        next_token_logits,
        dim=-1
    )

    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    )

    input_ids = torch.cat(
        [input_ids, next_token],
        dim=1
    )

Then run the model again.

This repeats until the stopping condition is reached.


# 47. What Is Temperature?

Temperature controls randomness.

The logits are divided by temperature before softmax:

    logits / temperature

Lower temperature such as:

    0.2

makes the distribution sharper and generation more predictable.

Higher temperature such as:

    1.5

makes the distribution flatter and generation more random.


# 48. What Is Top-K Sampling?

Top-K sampling keeps only the K most likely tokens.

For example:

    top_k = 5

means only the five most likely tokens remain candidates for sampling.


# 49. Argmax vs Sampling

Argmax always chooses the most likely token:

    next_token = torch.argmax(probabilities)

Sampling allows less-likely tokens to occasionally be selected according to their probabilities.

For educational purposes, it is useful to experiment with both approaches.


# 50. Understanding Our Generated Result

Suppose we get:

    the dog is friendly the dog likes food
    the dog likes running the cat is happy

We should not interpret this as proof that the model understands dogs.

A more accurate interpretation is:

    The model has learned statistical relationships
    between tokens in the training data.

If training frequently contains:

    the dog is friendly

then the model learns that:

    "the dog"

is often followed by:

    "is"

and:

    "is"

is often followed by:

    "friendly"


# 51. Why Does Our Model Sometimes Produce Strange Text?

Our training dataset is extremely small.

For example:

    the dog is happy
    the dog is friendly
    the dog likes food
    the dog likes running

    the cat is happy
    the cat is friendly

The model may generate combinations of learned patterns.

It has not learned the broad world knowledge of a production LLM.

This is one reason why EOS tokens and properly structured training data are important.


# 52. Example Tensor Flow

Suppose input is:

    the dog

After tokenization:

    [0, 28, 12]

Input tensor shape:

    [1, 3]

After embedding:

    [1, 3, 128]

For attention:

    Q:
    [1, 8, 3, 16]

    K:
    [1, 2, 3, 16]

    V:
    [1, 2, 3, 16]

After GQA expansion:

    Q:
    [1, 8, 3, 16]

    K:
    [1, 8, 3, 16]

    V:
    [1, 8, 3, 16]

Attention scores:

    [1, 8, 3, 3]

Final logits:

    [1, 3, 31]


# 53. Why Are Final Logits [1, 3, 31]?

The dimensions are:

    [Batch, Sequence, Vocabulary]

So:

    [1, 3, 31]

means:

    1 sequence
    3 positions
    31 possible vocabulary tokens

For generation we need only the final position:

    logits[:, -1, :]

which gives:

    [1, 31]

This means:

    What should the next token be?


# 54. Understanding debug.py

The purpose of debug.py is to make the model transparent.

Instead of:

    Input
      ↓
    Model
      ↓
    Output

we inspect:

    Input IDs
        ↓
    Embedding
        ↓
    RMSNorm
        ↓
    Q / K / V
        ↓
    RoPE
        ↓
    GQA
        ↓
    Attention Scores
        ↓
    Causal Mask
        ↓
    Softmax
        ↓
    Attention Probabilities
        ↓
    Attention Output
        ↓
    Residual
        ↓
    RMSNorm
        ↓
    SwiGLU
        ↓
    Residual
        ↓
    Final RMSNorm
        ↓
    LM Head
        ↓
    Logits
        ↓
    Probability Distribution


# 55. Recommended Learning Order

If you are an intern learning this project, study the concepts in this order:

    1. Python
    2. PyTorch tensors
    3. Tensor shapes
    4. Tokenization
    5. BPE
    6. Embeddings
    7. Attention
    8. Q / K / V
    9. Causal masking
    10. RoPE
    11. GQA
    12. RMSNorm
    13. SiLU
    14. SwiGLU
    15. Residual connections
    16. Transformer Block
    17. Logits
    18. Softmax
    19. Probability distributions
    20. Cross-entropy
    21. Backpropagation
    22. Training
    23. Generation
    24. Debugging

Study the corresponding files in this order:

    tokenizer.py
    attention.py
    feedforward.py
    transformer.py
    train.py
    generate.py
    debug.py


# 56. How to Run the Project

Create a virtual environment:

    python -m venv .venv

Activate it on macOS/Linux:

    source .venv/bin/activate

Install dependencies:

    pip install torch numpy


# 57. Train the Model

Run:

    python train.py

After training, you should have:

    mini_llama.pt

This contains the learned model parameters.


# 58. Generate Text

Run:

    python generate.py

For example:

    Prompt:
    the dog

    Result:
    the dog is friendly the dog likes food ...

The exact output may vary depending on:

- Training
- Random seed
- Temperature
- Top-K
- Model initialization
- Dataset
- Sampling method


# 59. Debug the Model

Run:

    python debug.py

This allows inspection of the internal operations and tensor shapes.


# 60. Suggested Experiments

## Experiment 1 — Change the Prompt

Try:

    the dog

then:

    the cat

then:

    the

Compare the outputs.


## Experiment 2 — Change Temperature

Try:

    temperature = 0.2

then:

    temperature = 0.7

then:

    temperature = 1.5

Observe how randomness changes.


## Experiment 3 — Change Top-K

Try:

    top_k = 1

    top_k = 5

    top_k = 10

Observe how randomness changes.


## Experiment 4 — Inspect Attention

Run:

    python debug.py

Look at attention probabilities.

Ask:

    Which previous tokens is each attention head
    focusing on?


## Experiment 5 — Increase Hidden Size

Try changing:

    hidden_size = 128

to:

    hidden_size = 256

Observe:

- Parameter count
- Memory usage
- Training speed
- Output quality


## Experiment 6 — Increase Number of Layers

Try:

    num_hidden_layers = 2

and then:

    num_hidden_layers = 4

Observe how the number of parameters changes.


# 61. Important Engineering Concepts

## Separation of Concerns

Each file has one primary responsibility:

    tokenizer.py
        ↓
    Text processing

    attention.py
        ↓
    Attention

    feedforward.py
        ↓
    Feed-forward network

    transformer.py
        ↓
    Model architecture

    train.py
        ↓
    Training

    generate.py
        ↓
    Inference

    debug.py
        ↓
    Debugging

This is much easier to maintain than putting everything into one large Python file.


# 62. Tensor Shapes Are Contracts

When debugging a neural network, always ask:

    What is the shape of this tensor?

For example:

    [B, S, D]

means:

    B = Batch Size
    S = Sequence Length
    D = Hidden Dimension

For attention:

    [B, H, S, D_head]

means:

    B      = Batch
    H      = Number of Heads
    S      = Sequence Length
    D_head = Dimension per Head

Understanding tensor shapes is one of the most important skills for working with PyTorch and LLMs.


# 63. Reproducibility

Model output depends on:

- Model architecture
- Model weights
- Tokenizer vocabulary
- Tokenizer merges
- Training data
- Random seed
- Temperature
- Top-K
- Sampling method

For reproducible experiments, control the random seed:

    torch.manual_seed(42)


# 64. Production vs Educational LLM

This project intentionally leaves out many production concerns, such as:

- Distributed training
- Mixed precision
- Flash Attention
- KV caching
- Tensor parallelism
- Pipeline parallelism
- Quantization
- Large-scale datasets
- Data filtering
- Checkpoint sharding
- Distributed optimizers
- Advanced sampling
- Evaluation frameworks
- Monitoring
- Inference servers

Our goal is different:

    Understand what happens inside an LLM

before learning how to train and serve massive LLMs efficiently.


# 65. The Most Important Mental Model

    TEXT
      │
      ▼
    TOKENIZER
      │
      ▼
    TOKEN IDs
      │
      ▼
    EMBEDDING
      │
      ▼
    TRANSFORMER
      │
      ├── RMSNorm
      │
      ├── Attention
      │
      │    ├── Q
      │    ├── K
      │    ├── V
      │    ├── RoPE
      │    ├── GQA
      │    ├── Causal Mask
      │    └── Softmax
      │
      ├── Residual
      │
      ├── RMSNorm
      │
      ├── SwiGLU
      │
      └── Residual
      │
      ▼
    FINAL RMSNORM
      │
      ▼
    LM HEAD
      │
      ▼
    LOGITS
      │
      ▼
    SOFTMAX
      │
      ▼
    PROBABILITY DISTRIBUTION
      │
      ▼
    NEXT TOKEN
      │
      ▼
    APPEND TOKEN
      │
      └──────────────┐
                     │
                     ▼
                   REPEAT


# 66. The Entire LLM in One Sentence

A language model is essentially learning:

    Given the tokens I have seen so far,
    what token is most likely to come next?

Everything in the Transformer exists to make this prediction better.


# 67. A Concrete Example

Let's walk through:

    the dog


## Step 1 — Tokenization

    "the dog"

becomes:

    [0, 28, 12]


## Step 2 — Embedding

    [0, 28, 12]
           ↓
       Embedding
           ↓
       [1, 3, 128]


## Step 3 — Transformer

    RMSNorm
       ↓
    Attention
       ↓
    Residual
       ↓
    RMSNorm
       ↓
    SwiGLU
       ↓
    Residual


## Step 4 — Final Representation

    [1, 3, 128]


## Step 5 — LM Head

    [1, 3, 128]
           ↓
        LM Head
           ↓
      [1, 3, 31]


## Step 6 — Select Last Position

    logits[:, -1, :]

produces:

    [1, 31]


## Step 7 — Softmax

The model might produce:

    is       61%
    likes    24%
    can       7%
    happy     3%
    ...


## Step 8 — Choose a Token

Suppose the selected token is:

    is

Now:

    the dog is


## Step 9 — Repeat

    the dog is
           ↓
       friendly

Now:

    the dog is friendly

Continue until EOS or max_new_tokens is reached.


# 68. Final Architecture

    INPUT TEXT
         │
         ▼
    BPE TOKENIZER
         │
         ▼
    TOKEN IDs
         │
         ▼
    TOKEN EMBEDDING
         │
         ▼
    ┌───────────────────────────┐
    │     TRANSFORMER LAYER     │
    │                           │
    │         RMSNorm           │
    │            ↓              │
    │        Attention          │
    │        ├── Q              │
    │        ├── K              │
    │        ├── V              │
    │        ├── RoPE           │
    │        ├── GQA            │
    │        ├── Mask           │
    │        └── Softmax        │
    │            ↓              │
    │        Residual           │
    │            ↓              │
    │         RMSNorm           │
    │            ↓              │
    │         SwiGLU            │
    │            ↓              │
    │        Residual           │
    └─────────────┬─────────────┘
                  │
             Repeat N Layers
                  │
                  ▼
             Final RMSNorm
                  │
                  ▼
                LM Head
                  │
                  ▼
                Logits
                  │
                  ▼
               Softmax
                  │
                  ▼
          Token Probabilities
                  │
                  ▼
             Sample Token
                  │
                  ▼
          Append to Context
                  │
                  └──────────────┐
                                 │
                                 ▼
                               Repeat


# 69. Recommended Next Steps

Once the intern understands this project:

1. Improve tokenizer persistence.
2. Store vocabulary and BPE merges with the checkpoint.
3. Properly train with BOS/EOS boundaries.
4. Improve the dataset pipeline.
5. Add a validation dataset.
6. Track training and validation loss.
7. Plot the loss curve.
8. Add deterministic generation.
9. Add KV caching.
10. Visualize attention weights.
11. Add unit tests for tensor shapes.
12. Add model configuration files.
13. Compare this implementation with a standard Transformer.
14. Study the differences between this educational implementation and production Llama architectures.


# 70. Final Mental Model for an Intern

Think of the model as a machine that repeatedly performs this operation:

    "I have these tokens."
             ↓
    "What relationships exist between them?"
             ↓
    "Which tokens should I pay attention to?"
             ↓
    "What representation should I create?"
             ↓
    "What should the next token be?"
             ↓
    "Add that token to the input."
             ↓
    "Do it again."

For example:

    the dog
       ↓
      is

    the dog is
       ↓
    friendly

    the dog is friendly
       ↓
      the

This repeated next-token prediction is at the heart of autoregressive LLM generation.


# 71. Summary

The Mini Llama project demonstrates the complete journey:

                         TRAINING

    Training Text
         ↓
    Tokenizer
         ↓
    Token IDs
         ↓
    Embeddings
         ↓
    Transformer
         ↓
    Logits
         ↓
    Cross Entropy Loss
         ↓
    Backpropagation
         ↓
    Optimizer
         ↓
    Learned Weights
         ↓
    mini_llama.pt


Then:

                         INFERENCE

    Prompt
      ↓
    Tokenizer
      ↓
    Token IDs
      ↓
    Embeddings
      ↓
    Transformer
      ↓
    Logits
      ↓
    Softmax
      ↓
    Probability Distribution
      ↓
    Next Token
      ↓
    Append Token
      ↓
    Run Again
      ↓
    Repeat
      ↓
    Generated Text


The key idea is simple:

    An LLM repeatedly predicts the next token based
    on the tokens that came before it.

The Transformer architecture provides the machinery that allows the model to learn increasingly useful relationships between those tokens.


# 72. File-to-Concept Map

    File               Main Responsibility          Concepts

    tokenizer.py       Convert text to IDs           BPE, vocabulary, tokens

    attention.py       Attention mechanism           Q, K, V, RoPE, GQA, masking

    feedforward.py     Feed-forward network          SiLU, SwiGLU, gating

    transformer.py     Complete model                Transformer, RMSNorm,
                                                      residuals

    train.py           Train model                   Loss, backpropagation,
                                                      optimizer

    generate.py        Generate text                Logits, softmax, sampling

    debug.py           Inspect internals             Tensor shapes,
                                                      intermediate values

    mini_llama.pt      Learned weights               Model checkpoint

    README.md          Documentation                Architecture and concepts


# 73. Final Takeaway

If you can understand the following chain, you understand the core of this project:

    Text
      ↓
    Tokens
      ↓
    Token IDs
      ↓
    Embeddings
      ↓
    Q / K / V
      ↓
    Attention
      ↓
    RoPE
      ↓
    Causal Mask
      ↓
    GQA
      ↓
    SwiGLU
      ↓
    Residual Connections
      ↓
    RMSNorm
      ↓
    Logits
      ↓
    Softmax
      ↓
    Probability Distribution
      ↓
    Next Token
      ↓
    Repeat


That is the foundation to understand before moving on to larger and more complex LLM implementations.