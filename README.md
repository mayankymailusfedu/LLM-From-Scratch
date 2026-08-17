# Understanding LLM From the Ground Up

This directory contains three Python code that explain important building blocks used in modern Large Language Models (LLMs), using simplified implementations inspired by Llama 4.

The three main files are:

    tokenizer.py
    attention_weights.py
    feed_forward.py

Instead, they showcase the core ideas that make an LLM work:

    Text
      ↓
    Tokenization
      ↓
    Token IDs
      ↓
    Embeddings
      ↓
    Transformer
      ↓
    Attention
      ↓
    Feed Forward Network
      ↓
    Next-token prediction



# 1. WHAT ARE WE BUILDING?

A Large Language Model ultimately performs a surprisingly simple task:

    Given the tokens that came before,
    predict what token should come next.

For example:

    "The dog"

might result in:

    "is"

Then the model sees:

    "The dog is"

and predicts:

    "happy"

So generation becomes:

    The dog
       ↓
      is
       ↓
    happy
       ↓
      ...

The Transformer architecture is the machinery that helps the model make these predictions.



# 2. FILE STRUCTURE

    LLM-From-Scratch/
    │
    ├── tokenizer.py
    │
    ├── attention_weights.py
    │
    └── feed_forward.py


Each file focuses on a different part of the LLM.


------------------------------------------------------------
tokenizer.py
------------------------------------------------------------

Purpose:

    Build a simple Byte Pair Encoding (BPE) tokenizer from scratch.

Main concepts:

    - Vocabulary
    - Characters
    - Tokens
    - Words
    - End-of-word markers
    - Pair frequencies
    - BPE merges
    - Subword tokens


------------------------------------------------------------
attention_weights.py
------------------------------------------------------------

Purpose:

    Understand the attention mechanism.

Main concepts:

    - Hidden states
    - Query
    - Key
    - Value
    - Multi-Head Attention
    - Grouped Query Attention
    - RoPE
    - Q/K normalization
    - Causal masking
    - Attention scores
    - Softmax
    - Attention probabilities
    - Weighted values
    - Output projection


------------------------------------------------------------
feed_forward.py
------------------------------------------------------------

Purpose:

    Understand the Feed Forward Network inside a Transformer layer.

Main concepts:

    - RMSNorm
    - Intermediate dimension
    - Gate projection
    - Up projection
    - SiLU / Swish
    - SwiGLU
    - Element-wise multiplication
    - Down projection
    - Residual connection


# 3. BIG PICTURE

The three files fit together like this:

                         TEXT
                           │
                           ▼
                      TOKENIZER
                           │
                           ▼
                       TOKENS
                           │
                           ▼
                      TOKEN IDs
                           │
                           ▼
                      EMBEDDING
                           │
                           ▼
                ┌─────────────────────┐
                │   TRANSFORMER       │
                │                     │
                │  RMSNorm            │
                │     ↓               │
                │  Attention          │
                │     ↓               │
                │  Residual           │
                │     ↓               │
                │  RMSNorm            │
                │     ↓               │
                │  Feed Forward       │
                │     ↓               │
                │  Residual           │
                └─────────┬───────────┘
                          │
                          ▼
                       LOGITS
                          │
                          ▼
                       SOFTMAX
                          │
                          ▼
                   NEXT TOKEN


The three files implement the middle part of this pipeline.


# 4. File 2 — TOKENIZATION

File:

    tokenizer.py


The Transformer cannot directly process:

    "The dog is happy"


Neural networks operate on numbers.

Therefore we need:

    Text
      ↓
    Tokens
      ↓
    Token IDs


For example:

    "the dog"

could become:

    ["the</w>", "dog</w>"]

and then:

    [28, 12]


The actual numbers depend on the vocabulary.


# 5. WHAT IS A TOKEN?

A token is a piece of text.

A token might be:

    a complete word

or:

    part of a word

or:

    punctuation

or:

    whitespace-related text


For example:

    "playing"

could potentially be represented as:

    ["play", "ing"]


This allows the tokenizer to represent many words without needing a separate vocabulary entry for every possible word.


# 6. WHAT IS BPE?

BPE means:

    Byte Pair Encoding


The basic idea is:

    Start with small pieces
          ↓
    Count frequently occurring pairs
          ↓
    Merge the most frequent pair
          ↓
    Add the merged token to vocabulary
          ↓
    Repeat


For example, suppose we start with:

    i s


If:

    "is"

appears frequently, we can merge:

    i + s

into:

    is


So:

    ["i", "s"]

becomes:

    ["is"]


This is the fundamental idea behind BPE.


# 7. BPE EXAMPLE

Suppose our training corpus contains:

    is
    this
    his
    list


Initially we may have:

    i
    s
    t
    h
    l
    ...


The pair:

    ("i", "s")

appears frequently.

BPE calculates pair frequencies.

For example:

    ("i", "s")      → 10
    ("t", "h")      → 8
    ("s", "t")      → 4


The most frequent pair is:

    ("i", "s")


So we merge it:

    i + s → is


Then the vocabulary grows:

    i
    s
    t
    h
    l
    is


BPE continues doing this repeatedly.


# 8. THE CODE: get_pair_stats()


The function:

    get_pair_stats(splits)

counts adjacent symbol pairs.

For example:

    ("t", "h", "i", "s", "</w>")


contains:

    ("t", "h")
    ("h", "i")
    ("i", "s")
    ("s", "</w>")


If the word appears 5 times, each pair receives a frequency contribution of 5.


The purpose is:

    Find which neighboring pieces occur most frequently.



# 9. THE CODE: merge_pair()


Suppose:

    pair_to_merge = ("i", "s")


and the word is:

    ("t", "h", "i", "s", "</w>")


After merging:

    ("t", "h", "is", "</w>")


The new token is:

    "is"


The function performs this transformation across the current vocabulary representation.



# 10. THE BPE TRAINING LOOP


The main BPE loop does:

    1. Count pair frequencies

    2. Find the most frequent pair

    3. Merge that pair

    4. Add the new token to vocabulary

    5. Store the merge rule

    6. Repeat


Conceptually:

    Initial tokens

        ↓

    Count pairs

        ↓

    Find best pair

        ↓

    Merge

        ↓

    New vocabulary

        ↓

    Repeat


The repository uses a small number of merges so that the process can be observed and understood.



# 11. WHAT IS </w>?


The tokenizer uses:

    </w>


This means:

    End Of Word


For example:

    dog</w>


means:

    dog

followed by a word boundary.


This is useful because the tokenizer needs to know where words end.

For example:

    dog
    dogs


can be represented differently while still sharing:

    dog


The exact representation used by production tokenizers can differ.


# 12. IMPORTANT TOKENIZER LIMITATION

The tokenizer in this file is intentionally simple.

It:

    - Uses a small training corpus
    - Splits words using spaces
    - Starts from characters
    - Uses </w>
    - Learns a small number of merges


It should NOT be considered a production Llama tokenizer.


This distinction is extremely important.

There are two different ideas:

    Simplified BPE tokenizer

versus:

    Production Llama tokenizer


The file is implementing:

    "How does BPE work?"


It is not attempting to reproduce the exact tokenizer used by a production Llama model.


# 13. TOKENIZER MENTAL MODEL

Think:

    "The dog is happy"

            ↓

    Tokenizer

            ↓

    ["The</w>", "dog</w>", "is</w>", "happy</w>"]

            ↓

    Vocabulary lookup

            ↓

    [12, 25, 7, 31]


The Transformer receives:

    [12, 25, 7, 31]

not:

    "The dog is happy"



# 14. FILE 3 — ATTENTION

File:

    attention_weights.py


Attention is one of the most important concepts in an LLM.

Its purpose is to allow each token to determine:

    "Which other tokens should I pay attention to?"


For example:

    The dog chased the cat because it was hungry.


What does:

    "it"

refer to?


Attention allows the model to learn relationships between tokens.


# 15. HIDDEN STATES

After tokenization, token IDs are converted into vectors.

For example:

    Token ID
       ↓
    Embedding
       ↓
    Vector


Suppose:

    hidden_size = 128


Then every token can be represented using:

    128 numbers


If we have:

    batch_size = 2
    sequence_length = 10
    hidden_size = 128


then the hidden states have shape:

    [2, 10, 128]


Meaning:

    2 sequences
    10 tokens per sequence
    128 values per token


# 16. Q, K AND V

Attention transforms the hidden states into:

    Q = Query
    K = Key
    V = Value


Conceptually:

    Hidden States
          │
          ├──────────► Query
          │
          ├──────────► Key
          │
          └──────────► Value


The code creates:

    q_proj
    k_proj
    v_proj


using linear layers.


# 17. WHAT DOES QUERY MEAN?

Think of Query as:

    "What information am I looking for?"


Key means:

    "What information do I represent?"


Value means:

    "Here is the information I can provide."


Attention compares:

    Query

against:

    Keys


and uses the resulting scores to decide how much of each:

    Value

should be used.


# 18. MULTI-HEAD ATTENTION

Instead of performing one giant attention operation, the model divides the hidden representation into multiple heads.


For example:

    hidden_size = 128

    num_attention_heads = 16


Then:

    head_dim = 128 / 16

             = 8


So each attention head operates on:

    8 dimensions


Conceptually:

    Hidden State
          │
          ├── Head 1
          ├── Head 2
          ├── Head 3
          ├── ...
          └── Head 16


Each head can learn different relationships.


# 19. GROUPED QUERY ATTENTION

The file demonstrates:

    Grouped Query Attention

or:

    GQA


Suppose:

    Query heads = 16

    Key/Value heads = 4


Then:

    16 / 4 = 4


So every Key/Value head is shared by:

    4 Query heads


Conceptually:

    Q1 Q2 Q3 Q4
          │
          ▼
         K1 V1


    Q5 Q6 Q7 Q8
          │
          ▼
         K2 V2


    Q9 Q10 Q11 Q12
          │
          ▼
         K3 V3


    Q13 Q14 Q15 Q16
          │
          ▼
         K4 V4


This reduces the amount of Key/Value state that must be maintained.


# 20. WHY GQA?

Traditional Multi-Head Attention might have:

    16 Q heads
    16 K heads
    16 V heads


GQA might instead use:

    16 Q heads
    4 K heads
    4 V heads


This means fewer K/V representations.

This is particularly useful for reducing memory requirements during inference.


# 21. ATTENTION SHAPES

Suppose:

    batch_size = 2

    sequence_length = 10

    hidden_size = 128

    query_heads = 16

    kv_heads = 4

    head_dim = 8


Then:

    Q:

    [2, 16, 10, 8]


    K:

    [2, 4, 10, 8]


    V:

    [2, 4, 10, 8]


After GQA repeats K and V:

    K:

    [2, 16, 10, 8]


    V:

    [2, 16, 10, 8]


This allows Q, K and V to participate in the attention calculation head-by-head.


# 22. ROTARY POSITIONAL EMBEDDINGS

Attention by itself does not inherently know the order of tokens.

For example:

    dog bites man

and:

    man bites dog


contain the same words but have different meanings.


The model therefore needs positional information.


Llama-family models use:

    RoPE

which means:

    Rotary Positional Embeddings


# 23. INTUITION FOR RoPE

RoPE rotates parts of Query and Key vectors based on token position.


Conceptually:

    Query
      ↓
    Rotate according to position
      ↓
    New Query


and:

    Key
      ↓
    Rotate according to position
      ↓
    New Key


The important property is that the attention calculation can capture relative positional relationships.


# 24. WHY DOES RoPE USE ROTATION?

Imagine a 2D vector:

    [x, y]


A rotation transforms it based on an angle.


RoPE applies this idea to pairs of dimensions in the Query and Key vectors.


The file implements this using complex numbers:

    real + imaginary


and complex multiplication.


Conceptually:

    vector pair
        ↓
    complex number
        ↓
    multiply by rotation
        ↓
    rotated vector


# 25. torch.outer() IN RoPE

The file contains:

    torch.outer(t, inv_freq)


`torch.outer()` computes the outer product.


If:

    t = [0, 1, 2]

and:

    inv_freq = [a, b]


then:

    outer(t, inv_freq)


produces:

    [
        [0a, 0b],
        [1a, 1b],
        [2a, 2b]
    ]


This creates the position/frequency combinations needed for RoPE.


# 26. RoPE IS APPLIED TO Q AND K

An important concept:

    RoPE is applied to Query and Key.


The Value vectors are not rotated.


The file explicitly applies RoPE before repeating K/V for GQA.


So the conceptual order is:

    Q/K projection
          ↓
        RoPE
          ↓
       QK Norm
          ↓
       GQA repeat
          ↓
      Attention


# 27. Q/K NORMALIZATION

The file includes optional:

    QK normalization


The purpose is to normalize Query and Key vectors before calculating attention scores.


Conceptually:

    Q
     ↓
    Normalize
     ↓
    Q'


and:

    K
     ↓
    Normalize
     ↓
    K'


Then:

    Q' × K'


is used for attention.


The file uses an epsilon value such as:

    1e-6


to improve numerical stability.


# 28. WHAT IS EPSILON?

Epsilon is a tiny positive number.


Example:

    1e-6


It is added to calculations such as:

    sqrt(x + epsilon)


to prevent numerical problems such as division by zero.


This is a very common technique in neural networks.


# 29. CAUSAL MASK

When generating text, the current token cannot look into the future.


Suppose:

    I like dogs


The token:

    like


can see:

    I
    like


but cannot see:

    dogs


The attention matrix therefore looks conceptually like:

             I     like    dogs

    I        YES    NO      NO

    like     YES    YES     NO

    dogs     YES    YES     YES


This is called:

    Causal Attention


# 30. HOW THE CAUSAL MASK WORKS

The file creates an upper-triangular mask with:

    -inf


for future positions.


Conceptually:

    [ 0   -inf  -inf ]

    [ 0    0    -inf ]

    [ 0    0     0  ]


Then:

    attention_scores + mask


Future positions receive:

    -inf


After softmax:

    softmax(-inf) ≈ 0


Therefore future tokens receive effectively zero attention probability.


# 31. SCALED DOT-PRODUCT ATTENTION

The fundamental attention equation is:

    Attention(Q,K,V)
        =
    softmax(QK^T / sqrt(d)) V


The file follows this sequence:

    1. Q × Kᵀ

    2. Divide by sqrt(head_dim)

    3. Apply causal mask

    4. Softmax

    5. Multiply by V


# 32. STEP 1 — Q × Kᵀ

Suppose:

    Q shape:

    [batch, heads, sequence, head_dim]


and:

    K shape:

    [batch, heads, sequence, head_dim]


Transpose K:

    Kᵀ:

    [batch, heads, head_dim, sequence]


Then:

    Q @ Kᵀ


produces:

    [batch, heads, sequence, sequence]


For example:

    [2, 16, 10, 10]


The final:

    10 × 10


matrix represents how every token relates to every other token.


# 33. STEP 2 — SCALE

The attention scores are scaled:

    score / sqrt(head_dim)


Why?


Without scaling, dot products can become very large.

Very large values can make softmax extremely sharp.


Scaling keeps the values in a more useful numerical range.


# 34. STEP 3 — APPLY CAUSAL MASK

We add the mask:

    attention_scores + mask


Future positions contain:

    -inf


Therefore after softmax:

    future probability ≈ 0


# 35. STEP 4 — SOFTMAX

Softmax converts attention scores into probabilities.


For example:

    Scores:

    [2.0, 1.0, 0.0]


might become:

    [0.665, 0.245, 0.090]


These values:

    - are positive
    - sum to approximately 1


Therefore they form an attention probability distribution.


# 36. STEP 5 — MULTIPLY BY V

The attention probabilities are multiplied by Value vectors.


Suppose:

    Attention probabilities:

    [0.7, 0.2, 0.1]


and:

    V1
    V2
    V3


Then:

    output =
        0.7V1 +
        0.2V2 +
        0.1V3


So attention is effectively:

    "Take a weighted combination of information from other tokens."


# 37. ATTENTION OUTPUT

After calculating attention:

    [batch, heads, sequence, head_dim]


the heads are combined.


The file does:

    transpose
        ↓
    contiguous
        ↓
    reshape


to produce:

    [batch, sequence, hidden_size]


Then the output projection:

    o_proj


maps the result back into the hidden dimension.


# 38. COMPLETE ATTENTION PIPELINE

The entire attention operation can be remembered as:

    Hidden States
          │
          ▼
       Q / K / V
          │
          ▼
        RoPE
          │
          ▼
       Q/K Norm
          │
          ▼
        GQA
          │
          ▼
        QKᵀ
          │
          ▼
        Scale
          │
          ▼
      Causal Mask
          │
          ▼
        Softmax
          │
          ▼
      Attention Weights
          │
          ▼
       Weighted V
          │
          ▼
      Concatenate Heads
          │
          ▼
       Output Projection
          │
          ▼
        Attention Output


# 39. FILE 4 — FEED FORWARD

File:

    feed_forward.py


Attention allows tokens to communicate with one another.


The Feed Forward Network performs additional processing on each token representation.


A simplified Transformer block is:

    Input
      │
      ▼
    RMSNorm
      │
      ▼
    Attention
      │
      ▼
    Residual
      │
      ▼
    RMSNorm
      │
      ▼
    Feed Forward
      │
      ▼
    Residual
      │
      ▼
    Output


# 40. RMSNORM

RMSNorm means:

    Root Mean Square Normalization


The file computes the mean of squared values:

    mean(x²)


then:

    sqrt(mean(x²) + epsilon)


and normalizes:

    x / sqrt(mean(x²) + epsilon)


A learned weight is then applied.


The goal is to keep activations numerically well behaved.


# 41. WHY NORMALIZATION?

Neural networks repeatedly transform values.


Without normalization, activations can become poorly scaled.


Normalization helps maintain a stable numerical range.


Think of it as:

    Keep the representation at a useful scale
    before performing another large transformation.


# 42. RMSNORM EQUATION

A simplified RMSNorm equation is:

    RMS(x) =
        sqrt(mean(x²) + epsilon)


Then:

    normalized_x =
        x / RMS(x)


Then:

    output =
        normalized_x * learned_weight


The learned weight allows the model to determine an appropriate scale during training.


# 43. INTERMEDIATE SIZE

The Feed Forward Network expands the hidden representation.


For example:

    hidden_size = 128


The intermediate dimension might be larger:

    intermediate_size = 256


or another configured value.


Conceptually:

    128
     ↓
    256
     ↓
    128


This expansion gives the model additional capacity to transform the representation.


# 44. SWIGLU

The file uses a gated feed-forward architecture called:

    SwiGLU


The core equation is:

    down_proj(
        SiLU(gate_proj(x))
        *
        up_proj(x)
    )


There are three important projections:

    gate_proj
    up_proj
    down_proj


# 45. GATE PROJECTION

The first projection is:

    gate_proj


It transforms:

    hidden_size


into:

    intermediate_size


For example:

    [batch, sequence, 128]


becomes:

    [batch, sequence, 256]


The result is called:

    gate


# 46. UP PROJECTION

The second projection is:

    up_proj


It also transforms:

    hidden_size


into:

    intermediate_size


So:

    gate:

    [batch, sequence, 256]


and:

    up:

    [batch, sequence, 256]


have matching shapes.


# 47. SILU

SiLU means:

    Sigmoid Linear Unit


The equation is:

    SiLU(x) = x * sigmoid(x)


It is also commonly called:

    Swish


In PyTorch:

    nn.SiLU()


can be used.


# 48. WHY HAVE A GATE?

The gate allows the network to control which parts of the representation should be emphasized.


The model computes:

    SiLU(gate)


and then:

    SiLU(gate) * up


This is element-wise multiplication.


For example:

    gate:

    [2, 3, 4]


after SiLU:

    [1.76, 2.86, 3.93]


and:

    up:

    [1, 2, 3]


then:

    element-wise multiplication:

    [1.76, 5.72, 11.79]


This creates the gated representation.


# 49. ELEMENT-WISE MULTIPLICATION

This is very important.


The operation:

    a * b


in:

    activated_gate * up_output


means element-by-element multiplication.


Example:

    [1, 2, 3]

    *

    [4, 5, 6]

    =

    [4, 10, 18]


This is NOT matrix multiplication.


Matrix multiplication would be represented conceptually as:

    A @ B


in PyTorch.


# 50. DOWN PROJECTION

After gating:

    [batch, sequence, intermediate_size]


is passed through:

    down_proj


which returns:

    [batch, sequence, hidden_size]


So the overall shape transformation is:

    hidden_size
         ↓
    intermediate_size
         ↓
    hidden_size


# 51. COMPLETE SWIGLU PIPELINE

The Feed Forward Network is:

    x
    │
    ├───────────────┐
    │               │
    ▼               ▼
    gate_proj      up_proj
    │               │
    ▼               │
    SiLU             │
    │               │
    └───────┬───────┘
            │
            ▼
       element-wise *
            │
            ▼
        down_proj
            │
            ▼
          output


Mathematically:

    FFN(x) =
        W_down(
            SiLU(W_gate(x))
            *
            W_up(x)
        )


# 52. RESIDUAL CONNECTION

The Feed Forward output is not simply returned by itself.


The original input is added:

    output =
        input + FFN(input)


This is called a:

    Residual Connection


The file explicitly demonstrates:

    final_output =
        input_to_ffn_block + ffn_output


# 53. WHY RESIDUAL CONNECTIONS?

Imagine stacking many Transformer layers.


Without residual connections, information would have to pass through every transformation.


With residual connections:

    input
      │
      ├───────────────┐
      │               │
      │           Transformation
      │               │
      └─────── + ─────┘
              │
              ▼
            output


The original information has a direct path through the network.


Residual connections also help gradient flow during training.


# 54. COMPLETE TRANSFORMER BLOCK

Now we can combine the attention and Feed Forward files.


The structure is:

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
                   SwiGLU FFN
                        │
                        ▼
                  Residual Add
                        │
                        ▼
                      Output


This is the core Transformer block pattern.


# 55. HOW THE THREE FILES CONNECT

The connection is:

    FILE 2

    Text
      ↓
    BPE
      ↓
    Token IDs


          ↓


    EMBEDDING

    Token IDs
      ↓
    Vectors


          ↓


    FILE 3

    Attention

    Q/K/V
      ↓
    RoPE
      ↓
    Q/K Norm
      ↓
    GQA
      ↓
    Causal Attention
      ↓
    Attention Output


          ↓


    FILE 4

    Feed Forward

    RMSNorm
      ↓
    SwiGLU
      ↓
    Residual


          ↓


    NEXT TRANSFORMER LAYER


# 56. EXAMPLE WITH "THE DOG"

Suppose the user enters:

    "the dog"


------------------------------------------------------------
Step 1
------------------------------------------------------------

Tokenizer:

    "the dog"

becomes something similar to:

    ["the</w>", "dog</w>"]


------------------------------------------------------------
Step 2
------------------------------------------------------------

Vocabulary lookup:

    ["the</w>", "dog</w>"]

becomes:

    [28, 12]


------------------------------------------------------------
Step 3
------------------------------------------------------------

Embedding:

    [28, 12]

becomes:

    vectors


For example:

    [
      [0.2, -0.1, ...],
      [0.5,  0.3, ...]
    ]


------------------------------------------------------------
Step 4
------------------------------------------------------------

Attention creates:

    Q
    K
    V


------------------------------------------------------------
Step 5
------------------------------------------------------------

RoPE adds positional information to Q and K.


------------------------------------------------------------
Step 6
------------------------------------------------------------

GQA expands the smaller number of K/V heads to match Q heads.


------------------------------------------------------------
Step 7
------------------------------------------------------------

Attention calculates:

    QKᵀ


then:

    QKᵀ / sqrt(head_dim)


then:

    causal mask


then:

    softmax


then:

    attention_weights × V


------------------------------------------------------------
Step 8
------------------------------------------------------------

Attention output goes through:

    output projection


------------------------------------------------------------
Step 9
------------------------------------------------------------

Feed Forward:

    RMSNorm
       ↓
    gate_proj
       ↓
    SiLU

and:

    up_proj

then:

    SiLU(gate) * up


then:

    down_proj


------------------------------------------------------------
Step 10
------------------------------------------------------------

Residual connection:

    output =
        input + FFN_output


------------------------------------------------------------
Step 11
------------------------------------------------------------

The Transformer eventually produces logits for the vocabulary.


For example:

    is       → 0.61
    likes    → 0.24
    happy    → 0.07
    cat      → 0.03
    ...


The model can choose:

    "is"


Then the new sequence becomes:

    "the dog is"


The process repeats.


# 57. TENSOR SHAPE CHEAT SHEET

Assume:

    batch_size = 2

    sequence_length = 10

    hidden_size = 128

    attention_heads = 16

    kv_heads = 4

    head_dim = 8


Input hidden states:

    [2, 10, 128]


After Q projection:

    [2, 10, 128]


After reshape:

    Q:

    [2, 16, 10, 8]


K:

    [2, 4, 10, 8]


V:

    [2, 4, 10, 8]


After GQA:

    K:

    [2, 16, 10, 8]


V:

    [2, 16, 10, 8]


Attention scores:

    [2, 16, 10, 10]


Attention output:

    [2, 16, 10, 8]


After combining heads:

    [2, 10, 128]


After output projection:

    [2, 10, 128]


Feed Forward intermediate representation:

    [2, 10, intermediate_size]


Final FFN output:

    [2, 10, 128]


# 58. IMPORTANT PYTORCH OPERATIONS

Several PyTorch operations in these files are worth understanding deeply.


------------------------------------------------------------
torch.matmul()
------------------------------------------------------------

Used for matrix multiplication.


Example:

    A @ B


or:

    torch.matmul(A, B)


In attention:

    Q @ Kᵀ


calculates similarity scores.


------------------------------------------------------------
torch.transpose()
------------------------------------------------------------

Changes the order of tensor dimensions.


For example:

    [B, S, H, D]


can become:

    [B, H, S, D]


This is important for attention.


------------------------------------------------------------
torch.view()
------------------------------------------------------------

Changes how the tensor is interpreted without changing the underlying number of values.


For example:

    [2, 10, 128]


can become:

    [2, 10, 16, 8]


because:

    128 = 16 × 8


------------------------------------------------------------
torch.reshape()
------------------------------------------------------------

Similar to view, but can handle cases where the underlying memory layout does not allow a direct view.


------------------------------------------------------------
torch.unsqueeze()
------------------------------------------------------------

Adds a dimension.


Example:

    [10]


becomes:

    [1, 10]


------------------------------------------------------------
torch.expand()
------------------------------------------------------------

Creates a broadcasted view with expanded dimensions.


This is useful when repeating K/V heads for GQA.


------------------------------------------------------------
torch.outer()
------------------------------------------------------------

Computes an outer product.


If:

    a = [1, 2, 3]

and:

    b = [4, 5]


then:

    outer(a, b)


is:

    [
      [4, 5],
      [8, 10],
      [12, 15]
    ]


The RoPE implementation uses this idea to create position/frequency combinations.


# 59. IMPORTANT DIFFERENCE: * VS @

This is one of the most important PyTorch concepts.


    *


means element-wise multiplication.


Example:

    [1, 2, 3] * [4, 5, 6]

becomes:

    [4, 10, 18]


While:

    @


means matrix multiplication.


For example:

    A @ B


In the attention code:

    Q @ Kᵀ


is matrix multiplication.


In SwiGLU:

    activated_gate * up_output


is element-wise multiplication.


# 60. WHAT IS SOFTMAX?

Softmax converts numbers into a probability distribution.


Suppose:

    logits = [2, 1, 0]


Softmax produces approximately:

    [0.665, 0.245, 0.090]


Notice:

    0.665 + 0.245 + 0.090 ≈ 1


The largest logit gets the largest probability.


Softmax is used both conceptually in:

    Attention

and later in:

    Next-token prediction


# 61. ATTENTION PROBABILITIES VS TOKEN PROBABILITIES

Do not confuse these two.


Attention probabilities answer:

    "Which previous tokens should I pay attention to?"


Token probabilities answer:

    "Which token should I generate next?"


For example:


Attention:

    "dog" token may attend to:

    the   → 0.2
    dog   → 0.5
    is    → 0.3


Token prediction:

    next token:

    is       → 0.61
    likes    → 0.24
    happy    → 0.07


These are different probability distributions.


# 62. SIMPLIFIED IMPLEMENTATION VS REAL LLM

This distinction is extremely important.


The repository is designed to implement concepts.

It is NOT a complete implementation of the production LLM model.


The implementation simplifies many things so that the concepts are easier to understand.


For example, the attention file uses:

    hidden_size = 128

    attention_heads = 16

    kv_heads = 4

    max_position_embeddings = 256


These are small implementation values.


A production model uses much larger dimensions and additional engineering.


# 63. SIMPLIFIED RoPE

The file explicitly describes its RoPE implementation as:

    Simplified


It demonstrates the mathematical concept and implementation mechanics.


Production Llama 4 has additional behavior around positional embeddings.


Therefore:

    Do not copy this simplified RoPE implementation
    and assume it is a drop-in production LLM implementation.


The file's goal is:

    Understand how RoPE works.


# 64. SIMPLIFIED ATTENTION MASK

The file creates a simple causal mask.


The comments explicitly note that actual Llama 4 masking can be more complex.


The simplified mask is essentially:

    future position → -inf

which is enough to understand causal attention.


Production implementations may need to consider:

    - padding
    - cache length
    - different attention modes
    - local/global attention
    - longer context
    - implementation-specific optimizations


# 65. QK NORMALIZATION TERMINOLOGY

The file calls the operation:

    L2 normalization


but the demonstrated formula is:

    x * rsqrt(mean(x²) + epsilon)


This is worth noticing.


For learning purposes, the important concept is:

    Normalize Q and K before calculating
    their attention similarity.


Do not get too distracted by the naming at this stage.


# 66. RMSNORM VS LAYERNORM

The Feed Forward file discusses normalization and uses:

    RMSNorm


RMSNorm is different from traditional LayerNorm.


LayerNorm generally normalizes based on:

    mean
    variance


RMSNorm uses the root mean square of the values and does not require subtracting the mean in the same way.


For this project, remember:

    Llama-family architectures use RMSNorm-style normalization.


# 67. WHY DOES THE FEED FORWARD NETWORK EXIST?

Attention answers:

    "Which information from other tokens is relevant?"


The Feed Forward Network then transforms each token's representation.


A useful mental model is:

    Attention:

    Communication between tokens


    Feed Forward:

    Processing of each token's representation


The Transformer alternates these operations.


# 68. ATTENTION VS FEED FORWARD

Think of a classroom.


Attention:

    Students talk to each other.


Feed Forward:

    Each student thinks about what
    they just heard and processes it.


Then:

    Attention
       ↓
    Feed Forward
       ↓
    Attention
       ↓
    Feed Forward
       ↓
    ...


This repeated processing is what gives Transformers substantial representational power.


# 69. THE COMPLETE TRANSFORMER MENTAL MODEL

A Transformer layer can be remembered as:


                Input
                  │
                  ▼
               RMSNorm
                  │
                  ▼
              Attention
                  │
                  ▼
              Residual
                  │
                  ▼
               RMSNorm
                  │
                  ▼
               SwiGLU
                  │
                  ▼
              Residual
                  │
                  ▼
                Output


And multiple layers are stacked:

    Layer 1
       ↓
    Layer 2
       ↓
    Layer 3
       ↓
      ...
       ↓
    Layer N


# 70. WHAT THE MODEL ACTUALLY LEARNS

During training, the model learns the parameters of:

    q_proj
    k_proj
    v_proj
    o_proj

and:

    gate_proj
    up_proj
    down_proj

as well as:

    RMSNorm weights

and many other parameters in a complete model.


Initially these parameters are essentially random.


Training changes them so that the model becomes better at predicting the next token.


# 71. WHY THIS MATTERS

When you see:

    model.generate("The dog")


it can look like magic.


But internally, the process is approximately:


    "The dog"

        ↓

    Tokenizer

        ↓

    Token IDs

        ↓

    Embeddings

        ↓

    Transformer layers

        ↓

    Attention

        ↓

    Feed Forward

        ↓

    Logits

        ↓

    Probability distribution

        ↓

    Pick next token

        ↓

    Add token to sequence

        ↓

    Run again


There is no single "generate sentence" operation inside the neural network.


It repeatedly predicts the next token.


# 72. HOW TO STUDY THESE FILES

Do not read the files from top to bottom trying to memorize every line.


Instead use this order:


STEP 1

Understand:

    What is a token?


STEP 2

Understand:

    What is BPE?


STEP 3

Run:

    tokenizer.py


STEP 4

Look at:

    vocab

    word_splits

    pair_stats

    merges


STEP 5

Understand:

    Token ID


STEP 6

Move to attention.


STEP 7

Understand:

    Q
    K
    V


STEP 8

Understand:

    head
    head_dim


STEP 9

Understand:

    GQA


STEP 10

Understand:

    RoPE


STEP 11

Understand:

    causal masking


STEP 12

Understand:

    QKᵀ


STEP 13

Understand:

    softmax


STEP 14

Understand:

    weighted sum of V


STEP 15

Move to Feed Forward.


STEP 16

Understand:

    RMSNorm


STEP 17

Understand:

    gate_proj


STEP 18

Understand:

    up_proj


STEP 19

Understand:

    SiLU


STEP 20

Understand:

    element-wise multiplication


STEP 21

Understand:

    down_proj


STEP 22

Understand:

    residual connection


# 73. RECOMMENDED DEBUGGING APPROACH

When learning neural networks, always print tensor shapes.


For example:

    print(hidden_states.shape)

    print(query_states.shape)

    print(key_states.shape)

    print(value_states.shape)

    print(attention_weights.shape)

    print(attn_output.shape)


For this file, the shapes are often more useful than the actual numerical values.


Ask yourself:

    Why is this dimension 16?

    Why is this dimension 4?

    Why did 128 become 16 × 8?

    Why is attention_weights 10 × 10?

    Why does the final output return to 128?


If you can answer those questions, you are understanding the implementation.


# 74. KEY SHAPES TO MEMORIZE

General hidden state:

    [B, S, D]


where:

    B = batch
    S = sequence length
    D = hidden size


Attention:

    [B, H, S, Dh]


where:

    H  = number of attention heads

    Dh = head dimension


And:

    D = H × Dh


Attention scores:

    [B, H, S, S]


That final:

    S × S


is extremely important.


It represents relationships between:

    every query position

and:

    every key position.


# 75. COMMON BEGINNER CONFUSIONS


CONFUSION 1:

    "Why does the model need Q, K and V?"

ANSWER:

    Q and K determine relevance.

    V contains the information being aggregated.


------------------------------------------------------------

CONFUSION 2:

    "Why do we need multiple heads?"

ANSWER:

    Multiple heads allow different attention projections
    to learn different relationships.


------------------------------------------------------------

CONFUSION 3:

    "Why are there fewer K/V heads?"

ANSWER:

    GQA allows multiple Q heads to share K/V heads,
    reducing K/V memory requirements.


------------------------------------------------------------

CONFUSION 4:

    "Why do we need RoPE?"

ANSWER:

    Attention needs positional information.


------------------------------------------------------------

CONFUSION 5:

    "Why do we need a causal mask?"

ANSWER:

    The model must not see future tokens when predicting
    the next token.


------------------------------------------------------------

CONFUSION 6:

    "Why do we need Feed Forward after attention?"

ANSWER:

    Attention allows information exchange between tokens.

    Feed Forward performs additional nonlinear processing
    on each token representation.


------------------------------------------------------------

CONFUSION 7:

    "Why do we need residual connections?"

ANSWER:

    They provide a direct path for information and gradients
    through deep networks.


# 76. COMMON ENGINEERING MISTAKES

When implementing attention, watch for:

    - Incorrect tensor dimensions
    - Wrong transpose
    - Wrong head dimension
    - Incorrect number of K/V heads
    - Applying RoPE to the wrong dimension
    - Applying RoPE to V
    - Incorrect causal mask
    - Softmax over the wrong dimension
    - Forgetting to scale attention scores
    - Incorrect GQA repetition


When implementing SwiGLU, watch for:

    - Confusing * with @
    - Applying SiLU to the wrong branch
    - Incorrect intermediate dimension
    - Incorrect projection order
    - Forgetting the residual connection


When implementing BPE, watch for:

    - Incorrect pair counting
    - Incorrect merge replacement
    - Losing word boundaries
    - Not storing merge rules
    - Vocabulary/index mismatch


# 77. WHAT THIS REPOSITORY DOES NOT IMPLEMENT

These three files do not constitute a complete production Llama 4 model.


They do not provide the entire:

    tokenizer
    +
    embedding
    +
    transformer stack
    +
    output head
    +
    training pipeline
    +
    inference engine


Instead they focus on three important building blocks:

    1. Tokenization

    2. Attention

    3. Feed Forward Network


A complete model requires additional components.


# 78. WHAT IS STILL NEEDED FOR A COMPLETE LLM?

To turn these files into a complete language model, we would need:


    Tokenizer
       ↓
    Token Embedding
       ↓
    Transformer Block
       │
       ├── RMSNorm
       ├── Attention
       ├── Residual
       ├── RMSNorm
       ├── Feed Forward
       └── Residual
       ↓
    Repeat N times
       ↓
    Final RMSNorm
       ↓
    Language Model Head
       ↓
    Logits
       ↓
    Softmax
       ↓
    Token Selection
       ↓
    Generated Text


For training we would additionally need:

    Dataset
       ↓
    Batching
       ↓
    Input / Target sequences
       ↓
    Forward pass
       ↓
    Cross Entropy Loss
       ↓
    Backpropagation
       ↓
    Optimizer
       ↓
    Weight updates
       ↓
    Checkpoint


# 79. SIMPLIFIED MODEL VS PRODUCTION MODEL

This is perhaps the most important engineering file.


The code in this repository is intentionally:

    small
    readable
    verbose
    heavily commented


Production LLM implementations are typically:

    optimized
    distributed
    memory efficient
    hardware aware
    heavily vectorized
    checkpoint compatible
    inference optimized


Therefore:

    Do not judge the simplified implementation
    by production engineering standards.


Instead ask:

    "What concept is this code trying to implement?"


# 80. RELATIONSHIP TO REAL LLAMA 4

The files are inspired by real Llama 4 architectural concepts.


Real Llama 4 implementations contain concepts such as:

    - Transformer layers
    - RMSNorm
    - GQA
    - RoPE
    - SwiGLU-style feed-forward components
    - causal attention
    - residual connections


However, real Llama 4 contains substantially more engineering and architectural detail than these files.


For example, production implementations include additional handling for:

    - model configuration
    - large-scale dimensions
    - long-context behavior
    - attention masks
    - inference caching
    - distributed execution
    - memory optimization
    - model checkpoint loading
    - hardware acceleration
    - specialized attention behavior


Therefore this project should be viewed as:

    "Llama 4 concepts explained from scratch"


rather than:

    "A complete Llama 4 implementation"


# 81. THE MOST IMPORTANT EQUATION

For attention, remember:

    Attention(Q,K,V)
        =
    softmax(
        QKᵀ / sqrt(d)
    )V


Break it into steps:


    Q
     │
     │
     ├─────── Kᵀ
     │
     ▼
    QKᵀ
     │
     ▼
    Scale
     │
     ▼
    Mask
     │
     ▼
    Softmax
     │
     ▼
    Attention probabilities
     │
     ▼
    × V
     │
     ▼
    Attention output


If you understand this equation and the tensor shapes behind it, you understand the core of Transformer attention.


# 82. THE MOST IMPORTANT SWIGLU EQUATION

Remember:

    FFN(x) =
        W_down(
            SiLU(W_gate(x))
            *
            W_up(x)
        )


Break it into:

    x
    │
    ├───────────────┐
    │               │
    ▼               ▼
    W_gate          W_up
    │               │
    ▼               │
    SiLU             │
    │               │
    └───────*────────┘
            │
            ▼
          W_down
            │
            ▼
          output


If you understand this equation, you understand the core of the Feed Forward portion of the file.


# 83. THE MOST IMPORTANT BPE IDEA

Remember:

    Start with small pieces

         ↓

    Count adjacent pairs

         ↓

    Find most frequent pair

         ↓

    Merge pair

         ↓

    Add new token

         ↓

    Repeat


For example:

    i + s
      ↓
     is


then perhaps:

    t + h
      ↓
     th


and later:

    th + is
       ↓
      this


The tokenizer gradually learns useful subword units.


# 84. THE THREE MOST IMPORTANT MENTAL MODELS


BPE:

    "What pieces of text should I use?"


Attention:

    "Which other tokens should I pay attention to?"


Feed Forward:

    "How should I transform this token representation?"


Together:

    Tokenizer
        ↓
    Attention
        ↓
    Feed Forward


form important pieces of the LLM.


# 85. FINAL MENTAL MODEL

If you remember only one diagram from this README, remember this:


    "The dog is happy"
             │
             ▼
        TOKENIZER
             │
             ▼
       [token IDs]
             │
             ▼
        EMBEDDINGS
             │
             ▼
        ┌───────────┐
        │ ATTENTION │
        │           │
        │ Q         │
        │ K         │
        │ V         │
        │           │
        │ RoPE      │
        │ GQA       │
        │ Mask      │
        │ Softmax   │
        └─────┬─────┘
              │
              ▼
          RESIDUAL
              │
              ▼
        ┌───────────┐
        │   FFN     │
        │           │
        │ RMSNorm   │
        │ Gate      │
        │ SiLU      │
        │ Up        │
        │ Multiply  │
        │ Down      │
        └─────┬─────┘
              │
              ▼
          RESIDUAL
              │
              ▼
        NEXT LAYER
              │
              ▼
           LOGITS
              │
              ▼
           SOFTMAX
              │
              ▼
        NEXT TOKEN
              │
              ▼
        REPEAT


# 86. RECOMMENDED NEXT STEP

After understanding these three files, the next useful step is to build a small complete model around them.


The progression should be:


    STEP 1

    tokenizer.py

        ↓

    STEP 2

    attention.py

        ↓

    STEP 3

    feedforward.py

        ↓

    STEP 4

    transformer.py

        ↓

    STEP 5

    train.py

        ↓

    STEP 6

    mini_llama.pt

        ↓

    STEP 7

    generate.py

        ↓

    STEP 8

    debug.py


At that point you will have gone from:

    "I understand individual LLM components"


to:

    "I can trace a token through an entire
     Transformer and generate text."


# 87. FINAL TAKEAWAY

The three files implements three fundamental parts of an LLM.


TOKENIZER:

    Text
      ↓
    Tokens
      ↓
    Numbers


ATTENTION:

    Numbers
      ↓
    Q/K/V
      ↓
    Relationships between tokens


FEED FORWARD:

    Token representation
      ↓
    Normalize
      ↓
    Expand
      ↓
    Gate
      ↓
    Transform
      ↓
    Compress
      ↓
    Residual


And the overall language-model idea is:


    Given:

        "The dog"


    Predict:

        "is"


    Then:

        "The dog is"


    Predict:

        "happy"


    Then:

        "The dog is happy"


    Predict the next token again.


That repeated next-token prediction is the fundamental behavior of an autoregressive language model.


# 88. ONE-SENTENCE SUMMARY

The tokenizer converts text into numbers, attention lets token representations interact with each other, and the Feed Forward Network transforms those representations so that the Transformer can ultimately predict the next token.
