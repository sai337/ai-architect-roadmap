# Transformers & Deep Learning Foundations

## A Technical Guide for Cloud Architects Entering AI/ML

> **Audience**: Experienced Cloud Architect who understands distributed systems, APIs, and infrastructure — now upskilling into the ML/AI stack.
>
> **Philosophy**: We explain HOW things work, not just WHAT they do. Every concept builds on the last.

---

## Table of Contents

1. [Neural Network Fundamentals](#1-neural-network-fundamentals)
2. [Key Architectures Before Transformers](#2-key-architectures-before-transformers)
3. [The Transformer Architecture (Deep Dive)](#3-the-transformer-architecture-deep-dive)
4. [Model Families & Architectures](#4-model-families--architectures)
5. [Training Concepts](#5-training-concepts)
6. [Scaling Laws](#6-scaling-laws)
7. [Fine-Tuning Spectrum](#7-fine-tuning-spectrum)
8. [Inference Optimization](#8-inference-optimization)
9. [Alignment: Making Models Helpful](#9-alignment-making-models-helpful)
10. [Practical Exercises & Learning Path](#10-practical-exercises--learning-path)

---

## 1. Neural Network Fundamentals

### 1.1 The Neuron — A Weighted Decision Maker

Think of a neuron like a load balancer that makes a binary routing decision:

```
                    ┌─────────────────────────┐
  x₁ ──(w₁)──┐    │                         │
              │    │  z = w₁x₁ + w₂x₂ + b   │
  x₂ ──(w₂)──┼───▶│                         │──▶ a = σ(z) ──▶ output
              │    │  Apply activation: σ(z)  │
  x₃ ──(w₃)──┘    │                         │
              +b   └─────────────────────────┘
```

**Cloud analogy**: A neuron is like a weighted scoring function that takes multiple metrics (CPU %, memory, latency), multiplies each by a learned importance weight, sums them, and passes through a threshold function to decide "scale up? yes/no."

**The math**:
```
z = Σ(wᵢ · xᵢ) + b     ← weighted sum + bias
a = activation(z)        ← non-linear transformation
```

### 1.2 Activation Functions — Why Non-Linearity Matters

Without activation functions, stacking layers gives you nothing — 100 linear transformations collapse into 1 linear transformation. Non-linearity lets networks learn curved decision boundaries.

```
ReLU (most common):          Sigmoid (historical):       GELU (transformers):
                                                         
  y│    /                    y│      ___──              y│      __──
   │   /                      │   __/                    │    _/
   │  /                       │  /                       │  _/
───┼─/──────── x            ──┼─/──────── x           ──┼─/──────── x
   │                          │                          │/
   │                          │                         /│

f(x) = max(0, x)          f(x) = 1/(1+e⁻ˣ)         f(x) = x·Φ(x)
                                                     (smooth ReLU)
```

| Activation | Used In | Why |
|------------|---------|-----|
| **ReLU** | CNNs, general | Fast, avoids vanishing gradients |
| **GELU** | Transformers (GPT, BERT) | Smooth, probabilistic gating |
| **SiLU/Swish** | Llama, modern LLMs | x·sigmoid(x), smooth like GELU |
| **Softmax** | Output layers (classification) | Converts logits → probabilities |

### 1.3 Layers — Composable Feature Extractors

```
Input Layer        Hidden Layer 1      Hidden Layer 2      Output Layer
(raw features)     (low-level)         (high-level)        (prediction)

  ○                   ○                   ○                   ○
  ○   ──────────▶     ○   ──────────▶     ○   ──────────▶     ○
  ○                   ○                   ○                   ○
  ○                   ○                   ○
                      ○

 768 dims            2048 dims           2048 dims           vocab_size
 (e.g., pixels)     (edges, textures)   (shapes, parts)    (cat/dog/...)
```

**Key insight**: Each layer transforms its input into a more useful representation. Layer 1 might detect edges, Layer 2 combines edges into shapes, Layer 3 recognizes objects. The network learns WHAT to extract at each level.

### 1.4 Backpropagation — The Learning Algorithm

**Cloud analogy**: Backpropagation is like a post-mortem analysis after an incident. You trace the error backward through the pipeline to identify which component contributed most to the failure, then adjust each component proportionally.

```
FORWARD PASS (inference):
Input ──▶ Layer 1 ──▶ Layer 2 ──▶ Layer 3 ──▶ Prediction ──▶ Loss
                                                               │
                                                          Compare with
                                                          correct answer
BACKWARD PASS (learning):                                      │
                                                               ▼
Input ◀── Layer 1 ◀── Layer 2 ◀── Layer 3 ◀── Prediction ◀── ∂Loss
          ∂w₁           ∂w₂          ∂w₃                    (gradient)
          ↓              ↓            ↓
      w₁ -= lr·∂w₁  w₂ -= lr·∂w₂  w₃ -= lr·∂w₃

       Each weight is nudged proportional to its contribution to the error
```

**The chain rule in action**:
```
∂Loss/∂w₁ = ∂Loss/∂output × ∂output/∂layer3 × ∂layer3/∂layer2 × ∂layer2/∂w₁
             └─────────────────────────────────────────────────────────────────┘
                          Chain of partial derivatives (the "backpropagation")
```

**Why this matters for scale**: Gradients can vanish (multiply many small numbers → 0) or explode (multiply many large numbers → ∞) in deep networks. This is exactly the problem that:
- Residual connections solve (Section 3)
- Layer normalization addresses (Section 3)
- Careful initialization and learning rate schedules manage (Section 5)

---

## 2. Key Architectures Before Transformers

### 2.1 CNNs — Convolutional Neural Networks

**What they solve**: Processing grid-structured data (images) efficiently.

```
Input Image      Conv Filter     Feature Map      Pooling         Flatten → FC
  28×28          3×3 kernel      26×26           13×13            → prediction
 ┌──────┐       ┌───┐          ┌──────┐        ┌────┐
 │      │       │░░░│          │      │        │    │
 │ 🐈   │  ★    │░░░│    =     │edges │   ▶    │    │  ▶  [cat: 0.92]
 │      │       │░░░│          │      │        │    │     [dog: 0.08]
 └──────┘       └───┘          └──────┘        └────┘

 Key insight: The same small filter slides across the ENTIRE image.
 This is "weight sharing" — one filter detects one pattern everywhere.
```

**Why CNNs work**:
- **Translation invariance**: A cat in the top-left is still a cat
- **Hierarchical features**: Edge → Texture → Part → Object
- **Parameter efficiency**: A 3×3 filter has 9 params regardless of image size

**Limitations for language**: Words aren't spatially local. "The cat that sat on the mat was hungry" — "cat" and "hungry" are far apart but semantically linked. CNNs can't easily capture this.

### 2.2 RNNs/LSTMs — Sequential Processing

**What they solve**: Processing sequences (text, time series) one step at a time.

```
RNN: Process words sequentially, maintaining a "hidden state"

 "The"    "cat"    "sat"    "on"     "the"    "mat"
   │        │        │        │        │        │
   ▼        ▼        ▼        ▼        ▼        ▼
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│ h₁  │─▶│ h₂  │─▶│ h₃  │─▶│ h₄  │─▶│ h₅  │─▶│ h₆  │──▶ output
└─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘

Each hₜ = f(hₜ₋₁, xₜ)  ← current state depends on previous state + new input
```

**LSTM (Long Short-Term Memory)**: Adds "gates" to control what to remember/forget:

```
┌─────────────────────────────────────────────────┐
│                   LSTM Cell                       │
│                                                  │
│  Cell State (long-term memory) ─────────────▶    │
│       │              │              │             │
│    ┌──┴──┐       ┌──┴──┐       ┌──┴──┐          │
│    │Forget│       │Input │       │Output│         │
│    │ Gate │       │ Gate │       │ Gate │         │
│    └──┬──┘       └──┬──┘       └──┬──┘          │
│       │              │              │             │
│   "What to        "What new      "What to        │
│    discard"        info to add"   output now"     │
└─────────────────────────────────────────────────┘
```

### 2.3 Why Transformers Replaced RNNs

| Problem with RNNs | How Transformers Fix It |
|---|---|
| **Sequential processing** — can't parallelize. Token 50 must wait for tokens 1-49. | **Parallel** — all tokens processed simultaneously via attention |
| **Vanishing gradients** — distant tokens "fade" from memory. 500 tokens back is nearly invisible. | **Direct connections** — every token attends to every other token directly |
| **Bottleneck** — all context crammed into fixed-size hidden state vector | **No bottleneck** — each token maintains its own representation |
| **Slow training** — O(n) sequential steps | **Fast training** — O(1) parallel depth, O(n²) attention (manageable with hardware) |

**The key insight**: Instead of processing text left-to-right and hoping context survives, just let every word LOOK AT every other word directly.

---

## 3. The Transformer Architecture (Deep Dive)

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRANSFORMER                                    │
│                                                                       │
│  ┌──────────────────────┐         ┌──────────────────────┐          │
│  │       ENCODER         │         │       DECODER         │          │
│  │                       │         │                       │          │
│  │  ┌─────────────────┐ │         │  ┌─────────────────┐ │          │
│  │  │   Feed-Forward   │ │    ×N   │  │   Feed-Forward   │ │   ×N    │
│  │  ├─────────────────┤ │         │  ├─────────────────┤ │          │
│  │  │  Add & LayerNorm │ │         │  │  Add & LayerNorm │ │          │
│  │  ├─────────────────┤ │         │  ├─────────────────┤ │          │
│  │  │  Self-Attention   │ │         │  │  Cross-Attention  │ │          │
│  │  ├─────────────────┤ │         │  ├─────────────────┤ │          │
│  │  │  Add & LayerNorm │ │         │  │  Add & LayerNorm │ │          │
│  │  ├─────────────────┤ │         │  ├─────────────────┤ │          │
│  │  │  Self-Attention   │ │         │  │ Masked Self-Attn  │ │          │
│  │  ├─────────────────┤ │         │  ├─────────────────┤ │          │
│  │  │  Add & LayerNorm │ │         │  │  Add & LayerNorm │ │          │
│  │  └─────────────────┘ │         │  └─────────────────┘ │          │
│  │          ▲            │         │          ▲            │          │
│  │  ┌───────────────┐   │         │  ┌───────────────┐   │          │
│  │  │ Pos. Encoding  │   │         │  │ Pos. Encoding  │   │          │
│  │  │ + Embedding    │   │         │  │ + Embedding    │   │          │
│  │  └───────────────┘   │         │  └───────────────┘   │          │
│  │          ▲            │         │          ▲            │          │
│  └──────────│────────────┘         └──────────│────────────┘          │
│        Input tokens              Output tokens (shifted right)         │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Self-Attention — The Core Innovation

**Intuition**: For each word, compute how much attention it should pay to every other word. "The animal didn't cross the street because **it** was too tired" — "it" needs to attend heavily to "animal" to resolve the reference.

#### Step-by-Step with a Concrete Example

Let's process: `["I", "love", "ML"]` (3 tokens, embedding dim = 4 for simplicity)

**Step 1: Input Embeddings**

Each token becomes a vector:
```
X = ┌ 1.0  0.5  0.3  0.2 ┐   ← "I"
    │ 0.2  1.0  0.7  0.1 │   ← "love"
    └ 0.8  0.3  1.0  0.9 ┘   ← "ML"
    
    (3 tokens × 4 dimensions)
```

**Step 2: Create Q, K, V matrices via learned linear projections**

```
Q = X · Wq     K = X · Wk     V = X · Wv

Where Wq, Wk, Wv are learned weight matrices (4×4 in this example)
```

Think of these as:
- **Q (Query)**: "What am I looking for?" — like a database query
- **K (Key)**: "What do I contain?" — like a database index key
- **V (Value)**: "What information do I provide?" — like the actual data

```
Cloud analogy:
┌───────────────────────────────────────────────────────────┐
│  Token "it" generates Query: "I need to find my referent" │
│  Token "animal" has Key: "I'm a noun, a subject"          │
│  Token "animal" has Value: [semantic content of 'animal'] │
│                                                           │
│  Q·Kᵀ = high score → "it" attends to "animal"            │
│  Output = weighted sum of Values                          │
└───────────────────────────────────────────────────────────┘
```

**Step 3: Compute Attention Scores**

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √dₖ) · V

Step by step:

1. Q · Kᵀ = raw attention scores (how similar is each query to each key?)

        K₁    K₂    K₃
   Q₁ [ 2.1   1.3   0.8 ]   ← "I" attends most to itself
   Q₂ [ 0.9   2.5   1.7 ]   ← "love" attends most to itself
   Q₃ [ 1.1   1.8   2.3 ]   ← "ML" attends most to itself

2. Divide by √dₖ (= √4 = 2) — prevents scores from getting too large

        K₁    K₂    K₃
   Q₁ [ 1.05  0.65  0.40 ]
   Q₂ [ 0.45  1.25  0.85 ]
   Q₃ [ 0.55  0.90  1.15 ]

3. Apply softmax (each ROW sums to 1.0) — convert to probabilities

        K₁    K₂    K₃
   Q₁ [ 0.43  0.29  0.28 ]   ← "I" pays 43% attention to "I"
   Q₂ [ 0.21  0.45  0.34 ]   ← "love" pays 45% attention to "love"
   Q₃ [ 0.22  0.32  0.46 ]   ← "ML" pays 46% attention to "ML"

4. Multiply by V — weighted combination of value vectors

   Output₁ = 0.43·V₁ + 0.29·V₂ + 0.28·V₃   ← new representation for "I"
   Output₂ = 0.21·V₁ + 0.45·V₂ + 0.34·V₃   ← new representation for "love"
   Output₃ = 0.22·V₁ + 0.32·V₂ + 0.46·V₃   ← new representation for "ML"
```

**Why √dₖ scaling?** Without it, dot products grow with dimension size, pushing softmax into extreme regions where gradients vanish. It's variance normalization.

### 3.3 Multi-Head Attention — Parallel Perspectives

**Intuition**: One attention head might focus on syntactic relationships ("what's the subject?"), another on semantic similarity ("what words mean similar things?"), another on positional proximity.

```
┌──────────────────────────────────────────────────────────────┐
│                    MULTI-HEAD ATTENTION                        │
│                                                               │
│   Input X                                                     │
│     │                                                         │
│     ├──────────┬──────────┬──────────┬──────────┐            │
│     ▼          ▼          ▼          ▼          │            │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐       │            │
│  │Head 1│  │Head 2│  │Head 3│  │Head 4│  ...   │   h heads  │
│  │      │  │      │  │      │  │      │       │            │
│  │Syntax│  │Coref │  │Seman-│  │Posit-│       │            │
│  │rels  │  │rence │  │tics  │  │ional │       │            │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘       │            │
│     │          │          │          │          │            │
│     └──────────┴──────────┴──────────┘          │            │
│                    │                             │            │
│                    ▼                             │            │
│            ┌─────────────┐                      │            │
│            │ Concatenate  │                      │            │
│            │   + Linear   │                      │            │
│            │  Projection  │                      │            │
│            └─────────────┘                      │            │
│                    │                             │            │
│                    ▼                             │            │
│               Output (same dim as input)         │            │
└──────────────────────────────────────────────────────────────┘
```

**Implementation detail**:
```
If model_dim = 768 and num_heads = 12:
  - Each head works with dim 768/12 = 64
  - Each head has its own Wq, Wk, Wv of size (768 × 64)
  - All 12 heads run in PARALLEL (on GPU, this is a single matmul)
  - Outputs concatenated: 12 × 64 = 768, then projected back to 768
```

**What different heads learn** (empirically observed):
- Head A: Attends to the previous word (bigram patterns)
- Head B: Attends to the subject of the sentence
- Head C: Attends to semantically similar words
- Head D: Attends to separator/punctuation tokens
- Head E: Attends to the beginning of the sequence

### 3.4 Positional Encoding — Giving Order to Parallel Processing

Since attention processes all tokens simultaneously (no left-to-right), we must inject position information. Without it, "dog bites man" = "man bites dog".

#### Sinusoidal Positional Encoding (Original Transformer, 2017)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Position 0: [sin(0), cos(0), sin(0), cos(0), ...]  = [0, 1, 0, 1, ...]
Position 1: [sin(1), cos(1), sin(1/10000), cos(1/10000), ...]
Position 2: [sin(2), cos(2), sin(2/10000), cos(2/10000), ...]

Different dimensions oscillate at different frequencies:
  dim 0,1:  high frequency (changes every position)
  dim 2,3:  medium frequency
  dim d-2,d-1: very low frequency (like a "carry bit")
```

**Why this works**: Relative positions can be computed via linear transformation. PE(pos+k) can be expressed as a linear function of PE(pos), enabling the model to learn relative position patterns.

#### RoPE — Rotary Position Embedding (Llama, Mistral)

```
Key idea: Encode position by ROTATING the Q and K vectors

For a 2D subspace of Q/K at position m:
  [q₀]     [cos(mθ)  -sin(mθ)] [q₀]
  [q₁]  →  [sin(mθ)   cos(mθ)] [q₁]

The dot product between positions m and n naturally encodes (m-n):
  <Rotate(q,m), Rotate(k,n)> = f(q, k, m-n)
                                       └─ only depends on RELATIVE position!
```

**Advantages of RoPE**:
- Naturally captures relative positions (not absolute)
- Decays with distance (far-away tokens naturally get less attention)
- Can extrapolate to longer sequences than trained on (with NTK-aware scaling)

#### ALiBi — Attention with Linear Biases (BLOOM)

```
Simplest approach: Just subtract a linear penalty from attention scores

attention_score(i,j) = q_i · k_j - m · |i - j|
                                    └─ penalty proportional to distance

Head 1: m = 1/1   (strong locality bias)
Head 2: m = 1/2   (moderate locality)  
Head 3: m = 1/4   (weak locality)
Head 8: m = 1/128 (nearly global attention)

No learned parameters! Just a geometric sequence of slopes across heads.
```

**Comparison**:

| Method | Position Type | Extrapolation | Used In |
|--------|--------------|---------------|---------|
| Sinusoidal | Absolute | Poor | Original Transformer |
| Learned | Absolute | Poor | GPT-2, BERT |
| RoPE | Relative | Good (with scaling) | Llama, Mistral, Gemma |
| ALiBi | Relative (bias) | Excellent | BLOOM, MPT |

### 3.5 Layer Normalization & Residual Connections

#### Residual Connections — The Highway System

```
                    ┌───────────────┐
          ┌────────│  Skip/Residual │────────┐
          │        │  Connection    │        │
          │        └───────────────┘        │
          │                                  │
Input ────┤                                  ├───▶ Output = Input + SubLayer(Input)
          │                                  │
          │        ┌───────────────┐        │
          └───────▶│  Sub-Layer     │────────┘
                   │  (Attention or │
                   │   FFN)         │
                   └───────────────┘
```

**Why residual connections are critical**:
1. **Gradient highway**: Gradients flow directly backward without being multiplied by layer weights — prevents vanishing gradients in deep networks (GPT-4 has 120+ layers!)
2. **Identity initialization**: At the start of training, layers learn small perturbations around identity. Network starts as "just pass through" and gradually learns useful transformations.
3. **Ensemble effect**: Each layer adds to the representation rather than replacing it. The output is a sum of contributions from all previous layers.

#### Layer Normalization

```
Pre-norm (modern: GPT-2+, Llama):     Post-norm (original transformer):

   Input                                  Input
     │                                      │
     ▼                                      ▼
  LayerNorm                              SubLayer
     │                                      │
     ▼                                      ▼
  SubLayer                                 Add (residual)
     │                                      │
     ▼                                      ▼
   Add (residual)                        LayerNorm
     │                                      │
     ▼                                      ▼
  Output                                 Output
```

**LayerNorm formula**:
```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β

Where:
  μ = mean across the feature dimension (per token)
  σ = std across the feature dimension
  γ, β = learned scale and shift parameters
  ε = small constant for numerical stability (1e-5)
```

**Cloud analogy**: LayerNorm is like auto-scaling that normalizes all metrics to the same range before combining them. Without it, one "loud" dimension would dominate all the others.

**Pre-norm vs Post-norm**: Pre-norm is more stable during training (the input to each sublayer is always normalized), allowing larger learning rates and easier training of very deep models. Almost all modern LLMs use pre-norm.

### 3.6 Feed-Forward Networks (FFN) — Per-Token Processing

After attention mixes information BETWEEN tokens, the FFN processes each token INDEPENDENTLY:

```
┌──────────────────────────────────────────────────────────┐
│              Feed-Forward Network (per token)              │
│                                                           │
│  Input (d_model = 768)                                    │
│         │                                                 │
│         ▼                                                 │
│  ┌─────────────────┐                                      │
│  │ Linear: 768 → 3072 │  (expand 4×)                     │
│  └────────┬────────┘                                      │
│           │                                               │
│           ▼                                               │
│  ┌─────────────────┐                                      │
│  │ Activation (GELU) │                                    │
│  └────────┬────────┘                                      │
│           │                                               │
│           ▼                                               │
│  ┌─────────────────┐                                      │
│  │ Linear: 3072 → 768 │  (compress back)                 │
│  └────────┬────────┘                                      │
│           │                                               │
│           ▼                                               │
│  Output (d_model = 768)                                   │
└──────────────────────────────────────────────────────────┘
```

**Modern variant — SwiGLU (Llama, Mistral)**:
```
FFN_SwiGLU(x) = (W₁·x ⊗ SiLU(W_gate·x)) · W₂

The "gate" learns which neurons to activate — like a learned ReLU.
This outperforms standard FFNs at the same parameter count.
```

**What the FFN does**: Attention is the "communication" step (tokens share info). FFN is the "thinking" step (each token processes its gathered information). Research suggests factual knowledge is primarily stored in FFN weights.

### 3.7 Encoder vs Decoder vs Encoder-Decoder

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ENCODER-ONLY (BERT)           DECODER-ONLY (GPT)                       │
│  ─────────────────            ─────────────────                          │
│                                                                          │
│  Bidirectional attention:      Causal (masked) attention:                │
│                                                                          │
│  "The cat sat"                 "The cat sat"                             │
│   The → sees [The,cat,sat]      The → sees [The]                        │
│   cat → sees [The,cat,sat]      cat → sees [The,cat]                    │
│   sat → sees [The,cat,sat]      sat → sees [The,cat,sat]                │
│                                                                          │
│  Attention mask:               Attention mask:                           │
│  [1 1 1]                       [1 0 0]                                   │
│  [1 1 1]                       [1 1 0]                                   │
│  [1 1 1]                       [1 1 1]                                   │
│                                                                          │
│  Best for: understanding       Best for: generation                      │
│  (classification, NER,         (text gen, code gen,                      │
│   retrieval, embeddings)        chat, reasoning)                         │
│                                                                          │
│                                                                          │
│  ENCODER-DECODER (T5)                                                    │
│  ─────────────────────                                                   │
│                                                                          │
│  Encoder: bidirectional (sees full input)                                │
│  Decoder: causal + cross-attention to encoder                            │
│                                                                          │
│  Input: "Translate English to French: The cat sat"                       │
│         └─────── encoder processes this ──────┘                          │
│                          │                                               │
│                    cross-attention                                        │
│                          │                                               │
│  Output: "Le" → "chat" → "s'est" → "assis"                             │
│          └─── decoder generates this autoregressively ───┘               │
│                                                                          │
│  Best for: seq2seq (translation, summarization)                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**The causal mask in decoder-only models**:
```python
# Why causal masking? During training, we process the whole sequence at once
# but each position should only "see" previous positions (no cheating!)

mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * -inf

# Before softmax, add this mask to attention scores:
# Position 3 trying to attend to Position 5: score + (-inf) = -inf → softmax → 0
```

---

## 4. Model Families & Architectures

### 4.1 GPT Family (Decoder-Only, Autoregressive)

```
GPT Architecture:
┌─────────────────────────────┐
│  Token + Position Embedding  │
├─────────────────────────────┤
│  ┌────────────────────────┐ │
│  │ Masked Self-Attention   │ │
│  │ + Residual + LayerNorm  │ │  × N layers
│  │ Feed-Forward Network    │ │
│  │ + Residual + LayerNorm  │ │
│  └────────────────────────┘ │
├─────────────────────────────┤
│  Linear → Vocab Logits      │
│  → Softmax → Next Token     │
└─────────────────────────────┘
```

**Training objective**: Predict the next token given all previous tokens.

```
Input:  "The cat sat on the"
Target: "cat sat on the mat"

P(cat | The) × P(sat | The cat) × P(on | The cat sat) × ...
```

| Model | Params | Layers | d_model | Heads | Context |
|-------|--------|--------|---------|-------|---------|
| GPT-2 | 1.5B | 48 | 1600 | 25 | 1024 |
| GPT-3 | 175B | 96 | 12288 | 96 | 2048 |
| GPT-4 | ~1.8T (MoE) | ~120 | ~? | ~? | 128K |

### 4.2 BERT Family (Encoder-Only, MLM)

**Training objective**: Masked Language Modeling — predict randomly masked tokens.

```
Input:  "The [MASK] sat on the [MASK]"
Target: predict "cat" and "mat"

Unlike GPT, BERT sees the FULL context (bidirectional).
This makes it excellent for understanding but terrible for generation.
```

**Variants**:
- **RoBERTa**: BERT trained longer, with more data, no NSP objective
- **DeBERTa**: Disentangled attention (separate content and position)
- **ALBERT**: Parameter sharing across layers (efficient BERT)

### 4.3 T5/BART (Encoder-Decoder, Seq2Seq)

**T5 — "Text-to-Text Transfer Transformer"**:
Every task framed as text → text:
```
Classification: "sentiment: I love this movie"     → "positive"
Translation:    "translate en to de: Hello world"   → "Hallo Welt"  
Summarization:  "summarize: [long article]"         → "[short summary]"
QA:             "question: What is... context: ..." → "The answer is..."
```

**BART**: Denoising autoencoder. Corrupt input in various ways, learn to reconstruct.

### 4.4 Modern Architectures — Key Innovations

#### Llama (Meta, 2023-2024)

```
Llama innovations over vanilla Transformer:
┌─────────────────────────────────────────────┐
│ 1. RoPE positional encoding                  │
│    (relative positions, better extrapolation)│
│                                              │
│ 2. SwiGLU activation in FFN                  │
│    FFN(x) = SiLU(xW₁) ⊗ (xW₃) × W₂       │
│    (gated activation, 4/3 × intermediate)    │
│                                              │
│ 3. RMSNorm instead of LayerNorm              │
│    RMSNorm(x) = x / RMS(x) × γ             │
│    (no mean subtraction, 15% faster)         │
│                                              │
│ 4. Grouped Query Attention (GQA)             │
│    Multiple Q heads share fewer K,V heads    │
│    (reduces KV-cache memory by 4-8×)         │
│                                              │
│ 5. Pre-norm architecture                     │
│    (normalize before attention/FFN)          │
└─────────────────────────────────────────────┘
```

**Grouped Query Attention (GQA) — Critical for Inference**:
```
Multi-Head Attention (MHA):         Grouped Query Attention (GQA):
32 Q heads, 32 K heads, 32 V heads  32 Q heads, 8 K heads, 8 V heads

Q: [h1][h2][h3]...[h32]            Q: [h1][h2][h3][h4] [h5]...[h8]...[h32]
K: [h1][h2][h3]...[h32]            K: [g1]             [g2]  ...  [g8]
V: [h1][h2][h3]...[h32]            V: [g1]             [g2]  ...  [g8]

Every Q head has its own K,V         4 Q heads share 1 K,V group
KV-cache: 32× head_dim × 2          KV-cache: 8× head_dim × 2 (4× smaller!)
```

#### Mistral (2023-2024)

```
Key innovations:
1. Sliding Window Attention (SWA)
   - Each token only attends to W nearby tokens (e.g., W=4096)
   - But with multiple layers, effective receptive field grows:
     Layer 1: each token sees 4096 tokens
     Layer 2: each token indirectly sees 8192 tokens (via Layer 1 outputs)
     Layer 32: effective context = 32 × 4096 = 131,072

2. Rolling KV-Cache
   - Only store W positions in cache (not full sequence)
   - Dramatically reduces memory for long sequences
```

#### Mixture of Experts (MoE) — Mixtral, GPT-4

```
┌──────────────────────────────────────────────────┐
│            Mixture of Experts FFN                  │
│                                                   │
│  Input x                                          │
│     │                                             │
│     ├──────▶ Router Network ──▶ Top-K selection   │
│     │         (learned gate)     (e.g., K=2)      │
│     │              │                              │
│     │         [0.1, 0.7, 0.05, 0.02, 0.6, ...]  │
│     │              │                              │
│     │         Select top-2: Expert 2, Expert 5    │
│     │              │                              │
│     ├──────▶ Expert 2 (FFN) ──┐                  │
│     │                          ├──▶ 0.7·E₂ + 0.6·E₅ = output
│     └──────▶ Expert 5 (FFN) ──┘                  │
│                                                   │
│  8 experts total, but only 2 active per token     │
│  Total params: 8× FFN, but compute cost: ~2× FFN │
└──────────────────────────────────────────────────┘
```

**Why MoE**: You get a model with 8× the "knowledge capacity" (parameters) but only pay ~2× the compute cost per token. This is how GPT-4 can have ~1.8T total parameters but still be fast.

#### Claude (Anthropic) — Known Innovations

- Constitutional AI training (self-critique during RLHF)
- Very long context windows (200K tokens)
- Likely uses techniques similar to Llama (RoPE, GQA, SwiGLU) but details are proprietary

#### Gemini (Google) — Known Innovations

- Native multimodal (text, image, audio, video in one model)
- Likely MoE architecture
- Very large context (1M+ tokens via techniques like ring attention)

---

## 5. Training Concepts

### 5.1 Loss Functions

#### Cross-Entropy Loss (The Standard for Language Models)

```
For next-token prediction:

Vocabulary = [the, cat, sat, on, mat, ...]  (50,000+ tokens)

Model predicts probability distribution over ALL tokens:
  P(next | "The cat") = [the:0.01, cat:0.02, sat:0.60, on:0.05, mat:0.01, ...]

True answer (one-hot): [0, 0, 1, 0, 0, ...]  (it was "sat")

Cross-entropy loss = -log(P(correct token))
                   = -log(0.60)
                   = 0.51

If model was very confident:  -log(0.99) = 0.01  (low loss ✓)
If model was very wrong:      -log(0.01) = 4.61  (high loss ✗)
```

**Perplexity** (commonly reported metric):
```
Perplexity = e^(average cross-entropy loss)
           = e^0.51 = 1.67

Interpretation: The model is "confused" between ~1.67 choices on average.
Lower perplexity = better model.
GPT-4 level: perplexity ~5-10 on typical benchmarks.
```

### 5.2 Optimizers

#### Adam & AdamW

```
Standard SGD:          θ = θ - lr × gradient
                       Problem: same learning rate for all parameters,
                       noisy gradients cause oscillation

Adam:                  Maintains per-parameter:
                       m = moving average of gradient (momentum)
                       v = moving average of gradient² (adaptive rate)
                       
                       θ = θ - lr × m / (√v + ε)
                       
                       Parameters with large/frequent gradients get smaller updates
                       Parameters with small/rare gradients get larger updates

AdamW:                 Adam + decoupled weight decay
                       θ = θ × (1 - wd) - lr × m / (√v + ε)
                                  └─ weight decay applied to weights directly,
                                     not to the gradient (better regularization)
```

**Practical defaults for LLM training**:
```python
optimizer = AdamW(
    params,
    lr=3e-4,          # peak learning rate
    betas=(0.9, 0.95), # momentum terms
    eps=1e-8,          # numerical stability
    weight_decay=0.1   # regularization
)
```

### 5.3 Learning Rate Schedules

```
Learning Rate over Training Steps:

lr │
   │   ╱╲
   │  ╱  ╲
   │ ╱    ╲
   │╱      ╲
   │ Warmup  ╲──── Cosine Decay ────────▶
   │           ╲
   │            ╲
   │             ╲________________________  (min_lr = 0.1 × max_lr)
   └──────────────────────────────────────── steps
   0  2000                              100000

Warmup (first ~2000 steps):
  - Start from near-zero LR, linearly increase
  - Why: Adam's moving averages aren't calibrated yet
  - Cold start with high LR = divergence

Cosine Decay (rest of training):
  - Smoothly decrease LR following cosine curve
  - Why: As model converges, smaller updates prevent oscillation
  - Typically decays to 10% of peak LR
```

### 5.4 Batch Size & Gradient Accumulation

```
Problem: GPU memory can fit batch_size=2, but optimal is batch_size=256

Solution: Gradient Accumulation

┌─ Step 1: Forward+Backward on micro_batch=2, store gradients ─┐
├─ Step 2: Forward+Backward on micro_batch=2, accumulate grads  ─┤
├─ Step 3: Forward+Backward on micro_batch=2, accumulate grads  ─┤
│  ... (128 accumulation steps)                                  │
├─ Step 128: Forward+Backward on micro_batch=2, accumulate grads─┤
└─ Now: Apply optimizer update with effective_batch = 2×128 = 256 ┘

Same math as true batch=256, but fits in GPU memory!
Trade-off: 128× slower per optimizer step (but same final result)
```

**Practical batch sizes for LLM training**:
- GPT-3: 3.2M tokens per batch (gradually increased during training)
- Llama 2: 4M tokens per batch
- This requires distributed training across many GPUs

### 5.5 Mixed Precision Training

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRECISION FORMATS                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FP32 (Full precision):  1 sign + 8 exponent + 23 mantissa     │
│  ████████████████████████████████  (32 bits, baseline)           │
│  Range: ±3.4×10³⁸, Precision: 7 decimal digits                 │
│                                                                  │
│  FP16 (Half precision):  1 sign + 5 exponent + 10 mantissa     │
│  ████████████████  (16 bits, 2× memory savings)                  │
│  Range: ±65504, Precision: 3.3 decimal digits                   │
│  Problem: small range → overflow/underflow common               │
│                                                                  │
│  BF16 (Brain Float):    1 sign + 8 exponent + 7 mantissa       │
│  ████████████████  (16 bits, same range as FP32!)                │
│  Range: ±3.4×10³⁸, Precision: 2.3 decimal digits               │
│  Best of both: large range + 2× memory savings                  │
│  ★ Used in most modern LLM training (Llama, GPT-4, etc.)       │
│                                                                  │
│  FP8 (Quarter precision): 1 sign + 4/5 exp + 3/2 mantissa      │
│  ████████  (8 bits, 4× savings)                                  │
│  Used in: inference, some training with careful scaling          │
│  Emerging: H100/H200 native FP8 support                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Mixed Precision Training Strategy:
┌────────────────────────────────────────────────┐
│  Master weights:     FP32  (stored, updated)   │
│  Forward pass:       BF16  (fast matmuls)      │
│  Backward pass:      BF16  (fast matmuls)      │
│  Gradient accum:     FP32  (precision needed)  │
│  Optimizer states:   FP32  (Adam needs this)   │
└────────────────────────────────────────────────┘

Result: ~2× training speed, ~1.5× less memory, same quality!
```

---

## 6. Scaling Laws

### 6.1 The Neural Scaling Laws (Kaplan et al., 2020)

```
Key finding: Loss improves as a POWER LAW with three factors:

  L(N, D, C) ∝ 1/Nᵅ + 1/Dᵝ + constant

  N = number of parameters
  D = dataset size (tokens)
  C = compute budget (FLOPs)

  α ≈ 0.076 (for params)
  β ≈ 0.095 (for data)

Loss │
     │╲
     │ ╲
     │  ╲
     │   ╲
     │    ╲──────────────────  ← diminishing returns
     │                           but NEVER plateaus!
     └──────────────────────── log(Compute)

Key insight: A 10× bigger model trained on same data = better
             than a 1× model trained on 10× more data
             (This turned out to be WRONG — see Chinchilla)
```

### 6.2 Chinchilla Scaling (Hoffmann et al., 2022)

```
Chinchilla's finding: The Kaplan scaling laws over-prioritized parameters!

COMPUTE-OPTIMAL TRAINING:
  Optimal tokens ≈ 20 × parameters

  ┌──────────────────────────────────────────────────┐
  │  Model Size    │  Optimal Training Tokens          │
  ├──────────────────────────────────────────────────┤
  │  1B params     │  ~20B tokens                      │
  │  7B params     │  ~140B tokens (but Llama used 2T!)│
  │  70B params    │  ~1.4T tokens                     │
  │  175B params   │  ~3.5T tokens                     │
  │  Chinchilla:   │  70B params, 1.4T tokens          │
  │  Gopher:       │  280B params, 300B tokens ← WRONG │
  └──────────────────────────────────────────────────┘

  Chinchilla (70B) OUTPERFORMED Gopher (280B) because it was
  trained on proportionally more data!
```

**Post-Chinchilla reality** (2023-2024):
```
Modern models are "over-trained" on purpose:
  - Llama 2 7B: trained on 2T tokens (vs. optimal ~140B)
  - Llama 3 8B: trained on 15T tokens (100× "optimal"!)

Why "over-train"?
  - Inference cost dominates. A 7B model that's very well-trained
    is cheaper to serve than a compute-optimal 30B model.
  - Training is a one-time cost; inference runs forever.
  - "Inference-optimal" ≠ "Compute-optimal during training"
```

### 6.3 Emergent Abilities

```
Some capabilities appear "suddenly" at certain scales:

Performance │
            │                          ●
            │                        ●│
            │                      ●  │
            │                    ●    │  ← Sudden "emergence"
            │         ●  ●  ●  ●     │     (3-shot arithmetic, chain-of-thought)
            │   ●  ●                  │
            │●                        │
            └─────────────────────────── Model Size (log scale)
                                    ~100B

Examples of emergent abilities:
  - Multi-step reasoning (>100B params)
  - Few-shot learning without fine-tuning (>10B params)
  - Self-correction and chain-of-thought (>50B params)
```

---

## 7. Fine-Tuning Spectrum

### From Most to Least Expensive

```
┌───────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  ◄──── MORE EXPENSIVE / MORE POWERFUL ───── CHEAPER / MORE LIMITED ────►  │
│                                                                            │
│  Full        LoRA      QLoRA     Prefix     Prompt     In-Context         │
│  Fine-tune              Tuning   Tuning     Tuning     Learning            │
│                                                                            │
│  Update      Update    LoRA on   Prepend    Learn soft  Just provide       │
│  ALL         low-rank  quantized learned    embedding   examples in        │
│  parameters  adapters  model     "prefix"   "prompts"  the prompt          │
│                                  vectors    vectors                         │
│                                                                            │
│  Params:     Params:   Params:   Params:    Params:    Params:             │
│  100%        0.1-1%    0.1-1%    0.01%      0.001%     0%                  │
│                                                                            │
│  Memory:     Memory:   Memory:   Memory:    Memory:    Memory:             │
│  Full model  Full model 4-bit    Full model Full model Inference only      │
│  + gradients + adapters model    + prefix   + prompts                      │
│  + optimizer           + adapters                                          │
│                                                                            │
│  7B model:   7B model: 7B model: 7B model:  7B model:  7B model:          │
│  ~120GB      ~16GB     ~6GB      ~16GB      ~16GB      ~14GB (inf)        │
│  (A100 80GB  (1×A100)  (1×RTX    (1×A100)   (1×A100)   (API call)         │
│   ×2+)                  4090!)                                             │
└───────────────────────────────────────────────────────────────────────────┘
```

### 7.1 Full Fine-Tuning

Update every parameter in the model. Requires storing: model weights + gradients + optimizer states (Adam stores 2× model size in moving averages).

```
Memory requirement for full fine-tuning:
  Model params (BF16):     2 bytes × N
  Gradients (BF16):        2 bytes × N  
  Optimizer states (FP32): 8 bytes × N (Adam: m + v + master weights)
  Total:                   ~12 bytes × N

  7B model:  7×10⁹ × 12 = 84 GB  (+ activations ≈ 120 GB)
  70B model: 70×10⁹ × 12 = 840 GB (needs multi-node!)
```

### 7.2 LoRA — Low-Rank Adaptation

**Key insight**: Weight updates during fine-tuning are LOW RANK. Instead of updating a full d×d weight matrix, decompose the update into two small matrices.

```
Original weight matrix W (4096 × 4096 = 16.7M parameters):

Instead of:  W_new = W + ΔW   (ΔW is 4096 × 4096 = 16.7M params to learn)

LoRA does:   W_new = W + A × B
             where A is (4096 × r) and B is (r × 4096)
             with rank r = 8, 16, 32, or 64

┌────────────────────────────────────────────────────┐
│                                                     │
│   x ─────────┬──────────────────────── (+) ──▶ y   │
│              │          frozen W               ▲    │
│              │                                 │    │
│              │    ┌─────────┐  ┌─────────┐    │    │
│              └───▶│  A (↓r) │─▶│  B (r↑) │────┘    │
│                   │4096 × 16│  │16 × 4096│          │
│                   └─────────┘  └─────────┘          │
│                   └── trainable params ──┘           │
│                   = 4096×16 + 16×4096               │
│                   = 131K params (vs 16.7M!)         │
│                   = 0.8% of original                │
│                                                     │
└────────────────────────────────────────────────────┘
```

**Applied to**: Usually Q, K, V, and output projection matrices in attention layers.

### 7.3 QLoRA — Quantized LoRA

```
QLoRA = LoRA, but the frozen base model is quantized to 4-bit!

Normal LoRA:   Base model (BF16, 14GB for 7B) + LoRA adapters (BF16, ~20MB)
QLoRA:         Base model (NF4, 3.5GB for 7B) + LoRA adapters (BF16, ~20MB)

Key innovations:
1. NF4 (Normal Float 4-bit): Quantization format optimized for neural net 
   weight distributions (normally distributed → optimal 4-bit bins)

2. Double quantization: Even the quantization constants are quantized!
   (Saves additional ~0.5 GB for a 7B model)

3. Paged optimizers: Offload optimizer states to CPU when GPU runs out

Result: Fine-tune a 65B model on a single 48GB GPU! (vs. 780GB normally)
```

### 7.4 Prefix Tuning & Prompt Tuning

```
Prefix Tuning:
  Prepend learnable "virtual tokens" to every layer's key/value

  Normal input:       [This, movie, was, great]
  With prefix:        [P₁, P₂, P₃, P₄, This, movie, was, great]
                       └─ learned ─┘    └──── frozen model input ────┘
  
  The prefixes are different at EACH LAYER (not just input)
  ~0.01% of params, but modifies the attention pattern throughout

Prompt Tuning:
  Even simpler — learnable embeddings ONLY at the input layer

  [Soft₁, Soft₂, ..., Soft₂₀, <actual input tokens>]
   └── learned embedding vectors (not real words) ──┘

  Surprisingly effective for large models (>10B params)
```

### 7.5 In-Context Learning (ICL)

```
No training at all! Just provide examples in the prompt:

┌─────────────────────────────────────────────────┐
│ Classify the sentiment:                          │
│                                                  │
│ Text: "I love this!" → Positive                  │  ← few-shot examples
│ Text: "Terrible service" → Negative              │
│ Text: "It was okay" → Neutral                    │
│                                                  │
│ Text: "Best purchase ever!" →                    │  ← model generates
└─────────────────────────────────────────────────┘

Why this works: Large models have learned a meta-algorithm
for pattern matching during pre-training. The examples
"program" the model without changing any weights.
```

---

## 8. Inference Optimization

### 8.1 KV-Cache — Avoiding Redundant Computation

**The problem**: Autoregressive generation recomputes attention for all previous tokens at every step.

```
Without KV-Cache (naive):
  Step 1: "The"           → compute attention → predict "cat"
  Step 2: "The cat"       → recompute ALL attention → predict "sat"
  Step 3: "The cat sat"   → recompute ALL attention → predict "on"
  Step N: "The cat sat.." → recompute ALL attention → predict next
  
  Cost per token: O(n) where n = sequence length so far
  Total cost for N tokens: O(N²) — THIS IS TERRIBLE

With KV-Cache:
  Step 1: "The" → compute K₁,V₁, store in cache → predict "cat"
  Step 2: "cat" → compute K₂,V₂, append to cache
           Attention uses [K₁,K₂] and [V₁,V₂] → predict "sat"
  Step 3: "sat" → compute K₃,V₃, append to cache  
           Attention uses [K₁,K₂,K₃] and [V₁,V₂,V₃] → predict "on"

  Only compute new token's Q, K, V — reuse all previous K, V!
  Cost per token: O(n) for the attention lookup, but only O(1) new computation
```

**KV-Cache memory**:
```
Per token in cache:
  2 (K and V) × num_layers × num_kv_heads × head_dim × bytes_per_element

For Llama 2 70B at FP16:
  2 × 80 layers × 8 KV heads × 128 dim × 2 bytes = 327 KB per token
  
  At 4096 tokens: 327KB × 4096 = 1.3 GB (just for KV-cache!)
  This is why GQA (fewer KV heads) is critical for serving.
```

### 8.2 Speculative Decoding — Parallel Generation

```
Problem: Large model generates 1 token at a time (sequential, slow)
Idea: Use a SMALL model to "draft" multiple tokens, then verify in parallel

┌────────────────────────────────────────────────────────────────┐
│                    SPECULATIVE DECODING                          │
│                                                                  │
│  Draft model (small, fast):                                      │
│    "The cat" → drafts: ["sat", "on", "the", "mat"]              │
│    (4 tokens generated cheaply)                                  │
│                                                                  │
│  Verification (large model, parallel):                           │
│    Process ALL 4 draft tokens simultaneously                     │
│    Check: P_large(token) vs P_small(token)                       │
│                                                                  │
│    "sat" ✓ (accept)                                              │
│    "on"  ✓ (accept)                                              │
│    "the" ✓ (accept)                                              │
│    "mat" ✗ (reject, replace with "warm")                         │
│                                                                  │
│  Result: Generated 3 tokens in 1 large-model forward pass!       │
│  Speedup: ~2-3× in practice                                     │
│                                                                  │
│  Key property: Output is IDENTICAL to large model alone.         │
│  (Rejection sampling ensures mathematical equivalence)           │
└────────────────────────────────────────────────────────────────┘
```

### 8.3 Quantization — Shrinking Model Size

```
┌────────────────────────────────────────────────────────────────────┐
│                     QUANTIZATION METHODS                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Full precision:  w = 0.3827461  (FP32, 32 bits per weight)        │
│                                                                     │
│  8-bit quant:     w ≈ 0.38       (INT8, some quality loss)         │
│  4-bit quant:     w ≈ 0.4        (INT4, noticeable at small scale) │
│                                                                     │
│  Formula: quantized = round((w - min) / (max - min) × (2ⁿ - 1))   │
│                                                                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  GPTQ (Post-Training Quantization):                                │
│  ────                                                              │
│  - Layer-by-layer quantization using calibration data              │
│  - Minimizes output error: ||W·X - Q(W)·X||²                      │
│  - Uses Hessian information to quantize "less important" weights   │
│    more aggressively                                               │
│  - Fast inference on GPU (INT4 × FP16 matmul)                     │
│  - Best for: GPU inference servers                                  │
│                                                                     │
│  AWQ (Activation-Aware Weight Quantization):                        │
│  ───                                                                │
│  - Key insight: 1% of weights matter much more (high activation)    │
│  - Keep important weights at higher precision                       │
│  - Scale weights before quantization to protect important ones      │
│  - Better quality than GPTQ at same bit-width                       │
│  - Best for: Production GPU serving                                 │
│                                                                     │
│  GGUF (llama.cpp format):                                           │
│  ────                                                              │
│  - Designed for CPU inference (+ partial GPU offload)               │
│  - Multiple quant levels in one file (important layers get more     │
│    bits)                                                            │
│  - Supports: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0                  │
│  - Best for: Local inference on consumer hardware                   │
│                                                                     │
├────────────────────────────────────────────────────────────────────┤
│  Practical comparison (Llama 2 70B):                                │
│                                                                     │
│  Method  │ Bits │ Size   │ Quality │ Speed  │ Hardware               │
│  ────────┼──────┼────────┼─────────┼────────┼──────────             │
│  FP16    │ 16   │ 140 GB │ 100%    │ 1×     │ 2× A100 80GB         │
│  GPTQ    │ 4    │ 35 GB  │ ~97%    │ 1.5×   │ 1× A100 80GB         │
│  AWQ     │ 4    │ 35 GB  │ ~98%    │ 1.7×   │ 1× A100 80GB         │
│  GGUF Q4 │ 4    │ 40 GB  │ ~96%    │ 0.3×   │ CPU + 24GB GPU       │
│  GGUF Q2 │ 2    │ 25 GB  │ ~88%    │ 0.4×   │ 32GB RAM (CPU only)  │
└────────────────────────────────────────────────────────────────────┘
```

### 8.4 Other Inference Optimizations

```
Continuous Batching:
  Don't wait for all sequences to finish — slot new requests in as others complete.
  Like a restaurant that seats new diners as tables free up, not in fixed seatings.

Flash Attention:
  Standard attention: O(N²) memory (stores full attention matrix)
  Flash Attention: O(N) memory (computes attention in blocks, never materializes full matrix)
  2-4× faster, enables much longer contexts on same hardware.

PagedAttention (vLLM):
  KV-cache memory managed like virtual memory pages.
  Reduces memory waste from 60-80% to <4% (no more pre-allocated contiguous blocks).
  Enables 2-4× more concurrent requests.

Tensor Parallelism:
  Split model across GPUs by cutting weight matrices along hidden dimension.
  Each GPU computes a slice, then all-reduce to combine.
  Llama 70B: typically 4-8 GPUs with tensor parallelism.
```

---

## 9. Alignment: Making Models Helpful

### 9.1 The Alignment Problem

```
Pre-trained LLM (base model):
  Input: "How do I make a cake?"
  Output: "How do I make a cake? What are the ingredients? How long does..."
  (It just CONTINUES text — it's a document completer, not an assistant!)

After alignment:
  Input: "How do I make a cake?"
  Output: "Here's a simple recipe: 1. Preheat oven to 350°F..."
  (Now it's helpful, follows instructions, refuses harmful requests)
```

### 9.2 RLHF — Reinforcement Learning from Human Feedback

```
┌──────────────────────────────────────────────────────────────────────┐
│                          RLHF PIPELINE                                 │
│                                                                        │
│  STEP 1: Supervised Fine-Tuning (SFT)                                 │
│  ────────────────────────────────────                                  │
│  Train on (prompt, ideal_response) pairs from human demonstrations     │
│  This gives you an "okay" chatbot                                      │
│                                                                        │
│  STEP 2: Reward Model Training                                         │
│  ────────────────────────────────                                      │
│                                                                        │
│  Human annotators rank responses:                                      │
│                                                                        │
│  Prompt: "Explain quantum computing"                                   │
│  Response A: [technical, accurate, clear]     ← Rank 1 (best)         │
│  Response B: [accurate but confusing]         ← Rank 2                 │
│  Response C: [wrong, hallucinated]            ← Rank 3 (worst)         │
│                                                                        │
│  Train reward model: RM(prompt, response) → scalar score               │
│  Objective: RM(A) > RM(B) > RM(C)                                     │
│                                                                        │
│  STEP 3: PPO Optimization                                              │
│  ─────────────────────────                                             │
│                                                                        │
│  ┌──────┐     ┌──────────┐     ┌──────────┐                          │
│  │Prompt│────▶│ LLM (π)  │────▶│ Response │                           │
│  └──────┘     └──────────┘     └────┬─────┘                           │
│                    ▲                  │                                 │
│                    │                  ▼                                 │
│              PPO update         ┌──────────┐                           │
│                    │            │  Reward   │                           │
│                    │            │  Model    │                           │
│                    │            └────┬─────┘                           │
│                    │                  │                                 │
│                    └──── reward ──────┘                                 │
│                                                                        │
│  + KL penalty: Don't drift too far from the SFT model                 │
│    (prevents "reward hacking" — gaming the reward model)              │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

**Problems with RLHF**:
- Requires a separate reward model (expensive to train and maintain)
- PPO is unstable (many hyperparameters, reward hacking)
- Human preferences are noisy and inconsistent
- The reward model can be gamed

### 9.3 DPO — Direct Preference Optimization

```
Key insight: We can skip the reward model entirely!

RLHF:  SFT → Train Reward Model → PPO against RM (complex, unstable)
DPO:   SFT → Directly train on preference pairs (simple, stable)

┌──────────────────────────────────────────────────────────────┐
│                         DPO                                    │
│                                                               │
│  Training data: (prompt, chosen_response, rejected_response)  │
│                                                               │
│  Loss = -log σ( β × (                                        │
│      log π(chosen|prompt)/π_ref(chosen|prompt)                │
│    - log π(rejected|prompt)/π_ref(rejected|prompt)            │
│  ))                                                           │
│                                                               │
│  In English:                                                  │
│  "Increase probability of chosen responses relative to       │
│   rejected responses, but don't drift too far from the       │
│   reference (SFT) model"                                     │
│                                                               │
│  β controls how far from reference model you can drift       │
│  Higher β = more conservative updates                         │
│                                                               │
│  Advantages over RLHF:                                       │
│  ✓ No reward model needed                                    │
│  ✓ No RL training (just supervised loss)                     │
│  ✓ More stable optimization                                  │
│  ✓ Simpler to implement (standard cross-entropy style loss)  │
│  ✓ Mathematically equivalent under ideal conditions          │
└──────────────────────────────────────────────────────────────┘
```

### 9.4 ORPO — Odds Ratio Preference Optimization

```
Key insight: Skip SFT entirely! Single-stage alignment.

RLHF:  Pretrain → SFT → RM → PPO    (4 stages)
DPO:   Pretrain → SFT → DPO          (3 stages)
ORPO:  Pretrain → ORPO               (2 stages!)

┌──────────────────────────────────────────────────────────────┐
│                         ORPO                                   │
│                                                               │
│  Loss = L_NLL + λ × L_OR                                     │
│                                                               │
│  L_NLL: Standard language modeling loss on chosen responses   │
│         (replaces SFT step)                                   │
│                                                               │
│  L_OR:  Odds ratio loss                                       │
│         OR = odds(chosen) / odds(rejected)                    │
│         Maximize this ratio                                   │
│                                                               │
│  Combined: Learn to generate well AND prefer good outputs     │
│            in a single training run                            │
│                                                               │
│  Advantages:                                                  │
│  ✓ No separate SFT stage                                     │
│  ✓ No reference model needed during training                 │
│  ✓ Computationally cheapest alignment method                 │
│  ✓ Competitive quality with DPO                              │
└──────────────────────────────────────────────────────────────┘
```

### 9.5 Alignment Comparison

```
Method    │ Stages │ Extra Models    │ Stability │ Quality │ Compute
──────────┼────────┼─────────────────┼───────────┼─────────┼─────────
RLHF/PPO  │   3    │ Reward model    │ Low       │ High    │ Highest
                    │ + ref model     │           │         │
DPO       │   2    │ Reference model │ High      │ High    │ Medium
ORPO      │   1    │ None            │ High      │ Good    │ Lowest
SimPO     │   2    │ None            │ High      │ High    │ Medium
KTO       │   2    │ Reference model │ High      │ Good    │ Medium
          │        │ (pointwise,     │           │         │
          │        │  no pairs!)     │           │         │
```

---

## 10. Practical Exercises & Learning Path

### 10.1 Recommended Learning Path

```
┌─────────────────────────────────────────────────────────────────────┐
│                     4-WEEK LEARNING PATH                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  WEEK 1: Foundations                                                 │
│  ─────────────────                                                   │
│  □ Complete Andrej Karpathy's "Neural Networks: Zero to Hero"        │
│    (YouTube playlist — builds GPT from scratch)                      │
│  □ Implement a simple feed-forward network in PyTorch                │
│  □ Train a character-level language model                            │
│  □ Read: "Attention Is All You Need" (original paper)               │
│                                                                      │
│  WEEK 2: Transformers Deep Dive                                      │
│  ────────────────────────────                                        │
│  □ Implement self-attention from scratch (no libraries)              │
│  □ Build a small GPT (124M) following nanoGPT                        │
│  □ Visualize attention patterns using BertViz                        │
│  □ Read: "The Illustrated Transformer" (Jay Alammar's blog)         │
│  □ Experiment: Compare training with/without positional encoding     │
│                                                                      │
│  WEEK 3: Fine-Tuning & Practical Skills                              │
│  ──────────────────────────────────────                              │
│  □ Fine-tune a small model (GPT-2 or Llama-3-8B) with LoRA          │
│    - Use Hugging Face PEFT + Transformers                            │
│  □ Compare: Full fine-tune vs LoRA vs QLoRA (on same task)           │
│  □ Deploy a quantized model locally (llama.cpp, GGUF)                │
│  □ Benchmark: Measure latency, throughput, quality tradeoffs         │
│                                                                      │
│  WEEK 4: Production & Advanced Topics                                │
│  ────────────────────────────────────                                │
│  □ Set up vLLM serving with continuous batching                      │
│  □ Implement RAG (Retrieval-Augmented Generation)                    │
│  □ Run DPO alignment on a small model                                │
│  □ Profile GPU memory: where does it all go?                         │
│  □ Read: Chinchilla paper, LLM scaling considerations                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.2 Hands-On Exercises

#### Exercise 1: Self-Attention from Scratch

```python
import torch
import torch.nn.functional as F

def self_attention(X, Wq, Wk, Wv):
    """
    X:  (seq_len, d_model) - input embeddings
    Wq, Wk, Wv: (d_model, d_k) - projection matrices
    """
    Q = X @ Wq  # (seq_len, d_k)
    K = X @ Wk  # (seq_len, d_k)
    V = X @ Wv  # (seq_len, d_k)
    
    d_k = Q.shape[-1]
    scores = Q @ K.T / (d_k ** 0.5)  # (seq_len, seq_len)
    weights = F.softmax(scores, dim=-1)
    output = weights @ V  # (seq_len, d_k)
    
    return output, weights

# Try it:
seq_len, d_model, d_k = 4, 8, 4
X = torch.randn(seq_len, d_model)
Wq = torch.randn(d_model, d_k)
Wk = torch.randn(d_model, d_k)
Wv = torch.randn(d_model, d_k)

output, attn_weights = self_attention(X, Wq, Wk, Wv)
print(f"Attention weights (each row sums to 1):\n{attn_weights}")
```

#### Exercise 2: Build a Minimal Transformer Block

```python
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Pre-norm + self-attention + residual
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, attn_mask=mask)
        x = x + attn_out
        
        # Pre-norm + FFN + residual
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out
        
        return x

# Stack N blocks = a transformer!
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, tokens):
        x = self.embed(tokens)
        # Add causal mask
        seq_len = tokens.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.head(x)
```

#### Exercise 3: LoRA Implementation (Conceptual)

```python
class LoRALinear(nn.Module):
    """Replace nn.Linear with LoRA-augmented version"""
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        d_in = original_linear.in_features
        d_out = original_linear.out_features
        
        # Freeze original weights
        self.original = original_linear
        self.original.weight.requires_grad = False
        
        # LoRA matrices (trainable)
        self.A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, d_out))
        self.scale = alpha / rank
    
    def forward(self, x):
        # Original output + low-rank update
        original_out = self.original(x)
        lora_out = (x @ self.A @ self.B) * self.scale
        return original_out + lora_out

# Trainable params: d_in × rank + rank × d_out
# For d_in=d_out=4096, rank=16: 131K vs 16.7M (0.8%)
```

### 10.3 Key Resources

```
Papers (in reading order):
─────────────────────────
1. "Attention Is All You Need" (2017) - The original transformer
2. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
3. "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
4. "Scaling Laws for Neural Language Models" (Kaplan, 2020)
5. "Training Compute-Optimal LLMs" (Chinchilla, 2022)
6. "LoRA: Low-Rank Adaptation" (2021)
7. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
8. "Direct Preference Optimization" (DPO, 2023)
9. "LLaMA: Open and Efficient Foundation Language Models" (2023)
10. "Flash Attention: Fast and Memory-Efficient Exact Attention" (2022)

Courses & Tutorials:
────────────────────
• Andrej Karpathy: "Neural Networks: Zero to Hero" (YouTube)
• Andrej Karpathy: "Let's build GPT from scratch" (YouTube)
• Stanford CS224N: Natural Language Processing with Deep Learning
• Hugging Face NLP Course (free, hands-on)
• Jay Alammar: "The Illustrated Transformer" (blog)

Tools to master:
───────────────
• PyTorch (framework)
• Hugging Face Transformers (model library)
• PEFT (LoRA/QLoRA fine-tuning)
• vLLM (production serving)
• llama.cpp / Ollama (local inference)
• Weights & Biases (experiment tracking)
```

### 10.4 Mental Model: The Full Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE LLM STACK (for Cloud Architects)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  APPLICATION LAYER     │ Chat UI, API, Agent frameworks              │
│  ─────────────────     │ (what you build on top)                     │
│                        │                                             │
│  SERVING LAYER         │ vLLM, TGI, TensorRT-LLM                   │
│  ─────────────         │ Batching, KV-cache, quantization            │
│                        │ ← This is "load balancers + app servers"    │
│                        │                                             │
│  MODEL LAYER           │ The trained weights (Llama, GPT, Claude)    │
│  ───────────           │ Architecture choices (GQA, MoE, context)    │
│                        │ ← This is "the application code"            │
│                        │                                             │
│  TRAINING LAYER        │ Pre-training, Fine-tuning, Alignment        │
│  ──────────────        │ Distributed training, mixed precision       │
│                        │ ← This is "CI/CD + the build pipeline"      │
│                        │                                             │
│  DATA LAYER            │ Pre-training corpus, RLHF annotations       │
│  ──────────            │ Tokenization, data cleaning                 │
│                        │ ← This is "the database + data pipeline"    │
│                        │                                             │
│  INFRASTRUCTURE LAYER  │ GPU clusters (A100, H100), networking       │
│  ────────────────────  │ Storage (model weights, checkpoints)        │
│                        │ ← YOU ALREADY KNOW THIS PART! ✓             │
│                        │                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Key Numbers to Know

| What | Typical Value | Why It Matters |
|------|--------------|----------------|
| GPT-2 params | 1.5B | "Small" by today's standards |
| Llama 3 8B params | 8B | Sweet spot for local/edge deployment |
| Llama 3 70B params | 70B | Production quality, needs multi-GPU |
| GPT-4 params | ~1.8T (MoE, ~280B active) | Frontier, massive infrastructure |
| Typical d_model | 4096-8192 | Hidden dimension size |
| Typical context | 4K-128K tokens | How much text the model "sees" |
| Training tokens (Llama 3) | 15T | Takes months on thousands of GPUs |
| BF16 weight per param | 2 bytes | 7B model = 14GB just for weights |
| FP32 training overhead | ~12× params in bytes | 7B training ≈ 84GB |
| LoRA rank | 8-64 | Higher = more expressive, more memory |
| Chinchilla ratio | 20 tokens per param | Compute-optimal (but modern trains more) |

---

*Last updated: June 2026. The field moves fast — verify current state-of-the-art.*
