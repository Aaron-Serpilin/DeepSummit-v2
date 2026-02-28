# DeepSummit Transformer: Complete Mathematical Guide

This document provides a comprehensive, beginner-friendly explanation of every mathematical operation in the DeepSummit transformer. After reading this guide, you will understand:

1. What each component does and why it exists
2. The exact mathematical operations performed
3. How tensor shapes change through the network
4. How all components connect to form the complete model

---

## Table of Contents

1. [Prerequisites: Essential Concepts](#1-prerequisites-essential-concepts)
2. [Building Block #1: RMSNorm](#2-building-block-1-rmsnorm)
3. [Building Block #2: SwiGLU Feed-Forward Network](#3-building-block-2-swiglu-feed-forward-network)
4. [Building Block #3: DropPath (Stochastic Depth)](#4-building-block-3-droppath-stochastic-depth)
5. [Building Block #4: Multi-Head Attention](#5-building-block-4-multi-head-attention)
6. [Positional Encoding: Time2Vec](#6-positional-encoding-time2vec)
7. [Tokenization: Converting Features to Embeddings](#7-tokenization-converting-features-to-embeddings)
8. [Transformer Block: Combining Components](#8-transformer-block-combining-components)
9. [Transformer Encoder: Stacking Blocks](#9-transformer-encoder-stacking-blocks)
10. [Classification Head: Final Prediction](#10-classification-head-final-prediction)
11. [Complete Data Flow: End-to-End](#11-complete-data-flow-end-to-end)
12. [Shape Reference Summary](#12-shape-reference-summary)

---

## 1. Prerequisites: Essential Concepts

Before diving into the transformer, let's establish some foundational concepts.

### 1.1 Tensor Notation

Throughout this document, we use the following notation:

| Symbol | Meaning | Example |
|--------|---------|---------|
| `B` | Batch size | 32 samples processed together |
| `S` | Sequence length | 47 tokens |
| `H` | Hidden size | 256 dimensions |
| `d_k` | Head dimension | H / num_heads = 32 |
| `n_h` | Number of attention heads | 8 |

A tensor shape `(B, S, H)` means:
- First dimension: batch (how many examples)
- Second dimension: sequence (how many tokens)
- Third dimension: hidden (the embedding size)

### 1.2 What is a Token?

A **token** is a vector representation of a piece of information. In DeepSummit:
- Each expedition feature becomes a token (e.g., "climber age" → 256-dimensional vector)
- Each weather timestep becomes a token
- Tokens are what the transformer processes

### 1.3 Matrix Multiplication Basics

**Matrix-vector multiplication:**
If W is a matrix of shape `(out_features, in_features)` and x is a vector of shape `(in_features,)`:

$$y = Wx$$

The result y has shape `(out_features,)`.

**In PyTorch**, `nn.Linear(in_features, out_features)` computes `y = xW^T + b`.

### 1.4 Broadcasting

When we add tensors of different shapes, PyTorch "broadcasts" the smaller tensor:

```python
# x: (B, S, H) + bias: (H,) = output: (B, S, H)
# The bias is automatically expanded to match x's shape
```

---

## 2. Building Block #1: RMSNorm

### 2.1 What Problem Does Normalization Solve?

Neural networks can suffer from **internal covariate shift** — the distribution of layer inputs changes during training, making optimization difficult. Normalization stabilizes training by ensuring consistent input distributions.

### 2.2 The RMSNorm Formula

**Root Mean Square Normalization** simplifies LayerNorm by removing mean centering:

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

where the Root Mean Square is:

$$\text{RMS}(x) = \sqrt{\frac{1}{H} \sum_{i=1}^{H} x_i^2 + \epsilon}$$

- $x$: Input vector of dimension H
- $\gamma$: Learnable scale parameter (initialized to ones)
- $\epsilon$: Small constant (1e-6) to prevent division by zero

### 2.3 Comparison with LayerNorm

| Operation | LayerNorm | RMSNorm |
|-----------|-----------|---------|
| Mean centering | Yes: $x - \mu$ | No |
| Variance normalization | Yes: $/ \sigma$ | Yes: $/ \text{RMS}$ |
| Learnable scale | Yes: $\gamma$ | Yes: $\gamma$ |
| Learnable bias | Yes: $\beta$ | No |
| Speed | Baseline | 10-15% faster |

### 2.4 Code Implementation with Math Mapping

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # γ, shape: (H,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, S, H)

        # Step 1: Compute x²
        # x.pow(2) → (B, S, H)

        # Step 2: Compute mean(x²) along last dimension
        # .mean(dim=-1, keepdim=True) → (B, S, 1)

        # Step 3: Add epsilon and take sqrt to get RMS
        # torch.sqrt(...) → (B, S, 1)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Step 4: Normalize and scale
        # (x / rms) → (B, S, H)  [broadcasting: (B,S,H) / (B,S,1)]
        # * self.weight → (B, S, H)  [broadcasting: (B,S,H) * (H,)]
        return (x / rms) * self.weight
```

### 2.5 Shape Transformation

```
Input:  x        (B, S, H)      e.g., (32, 47, 256)
        ↓
        x²       (B, S, H)      element-wise square
        ↓
        mean     (B, S, 1)      mean across H dimension
        ↓
        RMS      (B, S, 1)      sqrt(mean + ε)
        ↓
        x/RMS    (B, S, H)      normalize (broadcast division)
        ↓
        ×γ       (B, S, H)      scale (broadcast multiplication)
        ↓
Output:          (B, S, H)      same shape as input
```

### 2.6 Why RMSNorm Works

The key insight is that mean centering (in LayerNorm) adds computational cost without significantly improving performance. Modern transformers like LLaMA and Mistral use RMSNorm and achieve state-of-the-art results.

---

## 3. Building Block #2: SwiGLU Feed-Forward Network

### 3.1 What is a Feed-Forward Network?

In a transformer, each token is processed independently through a feed-forward network (FFN). This allows the model to transform token representations through learned non-linear mappings.

### 3.2 The Gating Mechanism

**SwiGLU** (Swish-Gated Linear Unit) is a gated variant that outperforms standard FFNs:

$$\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_2) W_3$$

where:
- $W_1$: Gate projection (H → ffn_hidden)
- $W_2$: Value projection (H → ffn_hidden)
- $W_3$: Output projection (ffn_hidden → H)
- $\odot$: Element-wise multiplication
- $\text{Swish}(x) = x \cdot \sigma(x)$ (also called SiLU)

### 3.3 Understanding Swish/SiLU Activation

The **Swish** function is a smooth, non-monotonic activation:

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

```
      │
   2  │                    ╱
      │                  ╱
   1  │               ╱
      │             ╱
   0  │───────────●───────────
      │         ╱
  -1  │       ●
      │
      └──────────────────────
        -4  -2   0   2   4
```

Properties:
- Smooth (differentiable everywhere)
- Non-monotonic (has a small dip for negative values)
- Unbounded above, bounded below
- Self-gated: output depends on the input value itself

### 3.4 Why Gating?

The gating mechanism allows the network to selectively pass or block information:

```
Input x → splits into two paths
    │
    ├──→ Gate path: Swish(x·W₁)   ← "How much to let through"
    │
    └──→ Value path: x·W₂         ← "What to let through"

    ↓
    Gate × Value = Selective output
```

If the gate produces ~0, that feature is suppressed.
If the gate produces ~1, that feature passes through.

### 3.5 Code Implementation with Math Mapping

```python
class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int = None, dropout: float = 0.1):
        # Default: ffn_hidden_size = hidden_size × 8/3 ≈ 2.67 × hidden_size
        if ffn_hidden_size is None:
            ffn_hidden_size = int(hidden_size * 8 / 3)

        # W₁: gate projection
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        # W₂: value projection
        self.value_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        # W₃: output projection
        self.output_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, S, H)

        # Gate path: apply Swish activation
        # x·W₁ᵀ → (B, S, ffn_hidden)
        # F.silu applies Swish element-wise
        gate = F.silu(self.gate_proj(x))  # (B, S, ffn_hidden)

        # Value path: no activation
        value = self.value_proj(x)  # (B, S, ffn_hidden)

        # Element-wise gating: gate ⊙ value
        hidden = gate * value  # (B, S, ffn_hidden)

        # Output projection: hidden·W₃ᵀ
        output = self.output_proj(hidden)  # (B, S, H)
        output = self.dropout(output)

        return output
```

### 3.6 Shape Transformation

```
Input:  x               (B, S, H)           e.g., (32, 49, 256)
        │
        ├──→ gate_proj  (B, S, ffn_hidden)  e.g., (32, 49, 682)
        │    ↓
        │    Swish      (B, S, ffn_hidden)  element-wise
        │
        └──→ value_proj (B, S, ffn_hidden)  e.g., (32, 49, 682)

        ↓
        gate × value    (B, S, ffn_hidden)  element-wise multiply
        ↓
        output_proj     (B, S, H)           e.g., (32, 49, 256)
        ↓
        dropout         (B, S, H)
        ↓
Output:                 (B, S, H)           same shape as input, (32, 47, 256)
```

### 3.7 Parameter Count Comparison

For hidden_size H = 256:

| FFN Type | Structure | Parameter Count |
|----------|-----------|-----------------|
| Standard GELU | H → 4H → H | 2 × (256 × 1024) = 524K |
| SwiGLU | H → 2.67H → H (×3 projections) | 3 × (256 × 682) = 524K |

The 8/3 expansion factor keeps parameter counts equivalent.

---

## 4. Building Block #3: DropPath (Stochastic Depth)

### 4.1 What is Stochastic Depth?

**DropPath** randomly "skips" entire residual branches during training. Instead of:

$$y = x + F(x)$$

We have:

$$y = x + \text{DropPath}(F(x))$$

where DropPath outputs either:
- $F(x) / (1-p)$ with probability $(1-p)$ — **keep** (scaled up)
- $0$ with probability $p$ — **drop** (skip this branch)

### 4.2 The Math

During training:
$$\text{DropPath}(x) = \frac{x \cdot m}{1-p}$$

where $m \sim \text{Bernoulli}(1-p)$ is a random mask.

During inference:
$$\text{DropPath}(x) = x$$

The scaling factor $1/(1-p)$ ensures the expected value remains unchanged:
$$\mathbb{E}[\text{DropPath}(x)] = (1-p) \cdot \frac{x}{1-p} + p \cdot 0 = x$$

### 4.3 Why Drop Entire Branches (Not Individual Neurons)?

| Dropout Type | What's Dropped | Effect |
|--------------|----------------|--------|
| Standard Dropout | Individual neurons | Reduces co-adaptation within a layer |
| DropPath | Entire residual branch | Creates ensemble of networks with different depths |

DropPath is especially effective for deep networks because it implicitly trains an ensemble of shallower networks.

### 4.4 Linear Schedule

We use a **linear schedule** where drop probability increases with layer depth:

$$p_i = \frac{i}{L-1} \cdot p_{\text{max}}$$

| Layer | Formula | Drop Prob (if p_max = 0.1) |
|-------|---------|----------------------------|
| 0 | 0/5 × 0.1 | 0.00 |
| 1 | 1/5 × 0.1 | 0.02 |
| 2 | 2/5 × 0.1 | 0.04 |
| 3 | 3/5 × 0.1 | 0.06 |
| 4 | 4/5 × 0.1 | 0.08 |
| 5 | 5/5 × 0.1 | 0.10 |

**Why linear?** Early layers learn fundamental features (keep them). Later layers learn task-specific refinements (safe to regularize).

### 4.5 Code Implementation with Math Mapping

```python
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        self.drop_prob = drop_prob  # p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, S, H) — output of some residual branch

        # During eval or if p=0: return unchanged
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob  # (1-p)

        # Create random mask: shape (B, 1, 1, ...) for broadcasting
        # This ensures entire samples (not individual features) are dropped
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # e.g., (32, 1, 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)

        # Bernoulli mask: 1 if rand < keep_prob, else 0
        mask = (random_tensor < keep_prob).float()  # m

        # Scale and mask: x × m / (1-p)
        output = x * mask / keep_prob

        return output
```

### 4.6 Shape Transformation

```
Input:  x        (B, S, H)      e.g., (32, 47, 256)
        ↓
        mask     (B, 1, 1)      random 0 or 1 per sample
        ↓
        x × mask (B, S, H)      broadcast multiply
        ↓
        ÷ (1-p)  (B, S, H)      scale to maintain expected value
        ↓
Output:          (B, S, H)      same shape, some samples zeroed
```

---

## 5. Building Block #4: Multi-Head Attention

### 5.1 What Problem Does Attention Solve?

Traditional neural networks process each input independently. **Attention** allows the model to look at all positions in the sequence and decide which ones are relevant for processing each position.

### 5.2 Scaled Dot-Product Attention

The core attention mechanism computes a weighted sum of values, where weights come from query-key similarity:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Breaking this down:

1. **Q (Query):** "What am I looking for?"
2. **K (Key):** "What do I contain?"
3. **V (Value):** "What information should I pass?"
4. **QK^T:** Similarity scores between queries and keys
5. **÷√d_k:** Scaling to prevent softmax saturation
6. **softmax:** Convert scores to weights (sum to 1)
7. **× V:** Weighted sum of values

### 5.3 Visual Intuition

```
Query: "What weather patterns relate to success?"
        │
        │ Compare with all keys
        ▼
┌───────────────────────────────────────────────────┐
│ Sequence:                                         │
│ [CLS] [age] [o2] ... [SEP] [7d_w1] [7d_w2] ...    │
│                             ▲       ▲             │
│                             │       │             │
│                      High attention weights       │
│                     (weather is relevant!)        │
└───────────────────────────────────────────────────┘
        │
        ▼
Output: Weighted combination emphasizing weather tokens
```

### 5.4 Multi-Head: Multiple Perspectives

Instead of one attention function, we use **multiple heads** that attend to different aspects:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

where each head is:
$$\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

```
┌─────────────────────────────────────────────────────────┐
│                     Input X (B, S, H)                   │
│                            │                            │
│         ┌──────────────────┼──────────────────┐         │
│         │                  │                  │         │
│         ▼                  ▼                  ▼         │
│    ┌─────────┐       ┌─────────┐       ┌─────────┐      │
│    │ Head 1  │       │ Head 2  │  ...  │ Head 8  │      │
│    │ d_k=32  │       │ d_k=32  │       │ d_k=32  │      │
│    └────┬────┘       └────┬────┘       └────┬────┘      │
│         │                  │                  │         │
│         └──────────────────┼──────────────────┘         │
│                            │                            │
│                      Concatenate                        │
│                    (B, S, 8×32=256)                     │
│                            │                            │
│                            ▼                            │
│                      Output Proj W^O                    │
│                            │                            │
│                            ▼                            │
│                    Output (B, S, H)                     │
└─────────────────────────────────────────────────────────┘
```

### 5.5 Complete Math for Multi-Head Attention

Given input $X$ of shape (B, S, H):

**Step 1: Linear projections**
$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

where $W^Q, W^K, W^V$ have shape (H, H).

**Step 2: Reshape for multi-head**
$$Q, K, V: (B, S, H) \rightarrow (B, n_h, S, d_k)$$

where $d_k = H / n_h$ (head dimension).

**Step 3: Compute attention scores**
$$\text{scores} = \frac{Q K^T}{\sqrt{d_k}}$$

Shape: $(B, n_h, S, d_k) \times (B, n_h, d_k, S) = (B, n_h, S, S)$

**Step 4: Apply softmax**
$$\text{weights} = \text{softmax}(\text{scores}, \text{dim}=-1)$$

Shape: $(B, n_h, S, S)$ — each row sums to 1.

**Step 5: Weighted sum of values**
$$\text{context} = \text{weights} \cdot V$$

Shape: $(B, n_h, S, S) \times (B, n_h, S, d_k) = (B, n_h, S, d_k)$

**Step 6: Concatenate heads**
$$(B, n_h, S, d_k) \rightarrow (B, S, n_h \times d_k) = (B, S, H)$$

**Step 7: Output projection**
$$\text{output} = \text{context} \cdot W^O$$

Shape: $(B, S, H)$

### 5.6 Code Implementation with Math Mapping

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        self.hidden_size = hidden_size      # H = 256
        self.num_heads = num_heads          # n_h = 8
        self.head_dim = hidden_size // num_heads  # d_k = 32

        # W^Q, W^K, W^V: (H, H)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # W^O: (H, H)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        B, S, H = x.shape  # e.g., (32, 49, 256)

        # Step 1: Project to Q, K, V
        # x @ W^Q^T → (B, S, H)
        q = self.q_proj(x)  # (32, 49, 256)
        k = self.k_proj(x)  # (32, 49, 256)
        v = self.v_proj(x)  # (32, 49, 256)

        # Step 2: Reshape for multi-head
        # (B, S, H) → (B, S, n_h, d_k) → (B, n_h, S, d_k)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: q, k, v are (32, 8, 49, 32)

        # Steps 3-5: Scaled dot-product attention (Flash Attention)
        # PyTorch's optimized implementation
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        # attn_output: (32, 8, 49, 32)

        # Step 6: Reshape back
        # (B, n_h, S, d_k) → (B, S, n_h, d_k) → (B, S, H)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, S, H)
        # attn_output: (32, 49, 256)

        # Step 7: Output projection
        output = self.out_proj(attn_output)  # (32, 49, 256)

        return output
```

### 5.7 Shape Transformation Summary

```
Input:  x         (B, S, H)          (32, 47, 256)
        ↓
        Q,K,V     (B, S, H)          (32, 49, 256)
        ↓ reshape
        Q,K,V     (B, n_h, S, d_k)   (32, 8, 47, 32)
        ↓
        scores    (B, n_h, S, S)     (32, 8, 47, 47)  ← Q @ K^T
        ↓ softmax
        weights   (B, n_h, S, S)     (32, 8, 47, 47)  ← attention weights
        ↓
        context   (B, n_h, S, d_k)   (32, 8, 47, 32)  ← weights @ V
        ↓ reshape
        concat    (B, S, H)          (32, 47, 256)
        ↓
Output:           (B, S, H)          (32, 47, 256)
```

### 5.8 Why √d_k Scaling?

Without scaling, dot products grow with dimension:
- If $d_k = 32$, and Q,K entries are ~N(0,1)
- Then $QK^T$ entries have variance ~32
- Large values → softmax saturates → tiny gradients

Dividing by $\sqrt{d_k}$ normalizes variance back to ~1.

---

## 6. Positional Encoding: Time2Vec

### 6.1 Why Temporal Encoding?

Transformers are **permutation invariant** — they don't inherently know the order of tokens. For weather data, temporal ordering is crucial (yesterday's weather matters more than last month's for summit day).

### 6.2 Time2Vec Formula

Time2Vec learns a time representation with both linear and periodic components:

$$\text{Time2Vec}(t) = \begin{bmatrix} \omega_0 t + \phi_0 \\ \sin(\omega_1 t + \phi_1) \\ \vdots \\ \sin(\omega_k t + \phi_k) \end{bmatrix}$$

- **Linear component** $(\omega_0 t + \phi_0)$: Captures trends
- **Periodic components** $(\sin(\omega_i t + \phi_i))$: Captures cycles (seasons, weeks)
- **Learnable** $\omega_i, \phi_i$: Adapts to task-specific patterns

### 6.3 Visual Intuition

```
Input: t = [0, 1, 2, 3, 4, 5, 6]  (days before summit)

Linear component (trend):
    ───────────────────▶
    Captures: "closer to summit = higher urgency"

Periodic component (cycles):
    ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿
    Captures: "weather patterns repeat every N days"

Combined output:
    A rich representation of temporal position
```

### 6.4 Code Implementation with Math Mapping

```python
class Time2Vec(nn.Module):
    def __init__(self, num_features: int):
        # num_features = output dimension

        # Linear component: ω₀t + φ₀
        self.linear_weight = nn.Parameter(torch.randn(1))   # ω₀
        self.linear_bias = nn.Parameter(torch.randn(1))     # φ₀

        # Periodic components: sin(ωᵢt + φᵢ)
        num_periodic = num_features - 1
        self.periodic_weights = nn.Parameter(torch.randn(num_periodic))  # ω₁...ωₖ
        self.periodic_biases = nn.Parameter(torch.randn(num_periodic))   # φ₁...φₖ

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (B, S) — time values for each token

        # Linear component: ω₀ × t + φ₀
        linear = self.linear_weight * t + self.linear_bias  # (B, S)
        linear = linear.unsqueeze(-1)  # (B, S, 1)

        # Periodic components: sin(ωᵢ × t + φᵢ)
        t_expanded = t.unsqueeze(-1)  # (B, S, 1)
        # Broadcasting: (B, S, 1) * (num_periodic,) + (num_periodic,)
        periodic = torch.sin(self.periodic_weights * t_expanded + self.periodic_biases)
        # periodic: (B, S, num_periodic)

        # Concatenate: [linear, periodic]
        encoding = torch.cat([linear, periodic], dim=-1)  # (B, S, num_features)

        return encoding
```

### 6.5 Shape Transformation

```
Input:  t          (B, S)              e.g., (32, 26) — days before summit
        ↓
        linear     (B, S, 1)           ω₀t + φ₀
        ↓
        periodic   (B, S, k)           [sin(ω₁t+φ₁), ..., sin(ωₖt+φₖ)]
        ↓
        concat     (B, S, 1+k)         full Time2Vec encoding
        ↓
Output:            (B, S, num_features)
```

---

## 7. Tokenization: Converting Features to Embeddings

### 7.1 The FT-Transformer Approach

**Per-feature tokenization** gives each feature its own embedding:

```
Traditional approach:
    All features → Single shared embedding → One representation

FT-Transformer approach:
    Feature 1 → Embedding 1 → Token 1
    Feature 2 → Embedding 2 → Token 2
    ...
    Transformer learns feature interactions through attention
```

### 7.2 TabularTokenizer

Converts expedition metadata into tokens:

**Numeric features (8):** Linear projection
$$\text{token}_i = \text{Linear}_i(x_i)$$

**Categorical features (6):** Embedding lookup
$$\text{token}_i = \text{Embedding}_i[\text{category}]$$

**Binary features (6):** Binary embedding
$$\text{token}_i = \text{Embedding}_i[\text{0 or 1}]$$

### 7.3 TabularTokenizer Code with Math Mapping

```python
class TabularTokenizer(nn.Module):
    def __init__(self, hidden_size: int):
        # 8 numeric tokenizers: Linear(1, H)
        self.numeric_tokenizers = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(8)
        ])

        # 6 categorical embeddings: Embedding(vocab_size, H)
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, hidden_size)
            for vocab_size in [151, 101, 21, 5, 4, 15]
        ])

        # 6 binary embeddings: Embedding(2, H)
        self.binary_embeddings = nn.ModuleList([
            nn.Embedding(2, hidden_size) for _ in range(6)
        ])

    def forward(self, numeric, categorical, binary):
        # numeric: (B, 8), categorical: (B, 6), binary: (B, 6)
        tokens = []

        # Numeric: each feature → Linear → token
        for i, tokenizer in enumerate(self.numeric_tokenizers):
            feature = numeric[:, i:i+1]  # (B, 1)
            token = tokenizer(feature)    # (B, H)
            tokens.append(token)

        # Categorical: look up embedding
        for i, embedding in enumerate(self.categorical_embeddings):
            feature = categorical[:, i]   # (B,)
            token = embedding(feature)    # (B, H)
            tokens.append(token)

        # Binary: look up 0/1 embedding
        for i, embedding in enumerate(self.binary_embeddings):
            feature = binary[:, i]        # (B,)
            token = embedding(feature)    # (B, H)
            tokens.append(token)

        # Stack: 20 tokens
        tokens = torch.stack(tokens, dim=1)  # (B, 20, H)
        return tokens
```

### 7.4 WeatherTokenizer

Converts multi-scale weather data into tokens with temporal encoding:

$$\text{token}_t = \text{Linear}(\text{weather}_t) + \text{Time2Vec}(t)$$

### 7.5 WeatherTokenizer Code with Math Mapping

```python
class WeatherTokenizer(nn.Module):
    def __init__(self, hidden_size: int, weather_vars: int = 15):
        # Project weather features: Linear(15, H)
        self.weather_proj = nn.Linear(weather_vars, hidden_size)

        # Time2Vec encoders for two temporal dimensions
        self.time2vec_days_before = Time2Vec(32)  # Output: 32 features
        self.time2vec_day_of_year = Time2Vec(32)  # Output: 32 features

        # Project combined temporal encoding: Linear(64, H)
        self.temporal_proj = nn.Linear(64, hidden_size)

    def forward(self, weather, days_before_summit, day_of_year):
        # weather: (B, 26, 15) — 26 timesteps, 15 variables each
        # days_before_summit: (B, 26) — temporal position
        # day_of_year: (B, 26) — calendar position

        # Project weather variables
        weather_tokens = self.weather_proj(weather)  # (B, 26, H)

        # Encode temporal positions
        time_enc_before = self.time2vec_days_before(days_before_summit)  # (B, 26, 32)
        time_enc_year = self.time2vec_day_of_year(day_of_year)           # (B, 26, 32)

        # Concatenate temporal encodings
        temporal = torch.cat([time_enc_before, time_enc_year], dim=-1)   # (B, 26, 64)

        # Project to hidden size
        temporal_tokens = self.temporal_proj(temporal)  # (B, 26, H)

        # Add temporal encoding to weather tokens
        tokens = weather_tokens + temporal_tokens  # (B, 26, H)

        return tokens
```

### 7.6 Tokenization Shape Summary

```
TABULAR TOKENIZATION:
    numeric     (B, 8)       →  8 tokens  (B, 8, H)
    categorical (B, 6)       →  6 tokens  (B, 6, H)
    binary      (B, 6)       →  6 tokens  (B, 6, H)
                                ─────────────────
                                20 tokens (B, 20, H)

WEATHER TOKENIZATION:
    weather     (B, 26, 15)  →  26 tokens (B, 26, H)
  + days_before (B, 26)      →  Time2Vec encoding
  + day_of_year (B, 26)      →  Time2Vec encoding
                                ─────────────────
                                26 tokens (B, 26, H)

TOTAL SEQUENCE (with [CLS] token):
    [CLS] + 20 tabular + 26 weather = 47 tokens
```

---

## 8. Transformer Block: Combining Components

### 8.1 Block Architecture

A transformer block combines attention and FFN with normalization and residuals:

```
┌────────────────────────────────────────────────────────────────┐
│                      TRANSFORMER BLOCK                         │
│                                                                │
│    Input x                                                     │
│        │                                                       │
│        ├──────────────── residual ────────────────┐            │
│        │                                          │            │
│        ▼                                          │            │
│   ┌─────────┐                                     │            │
│   │ RMSNorm │  Pre-normalization                  │            │
│   └────┬────┘                                     │            │
│        │                                          │            │
│        ▼                                          │            │
│   ┌─────────────────┐                             │            │
│   │ Multi-Head Attn │                             │            │
│   └────────┬────────┘                             │            │
│            │                                      │            │
│            ▼                                      │            │
│       ┌─────────┐                                 │            │
│       │DropPath │  Stochastic regularization     │             │
│       └────┬────┘                                 │            │
│            │                                      │            │
│            └───────────────► + ◄──────────────────┘            │
│                              │                                 │
│        ├──────────────── residual ────────────────┐            │
│        │                                          │            │
│        ▼                                          │            │
│   ┌─────────┐                                     │            │
│   │ RMSNorm │  Pre-normalization                  │            │
│   └────┬────┘                                     │            │
│        │                                          │            │
│        ▼                                          │            │
│   ┌─────────────────┐                             │            │
│   │   SwiGLU FFN    │                             │            │
│   └────────┬────────┘                             │            │
│            │                                      │            │
│            ▼                                      │            │
│       ┌─────────┐                                 │            │
│       │DropPath │                                 │            │
│       └────┬────┘                                 │            │
│            │                                      │            │
│            └───────────────► + ◄──────────────────┘            │
│                              │                                 │
│                              ▼                                 │
│                           Output                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 8.2 The Math

$$\text{Block}(x) = x''$$

where:
$$x' = x + \text{DropPath}(\text{Attention}(\text{RMSNorm}(x)))$$
$$x'' = x' + \text{DropPath}(\text{FFN}(\text{RMSNorm}(x')))$$

### 8.3 Why Pre-Normalization?

| Approach | Formula | Stability |
|----------|---------|-----------|
| Post-Norm | $x + \text{Norm}(\text{F}(x))$ | Can have gradient issues |
| Pre-Norm | $x + \text{F}(\text{Norm}(x))$ | More stable training |

Pre-norm ensures the residual path is unscaled, improving gradient flow.

### 8.4 TransformerBlock Code with Math Mapping

```python
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_hidden_size, dropout, drop_path_prob):
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.ffn = SwiGLU(hidden_size, ffn_hidden_size, dropout)
        self.drop_path1 = DropPath(drop_path_prob)
        self.drop_path2 = DropPath(drop_path_prob)

    def forward(self, x):
        # x shape: (B, S, H)

        # Attention sub-block with residual
        # x' = x + DropPath(Attention(RMSNorm(x)))
        residual = x                              # (B, S, H)
        x_norm = self.norm1(x)                    # (B, S, H) — pre-norm
        attn_out, _ = self.attention(x_norm)      # (B, S, H)
        attn_out = self.drop_path1(attn_out)      # (B, S, H) — regularize
        x = residual + attn_out                   # (B, S, H) — residual connection

        # FFN sub-block with residual
        # x'' = x' + DropPath(FFN(RMSNorm(x')))
        residual = x                              # (B, S, H)
        x_norm = self.norm2(x)                    # (B, S, H) — pre-norm
        ffn_out = self.ffn(x_norm)                # (B, S, H)
        ffn_out = self.drop_path2(ffn_out)        # (B, S, H) — regularize
        x = residual + ffn_out                    # (B, S, H) — residual connection

        return x  # (B, S, H)
```

### 8.5 Shape Through Block

```
Input:      x          (B, S, H)      (32, 47, 256)
            │
            ├─────────────────────────────┐ residual
            ▼                             │
            RMSNorm    (B, S, H)           │
            ▼                             │
            Attention  (B, S, H)           │
            ▼                             │
            DropPath   (B, S, H)           │
            ▼                             │
            + ◄───────────────────────────┘
            │
            ├─────────────────────────────┐ residual
            ▼                             │
            RMSNorm    (B, S, H)           │
            ▼                             │
            SwiGLU     (B, S, H)           │
            ▼                             │
            DropPath   (B, S, H)           │
            ▼                             │
            + ◄───────────────────────────┘
            │
Output:                (B, S, H)      (32, 47, 256)
```

---

## 9. Transformer Encoder: Stacking Blocks

### 9.1 The Stack

The encoder stacks N transformer blocks with increasing drop path probability:

$$\text{Encoder}(x) = \text{RMSNorm}(\text{Block}_N(\text{Block}_{N-1}(...\text{Block}_1(x)...)))$$

### 9.2 Encoder Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER ENCODER                          │
│                                                                 │
│    Input tokens (B, S, H)                                       │
│           │                                                     │
│           ▼                                                     │
│    ┌─────────────────────┐                                      │
│    │ Transformer Block 0 │  drop_path = 0.00                    │
│    └──────────┬──────────┘                                      │
│               │                                                 │
│               ▼                                                 │
│    ┌─────────────────────┐                                      │
│    │ Transformer Block 1 │  drop_path = 0.02                    │
│    └──────────┬──────────┘                                      │
│               │                                                 │
│               ▼                                                 │
│    ┌─────────────────────┐                                      │
│    │ Transformer Block 2 │  drop_path = 0.04                    │
│    └──────────┬──────────┘                                      │
│               │                                                 │
│               ▼                                                 │
│    ┌─────────────────────┐                                      │
│    │ Transformer Block 3 │  drop_path = 0.06                    │
│    └──────────┬──────────┘                                      │
│               │                                                 │
│               ▼                                                 │
│    ┌─────────────────────┐                                      │
│    │ Transformer Block 4 │  drop_path = 0.08                    │
│    └──────────┬──────────┘                                      │
│               │                                                 │
│               ▼                                                 │
│    ┌─────────────────────┐                                      │
│    │ Transformer Block 5 │  drop_path = 0.10                    │
│    └──────────┬──────────┘                                      │
│               │                                                 │
│               ▼                                                 │
│    ┌─────────────────────┐                                      │
│    │      RMSNorm        │  Final normalization                 │
│    └──────────┬──────────┘                                      │
│               │                                                 │
│               ▼                                                 │
│    Output tokens (B, S, H)                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 TransformerEncoder Code

```python
class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, ffn_hidden_size,
                 dropout, drop_path_rate):

        # Linear drop path schedule
        drop_probs = [drop_path_rate * i / (num_layers - 1) for i in range(num_layers)]

        # Stack of blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_hidden_size,
                           dropout, drop_probs[i])
            for i in range(num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(hidden_size)

    def forward(self, x):
        # x shape: (B, S, H)

        # Pass through each block
        for block in self.blocks:
            x, _ = block(x)  # (B, S, H) → (B, S, H)

        # Final normalization
        x = self.norm(x)  # (B, S, H)

        return x
```

### 9.4 Shape Through Encoder

```
Input:       (B, S, H)       (32, 47, 256)
    │
    ▼
Block 0:     (B, S, H)       (32, 47, 256)
    │
    ▼
Block 1:     (B, S, H)       (32, 47, 256)
    │
    ▼
  ...
    │
    ▼
Block 5:     (B, S, H)       (32, 47, 256)
    │
    ▼
RMSNorm:     (B, S, H)       (32, 47, 256)
    │
    ▼
Output:      (B, S, H)       (32, 47, 256)
```

**Key insight:** The shape never changes through the encoder. All processing happens in the hidden dimension.

---

## 10. Classification Head: Final Prediction

### 10.1 From Sequence to Probability

The transformer outputs 47 token embeddings. We need a single probability. The **[CLS] token** (position 0) aggregates information from all tokens through attention.

$$P(\text{summit}) = \sigma(\text{MLP}(\text{encoder\_output}[0]))$$

### 10.2 Classification Head Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLASSIFICATION HEAD                          │
│                                                                 │
│    Encoder output: (B, S, H) = (32, 47, 256)                    │
│                                                                 │
│    Extract [CLS] token at position 0                            │
│           │                                                     │
│           │    encoder_output[:, 0, :]                          │
│           │                                                     │
│           ▼                                                     │
│                                                                 │
│    cls_token: (B, H) = (32, 256)                                │
│           │                                                     │
│           ▼                                                     │
│    ┌─────────────────┐                                          │
│    │ Linear(H → H)   │   256 → 256                              │
│    └────────┬────────┘                                          │
│             │                                                   │
│             ▼                                                   │
│    ┌─────────────────┐                                          │
│    │      GELU       │   Non-linearity                          │
│    └────────┬────────┘                                          │
│             │                                                   │
│             ▼                                                   │
│    ┌─────────────────┐                                          │
│    │    Dropout      │   Regularization                         │
│    └────────┬────────┘                                          │
│             │                                                   │
│             ▼                                                   │
│    ┌─────────────────┐                                          │
│    │ Linear(H → 1)   │   256 → 1                                │
│    └────────┬────────┘                                          │
│             │                                                   │
│             ▼                                                   │
│    ┌─────────────────┐                                          │
│    │    Sigmoid      │   Logit → Probability                    │
│    └────────┬────────┘                                          │
│             │                                                   │
│             ▼                                                   │
│                                                                 │
│    Output: P(summit success) ∈ [0, 1]                           │
│            Shape: (B,) = (32,)                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 The GELU Activation

**Gaussian Error Linear Unit:**

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the cumulative distribution function of the standard normal.

Approximation: $\text{GELU}(x) \approx 0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$

```
      │
   2  │                    ╱
      │                  ╱
   1  │               ╱
      │             ╱
   0  │───────────●───────────
      │        ╱
  -1  │      ╱  (smooth curve, not sharp like ReLU)
      │
      └──────────────────────
        -4  -2   0   2   4
```

### 10.4 The Sigmoid Function

Converts any real number to a probability in [0, 1]:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

| Input | Output |
|-------|--------|
| -∞ | 0 |
| -2 | 0.12 |
| 0 | 0.5 |
| +2 | 0.88 |
| +∞ | 1 |

### 10.5 ClassificationHead Code

```python
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        # encoder_output: (B, S, H) = (32, 49, 256)

        # Extract [CLS] token
        cls_token = encoder_output[:, 0, :]  # (B, H) = (32, 256)

        # Two-layer MLP
        x = self.dense(cls_token)            # (32, 256)
        x = F.gelu(x)                        # (32, 256)
        x = self.dropout(x)                  # (32, 256)
        logits = self.classifier(x)          # (32, 1)

        # Convert to probability
        probability = torch.sigmoid(logits)  # (32, 1)
        probability = probability.squeeze(-1)  # (32,)

        return probability
```

### 10.6 Shape Through Classification Head

```
Input:      encoder_output  (B, S, H)     (32, 47, 256)
            │
            │ [:, 0, :] — extract position 0
            ▼
            cls_token       (B, H)        (32, 256)
            │
            ▼ Linear
            hidden          (B, H)        (32, 256)
            │
            ▼ GELU
            activated       (B, H)        (32, 256)
            │
            ▼ Dropout
            regularized     (B, H)        (32, 256)
            │
            ▼ Linear
            logits          (B, 1)        (32, 1)
            │
            ▼ Sigmoid
            probability     (B, 1)        (32, 1)
            │
            ▼ Squeeze
Output:     probability     (B,)          (32,)
```

---

## 11. Complete Data Flow: End-to-End

### 11.1 The Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEEPSUMMIT TRANSFORMER                              │
│                         Complete Data Flow                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        RAW INPUT DATA                                │   │
│  │                                                                      │   │
│  │   numeric:     (B, 8)   = (32, 8)    — age, altitude, etc.          │    │
│  │   categorical: (B, 6)   = (32, 6)    — nationality, route, etc.     │    │
│  │   binary:      (B, 6)   = (32, 6)    — oxygen, commercial, etc.     │    │
│  │   weather:     (B, 26, 15) = (32, 26, 15) — 26 timesteps × 15 vars  │    │
│  │   days_before: (B, 26)  = (32, 26)   — temporal position            │    │
│  │   day_of_year: (B, 26)  = (32, 26)   — calendar position            │    │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       TOKENIZATION                                   │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │ TabularTokenizer                                            │   │    │
│  │   │   numeric → 8 Linear projections   → 8 tokens  (B, 8, H)    │   │    │
│  │   │   categorical → 6 Embedding lookups → 6 tokens (B, 6, H)    │   │    │
│  │   │   binary → 6 Embedding lookups      → 6 tokens (B, 6, H)    │   │    │
│  │   │   Stack → tabular_tokens            → (B, 20, H)            │   │    │
│  │   └─────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │ WeatherTokenizer                                            │   │    │
│  │   │   weather → Linear projection       → (B, 26, H)            │   │    │
│  │   │   days_before → Time2Vec            → (B, 26, 32)           │   │    │
│  │   │   day_of_year → Time2Vec            → (B, 26, 32)           │   │    │
│  │   │   Concat temporal → Project         → (B, 26, H)            │   │    │
│  │   │   Add weather + temporal            → weather_tokens (B,26,H)│   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    SEQUENCE ASSEMBLY                                 │   │
│  │                                                                      │   │
│  │   [CLS]          → learnable embedding → (1, H)                     │    │
│  │   tabular_tokens → from tokenizer      → (B, 20, H)                 │    │
│  │   weather_tokens → from tokenizer      → (B, 26, H)                 │    │
│  │                                                                      │   │
│  │   Concatenate: [CLS] + tabular + weather                            │    │
│  │   sequence: (B, 47, H) = (32, 47, 256)                              │    │
│  │                                                                      │   │
│  │   + Modality type embeddings (distinguishes token types)            │    │
│  │   Final sequence: (B, 47, H)                                        │    │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    TRANSFORMER ENCODER                               │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │                  × 6 Transformer Blocks                      │   │   │
│  │   │                                                              │   │   │
│  │   │   Input x: (B, 47, H)                                        │   │   │
│  │   │       │                                                      │   │   │
│  │   │       ├────────── residual ──────────┐                       │   │   │
│  │   │       ▼                              │                       │   │   │
│  │   │   RMSNorm → Attention → DropPath → + │                       │   │   │
│  │   │       │                              │                       │   │   │
│  │   │       ├────────── residual ──────────┤                       │   │   │
│  │   │       ▼                              │                       │   │   │
│  │   │   RMSNorm → SwiGLU → DropPath → +    │                       │   │   │
│  │   │       │                                                      │   │   │
│  │   │   Output x: (B, 47, H)                                       │   │   │
│  │   │                                                              │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │   │
│  │   Final RMSNorm → encoded: (B, 47, H) = (32, 47, 256)               │    │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    CLASSIFICATION HEAD                               │   │
│  │                                                                      │   │
│  │   Extract [CLS]: encoded[:, 0, :] → (B, H) = (32, 256)              │    │
│  │       │                                                              │   │
│  │       ▼                                                              │   │
│  │   Linear(H → H) → GELU → Dropout → (32, 256)                        │    │
│  │       │                                                              │   │
│  │       ▼                                                              │   │
│  │   Linear(H → 1) → Sigmoid → (32, 1) → squeeze → (32,)               │    │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         OUTPUT                                       │   │
│  │                                                                      │   │
│  │   probability: (B,) = (32,)                                         │    │
│  │                                                                      │   │
│  │   Each value ∈ [0, 1] represents P(summit success)                  │    │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 11.2 Token Sequence Visualization

```
Position:    0      1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
Token:     [CLS]  n₁  n₂  n₃  n₄  n₅  n₆  n₇  n₈  c₁  c₂  c₃  c₄  c₅  c₆  b₁  b₂  b₃  b₄  b₅  b₆
Type ID:     0     1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
           ────── ──────────────────────────────────────────────────────────────────────────────
           Special              Tabular Features (20 tokens)

Position:   21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46
Token:      w₁  w₂  w₃  w₄  w₅  w₆  w₇  w₈  w₉ w₁₀ w₁₁ w₁₂ w₁₃ w₁₄ w₁₅ w₁₆ w₁₇ w₁₈ w₁₉ w₂₀ w₂₁ w₂₂ w₂₃ w₂₄ w₂₅ w₂₆
Type ID:     2   2   2   2   2   2   2   3   3   3   3   3   3   3   3   3   3   4   4   4   4   4   4   4   4   4
           ─────────────────────── ───────────────────────────────────────────── ─────────────────────────────────
              7-day weather (7)              30-day weather (10)                        90-day weather (9)
```

### 11.3 What Each Token Type Learns

| Type | Tokens | Role |
|------|--------|------|
| [CLS] | 1 | Aggregates global information for classification |
| Tabular | 20 | Expedition metadata (who, what, when, how) |
| 7-day weather | 7 | Tactical conditions for summit push |
| 30-day weather | 10 | Acclimatization period patterns |
| 90-day weather | 9 | Seasonal context (monsoon, winter storms) |

---

## 12. Shape Reference Summary

### 12.1 Complete Shape Table

| Stage | Tensor | Shape | Example |
|-------|--------|-------|---------|
| **Input** | numeric | (B, 8) | (32, 8) |
| | categorical | (B, 6) | (32, 6) |
| | binary | (B, 6) | (32, 6) |
| | weather | (B, 26, 15) | (32, 26, 15) |
| | days_before | (B, 26) | (32, 26) |
| | day_of_year | (B, 26) | (32, 26) |
| **Tokenization** | tabular_tokens | (B, 20, H) | (32, 20, 256) |
| | weather_tokens | (B, 26, H) | (32, 26, 256) |
| **Sequence** | full_sequence | (B, 47, H) | (32, 47, 256) |
| **Encoder** | per_block | (B, 47, H) | (32, 47, 256) |
| | encoded | (B, 47, H) | (32, 47, 256) |
| **Attention** | Q, K, V | (B, n_h, S, d_k) | (32, 8, 47, 32) |
| | scores | (B, n_h, S, S) | (32, 8, 47, 47) |
| | context | (B, n_h, S, d_k) | (32, 8, 47, 32) |
| **FFN** | hidden | (B, S, ffn_hidden) | (32, 47, 682) |
| **Classification** | cls_token | (B, H) | (32, 256) |
| | logits | (B, 1) | (32, 1) |
| **Output** | probability | (B,) | (32,) |

### 12.2 Parameter Count Summary

| Component | Parameters | Formula |
|-----------|------------|---------|
| TabularTokenizer | ~86K | 8×(1×H+H) + Σvocab×H + 6×2×H |
| WeatherTokenizer | ~70K | 15×H+H + 2×Time2Vec + 64×H |
| CLS token | ~0.25K | 1×H |
| Modality embedding | ~1.3K | 5×H |
| Per TransformerBlock | ~787K | 2×H + 4×H² + 3×H×(8H/3) |
| × 6 blocks | ~4.7M | |
| ClassificationHead | ~66K | H² + H + H + 1 |
| **Total** | **~4.9M** | |

### 12.3 Computational Complexity

| Operation | Complexity | Why |
|-----------|------------|-----|
| Attention scores | O(S² · H) | QK^T for all heads |
| Attention context | O(S² · H) | weights × V |
| FFN | O(S · H · ffn_hidden) | Linear projections |
| Total per block | O(S² · H + S · H²) | Attention + FFN |
| Full encoder | O(L · (S² · H + S · H²)) | L blocks |

For DeepSummit: S=47, H=256, L=6 → ~130M FLOPs per forward pass.

---

## Conclusion

This guide has walked through every mathematical operation in the DeepSummit transformer:

1. **RMSNorm** normalizes by root mean square
2. **SwiGLU** uses gated activation for adaptive feature selection
3. **DropPath** regularizes by randomly skipping residual branches
4. **Multi-Head Attention** computes weighted sums based on query-key similarity
5. **Time2Vec** learns temporal representations with linear + periodic components
6. **Tokenizers** convert raw features into learnable embeddings
7. **Transformer Blocks** combine all components with pre-norm and residuals
8. **Classification Head** extracts [CLS] and projects to probability

The key insight: transformers are **remarkably uniform**. The same basic operations (linear projection, attention, FFN) repeat throughout, with the magic coming from:
- Learning the right projections (W matrices)
- Stacking depth (6 layers)
- Per-feature tokenization (each feature gets its own representation)
- Joint attention (all tokens can attend to all others)

Understanding these building blocks means you understand the entire architecture!
