# DeepSummit v2 — Model Architecture Design

## Overview

DeepSummit uses a **unified multimodal transformer** that jointly attends over expedition metadata (tabular features) and multi-scale historical weather data to predict summit success probability.

**Key Design Principles:**
- Modern efficient transformer (RMSNorm + SwiGLU + Stochastic Depth)
- Per-feature tokenization for tabular data (FT-Transformer style)
- Multi-scale temporal encoding for weather data
- Full joint attention from the start (no staged fusion)

---

## Part 1: Token Architecture

### 1.1 Token Sequence Structure

```
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Position:   0       1────────────20      21──────────46                            │
│              │            │                │           │                            │
│              ▼            ▼                ▼           ▼                            │
│           ┌─────┐   ┌───────────┐     ┌───────────────────┐                         │
│           │[CLS]│   │  Tabular  │     │     Weather       │                         │
│           │     │   │  Tokens   │     │     Tokens        │                         │
│           └─────┘   └───────────┘     └───────────────────┘                         │
│              │       (20 tokens)          (26 tokens)                               │
│              │            │                    │                                    │
│  Modality:   0            1              2 (7d) / 3 (30d) / 4 (90d)                 │
│           (CLS)       (tabular)           (weather scales)                          │
│                                                                                     │
│  Why no [SEP] tokens?                                                               │
│  → Modality embeddings distinguish token types more efficiently                     │
│  → No wasted sequence positions (47 vs 49 tokens)                                   │
│  → Learnable boundary representations instead of rigid separators                   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The [CLS] Token

```
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [CLS]  ─────────────────────────────────────────────────────►  │
│     │    Aggregates information from ALL tokens via attention    │
│     │    Used for final classification prediction                │
│     │                                                            │
│     └──► Learnable parameter: (1, hidden_size)                   │
│                                                                  │
│   Position 0 in every sequence                                   │
│   Modality type: 0                                               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.3 Modality Embeddings (Instead of [SEP] Tokens)

```
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Modern transformers use modality embeddings rather than [SEP]  │
│   tokens to distinguish between different data sources:          │
│                                                                  │
│   Implementation: Embedding(5, hidden_size)                      │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │  Type ID │ Token Type      │ Count │ Purpose            │    │
│   ├──────────┼─────────────────┼───────┼────────────────────┤    │
│   │    0     │ [CLS]           │   1   │ Classification     │    │
│   │    1     │ Tabular         │  20   │ Expedition metadata│    │
│   │    2     │ 7-day weather   │   7   │ Summit push        │    │
│   │    3     │ 30-day weather  │  10   │ Acclimatization    │    │
│   │    4     │ 90-day weather  │   9   │ Seasonal context   │    │
│   └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│   Applied as:  token += modality_embedding[type_id]              │
│                                                                  │
│   Advantages over [SEP]:                                         │
│   • More efficient (no wasted positions)                         │
│   • Learnable (adapts to task)                                   │
│   • Fine-grained (different embedding per scale)                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Tabular Tokenization (20 tokens)

### 2.1 Feature Breakdown

```
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                     NUMERIC FEATURES (8 tokens)                          │   │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                         │    │
│  │   age ──────────────┐                                                   │    │
│  │   prior_expeditions ├──► Each passes through its own Linear(1, H)      │     │
│  │   prior_summits ────┤    where H = hidden_size                          │    │
│  │   highest_prev_alt ─┤                                                   │    │
│  │   totmembers ───────┤    ┌────────┐     ┌────────────┐                  │    │
│  │   heightm ──────────┼───►│ Linear │────►│ (1, H)     │──► token        │     │
│  │   day_of_year ──────┤    │ (1, H) │     │ embedding  │                  │    │
│  │   route_success_rate┤    └────────┘     └────────────┘                  │    │
│  │   peak_success_rate─┘                                                   │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                   CATEGORICAL FEATURES (6 tokens)                        │   │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                         │    │
│  │   nationality (~150) ─┐                                                 │    │
│  │   route1 (~100) ──────┤   ┌──────────────────────┐                      │    │
│  │   himal (~20) ────────┼──►│ Embedding(vocab, H)  │──► token             │    │
│  │   season (4) ─────────┤   │ per feature          │                      │    │
│  │   style (3) ──────────┤   └──────────────────────┘                      │    │
│  │   peakid (8) ─────────┘   [UNK] token at index 0 for unknown values     │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                     BINARY FEATURES (6 tokens)                           │   │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                         │    │
│  │   sex_encoded ────────┐                                                 │    │
│  │   oxygen_planned ─────┤   ┌────────────────┐                            │    │
│  │   is_hired ───────────┼──►│ Embedding(2,H) │──► token                   │    │
│  │   is_sherpa ──────────┤   │ per feature    │                            │    │
│  │   is_commercial ──────┤   └────────────────┘                            │    │
│  │   o2_available ───────┘   0 → embedding[0], 1 → embedding[1]            │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Numeric Feature Normalization

All numeric features are z-score normalized using training set statistics:

```
                       x - μ_train
           x_norm  =  ─────────────
                        σ_train

Statistics computed once during training, saved for inference.
```

---

## Part 3: Weather Tokenization (26 tokens)

### 3.1 Multi-Scale Window Architecture

```
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  SUMMIT DATE ──────────────────────────────────────────────────────────────────►    │
│       │                                                                             │
│       ▼                                                                             │
│  ┌────┴────────────────────────────────────────────────────────────────────────┐    │
│  │                         90-DAY LOOKBACK WINDOW                               │   │
│  │                                                                              │   │
│  │  Day -89 ─────────────────────────────────────────────────────► Day 0       │    │
│  │                                                                 (summit)     │   │
│  │                                                                              │   │
│  │  ┌──────────────────────────────────────────────────────────────────────┐   │    │
│  │  │ 90d SCALE: 9 buckets × 10 days each                                  │   │    │
│  │  │ ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐              │   │     │
│  │  │ │ b0 ││ b1 ││ b2 ││ b3 ││ b4 ││ b5 ││ b6 ││ b7 ││ b8 │              │   │     │
│  │  │ └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘              │   │     │
│  │  │ Type ID: 4 (seasonal patterns, monsoon, winter storms)               │   │    │
│  │  └──────────────────────────────────────────────────────────────────────┘   │    │
│  │                                                                              │   │
│  │                        ┌──────────────────────────────────────────────┐     │    │
│  │                        │ 30d SCALE: 10 buckets × 3 days each          │     │    │
│  │                        │ ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐│     │
│  │                        │ │b0 ││b1 ││b2 ││b3 ││b4 ││b5 ││b6 ││b7 ││b8 ││b9 ││     │
│  │                        │ └───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘│     │
│  │                        │ Type ID: 3 (acclimatization period)          │     │    │
│  │                        └──────────────────────────────────────────────┘     │    │
│  │                                                                              │   │
│  │                                              ┌─────────────────────────┐    │    │
│  │                                              │ 7d SCALE: 7 daily       │    │    │
│  │                                              │ ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐  │    │     │
│  │                                              │ │0││1││2││3││4││5││6│  │    │     │
│  │                                              │ └─┘└─┘└─┘└─┘└─┘└─┘└─┘  │    │     │
│  │                                              │ Type ID: 2 (summit push)│    │    │
│  │                                              └─────────────────────────┘    │    │
│  │                                                                              │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Weather Variables (15 per token)

```
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Each weather token encodes 15 variables for its time bucket:                  │
│                                                                                 │
│   TEMPERATURE (6)                      PRECIPITATION (4)                        │
│   ├── temperature_2m_mean              ├── precipitation_sum (mm)               │
│   ├── temperature_2m_max               ├── rain_sum (mm)                        │
│   ├── temperature_2m_min               ├── snowfall_sum (cm)                    │
│   ├── apparent_temperature_mean        └── precipitation_hours                  │
│   ├── apparent_temperature_max                                                  │
│   └── apparent_temperature_min         WIND (3)                                 │
│                                        ├── wind_speed_10m_max (km/h)            │
│   RADIATION (2)                        ├── wind_gusts_10m_max (km/h)            │
│   ├── shortwave_radiation_sum          └── wind_direction_10m_dominant (°)      │
│   └── et0_fao_evapotranspiration                                                │
│                                                                                 │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │                    TOKENIZATION PROCESS                                │    │
│   │                                                                        │    │
│   │   weather_vector ──► Linear(15, hidden_size) ──► weather_token        │     │
│   │      (15,)                                          (hidden_size,)     │    │
│   │                                                                        │    │
│   │   One Linear layer shared across all 26 timesteps                      │    │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Embeddings and Positional Encoding

### 4.1 Modality Type Embeddings

```
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Implementation: Embedding(5, hidden_size)                                     │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  Type ID │ Token Type      │ Count │ Purpose                            │   │
│   ├──────────┼─────────────────┼───────┼────────────────────────────────────┤   │
│   │    0     │ [CLS]           │   1   │ Classification aggregator          │   │
│   │    1     │ Tabular         │  20   │ Expedition metadata features       │   │
│   │    2     │ 7-day weather   │   7   │ Tactical summit push conditions    │   │
│   │    3     │ 30-day weather  │  10   │ Acclimatization period patterns    │   │
│   │    4     │ 90-day weather  │   9   │ Seasonal context signals           │   │
│   │          │                 │  ──── │                                    │   │
│   │          │ TOTAL           │  47   │                                    │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   Applied as:  token_embedding += modality_type_embedding[type_id]              │
│                                                                                 │
│   Why modality embeddings instead of [SEP] tokens?                              │
│   • More efficient: 47 tokens vs 49 (2 fewer positions)                         │
│   • More expressive: Different embedding per weather scale (2, 3, 4)            │
│   • Learnable: Model adapts boundary behavior during training                   │
│   • FT-Transformer pattern: Standard for tabular transformers                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Time2Vec Positional Encoding

```
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Applied to: Weather tokens ONLY (not tabular or special tokens)               │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                         │   │
│   │   Time2Vec(t) = [ ω₀·t + φ₀ ,  sin(ω₁·t + φ₁) , ... , sin(ωₖ·t + φₖ) ] │    │
│   │                  └─────────┘   └────────────────────────────────────┘   │   │
│   │                   linear           k periodic components                │   │
│   │                  (trend)           (cyclical patterns)                  │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   Two temporal dimensions encoded:                                              │
│                                                                                 │
│   1. DAYS_BEFORE_SUMMIT (0-89)                                                  │
│      └── Relative position: "how far before the attempt?"                       │
│          Captures: urgency, proximity to decision point                         │
│                                                                                 │
│   2. DAY_OF_YEAR (1-365)                                                        │
│      └── Absolute calendar position: "what season?"                             │
│          Captures: monsoon timing, winter storms, climbing seasons              │
│                                                                                 │
│   Why Time2Vec over sinusoidal?                                                 │
│   • Learnable frequencies adapt to mountaineering-specific patterns             │
│   • Linear component captures non-periodic trends                               │
│   • More expressive for irregular temporal patterns                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 5: Transformer Architecture

### 5.1 Overall Architecture

```
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   Input: Token Sequence (batch, 47, hidden_size)                                    │
│        │                                                                            │
│        ▼                                                                            │
│   ╔═════════════════════════════════════════════════════════════════════════════╗   │
│   ║                       TRANSFORMER ENCODER                                    ║  │
│   ║                                                                              ║  │
│   ║   ┌────────────────────────────────────────────────────────────────────┐    ║   │
│   ║   │                    × num_layers (default: 6)                       │    ║   │
│   ║   │  ┌──────────────────────────────────────────────────────────────┐ │    ║    │
│   ║   │  │ residual = x                                                  │ │    ║   │
│   ║   │  │     │                                                         │ │    ║   │
│   ║   │  │     ▼                                                         │ │    ║   │
│   ║   │  │ ┌─────────┐                                                   │ │    ║   │
│   ║   │  │ │ RMSNorm │  ◄── Pre-normalization (stable training)          │ │    ║   │
│   ║   │  │ └────┬────┘                                                   │ │    ║   │
│   ║   │  │      ▼                                                        │ │    ║   │
│   ║   │  │ ┌───────────────────┐                                         │ │    ║   │
│   ║   │  │ │ Multi-Head Attn   │  ◄── Flash Attention 2 (PyTorch 2.2+)   │ │    ║   │
│   ║   │  │ │ (8 heads default) │                                         │ │    ║   │
│   ║   │  │ └─────────┬─────────┘                                         │ │    ║   │
│   ║   │  │           ▼                                                   │ │    ║   │
│   ║   │  │     ┌─────────┐                                               │ │    ║   │
│   ║   │  │     │ Dropout │  (0.1)                                        │ │    ║   │
│   ║   │  │     └────┬────┘                                               │ │    ║   │
│   ║   │  │          ▼                                                    │ │    ║   │
│   ║   │  │     ┌──────────┐                                              │ │    ║   │
│   ║   │  │     │ DropPath │  ◄── Stochastic depth (linear schedule)      │ │    ║   │
│   ║   │  │     └────┬─────┘                                              │ │    ║   │
│   ║   │  │          │                                                    │ │    ║   │
│   ║   │  │ x = residual + output  ◄── Residual connection                │ │    ║   │
│   ║   │  │          │                                                    │ │    ║   │
│   ║   │  │ residual = x                                                  │ │    ║   │
│   ║   │  │     │                                                         │ │    ║   │
│   ║   │  │     ▼                                                         │ │    ║   │
│   ║   │  │ ┌─────────┐                                                   │ │    ║   │
│   ║   │  │ │ RMSNorm │                                                   │ │    ║   │
│   ║   │  │ └────┬────┘                                                   │ │    ║   │
│   ║   │  │      ▼                                                        │ │    ║   │
│   ║   │  │ ┌───────────┐                                                 │ │    ║   │
│   ║   │  │ │ SwiGLU FFN│  ◄── Gated activation (outperforms GELU)        │ │    ║   │
│   ║   │  │ └─────┬─────┘                                                 │ │    ║   │
│   ║   │  │       ▼                                                       │ │    ║   │
│   ║   │  │  ┌─────────┐                                                  │ │    ║   │
│   ║   │  │  │ Dropout │                                                  │ │    ║   │
│   ║   │  │  └────┬────┘                                                  │ │    ║   │
│   ║   │  │       ▼                                                       │ │    ║   │
│   ║   │  │  ┌──────────┐                                                 │ │    ║   │
│   ║   │  │  │ DropPath │                                                 │ │    ║   │
│   ║   │  │  └────┬─────┘                                                 │ │    ║   │
│   ║   │  │       │                                                       │ │    ║   │
│   ║   │  │ x = residual + output                                         │ │    ║   │
│   ║   │  └───────┼──────────────────────────────────────────────────────┘ │    ║    │
│   ║   │          │                                                        │    ║    │
│   ║   └──────────┼────────────────────────────────────────────────────────┘    ║    │
│   ║              ▼                                                              ║   │
│   ║         ┌─────────┐                                                         ║   │
│   ║         │ RMSNorm │  ◄── Final normalization                                ║   │
│   ║         └────┬────┘                                                         ║   │
│   ╚══════════════╪══════════════════════════════════════════════════════════════╝   │
│                  │                                                                  │
│                  ▼                                                                  │
│   ╔══════════════════════════════════════════════════════════════════════════════╗  │
│   ║                      CLASSIFICATION HEAD                                      ║ │
│   ║                                                                               ║ │
│   ║    cls_token = output[:, 0, :]   ◄── Extract [CLS] token                     ║  │
│   ║         │                                                                     ║ │
│   ║         ▼                                                                     ║ │
│   ║    ┌──────────────────────┐                                                   ║ │
│   ║    │ Linear(H, H) → GELU  │                                                   ║ │
│   ║    └──────────┬───────────┘                                                   ║ │
│   ║               ▼                                                               ║ │
│   ║         ┌─────────┐                                                           ║ │
│   ║         │ Dropout │                                                           ║ │
│   ║         └────┬────┘                                                           ║ │
│   ║              ▼                                                                ║ │
│   ║    ┌──────────────────────┐                                                   ║ │
│   ║    │  Linear(H, 1)        │                                                   ║ │
│   ║    └──────────┬───────────┘                                                   ║ │
│   ║               ▼                                                               ║ │
│   ║         ┌─────────┐                                                           ║ │
│   ║         │ Sigmoid │                                                           ║ │
│   ║         └────┬────┘                                                           ║ │
│   ╚══════════════╪════════════════════════════════════════════════════════════════╝ │
│                  │                                                                  │
│                  ▼                                                                  │
│         P(summit_success) ∈ [0, 1]                                                  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 RMSNorm (Root Mean Square Normalization)

```
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Standard LayerNorm:                                                           │
│                                                                                 │
│                    x - mean(x)                                                  │
│        LN(x)  =  ─────────────  ×  γ  +  β                                      │
│                    std(x) + ε                                                   │
│                                                                                 │
│   RMSNorm (simplified — no mean centering):                                     │
│                                                                                 │
│                        x                                                        │
│        RMS(x)  =  ─────────────  ×  γ                                           │
│                   RMS(x) + ε                                                    │
│                                                                                 │
│                              ┌─────────────────┐                                │
│        where RMS(x) = sqrt( │ mean(x²)        │ )                               │
│                              └─────────────────┘                                │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  WHY RMSNORM?                                                           │   │
│   │                                                                         │   │
│   │  • 10-15% faster than LayerNorm (skips mean computation)                │   │
│   │  • Same or better performance in practice                               │   │
│   │  • Used by LLaMA, Mistral, and other modern LLMs                        │   │
│   │  • Learnable scale (γ) only — no learnable bias (β)                     │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   PyTorch Implementation:                                                       │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  class RMSNorm(nn.Module):                                              │   │
│   │      def __init__(self, dim, eps=1e-6):                                 │   │
│   │          self.eps = eps                                                 │   │
│   │          self.weight = nn.Parameter(torch.ones(dim))  # γ               │   │
│   │                                                                         │   │
│   │      def forward(self, x):                                              │   │
│   │          rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)   │   │
│   │          return x / rms * self.weight                                   │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 SwiGLU Feed-Forward Network

```
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Standard FFN (GELU):                                                          │
│                                                                                 │
│       FFN(x) = GELU(x · W₁) · W₂                                                │
│                                                                                 │
│   SwiGLU FFN (Gated Linear Unit with Swish):                                    │
│                                                                                 │
│       SwiGLU(x) = ( Swish(x · W₁) ⊙ (x · W₂) ) · W₃                             │
│                                                                                 │
│       where:  Swish(x) = x · σ(x)    [also called SiLU]                         │
│               ⊙ = element-wise multiplication (gating)                          │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                         │   │
│   │      Input x                                                            │   │
│   │         │                                                               │   │
│   │    ┌────┴────┐                                                          │   │
│   │    │         │                                                          │   │
│   │    ▼         ▼                                                          │   │
│   │  ┌────┐   ┌────┐                                                        │   │
│   │  │ W₁ │   │ W₂ │     (gate projection)  (value projection)              │   │
│   │  └──┬─┘   └──┬─┘                                                        │   │
│   │     │        │                                                          │   │
│   │     ▼        │                                                          │   │
│   │  ┌──────┐    │                                                          │   │
│   │  │Swish │    │       Swish(x) = x × sigmoid(x)                          │   │
│   │  └──┬───┘    │                                                          │   │
│   │     │        │                                                          │   │
│   │     └───┬────┘                                                          │   │
│   │         │                                                               │   │
│   │         ▼                                                               │   │
│   │      ┌─────┐                                                            │   │
│   │      │  ⊙  │         Element-wise multiplication (gating)               │   │
│   │      └──┬──┘                                                            │   │
│   │         │                                                               │   │
│   │         ▼                                                               │   │
│   │      ┌────┐                                                             │   │
│   │      │ W₃ │          Output projection                                  │   │
│   │      └──┬─┘                                                             │   │
│   │         │                                                               │   │
│   │         ▼                                                               │   │
│   │      Output                                                             │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   Dimensions (for hidden_size H):                                               │
│                                                                                 │
│      ffn_hidden = H × 8/3 ≈ 2.67H  (keeps param count similar to 4H GELU FFN)   │
│                                                                                 │
│      W₁: (H, ffn_hidden)    gate projection                                     │
│      W₂: (H, ffn_hidden)    value projection                                    │
│      W₃: (ffn_hidden, H)    output projection                                   │
│                                                                                 │
│   WHY SWIGLU?                                                                   │
│   • Consistently outperforms GELU/ReLU in transformer benchmarks                │
│   • Gating mechanism provides adaptive feature selection                        │
│   • Used in PaLM, LLaMA, and most modern LLMs                                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Stochastic Depth (DropPath)

```
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Core Idea: Randomly drop entire residual branches during training             │
│                                                                                 │
│   Standard Residual:    x = x + F(x)                                            │
│   With DropPath:        x = x + DropPath(F(x))                                  │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                         │   │
│   │   During Training:                                                      │   │
│   │                                                                         │   │
│   │                       ┌─────────────────────────────────────────┐       │   │
│   │                       │  With probability (1 - p):              │       │   │
│   │     Input x ─────────►│     output = F(x) / (1 - p)             │       │   │
│   │         │             │                                         │       │   │
│   │         │             │  With probability p:                    │       │   │
│   │         │             │     output = 0  (drop entire branch)    │       │   │
│   │         │             └─────────────────────────────────────────┘       │   │
│   │         │                              │                                │   │
│   │         └────────────► + ◄─────────────┘                                │   │
│   │                        │                                                │   │
│   │                        ▼                                                │   │
│   │                    Output x                                             │   │
│   │                                                                         │   │
│   │   During Inference:                                                     │   │
│   │     output = F(x)  (no dropping, no scaling)                            │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   LINEAR SCHEDULE (recommended):                                                │
│                                                                                 │
│      Drop probability increases with layer depth:                               │
│                                                                                 │
│      ┌──────────────────────────────────────────────────────────────────────┐   │
│      │                                                                      │   │
│      │  Layer 0:  p = 0.00  ─────────────────────────────  (keep early)    │    │
│      │  Layer 1:  p = 0.02  ─────────────────────────────                   │   │
│      │  Layer 2:  p = 0.04  ─────────────────────────────                   │   │
│      │  Layer 3:  p = 0.06  ─────────────────────────────                   │   │
│      │  Layer 4:  p = 0.08  ─────────────────────────────                   │   │
│      │  Layer 5:  p = 0.10  ─────────────────────────────  (regularize)    │    │
│      │                                                                      │   │
│      │  Formula: p_i = drop_path_rate × (i / (num_layers - 1))             │    │
│      │                                                                      │   │
│      └──────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   WHY LINEAR SCHEDULE?                                                          │
│   • Early layers learn fundamental representations → keep them                  │
│   • Later layers learn task-specific refinements → safe to regularize           │
│   • Empirically outperforms uniform drop probability                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.5 Multi-Head Attention

```
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Standard Scaled Dot-Product Attention:                                        │
│                                                                                 │
│                              Q · K^T                                            │
│       Attention(Q, K, V) = softmax( ───────── ) · V                             │
│                               √d_k                                              │
│                                                                                 │
│   Multi-Head (h = 8 heads default):                                             │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                         │   │
│   │            Input x: (batch, seq_len, hidden_size)                       │   │
│   │                │                                                        │   │
│   │    ┌───────────┼───────────┐                                            │   │
│   │    │           │           │                                            │   │
│   │    ▼           ▼           ▼                                            │   │
│   │  ┌────┐     ┌────┐     ┌────┐                                           │   │
│   │  │ Wq │     │ Wk │     │ Wv │    Linear projections                      │  │
│   │  └──┬─┘     └──┬─┘     └──┬─┘                                           │   │
│   │     │          │          │                                             │   │
│   │     ▼          ▼          ▼                                             │   │
│   │  Split into h heads (reshape + transpose)                               │   │
│   │     │          │          │                                             │   │
│   │     │    Q     │    K     │    V                                        │   │
│   │     │ (B,h,S,d)│ (B,h,S,d)│ (B,h,S,d)                                   │   │
│   │     │          │          │                                             │   │
│   │     └────┬─────┴──────────┘                                             │   │
│   │          │                                                              │   │
│   │          ▼                                                              │   │
│   │   ┌─────────────────────────┐                                           │   │
│   │   │   Scaled Dot-Product    │                                           │   │
│   │   │      Attention          │  ◄── Flash Attention 2 (efficient)        │   │
│   │   │                         │                                           │   │
│   │   │   softmax(QK^T/√d) · V  │                                           │   │
│   │   └───────────┬─────────────┘                                           │   │
│   │               │                                                         │   │
│   │               ▼                                                         │   │
│   │        Concatenate heads                                                │   │
│   │               │                                                         │   │
│   │               ▼                                                         │   │
│   │            ┌────┐                                                       │   │
│   │            │ Wo │    Output projection                                  │   │
│   │            └──┬─┘                                                       │   │
│   │               │                                                         │   │
│   │               ▼                                                         │   │
│   │        Output: (batch, seq_len, hidden_size)                            │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   Dimensions:                                                                   │
│      hidden_size = 256 (default)                                                │
│      num_heads = 8                                                              │
│      head_dim = 256 / 8 = 32                                                    │
│                                                                                 │
│   Flash Attention 2:                                                            │
│   • Memory-efficient: O(N) instead of O(N²) for attention matrix                │
│   • Built into PyTorch 2.2+ via torch.nn.functional.scaled_dot_product_attention│
│   • Enabled automatically when using SDPA                                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 6: Full Data Flow

```
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌─────────────────────┐     ┌─────────────────────┐                               │
│   │   features.csv      │     │    weather.csv      │                               │
│   │   (81,886 rows)     │     │    (6,224 rows)     │                               │
│   └─────────┬───────────┘     └──────────┬──────────┘                               │
│             │                            │                                          │
│             └──────────┬─────────────────┘                                          │
│                        │  JOIN on weather_id                                        │
│                        ▼                                                            │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                         ExpeditionDataset                                   │   │
│   │                                                                             │   │
│   │   __getitem__(idx) returns:                                                 │   │
│   │                                                                             │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │   │
│   │   │ numeric (8,)│  │ categ (6,)  │  │ binary (6,) │  │ weather (26,15)  │   │   │
│   │   │ float32     │  │ int64       │  │ int64       │  │ float32          │   │   │
│   │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘   │   │
│   │          │                │                │                  │             │   │
│   └──────────┼────────────────┼────────────────┼──────────────────┼─────────────┘   │
│              │                │                │                  │                 │
│              ▼                ▼                ▼                  ▼                 │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                           TokenEmbedding                                    │   │
│   │                                                                             │   │
│   │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │   │
│   │   │ NumericTokenizer│  │ CategoricalEmbed│  │ BinaryEmbedding │             │   │
│   │   │ Linear(1,H) × 8 │  │ Embed(V,H) × 6  │  │ Embed(2,H) × 6  │             │   │
│   │   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │   │
│   │            │                    │                    │                      │   │
│   │            └────────────────────┼────────────────────┘                      │   │
│   │                                 │ Concatenate                               │   │
│   │                                 ▼                                            │  │
│   │                    tabular_tokens: (batch, 20, H)                            │  │
│   │                                                                             │   │
│   │   ┌─────────────────────────────────────────────────────────────────────┐   │   │
│   │   │                    WeatherTokenizer                                 │   │   │
│   │   │                                                                     │   │   │
│   │   │    weather (26, 15) ──► Linear(15, H) ──► (26, H)                   │   │   │
│   │   │                                                                     │   │   │
│   │   │    + Time2Vec(days_before_summit, day_of_year)                      │   │   │
│   │   │                                                                     │   │   │
│   │   └───────────────────────────────┬─────────────────────────────────────┘   │   │
│   │                                   │                                         │   │
│   │                                   ▼                                         │   │
│   │                    weather_tokens: (batch, 26, H)                           │   │
│   │                                                                              │  │
│   └──────────────────────────────────────────────────────────────────────────────┘  │
│              │                                   │                                  │
│              ▼                                   ▼                                  │
│   ┌──────────────────────────────────────────────────────────────────────────────┐  │
│   │                          SequenceAssembly                                     │ │
│   │                                                                               │ │
│   │   [CLS] + tabular_tokens + weather_tokens                                    │  │
│   │     │          │                │                                            │  │
│   │     ▼          ▼                ▼                                            │  │
│   │   type=0     type=1          type=2/3/4                                      │  │
│   │                                                                               │ │
│   │   + modality_type_embedding (added to each token)                             │ │
│   │                                                                               │ │
│   │   sequence: (batch, 47, H)                                                    │ │
│   │                                                                               │ │
│   └───────────────────────────────────┬───────────────────────────────────────────┘ │
│                                       │                                             │
│                                       ▼                                             │
│   ┌───────────────────────────────────────────────────────────────────────────────┐ │
│   │                         TransformerEncoder                                     ││
│   │                                                                                ││
│   │   6 layers × [Pre-RMSNorm → MHA → DropPath → Pre-RMSNorm → SwiGLU → DropPath] │ │
│   │                                                                                ││
│   │   encoded: (batch, 47, H)                                                      ││
│   │                                                                                ││
│   └────────────────────────────────────┬──────────────────────────────────────────┘ │
│                                        │                                            │
│                                        ▼                                            │
│   ┌───────────────────────────────────────────────────────────────────────────────┐ │
│   │                         ClassificationHead                                     ││
│   │                                                                                ││
│   │   cls_token = encoded[:, 0, :]  →  Linear(H,H) → GELU → Dropout               │ │
│   │                                           ↓                                    ││
│   │                                    Linear(H, 1) → Sigmoid                      ││
│   │                                           ↓                                    ││
│   │                                  P(summit) ∈ [0, 1]                            ││
│   │                                                                                ││
│   └───────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 7: Hyperparameters

### 7.1 Base Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `hidden_size` | 256 | Good balance for ~82K samples; 8 heads × 32 head_dim |
| `num_layers` | 6 | Deeper but narrower works well for small datasets |
| `num_heads` | 8 | Standard; divides cleanly into 256 |
| `ffn_expansion` | 2.67 | SwiGLU standard (compensates for 3 weight matrices) |
| `dropout` | 0.1 | Standard for transformers |
| `drop_path_rate` | 0.1 | Linear schedule 0→0.1 across layers |
| `learning_rate` | 1e-4 | AdamW default for transformers |
| `weight_decay` | 0.05 | Moderate regularization |
| `batch_size` | 256 | Fits in GPU memory; good gradient statistics |

### 7.2 Hyperparameter Search Space

| Parameter | Search Range | Distribution |
|-----------|--------------|--------------|
| `hidden_size` | {128, 256, 512} | Categorical |
| `num_layers` | {4, 6, 8} | Categorical |
| `num_heads` | {4, 8} | Categorical |
| `dropout` | {0.1, 0.2} | Categorical |
| `drop_path_rate` | {0.1, 0.2} | Categorical |
| `learning_rate` | [1e-5, 1e-3] | Log-uniform |
| `weight_decay` | {0.01, 0.05, 0.1} | Categorical |
| `batch_size` | {128, 256, 512} | Categorical |

### 7.3 Training Configuration

| Setting | Value |
|---------|-------|
| **Optimizer** | AdamW (β₁=0.9, β₂=0.999) |
| **LR Schedule** | Cosine decay with linear warmup |
| **Warmup Steps** | 5% of total steps |
| **Max Epochs** | 100 (with early stopping) |
| **Early Stopping** | Patience=10 on val AUC |
| **Loss Function** | Binary Cross-Entropy |
| **Gradient Clipping** | Max norm = 1.0 |

---

## Part 8: Parameter Count Estimate

```
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   TOKENIZATION                                                                  │
│   ├── CLS token:              1 × 256 = 256                                     │
│   ├── Numeric Linear (8):     8 × (1 × 256 + 256) = 4,096                       │
│   ├── Categorical Embed:      ~285 × 256 = 72,960 (total vocab)                 │
│   ├── Binary Embed (6):       6 × (2 × 256) = 3,072                             │
│   ├── Weather Linear:         15 × 256 + 256 = 4,096                            │
│   ├── Modality Embed:         5 × 256 = 1,280                                   │
│   └── Time2Vec:               ~2 × 256 = 512                                    │
│       Subtotal:               ~86K                                              │
│                                                                                 │
│   TRANSFORMER (per layer)                                                       │
│   ├── RMSNorm (×2):           2 × 256 = 512                                     │
│   ├── Q, K, V projections:    3 × (256 × 256) = 196,608                         │
│   ├── Output projection:      256 × 256 = 65,536                                │
│   └── SwiGLU FFN:             3 × (256 × 683) = 524,544                         │
│       Subtotal per layer:     ~787K                                             │
│       × 6 layers:             ~4.7M                                             │
│                                                                                 │
│   CLASSIFICATION HEAD                                                           │
│   ├── Linear (H→H):           256 × 256 + 256 = 65,792                          │
│   └── Linear (H→1):           256 + 1 = 257                                     │
│       Subtotal:               ~66K                                              │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   TOTAL:                      ~4.9M parameters                                  │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   Memory footprint (fp32):    ~20 MB                                            │
│   Memory footprint (fp16):    ~10 MB                                            │
│                                                                                 │
│   Appropriate for 82K training samples (samples/params ratio ≈ 17)              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 9: Design Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Normalization | RMSNorm (Pre-Norm) | Faster than LayerNorm, same quality, modern standard |
| FFN Activation | SwiGLU | Outperforms GELU in all benchmarks, used by LLaMA/PaLM |
| Regularization | Stochastic Depth + Dropout | Proven effective for transformers on limited data |
| Attention | Standard MHA + Flash Attention 2 | Simple, efficient at 49 tokens |
| Tokenization | Per-feature (FT-Transformer) | Proven for tabular data |
| Weather encoding | 1 token per timestep | Simple, attention learns importance |
| Temporal encoding | Time2Vec | Learnable, captures periodic + linear patterns |
| Multi-scale windows | 7d/30d/90d | Tactical, acclimatization, seasonal contexts |
| Cross-modal fusion | Full joint attention | No staged fusion, learns interactions naturally |
| Classification | [CLS] token pooling | Standard practice, well understood |
| Model size | ~5M params | Appropriate for 82K samples |

---

## Part 10: Module Structure

```
ml/
├── model/
│   ├── __init__.py
│   ├── transformer.py          # DeepSummitTransformer (main model)
│   ├── encoder.py              # TransformerEncoder (layer stack)
│   ├── attention.py            # MultiHeadAttention with Flash Attention
│   ├── ffn.py                  # SwiGLU feed-forward network
│   ├── normalization.py        # RMSNorm implementation
│   ├── tokenizer.py            # TabularTokenizer, WeatherTokenizer
│   ├── embeddings.py           # ModalityEmbedding, Time2Vec
│   └── head.py                 # ClassificationHead
│
├── training/
│   ├── __init__.py
│   ├── dataset.py              # ExpeditionDataset (PyTorch Dataset)
│   ├── train.py                # Training loop with W&B
│   ├── scheduler.py            # Cosine LR with warmup
│   └── metrics.py              # AUC, accuracy, calibration
│
└── inference/
    ├── __init__.py
    ├── service.py              # BentoML service
    └── preprocessing.py        # Feature preprocessing
```

---

## References

1. **RMSNorm**: Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
2. **SwiGLU**: Shazeer (2020) "GLU Variants Improve Transformer"
3. **Stochastic Depth**: Huang et al. (2016) "Deep Networks with Stochastic Depth"
4. **FT-Transformer**: Gorishniy et al. (2021) "Revisiting Deep Learning Models for Tabular Data"
5. **Time2Vec**: Kazemi et al. (2019) "Time2Vec: Learning a General Time Representation"
6. **Flash Attention 2**: Dao (2023) "FlashAttention-2: Faster Attention with Better Parallelism"
