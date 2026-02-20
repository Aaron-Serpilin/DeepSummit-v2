# DeepSummit — Architecture

This document covers the high-level system design, component relationships, and the reasoning behind key technical decisions.

---

## System Overview

DeepSummit is a serverless ML system deployed on GCP. All services run on Cloud Run (scale-to-zero, pay-per-request). The system has five runtime components and an async event pipeline:

```
                        ┌──────────────────────────────┐
                        │  React Frontend (Vite)        │
                        │  CesiumJS Globe               │
                        │  Prediction Form + Results    │
                        └──────────────┬───────────────┘
                                       │ HTTPS
                        ┌──────────────▼───────────────┐
                        │  FastAPI  (Cloud Run)         │
                        │  REST API + auth + rate limit │
                        └────┬─────────┬───────────────┘
                             │         │
               ┌─────────────▼─┐   ┌───▼──────────────┐
               │  PostgreSQL   │   │  Redis            │
               │  + pgvector   │   │  (weather cache)  │
               └───────────────┘   └───────────────────┘
                             │
                   ┌─────────▼────────────┐
                   │  BentoML  (Cloud Run) │
                   │  Inference service    │
                   │  (internal only)      │
                   └──────────────────────┘

                   Pub/Sub (async)
                        │
          ┌─────────────┴─────────────┐
          │                           │
 ┌────────▼──────────┐   ┌────────────▼──────────┐
 │  Embedding        │   │  Monitoring           │
 │  Pipeline         │   │  (Evidently AI)       │
 │  (Cloud Run)      │   │  (Cloud Run, daily)   │
 └───────────────────┘   └───────────────────────┘
```

The API service is the only public-facing backend endpoint. The inference service has no public URL and is called internally. Redis sits between the API and the weather data source to eliminate Open-Meteo latency at prediction time.

---

## Inference Data Flow

When a prediction request arrives:

```
Request
  │
  ├─ 1. Validate schema (Pydantic)
  ├─ 2. Check API key
  ├─ 3. Check rate limit (10 req/min)
  │
  ├─ 4. Fetch weather data
  │      Redis HIT  → ~5ms
  │      Redis MISS → Open-Meteo API (~100ms) → cache with 1h TTL
  │
  ├─ 5. Build feature tensors
  │      Tabular features
  │      7-day / 30-day / 90-day weather windows
  │
  ├─ 6. Call inference service (internal HTTP)
  │      Single forward pass → logit, SHAP values, attention weights
  │
  └─ 7. Build and return response
         success_probability, risk_level, shap_values,
         attention_weights, top_risk_factors, current_conditions
```

Target p95 latency: **< 500ms** (typically ~220ms on a warm cache hit).

---

## ML Model

### Unified Multimodal Transformer

The core insight is that expedition metadata (who, when, what route) and weather data (what conditions prevailed) need to interact — the same wind speed means something different depending on the climber's experience level. A unified architecture with joint attention over all tokens handles this naturally.

```
Inputs
├── Tabular (~40 expedition features)
│   └── Each feature → linear projection → 1 token
└── Weather (multi-scale)
    ├── 7-day window  — 7 days × 55 variables   = 385 tokens (full resolution)
    ├── 30-day window — 10 aggregates × 55 vars  = 110 tokens (3-day buckets)
    └── 90-day window — 9 aggregates × 55 vars   =  99 tokens (10-day buckets)

All tokens → [CLS] + joint transformer encoder → sigmoid → P(success)
Total sequence length: ~635 tokens
```

**Why per-feature tokenization?**
Each tabular feature gets its own linear projection into the model's hidden dimension. The transformer then learns pairwise interactions between features, rather than receiving a pre-mixed feature vector. This is the FT-Transformer approach and consistently outperforms standard MLP baselines on tabular data.

**Why multi-scale weather windows?**
Summit success depends on conditions at different timescales simultaneously:
- **7-day**: The actual summit window — jet stream position, storm fronts
- **30-day**: Acclimatization period weather — fitness and adaptation
- **90-day**: Seasonal context — monsoon timing, snowpack

Aggregating the longer windows (3-day and 10-day buckets) keeps the sequence length manageable without losing temporal coverage.

**Why joint attention throughout?**
Some architectures encode modalities separately and fuse at the end. Here, all tokens attend to all other tokens at every layer. The model can discover cross-modal interactions (e.g., "this wind pattern matters more for a solo alpine climber than a commercial expedition") without those interactions being manually engineered.

### Temporal Encoding

Weather tokens are position-encoded with **Time2Vec**: a learned encoding with one linear component (trend) and `k` sinusoidal components (periodic patterns). This is combined with explicit `day_of_year` and `days_before_summit` features to help the model understand seasonal position and proximity to the target date.

### Architecture Details

| Component | Choice | Reason |
|-----------|--------|--------|
| Attention | Flash Attention 2 | Memory-efficient for 600+ token sequences |
| Normalization | Pre-LayerNorm | Stable training; modern convention |
| FFN | SwiGLU | State-of-the-art activation (LLaMA, PaLM); gated structure improves expressiveness |
| Regularization | Stochastic Depth (DropPath) | Effective for medium datasets; drops entire residual paths during training |
| Explainability | SHAP + attention weights | SHAP for feature attribution; attention for temporal focus |

---

## Data Architecture

### PostgreSQL Schema (Summary)

Six tables store all relational data and embeddings:

```
peaks ──< routes ──< expeditions ──< expedition_members
  │                      │
  └──< weather_cache      └──< expedition_embeddings (vector(512))
```

`expedition_embeddings` uses **pgvector** with an IVFFlat index for approximate nearest-neighbour search — enabling "similar past expeditions" lookups in the results view.

### Weather Data

Weather is sourced from **Open-Meteo**, which provides free access to ERA5 reanalysis data (1940–present) and forecasts across 55+ variables at ~100ms latency. All fetched data is:
1. Stored in Redis with a 1-hour TTL (fast repeated access)
2. Persisted to PostgreSQL (historical archive, avoids re-fetching)

At prediction time, weather data is always served from cache — the model never waits on an external API call during inference.

---

## Key Architectural Decisions

### Single unified transformer vs. dual-transformer (v1)

v1 used SAINT (tabular) and Stormer (meteorological) as separate encoders with an MLP fusion layer. This required two forward passes, introduced an artificial bottleneck at the fusion point, and made cross-modal interaction impossible within each encoder.

v2 uses a single transformer where tabular and weather tokens coexist from layer one. The model attends freely across modalities, learns its own fusion strategy, and produces a single forward pass — simpler, faster, and more expressive.

### Cloud Run over Kubernetes

Every service is a Cloud Run container. The tradeoffs:
- **Pros**: No cluster management, scale-to-zero, per-request billing, straightforward deployment
- **Cons**: Cold starts (mitigated with `min-instances: 1` on the inference service), no persistent connections

For a portfolio system at this scale, operational simplicity wins over the flexibility of Kubernetes.

### BentoML for model serving

BentoML packages the model, preprocessing pipeline, and SHAP computation into a single deployable unit (a "bento"). The API service stays thin — it builds tensors, calls the inference service via HTTP, and serialises the response. Model versions are tracked in W&B and promoted to BentoML independently of the API.

### Open-Meteo over direct ERA5

v1 downloaded ERA5 data directly, which took 30–60 minutes per inference. Open-Meteo wraps ERA5 (and more) in a free REST API with ~100ms response times. The data is identical; the latency drops by four orders of magnitude.

### Redis + PostgreSQL dual caching

Redis handles the hot path (sub-millisecond reads for recent weather). PostgreSQL stores the full historical archive. This avoids re-fetching data for peaks and date ranges already in the system, which matters for batch training runs as much as for inference.

---

## Async Pipeline

Two Pub/Sub topics decouple the synchronous request path from background work:

| Topic | Published by | Consumed by | Purpose |
|-------|-------------|-------------|---------|
| `expedition-data-added` | Data ingestion scripts | Embedding pipeline | Compute and store vector(512) embeddings for new expeditions |
| `prediction-logged` | API service (async) | Monitoring service | Feed Evidently AI drift detection |

The monitoring service runs on a Cloud Scheduler daily trigger. It loads the reference distribution (training data statistics) and the current prediction window, computes feature and prediction drift (Population Stability Index), and fires a W&B alert if PSI > 0.2.

---

## CI/CD

```
PR opened
  └── GitHub Actions
        ├── ruff (linting)
        ├── mypy (type checking)
        ├── pytest unit + integration
        └── model performance gate: accuracy ≥ 0.88, latency ≤ 500ms

Merge to main
  └── Cloud Build
        ├── Build Docker images (backend, ml, embedding-pipeline, frontend)
        ├── Push to Artifact Registry
        ├── Deploy to Cloud Run — canary (10% traffic)
        ├── Wait 10 minutes + health checks
        ├── Promote to 100% — or rollback
        └── Smoke tests against production
```

The model performance gate in CI ensures that no deployment regresses below the accuracy threshold established in v1.
