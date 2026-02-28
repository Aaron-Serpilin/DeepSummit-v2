# DeepSummit v2: Training → Deployment Pipeline

> **Purpose:** This document outlines the complete pipeline from model training to production deployment. It serves both as reader documentation and context for future development sessions.

---

## Table of Contents

1. [Overview](#overview)
2. [Decisions Summary](#decisions-summary)
3. [Architecture Overview](#architecture-overview)
4. [Training Pipeline](#section-1-training-pipeline)
5. [Model Artifacts](#section-2-model-artifacts)
6. [Inference Service](#section-3-inference-service)
7. [Deployment](#section-4-deployment)
8. [Monitoring](#section-5-monitoring)
9. [Implementation Phases](#implementation-phases)
10. [Files to Create](#files-to-create)

---

## Overview

The unified multimodal transformer (4.9M parameters) is fully implemented and tested. This document covers the remaining infrastructure needed to:

1. **Train** the model on expedition data with W&B tracking
2. **Serve** predictions via BentoML on Cloud Run
3. **Monitor** for data drift with Evidently AI

**Cost philosophy:** Training uses a one-time GCP spot instance (~$1). Ongoing hosting uses free tiers (~$0/month).

---

## Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Training environment | GCP Compute Engine (spot T4) | One-time cost, full control |
| Model hosting | Cloud Run | Serverless, free tier, scales to zero |
| Experiment tracking | Weights & Biases | Free tier, industry standard |
| Inference mode | Real-time (<500ms) | Best UX for predictions |
| Explainability | SHAP explanations | Users understand why predictions are made |
| Monitoring | Evidently AI (batch) | Free tier friendly, daily reports |
| Model storage | GCS bucket | 5GB free tier, simple integration |
| Serving framework | BentoML | Professional ML serving, single container |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PHASE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   data/training/              ml/training/           GCS Bucket         │
│   ┌─────────────┐            ┌─────────────┐       ┌─────────────┐      │
│   │features.csv │───────────▶│   Trainer   │──────▶│  model.pt   │      │
│   │weather.csv  │            │  (PyTorch)  │       │  scaler.pkl │      │
│   └─────────────┘            └──────┬──────┘       │  vocab.json │      │
│         │                           │              └─────────────┘      │
│         │                           ▼                                   │
│         │                    ┌─────────────┐                            │
│         └───────────────────▶│    W&B      │                            │
│                              │  (tracking) │                            │
│                              └─────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PHASE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   User Request                                                          │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────┐     ┌─────────────────────────────────────────┐       │
│   │  Cloud Run  │────▶│           BentoML Service               │       │
│   │ (serverless)│     │  ┌───────────-┬──────────┬───────────┐  │       │
│   └─────────────┘     │  │Preprocessor│  Model   │   SHAP    │  │       │
│                       │  │ (validate) │(predict) │(explain)  │  │       │
│                       │  └───────────-┴──────────┴──────────-┘  │       │
│                       └─────────────────────────────────────────┘       │
│                                      │                                  │
│                                      ▼                                  │
│                              ┌─────────────┐                            │
│                              │  Response   │                            │
│                              │ probability │                            │ 
│                              │ explanations│                            │
│                              └─────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         MONITORING PHASE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Predictions ───▶ BigQuery ───▶ Evidently ───▶ Drift Report            │
│                     (logs)       (daily)        (GCS)                   │
│                                      │                                  │
│                                      ▼                                  │
│                              Alert if drift                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Section 1: Training Pipeline

### 1.1 Data Loading (`ml/data/`)

The training data consists of:
- `data/training/features.csv` — 81,886 expedition records with 20 features
- `data/training/weather.csv` — 6,224 unique weather windows (26 timesteps × 15 variables)

```python
# ml/data/dataset.py
class ExpeditionWeatherDataset(Dataset):
    """Loads features.csv + weather.csv, joins on weather_id."""

    def __init__(self, features_path: str, weather_path: str, split: str = "train"):
        # Load and merge CSVs
        # Apply train/val/test split (70/15/15)
        # Store preprocessed tensors

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "tabular": self.tabular[idx],      # (20,) - expedition features
            "weather": self.weather[idx],       # (26, 15) - multi-scale weather
            "label": self.labels[idx],          # (1,) - success/failure
        }
```

**Preprocessing pipeline:**
- **Numeric features (8):** `StandardScaler` — fit on train split only, transform all
- **Categorical features (6):** Vocabulary mapping to integer indices
- **Binary features (6):** Direct 0/1 encoding
- **Weather data:** Already normalized in weather.csv (z-scores)

### 1.2 Training Loop (`ml/training/`)

```python
# ml/training/trainer.py
class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        self.criterion = BCEWithLogitsLoss()

    def train_epoch(self) -> dict[str, float]:
        """Single training epoch with gradient accumulation."""
        # Log to W&B: loss, learning rate, gradient norms

    def validate(self) -> dict[str, float]:
        """Validation on held-out data."""
        # Compute: accuracy, precision, recall, F1, AUC-ROC
        # Log metrics to W&B

    def fit(self, epochs: int) -> None:
        """Main training loop with early stopping."""
        # Save best checkpoint based on val_loss
        # Upload final model to GCS
```

### 1.3 Hyperparameters

| Category | Parameter | Range | Default |
|----------|-----------|-------|---------|
| **Optimizer** | learning_rate | 1e-5 → 1e-3 | 1e-4 |
| | weight_decay | 0 → 0.1 | 0.01 |
| | warmup_steps | 0 → 500 | 100 |
| **Architecture** | hidden_size | 128, 256, 384 | 256 |
| | num_layers | 4, 6, 8 | 6 |
| | num_heads | 4, 8 | 8 |
| | ffn_expansion | 2.0, 2.67, 4.0 | 2.67 |
| **Regularization** | dropout | 0.05 → 0.3 | 0.1 |
| | drop_path_rate | 0 → 0.2 | 0.1 |
| **Training** | batch_size | 32, 64, 128 | 64 |
| | epochs | 30 → 100 | 50 |
| | label_smoothing | 0 → 0.1 | 0 |

W&B Sweeps will enable Bayesian hyperparameter optimization across these ranges.

### 1.4 Training Execution

```bash
# On GCP Compute Engine (spot T4 instance, ~$0.11/hr)
python -m ml.training.train \
    --data_dir data/training \
    --output_dir gs://deepsummit-models/v1 \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --wandb_project deepsummit
```

**Estimated training cost:** 2-4 hours × $0.11/hr = **$0.40-1.00**

---

## Section 2: Model Artifacts

### 2.1 GCS Bucket Structure

```
gs://deepsummit-models/
├── v1/
│   ├── model.pt           # State dict (~20MB for 4.9M params)
│   ├── config.json        # Hyperparameters used for training
│   ├── scaler.pkl         # StandardScaler for numeric features
│   ├── vocab.json         # Categorical feature vocabularies
│   └── metrics.json       # Final accuracy, F1, AUC-ROC
├── v2/
│   └── ...
└── production/
    └── model.pt           # Symlink to current production model
```

### 2.2 Quality Gates

Before promoting any model to production, it must pass:

1. **Accuracy ≥ 0.88** on held-out test set
2. **Latency ≤ 500ms** (single sample, warm model)
3. **All unit tests pass**

```bash
# After validating quality gates:
gsutil cp gs://deepsummit-models/v2/model.pt gs://deepsummit-models/production/
```

---

## Section 3: Inference Service

### 3.1 BentoML Service Definition

```python
# ml/inference/service.py
import bentoml

@bentoml.service(
    resources={"cpu": "1"},       # Cloud Run free tier compatible
    traffic={"timeout": 60}       # 60 second timeout
)
class DeepSummitService:

    def __init__(self):
        # Load model and preprocessing artifacts from GCS
        self.model = load_model("gs://deepsummit-models/production/model.pt")
        self.scaler = load_scaler("gs://deepsummit-models/production/scaler.pkl")
        self.vocab = load_vocab("gs://deepsummit-models/production/vocab.json")
        self.explainer = shap.DeepExplainer(self.model, background_data)

    @bentoml.api
    def predict(self, expedition: ExpeditionInput) -> PredictionOutput:
        """
        Main prediction endpoint.

        1. Validate input against schema
        2. Fetch weather from Open-Meteo (or cache)
        3. Preprocess features (scale, encode)
        4. Run model.predict()
        5. Compute SHAP explanations
        6. Return structured response
        """
        pass

    @bentoml.api
    def health(self) -> dict:
        """Health check endpoint for Cloud Run."""
        return {"status": "healthy", "model_version": self.version}
```

### 3.2 Input/Output Schemas

```python
# ml/inference/schemas.py
from pydantic import BaseModel, Field

class ExpeditionInput(BaseModel):
    """Input schema for prediction requests."""
    peak_id: str = Field(..., description="Peak code (EVER, KANG, etc.)")
    year: int = Field(..., ge=1920, le=2030)
    season: str = Field(..., description="Spring, Autumn, Winter, Summer")
    team_size: int = Field(..., ge=1, le=100)
    oxygen_used: bool
    commercial_route: bool
    # ... remaining 14 features

class FeatureExplanation(BaseModel):
    """SHAP explanation for a single feature."""
    feature: str
    value: float
    contribution: float
    direction: str  # "positive" or "negative"

class PredictionOutput(BaseModel):
    """Output schema for predictions."""
    probability: float = Field(..., ge=0.0, le=1.0)
    confidence: str  # "low", "medium", "high"
    explanations: list[FeatureExplanation]
    model_version: str
    latency_ms: float
```

### 3.3 Weather Integration

Weather data is fetched at inference time:

1. Check if weather data is cached (future: Redis with 1-hour TTL)
2. If miss: fetch from Open-Meteo API (free, 10k requests/day)
3. Build multi-scale windows using `utils/weather.py`:
   - 7-day window: 7 daily tokens
   - 30-day window: 10 × 3-day aggregates
   - 90-day window: 9 × 10-day aggregates
4. Return 26 timesteps × 15 weather variables

**Note:** Initial deployment skips Redis caching. Open-Meteo adds ~200ms latency but stays under the 500ms budget for a portfolio project.

---

## Section 4: Deployment

### 4.1 Cloud Run Configuration

```yaml
# infrastructure/cloudrun.yaml
service:
  name: deepsummit-inference
  region: europe-west4          # Amsterdam (closest to developer)

  container:
    image: gcr.io/deepsummit/inference:latest
    resources:
      cpu: 1
      memory: 512Mi             # Sufficient for 4.9M param model

  scaling:
    minInstances: 0             # Scale to zero (free tier friendly)
    maxInstances: 2             # Cost control

  traffic:
    timeout: 60s
    concurrency: 80             # Requests per instance
```

### 4.2 Deployment Commands

```bash
# Local development
bentoml serve ml.inference.service:DeepSummitService --reload

# Build BentoML container
bentoml build
bentoml containerize deepsummit:latest

# Push to Google Container Registry
docker push gcr.io/deepsummit/inference:latest

# Deploy to Cloud Run
gcloud run deploy deepsummit-inference \
    --image gcr.io/deepsummit/inference:latest \
    --region europe-west4 \
    --allow-unauthenticated \
    --min-instances 0 \
    --max-instances 2
```

### 4.3 CI/CD (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]
    paths: ['ml/inference/**', 'Dockerfile']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install bentoml

      - name: Build BentoML container
        run: |
          bentoml build
          bentoml containerize deepsummit:latest

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Push to GCR
        run: docker push gcr.io/${{ secrets.GCP_PROJECT }}/inference:latest

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy deepsummit-inference \
            --image gcr.io/${{ secrets.GCP_PROJECT }}/inference:latest \
            --region europe-west4 \
            --allow-unauthenticated
```

### 4.4 Cost Estimate

| Resource | Free Tier Allowance | Expected Usage | Cost |
|----------|---------------------|----------------|------|
| Cloud Run | 2M requests/mo, 180k vCPU-sec | <1k requests/mo | $0 |
| GCS | 5GB storage | ~50MB (model + artifacts) | $0 |
| Container Registry | 0.5GB storage | ~200MB image | $0 |
| **Monthly Total** | — | — | **~$0** |

---

## Section 5: Monitoring

### 5.1 Evidently AI Integration

Monitoring uses batch analysis rather than real-time streaming to stay within free tier limits.

```python
# ml/monitoring/drift_detection.py
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame
) -> Report:
    """
    Compare current predictions against training distribution.

    Args:
        reference_data: Training data distribution
        current_data: Recent prediction inputs/outputs

    Returns:
        Evidently Report with drift metrics
    """
    report = Report(metrics=[
        DataDriftPreset(),      # Feature distribution drift
        TargetDriftPreset(),    # Prediction distribution drift
    ])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=ColumnMapping(target="prediction")
    )
    return report
```

### 5.2 Monitoring Metrics

| Metric | Detection Method | Action |
|--------|------------------|--------|
| **Input drift** | Feature distribution shift (KS test, PSI) | Alert + investigate |
| **Prediction drift** | Output probability distribution shift | Alert + potential retrain |
| **Data quality** | Missing values, out-of-range values | Block prediction + alert |
| **Model performance** | Accuracy on labeled data (when available) | Retrain trigger |

### 5.3 Execution Schedule

- **Daily batch analysis** triggered by Cloud Scheduler (free tier)
- **Reports stored** in GCS: `gs://deepsummit-monitoring/reports/YYYY-MM-DD.html`
- **Alerting** via simple email (Gmail API, free) when drift detected

---

## Implementation Phases

### Phase 1: Training Pipeline (P0 — Blocks everything)

| Step | Task | Output |
|------|------|--------|
| 1.1 | Implement `ml/data/dataset.py` | ExpeditionWeatherDataset class |
| 1.2 | Implement `ml/data/preprocessing.py` | StandardScaler pipeline |
| 1.3 | Implement `ml/training/config.py` | TrainingConfig dataclass |
| 1.4 | Implement `ml/training/trainer.py` | Trainer class with W&B |
| 1.5 | Implement `ml/training/train.py` | CLI entry point |
| 1.6 | Add tests for data loading | `tests/unit/test_dataset.py` |
| 1.7 | Run training on GCP | Trained model in GCS |
| 1.8 | Validate quality gates | accuracy ≥ 0.88 |

### Phase 2: Inference Service (P0 — Blocks deployment)

| Step | Task | Output |
|------|------|--------|
| 2.1 | Implement `ml/inference/schemas.py` | Pydantic models |
| 2.2 | Implement `ml/inference/preprocessor.py` | Input validation |
| 2.3 | Implement `ml/inference/service.py` | BentoML service |
| 2.4 | Add SHAP explanations | Feature contributions |
| 2.5 | Test locally with `bentoml serve` | Working /predict endpoint |

### Phase 3: Deployment (P0 — Enables production)

| Step | Task | Output |
|------|------|--------|
| 3.1 | Create `infrastructure/cloudrun.yaml` | Cloud Run config |
| 3.2 | Build and containerize | Docker image |
| 3.3 | Deploy to Cloud Run | Live endpoint |
| 3.4 | Set up CI/CD workflow | `.github/workflows/deploy.yml` |
| 3.5 | Verify latency ≤ 500ms | Performance validated |

### Phase 4: Monitoring (P1 — Production reliability)

| Step | Task | Output |
|------|------|--------|
| 4.1 | Implement `ml/monitoring/drift_detection.py` | Evidently reports |
| 4.2 | Set up Cloud Scheduler | Daily batch job |
| 4.3 | Configure alerting | Email notifications |
| 4.4 | Store reports in GCS | Historical tracking |

---

## Files to Create

```
ml/
├── data/
│   ├── __init__.py
│   ├── dataset.py           # ExpeditionWeatherDataset
│   └── preprocessing.py     # StandardScaler, vocab encoding
├── training/
│   ├── __init__.py
│   ├── config.py           # TrainingConfig dataclass
│   ├── trainer.py          # Trainer class with W&B
│   └── train.py            # CLI entry point
├── inference/
│   ├── __init__.py
│   ├── service.py          # BentoML DeepSummitService
│   ├── schemas.py          # Pydantic input/output models
│   └── preprocessor.py     # Input validation + transformation
└── monitoring/
    ├── __init__.py
    ├── drift_detection.py  # Evidently report generation
    └── alerts.py           # Simple email alerting

infrastructure/
├── cloudrun.yaml           # Cloud Run service config
└── deploy.sh               # Deployment helper script

.github/workflows/
└── deploy.yml              # CI/CD for Cloud Run deployment

tests/
└── unit/
    ├── test_dataset.py     # Data loading tests
    ├── test_trainer.py     # Training loop tests
    └── test_inference.py   # BentoML service tests
```

---

## Verification Checklist

Before considering each phase complete:

### Training Pipeline
- [ ] `ExpeditionWeatherDataset` loads and joins CSVs correctly
- [ ] Train/val/test splits are reproducible (seed=42)
- [ ] `StandardScaler` fits only on training data
- [ ] Training completes without OOM errors
- [ ] W&B logs loss, metrics, and hyperparameters
- [ ] Best checkpoint saved to GCS
- [ ] Model accuracy ≥ 0.88 on test set

### Inference Service
- [ ] BentoML service starts without errors
- [ ] `/predict` endpoint accepts valid input
- [ ] `/predict` returns probability in [0, 1]
- [ ] SHAP explanations included in response
- [ ] Invalid input returns 400 with clear error message
- [ ] `/health` endpoint returns healthy status

### Deployment
- [ ] Docker image builds successfully
- [ ] Container runs locally with same behavior
- [ ] Cloud Run deployment succeeds
- [ ] Endpoint responds within 500ms (cold start may be slower)
- [ ] CI/CD triggers on push to main
- [ ] Rollback procedure documented

### Monitoring
- [ ] Evidently report generates without errors
- [ ] Drift detected on synthetic shifted data
- [ ] Reports stored in GCS with correct naming
- [ ] Alert sent when drift threshold exceeded

---

## References

- [BentoML Documentation](https://docs.bentoml.com/)
- [Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [W&B Sweeps Guide](https://docs.wandb.ai/guides/sweeps)
- [Open-Meteo API](https://open-meteo.com/en/docs)
