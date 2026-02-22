import numpy as np

# STUB: deterministic random unit vector seeded by expedition ID.
# Phase 2: replace the body of compute() with a BentoML HTTP client call.
# The function signature and MODEL_VERSION constant stay — only the body changes.
MODEL_VERSION = "stub-v0"

def compute(expedition: dict) -> np.ndarray:
    """Compute a 512-dim unit-normalised embedding for an expedition.

    Currently a deterministic random stub — same expedition ID always yields
    the same vector. Replace this body (not the signature) with the BentoML
    HTTP call when the transformer model is ready in Phase 2.
    """
    rng = np.random.default_rng(seed=expedition["id"])
    vector = rng.standard_normal(512)
    return (vector / np.linalg.norm(vector)).astype(np.float32)
