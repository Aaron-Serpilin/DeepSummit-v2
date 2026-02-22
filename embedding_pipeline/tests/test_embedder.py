import numpy as np

import embedder

SAMPLE_EXPEDITION = {"id": 42, "peak_id": 1, "year": 2023, "members": []}

def test_compute_returns_512_dim_array():
    result = embedder.compute(SAMPLE_EXPEDITION)

    assert result.shape == (512,)

def test_compute_returns_float32():
    result = embedder.compute(SAMPLE_EXPEDITION)

    assert result.dtype == np.float32

def test_compute_is_unit_normalized():
    result = embedder.compute(SAMPLE_EXPEDITION)

    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 1e-5

def test_compute_is_deterministic():
    """Same expedition ID must always produce the same vector."""
    result_a = embedder.compute(SAMPLE_EXPEDITION)
    result_b = embedder.compute(SAMPLE_EXPEDITION)

    np.testing.assert_array_equal(result_a, result_b)

def test_compute_differs_per_expedition():
    """Different expedition IDs must produce different vectors."""
    result_a = embedder.compute({"id": 1, "members": []})
    result_b = embedder.compute({"id": 2, "members": []})

    assert not np.array_equal(result_a, result_b)
