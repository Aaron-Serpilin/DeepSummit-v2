"""Tests for rate limiter."""

import threading
from time import time

from utils.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test time-spacing rate limiter."""

    def test_enforces_minimum_interval(self):
        """Requests are spaced at least min_interval apart."""
        limiter = RateLimiter(requests_per_minute=600)  # 100ms apart

        timestamps = []
        for _ in range(5):
            limiter.acquire()
            timestamps.append(time())

        # Check intervals between consecutive requests
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i - 1]
            assert interval >= 0.09  # Allow small tolerance (90ms for 100ms target)

    def test_parallel_workers_serialize(self):
        """Multiple workers wait their turn."""
        limiter = RateLimiter(requests_per_minute=300)  # 200ms apart
        timestamps = []
        lock = threading.Lock()

        def worker():
            for _ in range(3):
                limiter.acquire()
                with lock:
                    timestamps.append(time())

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(timestamps) == 9  # 3 workers × 3 requests

        # All requests should be reasonably spaced
        timestamps.sort()
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i - 1]
            assert interval >= 0.15  # At least 150ms (allowing tolerance for 200ms target)

    def test_stats_tracking(self):
        """Stats accurately reflect usage."""
        limiter = RateLimiter(requests_per_minute=1200)  # 50ms apart

        for _ in range(10):
            limiter.acquire()

        stats = limiter.get_stats()
        assert stats["total_requests"] == 10
        assert stats["target_rate_per_min"] == 1200

    def test_first_request_immediate(self):
        """First request should proceed immediately."""
        limiter = RateLimiter(requests_per_minute=60)  # 1 second apart

        start = time()
        limiter.acquire()
        elapsed = time() - start

        # First request should be nearly instant (< 50ms)
        assert elapsed < 0.05

    def test_min_interval_calculation(self):
        """Min interval is calculated correctly from requests_per_minute."""
        limiter = RateLimiter(requests_per_minute=600)
        assert abs(limiter.min_interval - 0.1) < 0.001  # 100ms ± 1ms

        limiter2 = RateLimiter(requests_per_minute=500)
        assert abs(limiter2.min_interval - 0.12) < 0.001  # 120ms ± 1ms
