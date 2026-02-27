"""Rate limiter for coordinating parallel API calls within rate limits."""

import logging
from threading import Lock
from time import sleep, time

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Time-spacing rate limiter that enforces minimum intervals between requests.

    Unlike token bucket, this guarantees evenly-spaced requests which matches
    how most API rate limiters work (rolling window detection).

    Workers call acquire() before making requests - this blocks until the
    minimum interval has passed since the last request from ANY worker.

    Args:
        requests_per_minute: Target request rate (use conservative value, e.g., 500 for 600 limit)
    """

    def __init__(self, requests_per_minute: int = 500) -> None:
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute  # seconds between requests
        self.last_request_time = 0.0
        self.lock = Lock()

        # Stats for monitoring
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.start_time = time()

    def acquire(self) -> None:
        """
        Block until enough time has passed since the last request.

        Thread-safe: only one request will proceed at a time globally.
        """
        with self.lock:
            now = time()
            time_since_last = now - self.last_request_time

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                self.total_wait_time += wait_time
                sleep(wait_time)

            # Update timestamp AFTER waiting
            self.last_request_time = time()
            self.total_requests += 1

    def get_stats(self) -> dict[str, float]:
        """Get rate limiting statistics."""
        with self.lock:
            elapsed = time() - self.start_time
            actual_rate = self.total_requests / elapsed * 60 if elapsed > 0 else 0

            return {
                "total_requests": self.total_requests,
                "total_wait_time_seconds": round(self.total_wait_time, 2),
                "elapsed_minutes": round(elapsed / 60, 2),
                "actual_rate_per_min": round(actual_rate, 1),
                "target_rate_per_min": self.requests_per_minute,
            }
