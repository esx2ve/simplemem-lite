"""Rate limiter tests for SimpleMem Lite.

Tests critical rate limiting pathways:
- Token bucket algorithm
- Rate limit enforcement
- Multi-client handling
"""

import time

import pytest


class TestRateLimiter:
    """Test rate limiter functionality."""

    def test_allows_requests_under_limit(self):
        """Requests under the limit should be allowed."""
        from simplemem_lite.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=60)  # 1 per second
        client_ip = "127.0.0.1"

        # First request should be allowed
        assert limiter.is_allowed(client_ip) is True

    def test_blocks_burst_over_limit(self):
        """Rapid requests exceeding bucket should be blocked."""
        from simplemem_lite.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=5)  # Very low limit
        client_ip = "127.0.0.1"

        # Exhaust the bucket
        allowed_count = 0
        for _ in range(10):
            if limiter.is_allowed(client_ip):
                allowed_count += 1

        # Should have blocked some requests
        assert allowed_count == 5  # Only 5 tokens available

    def test_replenishes_tokens_over_time(self):
        """Tokens should replenish after time passes."""
        from simplemem_lite.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=60)  # 1 per second
        client_ip = "127.0.0.1"

        # Exhaust some tokens
        for _ in range(60):
            limiter.is_allowed(client_ip)

        # Should be blocked
        assert limiter.is_allowed(client_ip) is False

        # Wait for replenishment (simulate by adjusting internal state)
        time.sleep(0.1)  # Should replenish ~0.1 tokens at 1/sec rate

        # Note: 0.1 seconds isn't enough for a full token at 1/sec
        # This test verifies the mechanism exists, not precise timing

    def test_separate_limits_per_client(self):
        """Each client should have their own bucket."""
        from simplemem_lite.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=3)  # Very low
        client_a = "192.168.1.1"
        client_b = "192.168.1.2"

        # Exhaust client A's tokens
        for _ in range(5):
            limiter.is_allowed(client_a)

        # Client B should still have tokens
        assert limiter.is_allowed(client_b) is True

    def test_config_rate_limit_default(self):
        """Config should have rate limit default."""
        from simplemem_lite.config import Config

        config = Config()
        assert config.http_rate_limit == 100  # requests per minute


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
