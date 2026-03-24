"""
Unit tests for the split rate limiter in api_v2.py.

Tests verify:
- general and upload buckets are independent
- each bucket enforces its own limit
- different tokens get separate counters
- bucket_prefix namespacing works correctly
"""
import os
import sys
import time
import pytest

# Ensure server modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api_v2 import _check_rate_limit, _rate_buckets


@pytest.fixture(autouse=True)
def clear_buckets():
    """Clear rate limit state before each test."""
    _rate_buckets.clear()
    yield
    _rate_buckets.clear()


class TestSplitRateLimiter:

    def test_general_bucket_enforces_limit(self):
        """General bucket should reject after its limit is reached."""
        limit = 5
        for i in range(limit):
            assert _check_rate_limit("tok", "general", limit) is True
        # Next request should be rejected
        assert _check_rate_limit("tok", "general", limit) is False

    def test_upload_bucket_enforces_limit(self):
        """Upload bucket should reject after its limit is reached."""
        limit = 10
        for i in range(limit):
            assert _check_rate_limit("tok", "upload", limit) is True
        assert _check_rate_limit("tok", "upload", limit) is False

    def test_general_and_upload_are_independent(self):
        """Exhausting the general bucket must NOT affect the upload bucket."""
        gen_limit = 3
        up_limit = 10
        # Exhaust general
        for _ in range(gen_limit):
            _check_rate_limit("tok", "general", gen_limit)
        assert _check_rate_limit("tok", "general", gen_limit) is False
        # Upload should still work
        assert _check_rate_limit("tok", "upload", up_limit) is True

    def test_upload_exhaustion_does_not_affect_general(self):
        """Exhausting the upload bucket must NOT affect the general bucket."""
        gen_limit = 10
        up_limit = 3
        for _ in range(up_limit):
            _check_rate_limit("tok", "upload", up_limit)
        assert _check_rate_limit("tok", "upload", up_limit) is False
        assert _check_rate_limit("tok", "general", gen_limit) is True

    def test_different_tokens_are_independent(self):
        """Different tokens should have separate counters within the same bucket."""
        limit = 2
        for _ in range(limit):
            _check_rate_limit("token_a", "general", limit)
        assert _check_rate_limit("token_a", "general", limit) is False
        # token_b should still be fresh
        assert _check_rate_limit("token_b", "general", limit) is True

    def test_bucket_key_includes_prefix_and_token(self):
        """Internal bucket keys should combine prefix + token hash."""
        _check_rate_limit("mytoken", "general", 100)
        _check_rate_limit("mytoken", "upload", 100)
        # Should have two distinct keys
        keys = list(_rate_buckets.keys())
        assert len(keys) == 2
        assert any(k.startswith("general:") for k in keys)
        assert any(k.startswith("upload:") for k in keys)

    def test_upload_limit_much_higher_than_general(self):
        """With realistic limits, uploads survive while general is exhausted."""
        gen_limit = 5
        up_limit = 50
        # Saturate general
        for _ in range(gen_limit):
            _check_rate_limit("tok", "general", gen_limit)
        assert _check_rate_limit("tok", "general", gen_limit) is False

        # Upload should handle many more
        for i in range(up_limit):
            assert _check_rate_limit("tok", "upload", up_limit) is True, (
                f"Upload request {i+1} rejected but limit is {up_limit}"
            )
        assert _check_rate_limit("tok", "upload", up_limit) is False


class TestConfigWiring:
    """Verify that the rate limit env vars are read correctly."""

    def test_general_limit_default(self):
        from api_v2 import RATE_LIMIT_GENERAL_PER_MINUTE
        # Default should be 60 (or whatever RATE_LIMIT_PER_MINUTE was in env)
        assert isinstance(RATE_LIMIT_GENERAL_PER_MINUTE, int)
        assert RATE_LIMIT_GENERAL_PER_MINUTE > 0

    def test_upload_limit_default(self):
        from api_v2 import RATE_LIMIT_UPLOAD_PER_MINUTE
        assert isinstance(RATE_LIMIT_UPLOAD_PER_MINUTE, int)
        assert RATE_LIMIT_UPLOAD_PER_MINUTE > 0

    def test_upload_limit_higher_than_general(self):
        from api_v2 import RATE_LIMIT_GENERAL_PER_MINUTE, RATE_LIMIT_UPLOAD_PER_MINUTE
        assert RATE_LIMIT_UPLOAD_PER_MINUTE >= RATE_LIMIT_GENERAL_PER_MINUTE, (
            f"Upload limit ({RATE_LIMIT_UPLOAD_PER_MINUTE}) should be >= general limit "
            f"({RATE_LIMIT_GENERAL_PER_MINUTE})"
        )
