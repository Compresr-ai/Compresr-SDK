"""
Fixtures for performance tests.
"""

import pytest

from compresr import CompressionClient


@pytest.fixture
def admin_client(admin_api_key):
    """Create CompressionClient with ADMIN key."""
    if not admin_api_key:
        pytest.skip("Admin API key not available")
    return CompressionClient(api_key=admin_api_key)
