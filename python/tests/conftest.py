"""
Pytest Configuration for Compresr SDK Tests

Usage:
    pytest                  # Run tests against production API
    pytest -v               # Verbose output
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env file from SDK root
env_file = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_file)


def pytest_configure(config):
    """Configure pytest."""
    print(f"\n{'='*50}")
    print(f"Testing against: https://api.compresr.ai")
    print(f"{'='*50}\n")


@pytest.fixture
def admin_api_key():
    """Get admin API key."""
    return (
        os.getenv("COMPRESR_API_KEY")
        or os.getenv("COMPRESSION_SERVICE_ADMIN_KEY")
    )


@pytest.fixture
def user_api_key():
    """Get user API key (rate limited)."""
    return os.getenv("COMPRESSION_SERVICE_USER_KEY")
