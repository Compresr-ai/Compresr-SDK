"""
Pytest Configuration for Compresr SDK Tests

Usage:
    pytest                      # Run tests against localhost (default)
    pytest --prod               # Run tests against production API
    pytest -v                   # Verbose output

Set COMPRESR_BASE_URL to test against a different environment.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env file from SDK root
env_file = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_file)


def pytest_addoption(parser):
    """Add --prod command line option."""
    parser.addoption(
        "--prod",
        action="store_true",
        default=False,
        help="Run tests against production API (api.compresr.ai)",
    )


def pytest_configure(config):
    """Configure pytest."""
    use_prod = config.getoption("--prod", default=False)

    if use_prod:
        os.environ["COMPRESR_BASE_URL"] = "https://api.compresr.ai"
    elif "COMPRESR_BASE_URL" not in os.environ:
        os.environ["COMPRESR_BASE_URL"] = "http://localhost:8000"

    base_url = os.environ["COMPRESR_BASE_URL"]
    env_name = "PRODUCTION" if "api.compresr.ai" in base_url else "LOCAL"

    print(f"\n{'='*50}")
    print(f"Testing against: {base_url}")
    print(f"Environment: {env_name}")
    print(f"{'='*50}\n")


@pytest.fixture
def admin_api_key():
    """Get admin API key."""
    return os.getenv("COMPRESR_API_KEY") or os.getenv("COMPRESSION_SERVICE_ADMIN_KEY")


@pytest.fixture
def user_api_key():
    """Get user API key (rate limited)."""
    return os.getenv("COMPRESSION_SERVICE_USER_KEY")
