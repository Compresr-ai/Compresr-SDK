"""
Pytest Configuration for Compresr SDK Tests

Usage:
    pytest                  # Test against production (default)
    pytest --env=dev        # Test against dev environment
    pytest --env=prod       # Test against production
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env file from SDK root
env_file = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_file)

ENVIRONMENT_URLS = {
    "dev": os.getenv(
        "DEV_COMPRESR_BASE_URL", os.getenv("COMPRESR_BASE_URL", "http://localhost:8000")
    ),
    "prod": os.getenv("PROD_COMPRESR_BASE_URL", "https://api.compresr.ai"),
}


def pytest_addoption(parser):
    """Add --env option."""
    parser.addoption(
        "--env",
        action="store",
        default="prod",
        choices=["dev", "prod"],
        help="Environment to test: 'dev' or 'prod' (default: prod)",
    )


def pytest_configure(config):
    """Configure pytest with environment settings."""
    env = config.getoption("--env")

    env_path = Path(__file__).parent.parent.parent.parent / ".env.test"
    load_dotenv(env_path)

    base_url = ENVIRONMENT_URLS.get(env)
    os.environ["COMPRESR_TEST_BASE_URL"] = base_url
    os.environ["COMPRESR_TEST_ENV"] = env

    print(f"\n{'='*50}")
    print(f"Testing: {env.upper()} | URL: {base_url}")
    print(f"{'='*50}\n")


@pytest.fixture(scope="session")
def base_url():
    """Get base URL for current environment."""
    return os.environ.get("COMPRESR_TEST_BASE_URL")


@pytest.fixture(scope="session")
def test_env():
    """Get current test environment."""
    return os.environ.get("COMPRESR_TEST_ENV")


@pytest.fixture
def admin_api_key():
    """Get admin API key."""
    env = os.environ.get("COMPRESR_TEST_ENV")
    if env == "dev":
        return os.getenv("COMPRESSION_SERVICE_ADMIN_KEY") or os.getenv("COMPRESR_API_KEY")
    elif env == "prod":
        return (
            os.getenv("PROD_COMPRESSION_SERVICE_ADMIN_KEY")
            or os.getenv("COMPRESSION_SERVICE_ADMIN_KEY")
            or os.getenv("COMPRESR_API_KEY")
        )
    return None


@pytest.fixture
def user_api_key():
    """Get user API key (rate limited)."""
    env = os.environ.get("COMPRESR_TEST_ENV")
    if env == "dev":
        return os.getenv("COMPRESSION_SERVICE_USER_KEY")
    elif env == "prod":
        return os.getenv("PROD_COMPRESSION_SERVICE_USER_KEY") or os.getenv(
            "COMPRESSION_SERVICE_USER_KEY"
        )
    return None
