# Compresr SDK Tests

Integration tests for the Compresr Python SDK.

## Quick Start

```bash
cd sdk/python

# Install
pip install -e .
pip install pytest pytest-asyncio python-dotenv

# Run tests
pytest tests/ --env=dev -v
pytest tests/ --env=prod -v
```

## Environment Selection

The `--env` flag is **required**:

| Environment | Flag | Description |
|-------------|------|-------------|
| Development | `--env=dev` | Dev/staging |
| Production | `--env=prod` | Production |

## Running Tests

```bash
# All tests
pytest tests/ --env=dev -v

# Specific tests
pytest tests/test_compression_client.py::TestBasicCompression --env=dev -v

# With coverage
pytest tests/ --env=dev --cov=compresr --cov-report=term-missing -v
```

## API Keys

Set in `.env.test` at project root:

```
COMPRESSION_SERVICE_ADMIN_KEY=cmp_...
COMPRESSION_SERVICE_USER_KEY=cmp_...
```
