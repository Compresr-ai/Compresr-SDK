# Contributing to Compresr SDK

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites
- Python>=3.9+
- Git

### Setup

```bash
# Clone the repository
git clone git@github.com:Compresr-ai/Compresr-SDK.git
cd Compresr-SDK

# Copy the environment template and add your API key
cp .env.example .env
```

## CI/CD Workflows

The repository has GitHub Actions workflows that run automatically on every push/PR:

1. **Python SDK** - Code style checks + tests
   - black (formatting)
   - isort (import sorting)
   - mypy (type checking)
   - ruff (linting)
   - Unit, integration, and performance tests

2. **curl SDK** - Shell script linting + tests
   - shellcheck (bash linting)
   - Test scripts validation

3. **Security** - Scans for hardcoded API keys

All workflows must pass before merging.

### GitHub Secrets Setup

Navigate to **Settings → Secrets and variables → Actions** and add:

- `COMPRESR_API_KEY` - Your Compresr API key for integration tests (optional, only needed for integration tests)

### Test Locally Before Pushing (Docker)

Run all CI workflows locally before pushing:

```bash
# Auto-installs act and runs all workflows
./test_ci_local.sh
```

For tests requiring an API key, create `.secrets` file:
```bash
echo "COMPRESR_API_KEY=your_key" > .secrets
./test_ci_local.sh
```

The script will:
1. Install `act` if not present (requires Homebrew on macOS)
2. Run Python SDK tests
3. Run curl SDK tests
4. Run security checks

See [.github/TESTING.md](.github/TESTING.md) for manual `act` usage.

## Questions?

- Open a [Discussion](https://github.com/compresr/sdk/discussions)
- Email: support@compresr.ai
