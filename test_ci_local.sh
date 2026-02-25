#!/bin/bash
set -e

# Parse arguments
USE_PROD=0
for arg in "$@"; do
    case $arg in
        --prod)
            USE_PROD=1
            shift
            ;;
        -h|--help)
            echo "Usage: ./test_ci_local.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --prod    Run tests against production API (api.compresr.ai)"
            echo "            Default: localhost:8000"
            exit 0
            ;;
    esac
done
export USE_PROD

echo "====================================="
echo "Compresr SDK - CI Test Runner"
if [ "$USE_PROD" = "1" ]; then
    echo "Environment: PRODUCTION (api.compresr.ai)"
else
    echo "Environment: LOCAL (localhost:8000)"
fi
echo "====================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load .env for API key
if [ -f .env ]; then
    set -a
    # shellcheck source=/dev/null
    . .env
    set +a
fi

echo ""
FAILED=0

# Run Python SDK linting
echo "Running Python SDK Linting..."
cd python
if black . --check --diff && isort . --check-only --diff && mypy compresr && ruff check .; then
    echo -e "${GREEN}✓ Python SDK linting passed${NC}"
else
    echo -e "${RED}✗ Python SDK linting failed${NC}"
    FAILED=$((FAILED + 1))
fi
cd ..

echo ""

# Run Python SDK unit tests
echo "Running Python SDK unit tests..."
cd python
if pytest tests/unit -v; then
    echo -e "${GREEN}✓ Python SDK unit tests passed${NC}"
else
    echo -e "${RED}✗ Python SDK unit tests failed${NC}"
    FAILED=$((FAILED + 1))
fi
cd ..

echo ""

# Run Python SDK integration tests
echo "Running Python SDK integration tests..."
cd python
PYTEST_PROD_FLAG=""
if [ "$USE_PROD" = "1" ]; then
    PYTEST_PROD_FLAG="--prod"
fi
if pytest tests/integration -v --timeout=60 $PYTEST_PROD_FLAG; then
    echo -e "${GREEN}✓ Python SDK integration tests passed${NC}"
else
    if [ "$USE_PROD" = "1" ]; then
        echo -e "${RED}✗ Python SDK integration tests failed${NC}"
    else
        echo -e "${YELLOW}⚠ Python SDK integration tests failed (may need backend running)${NC}"
        echo -e "${YELLOW}  Start backend: cd compresr-platform/backend && uvicorn app.main:app${NC}"
    fi
    FAILED=$((FAILED + 1))
fi
cd ..

echo ""

# Run curl SDK tests
echo "Running curl SDK tests..."
CURL_PROD_FLAG=""
if [ "$USE_PROD" = "1" ]; then
    CURL_PROD_FLAG="--prod"
fi
if bash curl/tests/run_tests.sh $CURL_PROD_FLAG; then
    echo -e "${GREEN}✓ curl SDK tests passed${NC}"
else
    echo -e "${RED}✗ curl SDK tests failed${NC}"
    FAILED=$((FAILED + 1))
fi

echo ""

# Run Security checks
echo "Running Security checks..."
if ! grep -rE "cp-[a-zA-Z0-9]{20,}" --include="*.py" --include="*.sh" python/ curl/ 2>/dev/null | grep -v "example" | grep -q .; then
    echo -e "${GREEN}✓ Security check passed (no hardcoded secrets)${NC}"
else
    echo -e "${RED}✗ Security check failed (hardcoded secrets found)${NC}"
    FAILED=$((FAILED + 1))
fi

echo ""
echo "====================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All CI tests passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILED CI test(s) failed${NC}"
    exit 1
fi
