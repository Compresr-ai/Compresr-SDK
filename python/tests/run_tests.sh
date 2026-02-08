#!/bin/bash
# SDK Test Runner Script
#
# Usage:
#   ./run_tests.sh prod          # Run all tests against production
#   ./run_tests.sh dev           # Run all tests against dev (localhost)
#   ./run_tests.sh prod quick    # Run only fast tests against production
#   ./run_tests.sh dev coverage  # Run with coverage report

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if environment argument is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Environment argument required!${NC}"
    echo ""
    echo "Usage:"
    echo "  ./run_tests.sh prod          # Run all tests against production"
    echo "  ./run_tests.sh dev           # Run all tests against dev"
    echo "  ./run_tests.sh prod quick    # Run only fast tests"
    echo "  ./run_tests.sh prod coverage # Run with coverage report"
    exit 1
fi

ENV=$1
MODE=${2:-"full"}

# Validate environment
if [ "$ENV" != "prod" ] && [ "$ENV" != "dev" ]; then
    echo -e "${RED}Error: Environment must be 'prod' or 'dev'${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  SDK Test Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Environment: ${GREEN}$ENV${NC}"
echo -e "Mode: ${GREEN}$MODE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Navigate to SDK directory
cd "$(dirname "$0")/.."

# Run tests based on mode
case $MODE in
    "quick")
        echo -e "${YELLOW}Running quick tests (skipping slow tests)...${NC}"
        python -m pytest tests/ --env=$ENV -v -m "not slow"
        ;;

    "coverage")
        echo -e "${YELLOW}Running tests with coverage report...${NC}"
        python -m pytest tests/ --env=$ENV -v \
            --cov=compresr \
            --cov-report=html \
            --cov-report=term-missing
        echo ""
        echo -e "${GREEN}Coverage report generated: htmlcov/index.html${NC}"
        ;;

    "full")
        echo -e "${YELLOW}Running all tests...${NC}"
        python -m pytest tests/ --env=$ENV -v --tb=line
        ;;

    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Available modes: full, quick, coverage"
        exit 1
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo ""
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
