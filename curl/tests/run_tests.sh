#!/bin/bash
set -e

# Parse arguments (only if COMPRESR_BASE_URL not already set)
if [ -z "$COMPRESR_BASE_URL" ]; then
    USE_PROD=0
    for arg in "$@"; do
        case $arg in
            --prod)
                USE_PROD=1
                shift
                ;;
        esac
    done
    export USE_PROD
fi

echo "====================================="
echo "Running curl SDK tests"
if [ "$COMPRESR_BASE_URL" = "https://api.compresr.ai" ] || [ "$USE_PROD" = "1" ]; then
    echo "Environment: PRODUCTION (api.compresr.ai)"
else
    echo "Environment: LOCAL (localhost:8000)"
fi
echo "====================================="

# Navigate to tests directory
cd "$(dirname "$0")"

# Make test scripts executable
chmod +x test_*.sh

# Track test results
PASSED=0
FAILED=0

# Run each test
for test in test_*.sh; do
    echo ""
    echo "Running $test..."
    if bash "$test"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "====================================="
echo "Test Results: $PASSED passed, $FAILED failed"
echo "====================================="

if [ $FAILED -eq 0 ]; then
    exit 0
else
    exit 1
fi
