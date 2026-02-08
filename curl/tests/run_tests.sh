#!/bin/bash
set -e

echo "====================================="
echo "Running curl SDK tests"
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
