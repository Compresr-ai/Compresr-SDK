#!/bin/bash
# Test configuration
# This file defines the base URL and can be sourced by all test scripts
#
# The URL is set by:
#   1. COMPRESR_BASE_URL env var if already exported (from test_ci_local.sh)
#   2. USE_PROD flag to select production vs localhost
#   3. Default to localhost

# Only set URL if not already exported
if [ -z "$COMPRESR_BASE_URL" ]; then
    if [ "${USE_PROD:-0}" = "1" ]; then
        export COMPRESR_BASE_URL="https://api.compresr.ai"
    else
        export COMPRESR_BASE_URL="http://localhost:8000"
    fi
fi
export BASE_URL="$COMPRESR_BASE_URL"

# Load API key from .env if not already set
if [ -z "$COMPRESR_API_KEY" ] && [ -f "../../.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "../../.env"
    set +a
fi

# Verify API key is set
if [ -z "$COMPRESR_API_KEY" ]; then
    echo "✗ COMPRESR_API_KEY not set. Please set it in .env or export it."
    exit 1
fi

echo "Using API endpoint: $BASE_URL"
