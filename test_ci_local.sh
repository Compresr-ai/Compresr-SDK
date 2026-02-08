#!/bin/bash
set -e

echo "====================================="
echo "Compresr SDK - CI Test Runner"
echo "====================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if act is installed
if ! command -v act &> /dev/null; then
    echo -e "${YELLOW}act is not installed. Installing...${NC}"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install act
        else
            echo -e "${RED}Error: Homebrew not found. Please install Homebrew first:${NC}"
            echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Installing act for Linux..."
        curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
    else
        echo -e "${RED}Error: Unsupported OS. Please install act manually:${NC}"
        echo "  https://github.com/nektos/act"
        exit 1
    fi
    
    echo -e "${GREEN}✓ act installed${NC}"
fi

echo ""
echo "Running all CI workflows locally with act..."
echo ""

# Check if .secrets file exists for API key
if [ -f .secrets ]; then
    echo -e "${YELLOW}Found .secrets file, using for tests${NC}"
    SECRET_FLAG="--secret-file .secrets"
else
    echo -e "${YELLOW}No .secrets file found, some tests may be skipped${NC}"
    echo "Create .secrets with: echo \"COMPRESR_API_KEY=your_key\" > .secrets"
    SECRET_FLAG=""
fi

echo ""
FAILED=0

# Run Python SDK workflow
echo "Running Python SDK CI..."
if act -W .github/workflows/python-ci.yml $SECRET_FLAG; then
    echo -e "${GREEN}✓ Python SDK CI passed${NC}"
else
    echo -e "${RED}✗ Python SDK CI failed${NC}"
    ((FAILED++))
fi

echo ""

# Run curl SDK workflow
echo "Running curl SDK CI..."
if act -W .github/workflows/curl-ci.yml $SECRET_FLAG; then
    echo -e "${GREEN}✓ curl SDK CI passed${NC}"
else
    echo -e "${RED}✗ curl SDK CI failed${NC}"
    ((FAILED++))
fi

echo ""

# Run Security workflow
echo "Running Security CI..."
if act -W .github/workflows/security.yml; then
    echo -e "${GREEN}✓ Security CI passed${NC}"
else
    echo -e "${RED}✗ Security CI failed${NC}"
    ((FAILED++))
fi

echo ""
echo "====================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All CI workflows passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILED CI workflow(s) failed${NC}"
    exit 1
fi
