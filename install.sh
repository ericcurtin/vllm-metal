#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Repository information
REPO_OWNER="ericcurtin"
REPO_NAME="vllm-metal"
PACKAGE_NAME="vllm-metal"

echo "=========================================="
echo "  vllm-metal Installer"
echo "=========================================="
echo ""

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}Error: This package is only supported on macOS.${NC}"
    exit 1
fi

# Check if running on Apple Silicon (arm64)
if [[ "$(uname -m)" != "arm64" ]]; then
    echo -e "${RED}Error: This package is only supported on Apple Silicon (arm64) Macs.${NC}"
    echo "Your architecture: $(uname -m)"
    exit 1
fi

echo -e "${GREEN}✓${NC} Running on macOS Apple Silicon"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    echo "Please install Python 3.10 or later from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')

echo "Python version: $PYTHON_VERSION"

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 10 ]]; then
    echo -e "${RED}Error: Python 3.10 or later is required.${NC}"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python version is compatible"

# Check for pip
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo -e "${RED}Error: pip is not installed.${NC}"
    echo "Please install pip: python3 -m ensurepip --upgrade"
    exit 1
fi

echo -e "${GREEN}✓${NC} pip is available"

# Fetch latest release information
echo ""
echo "Fetching latest release..."

LATEST_RELEASE_URL="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/latest"

if ! RELEASE_DATA=$(curl -fsSL "$LATEST_RELEASE_URL" 2>&1); then
    echo -e "${RED}Error: Failed to fetch release information.${NC}"
    echo "Please check your internet connection and try again."
    exit 1
fi

if [[ -z "$RELEASE_DATA" ]] || [[ "$RELEASE_DATA" == *"Not Found"* ]]; then
    echo -e "${RED}Error: No releases found for this repository.${NC}"
    echo "Please visit https://github.com/${REPO_OWNER}/${REPO_NAME}/releases"
    exit 1
fi

# Extract wheel URL using Python's json module for robust parsing
WHEEL_URL=$(python3 -c "
import sys
import json
try:
    data = json.loads('''$RELEASE_DATA''')
    assets = data.get('assets', [])
    for asset in assets:
        name = asset.get('name', '')
        if name.endswith('.whl'):
            print(asset.get('browser_download_url', ''))
            break
except Exception as e:
    print('', file=sys.stderr)
")

if [[ -z "$WHEEL_URL" ]]; then
    echo -e "${RED}Error: No wheel file found in the latest release.${NC}"
    exit 1
fi

WHEEL_NAME=$(basename "$WHEEL_URL")
echo "Latest release: $WHEEL_NAME"
echo -e "${GREEN}✓${NC} Found latest release"

# Create temporary directory
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo ""
echo "Downloading wheel..."
WHEEL_PATH="$TMP_DIR/$WHEEL_NAME"

if ! curl -fsSL "$WHEEL_URL" -o "$WHEEL_PATH"; then
    echo -e "${RED}Error: Failed to download wheel.${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Downloaded wheel"

# Install the wheel
echo ""
echo "Installing ${PACKAGE_NAME}..."

if python3 -m pip install --upgrade "$WHEEL_PATH"; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "  Installation successful!"
    echo "==========================================${NC}"
    echo ""
    echo "You can now use ${PACKAGE_NAME} in your Python projects:"
    echo ""
    echo "  from vllm import LLM, SamplingParams"
    echo ""
    echo "  llm = LLM(model=\"meta-llama/Llama-2-7b-hf\")"
    echo "  outputs = llm.generate([\"Hello, my name is\"])"
    echo ""
    echo "For more information, visit:"
    echo "  https://github.com/${REPO_OWNER}/${REPO_NAME}"
    echo ""
else
    echo -e "${RED}Error: Failed to install ${PACKAGE_NAME}.${NC}"
    exit 1
fi
