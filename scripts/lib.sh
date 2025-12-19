#!/bin/bash
# Common library functions for vllm-metal scripts

# Print an error message
error() {
  echo -e "Error: $*" >&2
}

# Print a success message
success() {
  echo -e "✓ $*"
}

# Print a section header
section() {
  echo "=== $* ==="
}

# Check if running on Apple Silicon
is_apple_silicon() {
  [ "$(uname -m)" = "arm64" ]
}

# Ensure uv is installed
ensure_uv() {
  if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    if ! curl -LsSf "https://astral.sh/uv/0.9.18/install.sh" | sh; then
      error "Failed to install uv"
      return 1
    fi

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

# Ensure virtual environment exists and is activated
ensure_venv() {
  if [ ! -d ".venv" ]; then
    section "Creating virtual environment"
    uv venv .venv
  fi

  # shellcheck source=/dev/null
  source .venv/bin/activate
}

# Get repository root directory
get_repo_root() {
  git rev-parse --show-toplevel
}

# Install dev dependencies
install_dev_deps() {
  section "Installing dependencies"
  uv pip install -e ".[dev]"
  
  section "Installing vllm (without CUDA dependencies)"
  # Install vllm with --no-deps to avoid pulling in CUDA dependencies
  # that don't work on non-CUDA systems (macOS/Apple Silicon, Linux without CUDA)
  uv pip install --no-deps "vllm>=0.12.0"
  
  section "Installing vllm dependencies"
  # Install vllm's essential dependencies needed for testing
  # These are the dependencies that vllm needs to import its modules
  uv pip install \
    msgspec \
    cloudpickle \
    prometheus-client \
    fastapi \
    uvicorn \
    pydantic \
    pillow \
    tiktoken \
    typing_extensions \
    filelock \
    py-cpuinfo \
    aiohttp \
    openai \
    einops \
    importlib_metadata \
    mistral_common \
    pyyaml \
    requests \
    tqdm \
    sentencepiece \
    compressed-tensors \
    gguf \
    partial-json-parser \
    blake3 \
    cbor2 \
    pyzmq \
    cachetools \
    regex \
    protobuf \
    python-multipart \
    lark \
    six \
    scipy \
    ninja \
    pybase64 \
    setproctitle \
    "tokenizers>=0.21.1" \
    diskcache \
    interegular \
    "torch>=2.5.0"
}

# Full development environment setup
setup_dev_env() {
  ensure_uv
  ensure_venv
  install_dev_deps
}

# Get version from pyproject.toml
get_version() {
  uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
}

check_python_version() {
  local min_major="${1:-3}"
  local min_minor="${2:-11}"

  if ! command -v python3 &> /dev/null; then
    error "Python 3 is not installed"
    return 1
  fi

  local major minor
  major=$(python3 -c 'import sys; print(sys.version_info[0])')
  minor=$(python3 -c 'import sys; print(sys.version_info[1])')

  if [ "$major" -lt "$min_major" ] || { [ "$major" -eq "$min_major" ] && [ "$minor" -lt "$min_minor" ]; }; then
    error "Python ${min_major}.${min_minor} or later is required (found ${major}.${minor})"
    return 1
  fi
}
