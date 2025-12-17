#!/bin/bash

main() {
  set -eu -o pipefail

  if [ "$(uname)" == "Darwin" ]; then
    brew install shellcheck
  fi

  shellcheck -- */*.sh

  if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/0.9.18/install.sh | sh
  fi

  if [ ! -d ".venv" ]; then
    echo "=== Creating virtual environment ==="
    uv venv .venv
  fi

  # shellcheck source=/dev/null
  source .venv/bin/activate

  echo "=== Installing dependencies ==="
  uv pip install -e ".[dev]"

  echo "=== Running ruff linter ==="
  ruff check .

  echo "=== Running ruff formatter check ==="
  ruff format --check .

  echo "=== Running mypy type checker ==="
  mypy vllm_metal

  echo "=== Running tests ==="
  pytest tests/ -v --tb=short

  echo "=== Verifying package import ==="
  python -c "import vllm_metal; print('vllm_metal imported successfully')"
}

main "$@"

