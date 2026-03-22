#!/usr/bin/env zsh

# Run with:
#   source start_server.sh
# Optional port:
#   source start_server.sh 8001
#
# Project layout:
# - script:      /chronicleux/src/start_server.sh
# - manage.py:   /chronicleux/src/manage.py
# - virtualenv:  /chronicleux/.venv

SCRIPT_PATH="${(%):-%x}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPECTED_VENV="$PROJECT_ROOT/.venv"

# Must be sourced, not executed.
if [[ "$ZSH_EVAL_CONTEXT" != *:file* ]]; then
  echo "Run this with: source start_server.sh"
  exit 1
fi

cd "$SCRIPT_DIR" || {
  echo "Could not cd to $SCRIPT_DIR"
  return 1
}

# Deactivate conda/base if active.
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "Conda environment detected: $CONDA_DEFAULT_ENV"
  conda deactivate 2>/dev/null || true
fi

# Ensure the expected project venv is active.
if [[ "${VIRTUAL_ENV:-}" == "$EXPECTED_VENV" ]]; then
  echo "Project virtualenv already active: $VIRTUAL_ENV"
else
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Different virtualenv detected: $VIRTUAL_ENV"
    deactivate 2>/dev/null || true
  fi

  if [[ -f "$EXPECTED_VENV/bin/activate" ]]; then
    echo "Activating virtualenv: $EXPECTED_VENV"
    source "$EXPECTED_VENV/bin/activate" || {
      echo "Failed to activate $EXPECTED_VENV"
      return 1
    }
  else
    echo "Could not find virtualenv at $EXPECTED_VENV/bin/activate"
    return 1
  fi
fi

if [[ ! -f "$SCRIPT_DIR/manage.py" ]]; then
  echo "manage.py not found in $SCRIPT_DIR"
  return 1
fi

PORT="${1:-8000}"

# Kill processes using the 8000 (or selected) port
kill -9 $(lsof -t -i :8000)

echo "Using python: $(which python)"
echo "Project root: $PROJECT_ROOT"
echo "Django dir: $SCRIPT_DIR"
echo "Starting Django on port $PORT"

python manage.py migrate && python manage.py runserver "$PORT"