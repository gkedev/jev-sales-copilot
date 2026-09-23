#!/usr/bin/env bash
# Launch the Sales Copilot server. Never prints the API key.
#
#   ./run.sh                 # serve on http://localhost:8000
#   PORT=9000 ./run.sh       # other port
#   ./run.sh cli --all       # headless replay of all calls (any extra args go to copilot.cli)
#   ./run.sh test            # run the test suite (unit + real-API integration + e2e)
set -euo pipefail
cd "$(dirname "$0")"

# Resolve keys: existing env > ./.env > a shared secrets file (COPILOT_SECRETS_FILE, default ~/.clawdia-secrets/.env)
SECRETS_FILE="${COPILOT_SECRETS_FILE:-$HOME/.clawdia-secrets/.env}"
if [[ -z "${TYPESAFE_API_KEY:-}" && -z "${JEV_API_KEY:-}" ]]; then
  if [[ -f .env ]]; then
    set -a; # shellcheck disable=SC1091
    source .env; set +a
  fi
fi
if [[ -z "${TYPESAFE_API_KEY:-}" && -z "${JEV_API_KEY:-}" && -f "$SECRETS_FILE" ]]; then
  JEV_API_KEY="$(grep -o '^JEV_API_KEY=.*' "$SECRETS_FILE" | cut -d= -f2-)"
  export JEV_API_KEY
fi
if [[ -z "${TYPESAFE_API_KEY:-}" && -n "${JEV_API_KEY:-}" ]]; then
  export TYPESAFE_API_KEY="$JEV_API_KEY"
fi
# Optional LLM key for tailored phrasing (same lookup order; the app runs fine without it)
if [[ -z "${ANTHROPIC_API_KEY:-}" && -f "$SECRETS_FILE" ]]; then
  ANTHROPIC_API_KEY="$(grep -o '^ANTHROPIC_API_KEY=.*' "$SECRETS_FILE" | cut -d= -f2-)"
  [[ -n "$ANTHROPIC_API_KEY" ]] && export ANTHROPIC_API_KEY
fi
if [[ -z "${TYPESAFE_API_KEY:-}" ]]; then
  echo "No API key. Set TYPESAFE_API_KEY (or JEV_API_KEY), or copy .env.example to .env." >&2
  exit 1
fi

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
uv sync --quiet

case "${1:-serve}" in
  serve)
    PORT="${PORT:-8000}"
    echo "Sales Copilot → http://localhost:${PORT}   (model: pinned in copilot/constants.py)"
    exec uv run uvicorn copilot.server:app --host 127.0.0.1 --port "$PORT" --log-level info
    ;;
  cli)
    shift
    exec uv run python -m copilot.cli "$@"
    ;;
  test)
    shift
    exec uv run pytest -q "$@"
    ;;
  *)
    echo "usage: ./run.sh [serve|cli ...|test ...]" >&2
    exit 2
    ;;
esac
