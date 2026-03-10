#!/usr/bin/env bash
# Providence — Local Development Launcher
#
# Seeds demo data and starts the API server with dashboard.
#
# Usage:
#   ./scripts/launch_local.sh              # seed + start
#   ./scripts/launch_local.sh --no-seed    # start only (reuse existing data)
#   ./scripts/launch_local.sh --seed-only  # seed data without starting server
#
# After launch, open:
#   Dashboard: http://localhost:8000/dashboard
#   API docs:  http://localhost:8000/docs
#   Health:    http://localhost:8000/api/v1/health

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data/demo"
PORT="${PROVIDENCE_PORT:-8000}"

cd "$PROJECT_DIR"

# ── Parse args ───────────────────────────────────────────────────
SKIP_SEED=false
SEED_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --no-seed)    SKIP_SEED=true ;;
        --seed-only)  SEED_ONLY=true ;;
        --help|-h)
            echo "Usage: $0 [--no-seed] [--seed-only]"
            echo ""
            echo "  --no-seed    Skip data seeding (reuse existing data/demo/)"
            echo "  --seed-only  Seed data and exit without starting the server"
            echo ""
            echo "Environment variables:"
            echo "  PROVIDENCE_PORT  Server port (default: 8000)"
            exit 0
            ;;
    esac
done

# ── Seed demo data ───────────────────────────────────────────────
if [ "$SKIP_SEED" = false ]; then
    echo "═══════════════════════════════════════════════════════════"
    echo "  Providence — Seeding demo data"
    echo "═══════════════════════════════════════════════════════════"
    python3 "${SCRIPT_DIR}/seed_demo_data.py" --data-dir "$DATA_DIR"
    echo ""
fi

if [ "$SEED_ONLY" = true ]; then
    echo "Seed complete. Data directory: $DATA_DIR"
    exit 0
fi

# ── Launch API server ────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  Providence — Starting API server"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Dashboard:  http://localhost:${PORT}/dashboard"
echo "  API docs:   http://localhost:${PORT}/docs"
echo "  Health:     http://localhost:${PORT}/api/v1/health"
echo ""
echo "  Press Ctrl+C to stop."
echo "═══════════════════════════════════════════════════════════"
echo ""

python3 -m providence.api.server \
    --data-dir "$DATA_DIR" \
    --port "$PORT" \
    --skip-perception \
    --skip-adaptive \
    --log-level info
