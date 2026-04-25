#!/usr/bin/env bash
# Providence — Local Development Launcher
#
# Seeds demo data, starts the API server AND the portal frontend.
#
# Usage:
#   ./scripts/launch_local.sh              # seed + start both
#   ./scripts/launch_local.sh --no-seed    # start only (reuse existing data)
#   ./scripts/launch_local.sh --seed-only  # seed data without starting server
#   ./scripts/launch_local.sh --api-only   # skip portal, only start API
#
# After launch, open:
#   Portal:    http://localhost:3000
#   API docs:  http://localhost:8000/docs
#   Health:    http://localhost:8000/api/v1/health

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PORTAL_DIR="${PROJECT_DIR}/../providence-portal"
DATA_DIR="${PROJECT_DIR}/data/demo"
API_PORT="${PROVIDENCE_PORT:-8000}"
PORTAL_PORT="${PORTAL_PORT:-3000}"

cd "$PROJECT_DIR"

# ── Parse args ───────────────────────────────────────────────────
SKIP_SEED=false
SEED_ONLY=false
API_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --no-seed)    SKIP_SEED=true ;;
        --seed-only)  SEED_ONLY=true ;;
        --api-only)   API_ONLY=true ;;
        --help|-h)
            echo "Usage: $0 [--no-seed] [--seed-only] [--api-only]"
            echo ""
            echo "  --no-seed    Skip data seeding (reuse existing data/demo/)"
            echo "  --seed-only  Seed data and exit without starting the server"
            echo "  --api-only   Only start the API server (skip portal frontend)"
            echo ""
            echo "Environment variables:"
            echo "  PROVIDENCE_PORT  API server port (default: 8000)"
            echo "  PORTAL_PORT      Portal dev server port (default: 3000)"
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

# ── Cleanup on exit ──────────────────────────────────────────────
API_PID=""
PORTAL_PID=""

cleanup() {
    echo ""
    echo "Shutting down..."
    [ -n "$PORTAL_PID" ] && kill "$PORTAL_PID" 2>/dev/null && echo "  Portal stopped."
    [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null && echo "  API stopped."
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# ── Launch API server (background) ───────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  Providence — Starting services"
echo "═══════════════════════════════════════════════════════════"
echo ""

python3 -m providence.api.server \
    --data-dir "$DATA_DIR" \
    --port "$API_PORT" \
    --skip-perception \
    --skip-adaptive \
    --log-level info &
API_PID=$!
echo "  API server starting on port ${API_PORT} (PID: $API_PID)"

# ── Launch Portal frontend ───────────────────────────────────────
if [ "$API_ONLY" = false ] && [ -d "$PORTAL_DIR" ]; then
    echo "  Portal starting on port ${PORTAL_PORT}..."
    cd "$PORTAL_DIR"
    PORT=$PORTAL_PORT npm run dev &
    PORTAL_PID=$!
    cd "$PROJECT_DIR"
    echo "  Portal dev server starting (PID: $PORTAL_PID)"
elif [ "$API_ONLY" = false ]; then
    echo "  ⚠ Portal directory not found at: $PORTAL_DIR"
    echo "    Skipping portal. Only API will be available."
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  ✦ Website:    http://localhost:${API_PORT}/"
echo "  ✦ Dashboard:  http://localhost:${API_PORT}/dashboard"
echo "  ✦ API docs:   http://localhost:${API_PORT}/docs"
echo "  ✦ Health:     http://localhost:${API_PORT}/api/v1/health"
echo ""
echo "  Press Ctrl+C to stop all services."
echo "═══════════════════════════════════════════════════════════"
echo ""

# Wait for either process to exit
wait
