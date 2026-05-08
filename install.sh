#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Hermes Model Router — Installer
# ═══════════════════════════════════════════════════════════════════════════
# Mechanical setup only — copies files, installs systemd unit, prints the
# Hermes custom_providers snippet. No LLM calls. Profile extraction happens
# lazily on first runtime.
#
# Usage: bash install.sh [--dry-run]

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUTER_DIR="/root/router-proxy"
SYSTEMD_UNIT_SRC="${SCRIPT_DIR}/router-config/systemd/hermes-router.service"
SYSTEMD_UNIT_DST="/root/.config/systemd/user/hermes-router.service"
CONFIG_SRC="${SCRIPT_DIR}/router_config.yaml"
CONFIG_DST="${ROUTER_DIR}/router_config.yaml"
SERVER_SRC="${SCRIPT_DIR}/server.py"
SERVER_DST="${ROUTER_DIR}/server.py"

echo -e "${BOLD}Hermes Model Router — Installer${NC}"
echo ""

# ── Pre-flight checks ──────────────────────────────────────────────────────
errors=0

if [[ ! -f "$SERVER_SRC" ]]; then
    echo -e "${RED}✗${NC} server.py not found at $SERVER_SRC"
    errors=1
fi
if [[ ! -f "$CONFIG_SRC" ]]; then
    echo -e "${RED}✗${NC} router_config.yaml not found at $CONFIG_SRC"
    errors=1
fi
if [[ ! -f "/root/.hermes/.env" ]]; then
    echo -e "${YELLOW}⚠${NC} /root/.hermes/.env not found — API keys may be missing"
fi
if [[ ! -d "/root/.hermes/hermes-agent/venv" ]]; then
    echo -e "${RED}✗${NC} Hermes venv not found at /root/.hermes/hermes-agent/venv"
    errors=1
fi

# Check for required Python packages
/root/.hermes/hermes-agent/venv/bin/python -c "import fastapi, httpx, yaml, uvicorn" 2>/dev/null || {
    echo -e "${YELLOW}⚠${NC} Missing Python dependencies — installing..."
    if [[ "$DRY_RUN" == false ]]; then
        /root/.hermes/hermes-agent/venv/bin/pip install fastapi httpx pyyaml uvicorn 2>&1 | tail -3
    fi
}

if [[ $errors -gt 0 ]]; then
    echo ""
    echo -e "${RED}Pre-flight checks failed. Fix errors above and retry.${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Pre-flight checks passed"
echo ""

# ── Install ─────────────────────────────────────────────────────────────────

install_file() {
    local src="$1"
    local dst="$2"
    local label="$3"

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "  ${YELLOW}[dry-run]${NC} Would copy $src → $dst"
        return
    fi

    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    echo -e "  ${GREEN}✓${NC} $label → $dst"
}

echo "Installing files..."
install_file "$SERVER_SRC" "$SERVER_DST" "server.py"
install_file "$CONFIG_SRC" "$CONFIG_DST" "router_config.yaml"
install_file "$SYSTEMD_UNIT_SRC" "$SYSTEMD_UNIT_DST" "systemd unit"
echo ""

# ── Systemd ─────────────────────────────────────────────────────────────────
echo "Enabling systemd service..."

if [[ "$DRY_RUN" == false ]]; then
    export XDG_RUNTIME_DIR=/run/user/0
    systemctl --user daemon-reload
    systemctl --user enable hermes-router.service 2>/dev/null || true

    # Stop if already running, then start fresh
    systemctl --user stop hermes-router.service 2>/dev/null || true
    systemctl --user reset-failed hermes-router.service 2>/dev/null || true
    systemctl --user start hermes-router.service

    sleep 3

    if systemctl --user is-active --quiet hermes-router.service; then
        echo -e "  ${GREEN}✓${NC} Service active"
    else
        echo -e "  ${YELLOW}⚠${NC} Service may not have started — check: journalctl --user -u hermes-router.service -n 20"
    fi
else
    echo -e "  ${YELLOW}[dry-run]${NC} Would daemon-reload + enable + start hermes-router.service"
fi

echo ""

# ── Hermes Config Snippet ───────────────────────────────────────────────────
echo -e "${BOLD}Add this to ~/.hermes/config.yaml under custom_providers:${NC}"
echo ""
echo -e "${YELLOW}custom_providers:${NC}"
echo -e "${YELLOW}  - name: auto-router${NC}"
echo -e "${YELLOW}    base_url: http://localhost:8766/v1${NC}"
echo -e "${YELLOW}    model: auto${NC}"
echo ""
echo -e "Then restart the gateway: ${BOLD}systemctl --user restart hermes-gateway${NC}"
echo ""

# ── Summary ─────────────────────────────────────────────────────────────────
echo -e "${BOLD}── What happens next ──${NC}"
echo ""
echo "  1. On the very first request, the router will read your SOUL.md"
echo "     and extract a profile hint via the flash classifier model."
echo "     (~3 seconds on first call, cached permanently after)"
echo ""
echo "  2. Every new session gets classified once: simple → cheap model,"
echo "     complex → capable model with fallback."
echo ""
echo "  3. Follow-up messages skip classification (sub-ms keyword scan)"
echo "     unless escalation/de-escalation keywords are detected."
echo ""
echo -e "  Check logs: ${BOLD}journalctl --user -u hermes-router.service -f${NC}"
echo -e "  Health:     ${BOLD}curl http://localhost:8766/health${NC}"
echo ""
echo -e "${GREEN}${BOLD}Done.${NC}"
