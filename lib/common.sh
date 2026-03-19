#!/bin/bash
# lib/common.sh — shared colours, logging helpers, and small utilities
# Source this file; do not execute directly.
# =============================================================================

# ── Colour support ────────────────────────────────────────────────────────────
if [[ -t 1 && "${NO_COLOR:-0}" != "1" ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; NC=''
fi

# ── Logging ───────────────────────────────────────────────────────────────────
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
step()    { echo -e "\n${BOLD}── $* ${NC}"; }

# ── Confirmation prompt ───────────────────────────────────────────────────────
# confirm "Do the thing?" && do_thing
confirm() {
    local msg="${1:-Continue?}"
    local answer
    read -r -p "$(echo -e "${YELLOW}${msg} [y/N]${NC} ")" answer
    [[ "${answer,,}" == "y" || "${answer,,}" == "yes" ]]
}

# ── Require commands ──────────────────────────────────────────────────────────
require_cmd() {
    for cmd in "$@"; do
        command -v "${cmd}" &>/dev/null || die "'${cmd}' is required but not found."
    done
}

# ── Template rendering via envsubst ──────────────────────────────────────────
# render_template src.tpl dest vars_to_substitute
# vars_to_substitute: space-separated list of ${VAR} strings to substitute
# (leave empty to substitute all exported vars)
render_template() {
    local src="$1" dest="$2" vars="${3:-}"
    [[ -f "${src}" ]] || die "Template not found: ${src}"
    if [[ -n "${vars}" ]]; then
        envsubst "${vars}" < "${src}" > "${dest}"
    else
        envsubst < "${src}" > "${dest}"
    fi
}

# ── Version comparison (a >= b) ───────────────────────────────────────────────
version_gte() {
    # Returns 0 if version $1 >= $2
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# ── Spinner ───────────────────────────────────────────────────────────────────
# Usage: long_command & spinner $!  "Doing the thing…"
spinner() {
    local pid="$1" msg="${2:-Working…}"
    local spin='-\|/'
    local i=0
    while kill -0 "${pid}" 2>/dev/null; do
        printf "\r${CYAN}[%c]${NC}  %s" "${spin:$((i % 4)):1}" "${msg}"
        (( i++ ))
        sleep 0.15
    done
    printf "\r"
}

# ── ASCII logo ────────────────────────────────────────────────────────────────
usage_logo() {
    cat <<'LOGO'
    ▗ ▌     ▜ ▜    
    ▜▘▛▌▛▌▛▘▐ ▐ ▛▛▌
    ▐▖▌▌▙▌▌ ▐▖▐▖▌▌▌
LOGO
}
