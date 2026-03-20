#!/bin/bash
# lib/common.sh — shared colours, logging helpers, and small utilities
# Source this file; do not execute directly.
# =============================================================================

# ── Colour support ────────────────────────────────────────────────────────────
if [[ -t 1 && "${NO_COLOR:-0}" != "1" ]]; then
    # NVIDIA Green palette
    NVIDIA_GREEN='\033[38;2;118;185;0m'   # #76b900
    NVIDIA_LIGHT='\033[38;2;160;216;50m'  # #a0d832
    GREEN="${NVIDIA_GREEN}"
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    CYAN="${NVIDIA_LIGHT}"
    MUTED='\033[38;2;90;122;32m'          # #5a7a20
    BOLD='\033[1m'
    DIM='\033[2m'
    NC='\033[0m'
else
    NVIDIA_GREEN=''; NVIDIA_LIGHT=''; GREEN=''
    RED=''; YELLOW=''; CYAN=''; MUTED=''; BOLD=''; DIM=''; NC=''
fi

# ── Logging ───────────────────────────────────────────────────────────────────
info()    { echo -e "${CYAN}[info]${NC}  $*"; }
success() { echo -e "${NVIDIA_GREEN}[ok]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()     { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }
step()    { echo -e "\n${BOLD}${NVIDIA_GREEN}──${NC}${BOLD} $* ${NC}"; }

# ── Confirmation prompt ───────────────────────────────────────────────────────
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
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# ── Spinner ───────────────────────────────────────────────────────────────────
spinner() {
    local pid="$1" msg="${2:-Working…}"
    local frames=('⠋' '⠙' '⠸' '⠴' '⠦' '⠇')
    local i=0
    while kill -0 "${pid}" 2>/dev/null; do
        printf "\r${NVIDIA_GREEN}${frames[$((i % 6))]}${NC}  %s" "${msg}"
        (( i++ ))
        sleep 0.1
    done
    printf "\r"
}

# ── ASCII logo ─────────────────────────────────────────────────────────────────
usage_logo() {
    echo -e "${NVIDIA_GREEN}${BOLD}"
    cat <<'LOGO'
  ▗ ▌     ▜ ▜    
  ▜▘▛▌▛▌▛▘▐ ▐ ▛▛▌
  ▐▖▌▌▙▌▌ ▐▖▐▖▌▌▌
LOGO
    echo -e "${NC}"
}

# ── Footer line ───────────────────────────────────────────────────────────────
print_footer() {
    local ver="${VERSION:-0.1.0}"
    local cols
    cols=$(tput cols 2>/dev/null || echo 80)
    local text=" thorllm · v${ver} · made by ms1design "
    local pad=$(( (cols - ${#text}) / 2 ))
    printf "\n${MUTED}%${pad}s%s${NC}\n\n" "" "${text}"
}
