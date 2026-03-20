#!/bin/bash
# =============================================================================
# thorllm — bootstrap installer
# Works both as a web-pipe install AND as a local repo script:
#
#   # From web (pipe mode — no local files available):
#   curl -fsSL https://raw.githubusercontent.com/ms1design/thorllm/main/install.sh | bash
#
#   # Local repo:
#   bash install.sh
#
# Environment overrides:
#   REPO_URL=https://github.com/your-fork/thorllm.git bash <(curl ...)
#   INSTALL_DIR=~/my-thorllm bash <(curl ...)
#   AUTO_SETUP=0   skip launching the wizard after install
# =============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ms1design/thorllm.git}"
INSTALL_DIR="${INSTALL_DIR:-${HOME}/.local/share/thorllm}"
BIN_DIR="${BIN_DIR:-${HOME}/.local/bin}"

# ── Detect whether we are running from a local file or piped from curl ────────
# When piped through bash, BASH_SOURCE[0] is either unset or equals "bash".
# We must NOT try to source any local files in that case.
_PIPE_MODE=1
if [[ -n "${BASH_SOURCE[0]:-}" && "${BASH_SOURCE[0]}" != "bash" && -f "${BASH_SOURCE[0]}" ]]; then
    _LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    _PIPE_MODE=0
fi

# ── Minimal inline helpers (used before we have the full lib available) ───────
_green='\033[38;2;118;185;0m'
_cyan='\033[38;2;160;216;50m'
_yellow='\033[1;33m'
_red='\033[0;31m'
_bold='\033[1m'
_nc='\033[0m'

_info()    { echo -e "${_cyan}[info]${_nc}  $*"; }
_success() { echo -e "${_green}[ok]${_nc}    $*"; }
_warn()    { echo -e "${_yellow}[warn]${_nc}  $*"; }
_die()     { echo -e "${_red}[error]${_nc} $*" >&2; exit 1; }

_logo() {
    echo -e "${_green}${_bold}"
    cat <<'LOGO'
  ▗ ▌     ▜ ▜    
  ▜▘▛▌▛▌▛▘▐ ▐ ▛▛▌
  ▐▖▌▌▙▌▌ ▐▖▐▖▌▌▌
LOGO
    echo -e "${_nc}"
}

# ── Preflight ─────────────────────────────────────────────────────────────────
command -v git  &>/dev/null || _die "git is required. Run: sudo apt-get install -y git"
command -v curl &>/dev/null || _die "curl is required. Run: sudo apt-get install -y curl"

_logo
_info "Installing thorllm to ${INSTALL_DIR}…"
echo ""

# ── Clone or update the repo into INSTALL_DIR ────────────────────────────────
if [[ -d "${INSTALL_DIR}/.git" ]]; then
    _info "Updating existing install…"
    git -C "${INSTALL_DIR}" pull --ff-only
elif [[ "${_PIPE_MODE}" == "0" && "${_LOCAL_DIR}" == "${INSTALL_DIR}" ]]; then
    # Running from inside the repo itself — nothing to clone
    _info "Running from repo at ${INSTALL_DIR} — skipping clone."
else
    # Fresh clone
    _info "Cloning ${REPO_URL} → ${INSTALL_DIR}"
    git clone --depth=1 "${REPO_URL}" "${INSTALL_DIR}"
fi

# ── If running locally and INSTALL_DIR is different, copy lib/tui/etc ─────────
# (handles: bash install.sh from a dev checkout into a separate INSTALL_DIR)
if [[ "${_PIPE_MODE}" == "0" && "${_LOCAL_DIR}" != "${INSTALL_DIR}" ]]; then
    _info "Copying repo files to ${INSTALL_DIR}…"
    for d in lib tui patches templates bin; do
        [[ -d "${_LOCAL_DIR}/${d}" ]] && cp -r "${_LOCAL_DIR}/${d}" "${INSTALL_DIR}/"
    done
    for f in VERSION; do
        [[ -f "${_LOCAL_DIR}/${f}" ]] && cp "${_LOCAL_DIR}/${f}" "${INSTALL_DIR}/"
    done
fi

# ── Verify lib/ exists (sanity check) ────────────────────────────────────────
[[ -d "${INSTALL_DIR}/lib" ]] \
    || _die "lib/ directory missing from ${INSTALL_DIR}. Clone may be incomplete."

# ── Install CLI symlink ───────────────────────────────────────────────────────
chmod +x "${INSTALL_DIR}/bin/thorllm"
mkdir -p "${BIN_DIR}"
ln -sf "${INSTALL_DIR}/bin/thorllm" "${BIN_DIR}/thorllm"

# ── Ensure BIN_DIR is on PATH (current session + rc files) ───────────────────
export PATH="${BIN_DIR}:${PATH}"

for rc in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
    if [[ -f "${rc}" ]] && ! grep -q "${BIN_DIR}" "${rc}" 2>/dev/null; then
        echo "export PATH=\"${BIN_DIR}:\${PATH}\"" >> "${rc}"
        _info "Added ${BIN_DIR} to PATH in ${rc}"
    fi
done

# ── Optional: install TAB completion ─────────────────────────────────────────
_completion_src="${INSTALL_DIR}/lib/completion.sh"
for rc in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
    if [[ -f "${rc}" ]] && ! grep -q 'thorllm.*completion' "${rc}" 2>/dev/null; then
        echo "# thorllm TAB completion" >> "${rc}"
        echo "source \"${_completion_src}\"" >> "${rc}"
    fi
done

echo ""
_logo
_success "thorllm CLI installed at ${BIN_DIR}/thorllm"
echo ""
echo -e "  ${_green}Run the setup wizard:${_nc}  thorllm setup"
echo -e "  ${_green}Or install directly:${_nc}   thorllm install"
echo -e "  ${_green}Help:${_nc}                  thorllm --help"
echo ""
echo -e "  ${_cyan}TAB completion will be active in new shells.${_nc}"
echo -e "  ${_cyan}To activate now: source ${_completion_src}${_nc}"
echo ""

# ── Auto-launch wizard if interactive ────────────────────────────────────────
if [[ -t 0 && -t 1 && "${AUTO_SETUP:-1}" == "1" ]]; then
    exec "${BIN_DIR}/thorllm" setup
fi
