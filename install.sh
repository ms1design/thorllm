#!/bin/bash
# =============================================================================
# thorllm — standalone web & local bootstrap installer
#
# Web install (pipe mode — no local files needed):
#   curl -fsSL https://raw.githubusercontent.com/ms1design/thorllm/main/install.sh | bash
#
# Local install from a cloned repo:
#   bash install.sh
#
# Environment overrides:
#   REPO_URL=https://github.com/your-fork/thorllm.git
#   INSTALL_DIR=~/.local/share/thorllm
#   BIN_DIR=~/.local/bin
#   AUTO_SETUP=0   (skip auto-launching wizard)
#   BRANCH=main    (branch/tag to clone)
# =============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ms1design/thorllm.git}"
INSTALL_DIR="${INSTALL_DIR:-${HOME}/.local/share/thorllm}"
BIN_DIR="${BIN_DIR:-${HOME}/.local/bin}"
BRANCH="${BRANCH:-main}"

# ── Detect pipe mode vs local file mode ──────────────────────────────────────
# In pipe mode (curl | bash), BASH_SOURCE[0] is unset or equals "bash".
# We must never try to source local files in that case.
_PIPE_MODE=1
_LOCAL_DIR=""
if [[ -n "${BASH_SOURCE[0]:-}" && "${BASH_SOURCE[0]}" != "bash" ]]; then
    _candidate="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
    if [[ -f "${_candidate}/lib/common.sh" ]]; then
        _LOCAL_DIR="${_candidate}"
        _PIPE_MODE=0
    fi
fi

# ── Inline minimal helpers (used before we have the repo available) ───────────
_i()  { echo "[info]  $*"; }
_ok() { echo "[ok]    $*"; }
_w()  { echo "[warn]  $*"; }
_e()  { echo "[error] $*" >&2; exit 1; }

_logo() {
    cat <<'LOGO'

  ▗ ▌     ▜ ▜    
  ▜▘▛▌▛▌▛▘▐ ▐ ▛▛▌
  ▐▖▌▌▙▌▌ ▐▖▐▖▌▌▌

LOGO
}

# ── Preflight ─────────────────────────────────────────────────────────────────
command -v git  &>/dev/null || _e "git is required. Run: sudo apt-get install -y git"
command -v curl &>/dev/null || _e "curl is required. Run: sudo apt-get install -y curl"

_logo
_i "Installing thorllm to ${INSTALL_DIR} ..."
echo ""

# ── Clone or update repo ──────────────────────────────────────────────────────
if [[ -d "${INSTALL_DIR}/.git" ]]; then
    _i "Updating existing install..."
    git -C "${INSTALL_DIR}" fetch --quiet origin
    git -C "${INSTALL_DIR}" reset --hard "origin/${BRANCH}" --quiet
    _ok "Updated to latest ${BRANCH}"
elif [[ "${_PIPE_MODE}" == "0" && "$(realpath "${_LOCAL_DIR}")" == "$(realpath "${INSTALL_DIR}" 2>/dev/null || echo __none__)" ]]; then
    # Running from inside INSTALL_DIR itself — nothing to clone
    _i "Running from repo at ${INSTALL_DIR} — skipping clone."
else
    _i "Cloning ${REPO_URL} (branch: ${BRANCH}) -> ${INSTALL_DIR}"
    git clone --depth=1 --branch "${BRANCH}" "${REPO_URL}" "${INSTALL_DIR}"
    _ok "Clone complete"
fi

# ── If running from a local dev checkout, sync files into INSTALL_DIR ─────────
if [[ "${_PIPE_MODE}" == "0" && -n "${_LOCAL_DIR}" ]]; then
    if [[ "$(realpath "${_LOCAL_DIR}")" != "$(realpath "${INSTALL_DIR}" 2>/dev/null || echo __none__)" ]]; then
        _i "Syncing repo files from ${_LOCAL_DIR} -> ${INSTALL_DIR} ..."
        for d in lib tui patches templates bin; do
            [[ -d "${_LOCAL_DIR}/${d}" ]] && cp -r "${_LOCAL_DIR}/${d}" "${INSTALL_DIR}/"
        done
        cp "${_LOCAL_DIR}/VERSION" "${INSTALL_DIR}/" 2>/dev/null || true
    fi
fi

# ── Sanity check ──────────────────────────────────────────────────────────────
[[ -d "${INSTALL_DIR}/lib" ]]       || _e "lib/ missing from ${INSTALL_DIR}"
[[ -f "${INSTALL_DIR}/lib/common.sh" ]] || _e "lib/common.sh missing from ${INSTALL_DIR}"
[[ -f "${INSTALL_DIR}/bin/thorllm" ]]   || _e "bin/thorllm missing from ${INSTALL_DIR}"
[[ -f "${INSTALL_DIR}/tui/wizard.py" ]] || _e "tui/wizard.py missing from ${INSTALL_DIR}"

# ── Now source the real shared libs from the freshly-installed repo ───────────
# shellcheck source=lib/common.sh
source "${INSTALL_DIR}/lib/common.sh"
source "${INSTALL_DIR}/lib/config.sh"

VERSION=$(cat "${INSTALL_DIR}/VERSION")

# ── Install CLI symlink ───────────────────────────────────────────────────────
chmod +x "${INSTALL_DIR}/bin/thorllm"
mkdir -p "${BIN_DIR}"
ln -sf "${INSTALL_DIR}/bin/thorllm" "${BIN_DIR}/thorllm"
success "CLI symlink: ${BIN_DIR}/thorllm -> ${INSTALL_DIR}/bin/thorllm"

# ── PATH persistence ─────────────────────────────────────────────────────────
export PATH="${BIN_DIR}:${PATH}"

for rc in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
    if [[ -f "${rc}" ]] && ! grep -qF "${BIN_DIR}" "${rc}" 2>/dev/null; then
        printf '\n# thorllm\nexport PATH="%s:${PATH}"\n' "${BIN_DIR}" >> "${rc}"
        info "Added ${BIN_DIR} to PATH in ${rc}"
    fi
done

# ── TAB completion ────────────────────────────────────────────────────────────
_comp="${INSTALL_DIR}/lib/completion.sh"
for rc in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
    if [[ -f "${rc}" ]] && ! grep -q 'thorllm.*completion' "${rc}" 2>/dev/null; then
        printf '\n# thorllm TAB completion\n[ -f "%s" ] && source "%s"\n' \
            "${_comp}" "${_comp}" >> "${rc}"
    fi
done

# ── Check Python / Textual ───────────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    if ! python3 -c "import textual" 2>/dev/null; then
        warn "Textual TUI not found. Install with:"
        warn "  pip install textual  (or: pip install textual --break-system-packages)"
        warn "thorllm will use fallback plain prompts until then."
    else
        success "Textual TUI available"
    fi
else
    warn "python3 not found — TUI setup wizard unavailable."
fi

echo ""
usage_logo
success "thorllm v${VERSION} installed at ${BIN_DIR}/thorllm"
echo ""
echo "  Run the setup wizard:  thorllm setup"
echo "  Or install directly:   thorllm install"
echo "  Help:                  thorllm --help"
echo "  TAB completion:        active in new shells"
echo ""

# ── Auto-launch wizard if interactive ────────────────────────────────────────
if [[ -t 0 && -t 1 && "${AUTO_SETUP:-1}" == "1" ]]; then
    exec "${BIN_DIR}/thorllm" setup
fi
