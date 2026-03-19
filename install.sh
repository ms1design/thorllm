#!/bin/bash
# =============================================================================
# thorllm — bootstrap installer
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ms1design/thorllm/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/ms1design/thorllm/main/install.sh | VLLM_MODEL=Qwen/Qwen3-VL-32B-Instruct bash
#   REPO_URL=https://github.com/ms1design/thorllm bash <(curl -fsSL ...)
# =============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ms1design/thorllm.git}"
INSTALL_DIR="${INSTALL_DIR:-${HOME}/.local/share/thorllm}"
BIN_DIR="${BIN_DIR:-${HOME}/.local/bin}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SELF_DIR}/lib/common.sh"

command -v git  &>/dev/null || die "git is required. Run: sudo apt-get install -y git"
command -v curl &>/dev/null || die "curl is required. Run: sudo apt-get install -y curl"

info "Installing thorllm to ${INSTALL_DIR}…"
echo ""

usage_logo
echo ""

if [[ -d "${INSTALL_DIR}/.git" ]]; then
    info "Updating existing install…"
    git -C "${INSTALL_DIR}" pull --ff-only
else
    git clone --depth=1 "${REPO_URL}" "${INSTALL_DIR}"
fi

chmod +x "${INSTALL_DIR}/bin/thorllm"

mkdir -p "${BIN_DIR}"
ln -sf "${INSTALL_DIR}/bin/thorllm" "${BIN_DIR}/thorllm"

# Ensure BIN_DIR is on PATH in current session
export PATH="${BIN_DIR}:${PATH}"

# Persist PATH in shell rc if not already present
for rc in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
    if [[ -f "${rc}" ]] && ! grep -q 'thorllm' "${rc}" 2>/dev/null; then
        echo 'export PATH="${HOME}/.local/bin:${PATH}"' >> "${rc}"
    fi
done

echo ""
usage_logo
echo ""
success "thorllm CLI installed at ${BIN_DIR}/thorllm"
echo "  Run the setup wizard:  thorllm setup"
echo "  Or install directly:   thorllm install"
echo "  Help:                  thorllm --help"
echo ""

# Auto-launch setup wizard if running interactively and a TTY is available
if [[ -t 0 && -t 1 && "${AUTO_SETUP:-1}" == "1" ]]; then
    exec "${BIN_DIR}/thorllm" setup
fi
