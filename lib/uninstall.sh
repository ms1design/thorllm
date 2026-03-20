#!/bin/bash
# lib/uninstall.sh — Remove thorllm and all installed components
# Sources: common.sh, config.sh (must be loaded first)
# =============================================================================

LIB_DIR="${LIB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BIN_DIR="${BIN_DIR:-${HOME}/.local/bin}"
_SHARE_DIR="${HOME}/.local/share/thorllm"

confirm_uninstall() {
    step "Uninstall thorllm"
    echo "This will permanently remove:"
    echo ""
    echo "  vLLM installation       ${BUILD_PATH}"
    echo "  Python venv             ${VENV_PATH}"
    echo "  Model configs           ${MODELS_DIR}"
    echo "  Cache files             ${CACHE_ROOT}"
    echo "  Systemd service         ${SERVICE_FILE}"
    echo "  CLI symlink             ${BIN_DIR}/thorllm"
    echo "  thorllm repo            ${_SHARE_DIR}"
    echo ""
    echo "NOTE: HuggingFace model weights in ~/.cache/huggingface will NOT be deleted."
    echo ""
    confirm "Are you sure you want to uninstall thorllm?" || die "Uninstall cancelled."
}

stop_service() {
    step "Stopping vLLM service"
    if sudo systemctl is-active --quiet "${SERVICE_NAME}" 2>/dev/null; then
        sudo systemctl stop "${SERVICE_NAME}" 2>/dev/null || warn "Failed to stop ${SERVICE_NAME}."
    else
        info "${SERVICE_NAME} not running."
    fi
    if sudo systemctl is-enabled --quiet "${SERVICE_NAME}" 2>/dev/null; then
        sudo systemctl disable "${SERVICE_NAME}" 2>/dev/null || warn "Failed to disable ${SERVICE_NAME}."
    fi
    success "Service stopped and disabled."
}

remove_systemd() {
    step "Removing systemd components"
    local sudoers_file="/etc/sudoers.d/vllm-drop-caches"

    if [[ -f "${SERVICE_FILE}" ]]; then
        sudo rm -f "${SERVICE_FILE}"
        success "Removed: ${SERVICE_FILE}"
    fi
    if [[ -f "${sudoers_file}" ]]; then
        sudo rm -f "${sudoers_file}"
        success "Removed: ${sudoers_file}"
    fi
    sudo systemctl daemon-reload 2>/dev/null || true
}

remove_build() {
    step "Removing build directory"
    if [[ -d "${BUILD_PATH}" ]]; then
        rm -rf "${BUILD_PATH}"
        success "Removed: ${BUILD_PATH}"
    else
        info "Not found: ${BUILD_PATH}"
    fi
}

remove_thorllm_install() {
    step "Removing thorllm installation"

    # Remove the share directory (cloned repo / installed files)
    if [[ -d "${_SHARE_DIR}" ]]; then
        rm -rf "${_SHARE_DIR}"
        success "Removed: ${_SHARE_DIR}"
    else
        info "Not found: ${_SHARE_DIR}"
    fi

    # Also remove LIB_DIR if it differs (e.g. custom INSTALL_DIR was used)
    local _lib_parent
    _lib_parent="$(dirname "${LIB_DIR}")"
    if [[ -d "${_lib_parent}" && "${_lib_parent}" != "${_SHARE_DIR}" && \
          "${_lib_parent}" != "${HOME}" && "${_lib_parent}" != "/" ]]; then
        if [[ -f "${_lib_parent}/bin/thorllm" ]]; then
            rm -rf "${_lib_parent}"
            success "Removed: ${_lib_parent}"
        fi
    fi

    # Remove CLI symlink
    if [[ -L "${BIN_DIR}/thorllm" ]]; then
        rm -f "${BIN_DIR}/thorllm"
        success "Removed: ${BIN_DIR}/thorllm"
    fi
}

remove_shellrc_entries() {
    step "Cleaning shell configuration"
    local cleaned=0
    for rc in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
        if [[ -f "${rc}" ]] && grep -q 'thorllm' "${rc}" 2>/dev/null; then
            local tmp="${rc}.tmp"
            grep -v 'thorllm' "${rc}" > "${tmp}" 2>/dev/null || true
            mv "${tmp}" "${rc}"
            success "Cleaned thorllm entries from ${rc}"
            cleaned=1
        fi
    done
    [[ ${cleaned} -eq 0 ]] && info "No thorllm entries found in shell configs."
}

run_uninstall() {
    config_export
    confirm_uninstall
    stop_service
    remove_systemd
    remove_build
    remove_thorllm_install
    remove_shellrc_entries

    step "Uninstall complete"
    echo "thorllm has been fully removed."
    echo ""
    echo "  Note: Run 'exec \$SHELL' to refresh your PATH."
    echo ""
}
