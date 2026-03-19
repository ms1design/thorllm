#!/bin/bash
# lib/uninstall.sh — Remove thorllm and all installed components
# Sources: common.sh, config.sh (must be loaded first)
# =============================================================================

LIB_DIR="${LIB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BIN_DIR="${BIN_DIR:-${HOME}/.local/bin}"

# ── Uninstall confirmation ────────────────────────────────────────────────────
confirm_uninstall() {
    step "Uninstall thorllm"
    echo ""
    echo -e "${RED}This will permanently remove:${NC}"
    echo ""
    echo "  • vLLM installation at ${BUILD_PATH}"
    echo "  • Python virtual environment at ${VENV_PATH}"
    echo "  • Model configs in ${MODELS_DIR}"
    echo "  • Cache files in ${CACHE_ROOT}"
    echo "  • Systemd service file at ${SERVICE_FILE}"
    echo "  • Environment file at ${BUILD_PATH}/vllm.env"
    echo "  • Activation script at ${BUILD_PATH}/activate_vllm.sh"
    echo "  • Thorllm CLI symlink at ${BIN_DIR}/thorllm"
    echo "  • Thorllm repo at ${LIB_DIR}"
    echo ""
    echo -e "${YELLOW}NOTE: Your home directory and HuggingFace cache will NOT be deleted.${NC}"
    echo ""
    confirm "Are you sure you want to uninstall thorllm? [y/N]" || die "Uninstall cancelled."
}

# ── Stop and disable service ──────────────────────────────────────────────────
stop_service() {
    step "Stopping vLLM service"

    if sudo systemctl is-active --quiet vllm 2>/dev/null; then
        info "Stopping vllm service…"
        sudo systemctl stop vllm 2>/dev/null || warn "Failed to stop vllm service."
    else
        info "vllm service not running."
    fi

    if sudo systemctl is-enabled --quiet vllm 2>/dev/null; then
        info "Disabling vllm service…"
        sudo systemctl disable vllm 2>/dev/null || warn "Failed to disable vllm service."
    else
        info "vllm service not enabled."
    fi

    success "Service stopped and disabled."
}

# ── Remove systemd components ─────────────────────────────────────────────────
remove_systemd() {
    step "Removing systemd components"

    local sudoers_file="/etc/sudoers.d/vllm-drop-caches"

    if [[ -f "${SERVICE_FILE}" ]]; then
        info "Removing service file: ${SERVICE_FILE}"
        sudo rm -f "${SERVICE_FILE}"
        success "Service file removed."
    else
        info "Service file not found — already removed."
    fi

    if [[ -f "${sudoers_file}" ]]; then
        info "Removing sudoers rule: ${sudoers_file}"
        sudo rm -f "${sudoers_file}"
        success "Sudoers rule removed."
    else
        info "Sudoers rule not found — already removed."
    fi

    info "Reloading systemd daemon…"
    sudo systemctl daemon-reload 2>/dev/null || warn "Failed to reload systemd daemon."

    success "Systemd components removed."
}

# ── Remove build directory and venv ───────────────────────────────────────────
remove_build() {
    step "Removing build directory"

    if [[ -d "${BUILD_PATH}" ]]; then
        info "Removing ${BUILD_PATH}…"
        rm -rf "${BUILD_PATH}"
        success "Build directory removed."
    else
        info "Build directory not found: ${BUILD_PATH}"
    fi
}

# ── Remove thorllm repo and CLI ───────────────────────────────────────────────
remove_thorllm() {
    step "Removing thorllm installation"

    if [[ -d "${LIB_DIR}" ]]; then
        info "Removing thorllm repo at ${LIB_DIR}…"
        rm -rf "${LIB_DIR}"
        success "Thorllm repo removed."
    else
        info "Thorllm repo not found: ${LIB_DIR}"
    fi

    if [[ -L "${BIN_DIR}/thorllm" ]]; then
        info "Removing CLI symlink at ${BIN_DIR}/thorllm…"
        rm -f "${BIN_DIR}/thorllm"
        success "CLI symlink removed."
    else
        info "CLI symlink not found: ${BIN_DIR}/thorllm"
    fi
}

# ── Remove shellrc entries ────────────────────────────────────────────────────
remove_shellrc_entries() {
    step "Cleaning shell configuration"

    local shellrcs=("${HOME}/.bashrc" "${HOME}/.zshrc")
    local cleaned=0

    for rc in "${shellrcs[@]}"; do
        if [[ -f "${rc}" ]] && grep -q "thorllm" "${rc}" 2>/dev/null; then
            info "Removing thorllm entries from ${rc}…"
            local tmp_rc="${rc}.tmp"
            grep -v 'thorllm' "${rc}" > "${tmp_rc}" 2>/dev/null || true
            mv "${tmp_rc}" "${rc}"
            cleaned=1
        fi
    done

    if [[ ${cleaned} -eq 1 ]]; then
        success "Shell configuration cleaned."
    else
        info "No thorllm entries found in shell configs."
    fi
}

# ── Main uninstall orchestrator ───────────────────────────────────────────────
run_uninstall() {
    config_export

    confirm_uninstall

    stop_service
    remove_systemd
    remove_build
    remove_thorllm
    remove_shellrc_entries

    step "Uninstall complete"
    echo ""
    echo -e "${GREEN}thorllm has been successfully uninstalled.${NC}"
    echo ""
    echo "  Note: You may need to reload your shell (e.g., 'exec $SHELL')"
    echo "  to update your PATH environment variable."
    echo ""
}
