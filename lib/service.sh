#!/bin/bash
# lib/service.sh — systemd unit management, env file, launcher generation
# Sources: common.sh, config.sh
# =============================================================================

# ── Health check ──────────────────────────────────────────────────────────────
_health_check() {
    local port="${VLLM_PORT:-8000}"
    curl -sf "http://localhost:${port}/v1/health" &>/dev/null
}

# ── EnvironmentFile ───────────────────────────────────────────────────────────
write_env_file() {
    local env_file="${BUILD_PATH}/vllm.env"
    info "Writing EnvironmentFile: ${env_file}"
    config_export

    local vars
    vars='${CUDA_HOME} ${TORCH_CUDA_ARCH} ${CACHE_ROOT} ${SERVE_MODEL}
          ${VLLM_CACHE_ROOT} ${VLLM_ASSETS_CACHE} ${VLLM_TUNED_CONFIG_FOLDER}
          ${FLASHINFER_JIT_DIR} ${TRITON_CACHE_DIR} ${TORCHINDUCTOR_CACHE_DIR}
          ${TORCH_COMPILE_CACHE_DIR} ${XDG_CACHE_HOME}
          ${TIKTOKEN_ENCODINGS_BASE} ${TIKTOKEN_RS_CACHE_DIR} ${HF_DOWNLOAD_DIR}'

    render_template "${TPL_DIR}/vllm.env" "${env_file}" "${vars}"
    chmod 600 "${env_file}"
    chown "${SERVICE_USER}:${SERVICE_USER}" "${env_file}" 2>/dev/null || true
    success "EnvironmentFile: ${env_file}"
}

# ── Launcher script ───────────────────────────────────────────────────────────
write_service_files() {
    local serve_script="${BUILD_PATH}/vllm-serve.sh"

    info "Writing launcher: ${serve_script}"
    local vars='${VENV_PATH} ${BUILD_PATH} ${MODELS_DIR}'
    render_template "${TPL_DIR}/vllm-serve.sh" "${serve_script}" "${vars}"
    chmod +x "${serve_script}"
    chown "${SERVICE_USER}:${SERVICE_USER}" "${serve_script}" 2>/dev/null || true
    success "Launcher: ${serve_script}"

    info "Writing systemd unit: ${SERVICE_FILE}"
    local svc_vars='${SERVICE_USER} ${BUILD_PATH}'
    local _rendered_svc="/tmp/vllm.service.rendered.$$"
    render_template "${TPL_DIR}/vllm.service" "${_rendered_svc}" "${svc_vars}"

    echo ""
    echo "[sudo] Installing systemd service unit requires root access."
    echo "       Destination: ${SERVICE_FILE}"

    # Wrap all sudo calls with || fallback — a sudo timeout or missing permission
    # must NOT abort the entire install under set -euo pipefail.
    local _svc_installed=0
    if sudo cp "${_rendered_svc}" "${SERVICE_FILE}" 2>/dev/null; then
        _svc_installed=1
        rm -f "${_rendered_svc}"
        success "Service file: ${SERVICE_FILE}"
    else
        warn "Could not install systemd service (sudo unavailable or timed out)."
        warn "  To install manually after this run:"
        warn "    sudo cp ${_rendered_svc} ${SERVICE_FILE}"
        warn "    sudo systemctl daemon-reload && sudo systemctl enable vllm"
    fi

    # sudoers rule for cache-drop without password
    if [[ "${_svc_installed}" == "1" ]]; then
        local sudoers_file="/etc/sudoers.d/vllm-drop-caches"
        echo ""
        echo "[sudo] Creating sudoers rule so vLLM can drop page cache without a password."
        echo "       File: ${sudoers_file}"
        if echo "${SERVICE_USER} ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" \
                | sudo tee "${sudoers_file}" > /dev/null 2>&1; then
            sudo chmod 440 "${sudoers_file}" 2>/dev/null || true
            sudo visudo -cf "${sudoers_file}" 2>/dev/null \
                && success "sudoers rule: ${sudoers_file}" \
                || { warn "sudoers rule invalid — removing"; sudo rm -f "${sudoers_file}" 2>/dev/null || true; }
        else
            warn "Could not write sudoers rule — skipping (non-fatal)."
        fi

        sudo systemctl daemon-reload 2>/dev/null \
            && success "Service installed: ${SERVICE_FILE}" \
            || warn "systemctl daemon-reload failed — run manually: sudo systemctl daemon-reload"
        info "Not enabled yet. Run: thorllm start  or  sudo systemctl enable vllm"
    fi
}

# ── Model config YAML ─────────────────────────────────────────────────────────
write_model_config() {
    local model="${1:-${SERVE_MODEL}}"
    local org; org=$(echo "${model}" | cut -d/ -f1)
    local name; name=$(echo "${model}" | cut -d/ -f2)
    local yaml_file="${MODELS_DIR}/${org}/${name}.yaml"

    # Ensure MODELS_DIR and TPL_DIR are available (config_export may not have
    # been called yet in some code paths, e.g. from install.sh directly).
    MODELS_DIR="${MODELS_DIR:-${BUILD_PATH}/models}"
    TPL_DIR="${TPL_DIR:-${SELF_DIR:+${SELF_DIR}/templates}}"
    TPL_DIR="${TPL_DIR:-${HOME}/.local/share/thorllm/templates}"
    local _tpl="${TPL_DIR}/models/example.yaml"

    if [[ ! -f "${_tpl}" ]]; then
        warn "Model template not found: ${_tpl} — skipping model config creation."
        warn "  Clone the thorllm repo to get templates: thorllm repo at ${HOME}/.local/share/thorllm"
        return 1
    fi

    mkdir -p "${MODELS_DIR}/${org}"

    if [[ -f "${yaml_file}" ]]; then
        warn "Model config exists — not overwriting: ${yaml_file}"
    else
        info "Writing model config: ${yaml_file}"
        export MODEL_NAME="${model}" MODEL_SHORT="${name}"
        render_template "${_tpl}" "${yaml_file}" '${MODEL_NAME} ${MODEL_SHORT}'
        success "Model config: ${yaml_file}"
    fi

    # Always ensure the generic example template exists — useful for
    # 'thorllm model add' reference and README examples.
    local ex="${MODELS_DIR}/example/gpt-oss-120b.yaml"
    if [[ ! -f "${ex}" ]]; then
        mkdir -p "${MODELS_DIR}/example"
        # Seed the example with the default model values so it is always readable.
        export MODEL_NAME="openai/gpt-oss-120b" MODEL_SHORT="gpt-oss-120b"
        render_template "${_tpl}" "${ex}" '${MODEL_NAME} ${MODEL_SHORT}'
        # Restore the actual model vars in case callers use them afterwards.
        export MODEL_NAME="${model}" MODEL_SHORT="${name}"
        info "Example template: ${ex}"
    fi

    chown -R "${SERVICE_USER}:${SERVICE_USER}" "${MODELS_DIR}" 2>/dev/null || true
    info "Add more models: thorllm model add <org/name>"
}

# ── Activation script ─────────────────────────────────────────────────────────
write_activation_script() {
    local act="${BUILD_PATH}/activate_vllm.sh"
    config_export
    render_template "${TPL_DIR}/activate.sh" "${act}" \
        '${CUDA_HOME} ${TORCH_CUDA_ARCH} ${CACHE_ROOT} ${VENV_PATH}
         ${SERVE_MODEL} ${VLLM_CACHE_ROOT} ${VLLM_ASSETS_CACHE}
         ${VLLM_TUNED_CONFIG_FOLDER} ${FLASHINFER_JIT_DIR} ${TRITON_CACHE_DIR}
         ${TORCHINDUCTOR_CACHE_DIR} ${TORCH_COMPILE_CACHE_DIR}
         ${XDG_CACHE_HOME} ${TIKTOKEN_ENCODINGS_BASE} ${TIKTOKEN_RS_CACHE_DIR}
         ${HF_DOWNLOAD_DIR}'
    chmod +x "${act}"
    success "Activation script: ${act}"
}

# ── Service control ───────────────────────────────────────────────────────────
service_ctl() {
    local cmd="$1"
    local port="${VLLM_PORT:-8000}"

    case "${cmd}" in
        start)
            step "Starting ${SERVICE_NAME}"
            echo ""
            echo -e "[sudo] Starting systemd service requires elevated privileges."
            sudo systemctl start "${SERVICE_NAME}"
            echo ""
            success "vLLM service started."
            echo "  Check status: thorllm status"
            echo "  View logs:    thorllm logs -f"
            ;;
        stop)
            step "Stopping ${SERVICE_NAME}"
            sudo systemctl stop "${SERVICE_NAME}"
            success "Service stopped."
            ;;
        restart)
            step "Restarting ${SERVICE_NAME}"
            sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true
            sudo systemctl restart "${SERVICE_NAME}"
            echo ""
            success "vLLM service restarted."
            echo "  Check status: thorllm status"
            echo "  View logs:    thorllm logs -f"
            ;;
        status)
            sudo systemctl status "${SERVICE_NAME}" --no-pager -l
            ;;
    esac
}

service_logs() {
    local follow="${1:-0}"
    if [[ "${follow}" == "1" ]]; then
        journalctl -u "${SERVICE_NAME}" -f
    else
        journalctl -u "${SERVICE_NAME}" -n 100 --no-pager
    fi
}

# ── Kill ──────────────────────────────────────────────────────────────────────
service_kill() {
    step "Force-killing vLLM"
    echo ""

    # 1) Try systemctl stop first (graceful)
    if sudo systemctl is-active --quiet "${SERVICE_NAME}" 2>/dev/null; then
        info "Stopping systemd service…"
        sudo systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
    fi

    # 2) Kill any remaining vllm / python processes holding GPU memory
    local killed=0
    for proc in "vllm" "vllm.entrypoints" "vllm_serve" "vllm-serve"; do
        if pgrep -f "${proc}" &>/dev/null; then
            sudo pkill -9 -f "${proc}" 2>/dev/null && (( killed++ )) || true
        fi
    done

    # 3) Release GPU memory if possible
    if command -v nvidia-smi &>/dev/null; then
        local pids
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
        for pid in ${pids}; do
            if kill -0 "${pid}" 2>/dev/null; then
                sudo kill -9 "${pid}" 2>/dev/null && (( killed++ )) || true
            fi
        done
    fi

    if (( killed > 0 )); then
        success "Killed ${killed} process(es)."
    else
        info "No vLLM processes found."
    fi

    # Drop GPU page cache
    sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true
    success "GPU page cache cleared."
}
