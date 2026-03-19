#!/bin/bash
# lib/service.sh — systemd unit management, env file, launcher generation
# Sources: common.sh, config.sh
# =============================================================================

# ── EnvironmentFile ───────────────────────────────────────────────────────────
write_env_file() {
    local env_file="${BUILD_PATH}/vllm.env"
    info "Writing EnvironmentFile: ${env_file}"

    config_export   # ensure all vars are set

    # List the variables envsubst should substitute in the template
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
    local env_file="${BUILD_PATH}/vllm.env"

    info "Writing launcher: ${serve_script}"
    local vars='${VENV_PATH} ${BUILD_PATH} ${MODELS_DIR}'
    render_template "${TPL_DIR}/vllm-serve.sh" "${serve_script}" "${vars}"
    chmod +x "${serve_script}"
    chown "${SERVICE_USER}:${SERVICE_USER}" "${serve_script}" 2>/dev/null || true
    success "Launcher: ${serve_script}"

    info "Writing systemd unit: ${SERVICE_FILE}"
    local svc_vars='${SERVICE_USER} ${BUILD_PATH}'
    render_template "${TPL_DIR}/vllm.service" "/tmp/vllm.service.rendered" "${svc_vars}"
    sudo cp "/tmp/vllm.service.rendered" "${SERVICE_FILE}"
    rm -f "/tmp/vllm.service.rendered"

    # sudoers rule so the launcher can drop page cache without a password prompt
    local sudoers_file="/etc/sudoers.d/vllm-drop-caches"
    echo "${SERVICE_USER} ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" \
        | sudo tee "${sudoers_file}" > /dev/null
    sudo chmod 440 "${sudoers_file}"
    sudo visudo -cf "${sudoers_file}" \
        && success "sudoers rule: ${sudoers_file}" \
        || { warn "sudoers rule invalid — removing"; sudo rm -f "${sudoers_file}"; }

    sudo systemctl daemon-reload
    success "Service installed: ${SERVICE_FILE}"
    info "Not enabled yet. Run: thorllm start  or  sudo systemctl enable vllm"
}

# ── Model config YAML ─────────────────────────────────────────────────────────
write_model_config() {
    local model="${1:-${SERVE_MODEL}}"
    local org; org=$(echo "${model}" | cut -d/ -f1)
    local name; name=$(echo "${model}" | cut -d/ -f2)
    local yaml_file="${MODELS_DIR}/${org}/${name}.yaml"

    mkdir -p "${MODELS_DIR}/${org}"

    if [[ -f "${yaml_file}" ]]; then
        warn "Model config exists — not overwriting: ${yaml_file}"
        return
    fi

    info "Writing model config: ${yaml_file}"
    export MODEL_NAME="${model}" MODEL_SHORT="${name}"
    render_template "${TPL_DIR}/models/example.yaml" "${yaml_file}" \
        '${MODEL_NAME} ${MODEL_SHORT}'

    # Also seed an example template if missing
    local ex="${MODELS_DIR}/example/gpt-oss-120b.yaml"
    if [[ ! -f "${ex}" ]]; then
        mkdir -p "${MODELS_DIR}/example"
        cp "${yaml_file}" "${ex}"
        info "Example template: ${ex}"
    fi

    chown -R "${SERVICE_USER}:${SERVICE_USER}" "${MODELS_DIR}" 2>/dev/null || true
    success "Model config: ${yaml_file}"
    info "Add more models: thorllm model add <org/name>"
}

# ── Activation script (manual shell use) ─────────────────────────────────────
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
    case "${cmd}" in
        start)
            info "Starting ${SERVICE_NAME}…"
            sudo systemctl start "${SERVICE_NAME}"
            sudo systemctl status "${SERVICE_NAME}" --no-pager -l
            ;;
        stop)
            info "Stopping ${SERVICE_NAME}…"
            sudo systemctl stop "${SERVICE_NAME}"
            ;;
        restart)
            info "Restarting ${SERVICE_NAME}…"
            sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true
            sudo systemctl restart "${SERVICE_NAME}"
            sudo systemctl status "${SERVICE_NAME}" --no-pager -l
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
