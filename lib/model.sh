#!/bin/bash
# lib/model.sh — model config management (add / list / switch / show / edit)
# Sources: common.sh, config.sh, service.sh
# =============================================================================

# LIB_DIR is exported by bin/thorllm before this file is sourced.
# THORLLM_LIB is kept as a legacy alias pointing to the same place.
LIB_DIR="${LIB_DIR:-${THORLLM_LIB:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
[[ -f "${LIB_DIR}/common.sh" ]] && source "${LIB_DIR}/common.sh"
source "${LIB_DIR}/config.sh"
source "${LIB_DIR}/service.sh"

_model_yaml_path() {
    local model="$1"
    local org; org=$(echo "${model}" | cut -d/ -f1)
    local name; name=$(echo "${model}" | cut -d/ -f2)
    echo "${MODELS_DIR}/${org}/${name}.yaml"
}

model_list() {
    local active; active=$(grep '^SERVE_MODEL=' "${BUILD_PATH}/vllm.env" 2>/dev/null \
        | cut -d= -f2 || echo "${SERVE_MODEL}")

    usage_logo
    echo -e "  Available models  (${MODELS_DIR})"
    echo ""
    if [[ ! -d "${MODELS_DIR}" ]] || [[ -z "$(ls -A "${MODELS_DIR}" 2>/dev/null)" ]]; then
        echo -e "  (none — run: thorllm model add <org/name>)"
    else
        find "${MODELS_DIR}" -name "*.yaml" ! -path "*/example/*" \
            | sort \
            | while read -r f; do
                local rel; rel=$(realpath --relative-to="${MODELS_DIR}" "${f}" | sed 's/\.yaml$//')
                if [[ "${rel}" == "${active}" ]]; then
                    printf "  ● %-40s (active)\n" "${rel}"
                else
                    printf "  ○ %s\n" "${rel}"
                fi
            done
    fi
    echo ""
    print_footer
}

model_add() {
    local model="$1"
    local yaml_file; yaml_file=$(_model_yaml_path "${model}")

    if [[ -f "${yaml_file}" ]]; then
        warn "Config already exists: ${yaml_file}"
        confirm "Overwrite?" || return 0
    fi

    write_model_config "${model}"

    echo ""
    info "Edit the config now? (${yaml_file})"
    confirm "Open in \$EDITOR?" && model_edit "${model}"
}

model_switch() {
    local model="$1"
    local yaml_file; yaml_file=$(_model_yaml_path "${model}")

    [[ -f "${yaml_file}" ]] \
        || die "Model config not found: ${yaml_file}\n  Create it first: thorllm model add ${model}"

    step "Switching active model"
    info "→ ${model}"

    local env_file="${BUILD_PATH}/vllm.env"
    [[ -f "${env_file}" ]] || die "EnvironmentFile not found: ${env_file}\n  Run: thorllm install"
    sed -i "s|^SERVE_MODEL=.*|SERVE_MODEL=${model}|" "${env_file}"

    SERVE_MODEL="${model}"
    config_save

    success "Active model: ${model}"
    confirm "Restart service now?" && service_ctl restart
}

model_show() {
    local model="$1"
    local yaml_file; yaml_file=$(_model_yaml_path "${model}")
    [[ -f "${yaml_file}" ]] || die "Config not found: ${yaml_file}"
    echo ""
    cat "${yaml_file}"
    echo ""
}

model_edit() {
    local model="$1"
    local yaml_file; yaml_file=$(_model_yaml_path "${model}")
    [[ -f "${yaml_file}" ]] || die "Config not found: ${yaml_file}\n  Create it first: thorllm model add ${model}"
    "${EDITOR:-nano}" "${yaml_file}"
}

# ── Interactive model select — delegates to TUI wizard or fallback ─────────────
_model_select_interactive() {
    # Delegated to wizard.sh where Textual / fallback logic lives
    source "${SELF_DIR}/tui/wizard.sh"
    _model_select_interactive
}
