#!/bin/bash
# tui/wizard.sh — shell bridge for the Python Textual TUI
#
# RELATIONSHIP BETWEEN wizard.sh AND wizard.py
# ─────────────────────────────────────────────
# wizard.py  — The actual TUI implementation.  Written in Python using the
#              Textual framework.  It renders all screens, handles user input,
#              and writes a JSON result file on completion.  Never run directly
#              from the shell in normal use; always invoked by wizard.sh.
#
# wizard.sh  — This file.  A pure-bash launcher / bridge that:
#                • checks whether Python + Textual are available,
#                • serialises current shell variables → JSON for wizard.py,
#                • calls wizard.py --output <tmpfile>,
#                • reads the JSON result back into shell variables,
#                • falls back to plain readline prompts when Textual is absent.
#
# There is NO code duplication: the two files have entirely different roles.
# wizard.sh knows about bash variables and system state; wizard.py knows about
# rendering a full-screen TUI.  Neither can replace the other.
# =============================================================================

_TUI_PY="${SELF_DIR}/tui/wizard.py"

# ── Check Textual availability ────────────────────────────────────────────────
_has_textual() {
    # Check system python3 first, then the vLLM venv (Textual may be there)
    if "${THORLLM_PYTHON:-python3}" -c "import textual" 2>/dev/null; then
        return 0
    fi
    local venv_py="${VENV_PATH:-${BUILD_PATH:-${HOME}/thorllm}/.vllm}/bin/python"
    if [[ -x "${venv_py}" ]] && "${venv_py}" -c "import textual" 2>/dev/null; then
        # Export so sub-invocations use the right interpreter
        export THORLLM_PYTHON="${venv_py}"
        return 0
    fi
    return 1
}

# ── Collect current config as JSON for the TUI ────────────────────────────────
_build_defaults_json() {
    "${THORLLM_PYTHON:-python3}" -c "
import json, sys
d = {
    'BUILD_PATH':    '${BUILD_PATH}',
    'CACHE_ROOT':    '${CACHE_ROOT}',
    'VLLM_VERSION':  '${VLLM_VERSION}',
    'TORCH_VERSION': '${TORCH_VERSION}',
    'SERVE_MODEL':   '${SERVE_MODEL}',
    'SERVICE_USER':  '${SERVICE_USER}',
}
print(json.dumps(d))
"
}

# ── Apply JSON result back to shell variables ─────────────────────────────────
_apply_config_json() {
    local json_file="$1"
    [[ -f "${json_file}" ]] || return 1

    BUILD_PATH=$("${THORLLM_PYTHON:-python3}" -c "import json,sys; d=json.load(open('${json_file}')); print(d.get('BUILD_PATH','${BUILD_PATH}'))")
    CACHE_ROOT=$("${THORLLM_PYTHON:-python3}" -c "import json,sys; d=json.load(open('${json_file}')); print(d.get('CACHE_ROOT','${CACHE_ROOT}'))")
    VLLM_VERSION=$("${THORLLM_PYTHON:-python3}" -c "import json,sys; d=json.load(open('${json_file}')); print(d.get('VLLM_VERSION',''))")
    TORCH_VERSION=$("${THORLLM_PYTHON:-python3}" -c "import json,sys; d=json.load(open('${json_file}')); print(d.get('TORCH_VERSION','${TORCH_VERSION}'))")
    SERVE_MODEL=$("${THORLLM_PYTHON:-python3}" -c "import json,sys; d=json.load(open('${json_file}')); print(d.get('SERVE_MODEL','${SERVE_MODEL}'))")
    SERVICE_USER=$("${THORLLM_PYTHON:-python3}" -c "import json,sys; d=json.load(open('${json_file}')); print(d.get('SERVICE_USER','${SERVICE_USER}'))")
    HF_TOKEN=$("${THORLLM_PYTHON:-python3}" -c "import json,sys; d=json.load(open('${json_file}')); print(d.get('HF_TOKEN',''))")
    export VENV_PATH="${BUILD_PATH}/${VENV_NAME}"
    export MODELS_DIR="${BUILD_PATH}/models"
}

# ── Fallback: plain readline prompts ─────────────────────────────────────────
_fallback_wizard() {
    warn "Textual not available — using plain prompts. Install with: pip install textual"
    echo ""
    usage_logo

    local val
    read -r -p "$(echo -e "BUILD_PATH [${BUILD_PATH}]: ")" val
    [[ -n "${val}" ]] && BUILD_PATH="${val}"

    read -r -p "$(echo -e "CACHE_ROOT [${CACHE_ROOT}]: ")" val
    [[ -n "${val}" ]] && CACHE_ROOT="${val}"

    read -r -p "$(echo -e "VLLM_VERSION (leave blank for latest) [${VLLM_VERSION}]: ")" val
    VLLM_VERSION="${val:-${VLLM_VERSION}}"

    read -r -p "$(echo -e "TORCH_VERSION [${TORCH_VERSION}]: ")" val
    [[ -n "${val}" ]] && TORCH_VERSION="${val}"

    read -r -p "$(echo -e "SERVE_MODEL [${SERVE_MODEL}]: ")" val
    [[ -n "${val}" ]] && SERVE_MODEL="${val}"

    read -r -s -p "$(echo -e "HF_TOKEN (hidden, optional): ")" val
    echo ""
    HF_TOKEN="${val}"

    read -r -p "$(echo -e "SERVICE_USER [${SERVICE_USER}]: ")" val
    [[ -n "${val}" ]] && SERVICE_USER="${val}"

    export VENV_PATH="${BUILD_PATH}/${VENV_NAME}"
    export MODELS_DIR="${BUILD_PATH}/models"
}

# ── Main wizard entry point ───────────────────────────────────────────────────
run_wizard() {
    config_load

    if ! _has_textual; then
        _fallback_wizard
    else
        local defaults_json tmp_out
        defaults_json=$(_build_defaults_json)
        tmp_out=$(mktemp /tmp/thorllm-config-XXXXXX.json)

        # Pass --output so the TUI writes JSON to a file; stdout/stderr stay
        # connected to the terminal so Textual can render its UI normally.
        if "${THORLLM_PYTHON:-python3}" "${_TUI_PY}" \
                --mode wizard \
                --defaults "${defaults_json}" \
                --version "${VERSION}" \
                --output "${tmp_out}"; then
            _apply_config_json "${tmp_out}"
            rm -f "${tmp_out}"
        else
            rm -f "${tmp_out}"
            info "Setup cancelled."
            return 0
        fi
    fi

    config_save
    export HF_TOKEN

    clear
    usage_logo
    step "Starting installation"
    echo ""

    source "${LIB_DIR}/install.sh"
    run_install 0
}

# ── Interactive model selection (called by `thorllm model select`) ─────────────
_model_select_interactive() {
    local active
    active=$(grep '^SERVE_MODEL=' "${BUILD_PATH}/vllm.env" 2>/dev/null \
        | cut -d= -f2 || echo "${SERVE_MODEL}")

    local models=()
    if [[ -d "${MODELS_DIR}" ]] && [[ -n "$(ls -A "${MODELS_DIR}" 2>/dev/null)" ]]; then
        while read -r f; do
            local rel
            rel=$(realpath --relative-to="${MODELS_DIR}" "${f}" | sed 's/\.yaml$//')
            models+=("${rel}")
        done < <(find "${MODELS_DIR}" -name "*.yaml" ! -path "*/example/*" | sort)
    fi

    if [[ ${#models[@]} -eq 0 ]]; then
        warn "No model configs found. Run: thorllm model add <org/name>"
        return 1
    fi

    if ! _has_textual; then
        # Fallback: numbered list
        echo ""
        step "Available models"
        local i=1
        for m in "${models[@]}"; do
            local marker="  ○"
            [[ "${m}" == "${active}" ]] && marker="  ●"
            echo -e "${marker} ${i}) ${m}"
            i=$(( i + 1 ))
        done
        echo ""
        read -r -p "$(echo -e "Select number [1-${#models[@]}]: ")" choice
        if [[ "${choice}" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#models[@]} )); then
            local selected="${models[$((choice-1))]}"
            source "${LIB_DIR}/model.sh"
            model_switch "${selected}"
        else
            warn "Invalid selection."
        fi
        return
    fi

    local models_json
    models_json=$("${THORLLM_PYTHON:-python3}" -c "import json,sys; print(json.dumps(sys.argv[1:]))" "${models[@]}")
    local tmp_out
    tmp_out=$(mktemp /tmp/thorllm-model-XXXXXX.json)

    # --output keeps stdout/stderr free for Textual to render the TUI.
    if "${THORLLM_PYTHON:-python3}" "${_TUI_PY}" \
            --mode model-select \
            --models "${models_json}" \
            --active "${active}" \
            --version "${VERSION}" \
            --output "${tmp_out}"; then
        local selected
        selected=$("${THORLLM_PYTHON:-python3}" -c "import json; d=json.load(open('${tmp_out}')); print(d.get('selected',''))")
        rm -f "${tmp_out}"
        if [[ -n "${selected}" ]]; then
            source "${LIB_DIR}/model.sh"
            model_switch "${selected}"
        fi
    else
        rm -f "${tmp_out}"
        info "Model selection cancelled."
    fi
}

# ── Interactive model management (called by `thorllm model`) ──────────────────
model_manage_interactive() {
    local active
    active=$(grep '^SERVE_MODEL=' "${BUILD_PATH}/vllm.env" 2>/dev/null \
        | cut -d= -f2 || echo "${SERVE_MODEL}")

    local models=()
    if [[ -d "${MODELS_DIR}" ]] && [[ -n "$(ls -A "${MODELS_DIR}" 2>/dev/null)" ]]; then
        while read -r f; do
            local rel
            rel=$(realpath --relative-to="${MODELS_DIR}" "${f}" | sed 's/\.yaml$//')
            models+=("${rel}")
        done < <(find "${MODELS_DIR}" -name "*.yaml" ! -path "*/example/*" | sort)
    fi

    if ! _has_textual; then
        # Fallback: delegate to regular subcommand display
        source "${LIB_DIR}/model.sh"
        model_list
        return
    fi

    local models_json
    models_json=$("${THORLLM_PYTHON:-python3}" -c "import json,sys; print(json.dumps(sys.argv[1:]))" "${models[@]}")
    local tmp_out
    tmp_out=$(mktemp /tmp/thorllm-model-manage-XXXXXX.json)

    if "${THORLLM_PYTHON:-python3}" "${_TUI_PY}" \
            --mode model-manage \
            --models "${models_json}" \
            --active "${active}" \
            --version "${VERSION}" \
            --output "${tmp_out}"; then
        local action model_arg
        action=$("${THORLLM_PYTHON:-python3}" -c "import json; d=json.load(open('${tmp_out}')); print(d.get('action',''))")
        model_arg=$("${THORLLM_PYTHON:-python3}" -c "import json; d=json.load(open('${tmp_out}')); print(d.get('model','') or d.get('selected',''))")
        rm -f "${tmp_out}"
        if [[ -n "${action}" && -n "${model_arg}" ]]; then
            source "${LIB_DIR}/model.sh"
            case "${action}" in
                add)    model_add    "${model_arg}" ;;
                switch) model_switch "${model_arg}" ;;
                show)   model_show   "${model_arg}" ;;
                edit)   model_edit   "${model_arg}" ;;
            esac
        fi
    else
        rm -f "${tmp_out}"
        info "Model management closed."
    fi
}