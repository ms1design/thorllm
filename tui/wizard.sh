#!/bin/bash
# tui/wizard.sh — interactive setup wizard bridge
# Launches the Python Textual TUI (tui/wizard.py) and reads back JSON config.
# Falls back to basic interactive prompts if Python/Textual is not available.
# =============================================================================

_TUI_PY="${SELF_DIR}/tui/wizard.py"

# ── Check Textual availability ────────────────────────────────────────────────
_has_textual() {
    python3 -c "import textual" 2>/dev/null
}

# ── Collect current config as JSON for the TUI ────────────────────────────────
_build_defaults_json() {
    python3 -c "
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

    BUILD_PATH=$(python3 -c "import json,sys; d=json.load(open('${json_file}')); print(d.get('BUILD_PATH','${BUILD_PATH}'))")
    CACHE_ROOT=$(python3 -c "import json,sys; d=json.load(open('${json_file}')); print(d.get('CACHE_ROOT','${CACHE_ROOT}'))")
    VLLM_VERSION=$(python3 -c "import json,sys; d=json.load(open('${json_file}')); print(d.get('VLLM_VERSION',''))")
    TORCH_VERSION=$(python3 -c "import json,sys; d=json.load(open('${json_file}')); print(d.get('TORCH_VERSION','${TORCH_VERSION}'))")
    SERVE_MODEL=$(python3 -c "import json,sys; d=json.load(open('${json_file}')); print(d.get('SERVE_MODEL','${SERVE_MODEL}'))")
    SERVICE_USER=$(python3 -c "import json,sys; d=json.load(open('${json_file}')); print(d.get('SERVICE_USER','${SERVICE_USER}'))")
    HF_TOKEN=$(python3 -c "import json,sys; d=json.load(open('${json_file}')); print(d.get('HF_TOKEN',''))")
    export VENV_PATH="${BUILD_PATH}/${VENV_NAME}"
    export MODELS_DIR="${BUILD_PATH}/models"
}

# ── Fallback: plain readline prompts ─────────────────────────────────────────
_fallback_wizard() {
    warn "Textual not available — using plain prompts. Install with: pip install textual"
    echo ""
    usage_logo

    local val
    read -r -p "$(echo -e "${CYAN}BUILD_PATH${NC} [${BUILD_PATH}]: ")" val
    [[ -n "${val}" ]] && BUILD_PATH="${val}"

    read -r -p "$(echo -e "${CYAN}CACHE_ROOT${NC} [${CACHE_ROOT}]: ")" val
    [[ -n "${val}" ]] && CACHE_ROOT="${val}"

    read -r -p "$(echo -e "${CYAN}VLLM_VERSION${NC} (leave blank for latest) [${VLLM_VERSION}]: ")" val
    VLLM_VERSION="${val:-${VLLM_VERSION}}"

    read -r -p "$(echo -e "${CYAN}TORCH_VERSION${NC} [${TORCH_VERSION}]: ")" val
    [[ -n "${val}" ]] && TORCH_VERSION="${val}"

    read -r -p "$(echo -e "${CYAN}SERVE_MODEL${NC} [${SERVE_MODEL}]: ")" val
    [[ -n "${val}" ]] && SERVE_MODEL="${val}"

    read -r -s -p "$(echo -e "${CYAN}HF_TOKEN${NC} (hidden, optional): ")" val
    echo ""
    HF_TOKEN="${val}"

    read -r -p "$(echo -e "${CYAN}SERVICE_USER${NC} [${SERVICE_USER}]: ")" val
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
        if python3 "${_TUI_PY}" \
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
            [[ "${m}" == "${active}" ]] && marker="${NVIDIA_GREEN}  ●${NC}"
            echo -e "${marker} ${i}) ${m}"
            (( i++ ))
        done
        echo ""
        read -r -p "$(echo -e "${CYAN}Select number [1-${#models[@]}]:${NC} ")" choice
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
    models_json=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1:]))" "${models[@]}")
    local tmp_out
    tmp_out=$(mktemp /tmp/thorllm-model-XXXXXX.json)

    # --output keeps stdout/stderr free for Textual to render the TUI.
    if python3 "${_TUI_PY}" \
            --mode model-select \
            --models "${models_json}" \
            --active "${active}" \
            --version "${VERSION}" \
            --output "${tmp_out}"; then
        local selected
        selected=$(python3 -c "import json; d=json.load(open('${tmp_out}')); print(d.get('selected',''))")
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
