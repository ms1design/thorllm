#!/bin/bash
# tui/wizard.sh — interactive setup wizard using whiptail
# Sources: common.sh, config.sh, install.sh, service.sh
# =============================================================================

# ── whiptail wrapper helpers ──────────────────────────────────────────────────
_wt_input() {
    # _wt_input "Title" "Prompt" "default"  →  stdout: user value
    local title="$1" prompt="$2" default="$3"
    whiptail --title "${title}" --inputbox "${prompt}" 10 70 "${default}" 3>&1 1>&2 2>&3
}

_wt_password() {
    local title="$1" prompt="$2"
    whiptail --title "${title}" --passwordbox "${prompt}" 10 70 "" 3>&1 1>&2 2>&3
}

_wt_menu() {
    # _wt_menu "Title" "Prompt" item1 desc1 item2 desc2 ...
    local title="$1" prompt="$2"; shift 2
    whiptail --title "${title}" --menu "${prompt}" 20 76 10 "$@" 3>&1 1>&2 2>&3
}

_wt_yesno() {
    # Returns 0 for Yes, 1 for No
    local title="$1" msg="$2"
    whiptail --title "${title}" --yesno "${msg}" 10 70
}

_wt_msg() {
    local title="$1" msg="$2"
    whiptail --title "${title}" --msgbox "${msg}" 14 76
}

_wt_checklist() {
    # _wt_checklist "Title" "Prompt" item1 desc1 ON/OFF ...
    local title="$1" prompt="$2"; shift 2
    whiptail --title "${title}" --checklist "${prompt}" 20 76 10 "$@" 3>&1 1>&2 2>&3
}

# ── Wizard pages ──────────────────────────────────────────────────────────────
_page_welcome() {
    _wt_msg "thorllm setup" \
"Welcome to thorllm — vLLM manager for NVIDIA Jetson Thor.

This wizard will configure your installation.

You will be asked about:
  • Installation and cache directories
  • vLLM version to install
  • Default model to serve
  • HuggingFace token (optional)
  • systemd service user

Settings are saved to ${BUILD_PATH}/thorllm.conf
and can be changed by re-running: thorllm setup"
}

_page_build_path() {
    local val
    val=$(_wt_input "Installation" \
        "Installation directory (BUILD_PATH):\nAll vllm files, venv, and model configs will be stored here." \
        "${BUILD_PATH}") || return 1
    [[ -n "${val}" ]] && BUILD_PATH="${val}"
    # Recompute derived paths (used by config_export when install runs)
    export VENV_PATH="${BUILD_PATH}/${VENV_NAME}"
    export MODELS_DIR="${BUILD_PATH}/models"
}

_page_cache_root() {
    local val
    val=$(_wt_input "Cache" \
        "Cache root directory (CACHE_ROOT):\nAll Triton / FlashInfer / HuggingFace caches will be stored here.\nPut this on your fastest storage (NVMe if available)." \
        "${CACHE_ROOT}") || return 1
    [[ -n "${val}" ]] && CACHE_ROOT="${val}"
}

_page_vllm_version() {
    info "Fetching available vLLM releases…"
    local latest
    latest=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest \
        | jq -r .tag_name | sed 's/^v//' 2>/dev/null || echo "")

    local recent
    recent=$(curl -s "https://api.github.com/repos/vllm-project/vllm/releases?per_page=6" \
        | jq -r '.[].tag_name' | sed 's/^v//' 2>/dev/null | head -5 || echo "")

    local menu_items=()
    menu_items+=("latest" "Always install the latest stable release")
    while IFS= read -r v; do
        [[ -n "${v}" ]] && menu_items+=("${v}" "vLLM v${v}")
    done <<< "${recent}"

    local choice
    choice=$(_wt_menu "vLLM version" \
        "Select vLLM version to install:\n(Latest detected: ${latest:-unknown})" \
        "${menu_items[@]}") || return 1

    if [[ "${choice}" == "latest" ]]; then
        VLLM_VERSION=""
    else
        VLLM_VERSION="${choice}"
    fi
}

_page_model() {
    local val
    val=$(_wt_input "Default model" \
        "Default model to serve (SERVE_MODEL):\nFormat: <org>/<model-name>  e.g. openai/gpt-oss-120b\nA YAML config will be created at:\n  \${BUILD_PATH}/models/<org>/<name>.yaml" \
        "${SERVE_MODEL}") || return 1
    [[ -n "${val}" ]] && SERVE_MODEL="${val}"
}

_page_hf_token() {
    local val
    val=$(_wt_password "HuggingFace" \
        "HuggingFace access token (optional):\nRequired only for gated models (Llama, Gemma, etc).\nLeave empty to skip.") || return 1
    HF_TOKEN="${val}"
}

_page_service_user() {
    local val
    val=$(_wt_input "Service user" \
        "User account to run the vLLM systemd service:" \
        "${SERVICE_USER}") || return 1
    [[ -n "${val}" ]] && SERVICE_USER="${val}"
}

_page_torch_version() {
    local val
    val=$(_wt_input "PyTorch version" \
        "PyTorch version to install (must have cu130 build on download.pytorch.org):" \
        "${TORCH_VERSION}") || return 1
    [[ -n "${val}" ]] && TORCH_VERSION="${val}"
}

_page_summary() {
    local summary
    summary="Configuration summary:

  BUILD_PATH       ${BUILD_PATH}
  CACHE_ROOT       ${CACHE_ROOT}
  VLLM_VERSION     ${VLLM_VERSION:-latest}
  TORCH_VERSION    ${TORCH_VERSION}
  SERVE_MODEL      ${SERVE_MODEL}
  SERVICE_USER     ${SERVICE_USER}
  HF_TOKEN         ${HF_TOKEN:+(set)}${HF_TOKEN:-not set}

Press OK to save and start installation,
or Cancel to go back."

    _wt_yesno "Ready to install" "${summary}" || return 1
}

# ── Main wizard ───────────────────────────────────────────────────────────────
run_wizard() {
    command -v whiptail &>/dev/null \
        || { warn "whiptail not found — running non-interactive install."
             run_install 0; return; }

    # Load existing config if present
    config_load

    _page_welcome       || { info "Setup cancelled."; return 0; }
    _page_build_path    || { info "Setup cancelled."; return 0; }
    _page_cache_root    || { info "Setup cancelled."; return 0; }
    _page_vllm_version  || { info "Setup cancelled."; return 0; }
    _page_model         || { info "Setup cancelled."; return 0; }
    _page_hf_token      || true   # optional — never abort on this page
    _page_service_user  || { info "Setup cancelled."; return 0; }
    _page_torch_version || { info "Setup cancelled."; return 0; }
    _page_summary       || { info "Setup cancelled."; return 0; }

    # Save config then run installer
    config_save
    export HF_TOKEN

    clear
    source "${LIB_DIR}/install.sh"
    run_install 0
}
