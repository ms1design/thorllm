#!/bin/bash
# lib/config.sh — persistent config read/write and env export
# Config file format: KEY=value (no 'export', no quotes needed for simple values)
# =============================================================================

# ── Defaults (all overridable via env or config file) ────────────────────────
BUILD_PATH="${BUILD_PATH:-${HOME}/thorllm}"
VENV_NAME="${VENV_NAME:-.vllm}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache/vllm}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"  # Keep CUDA from host — do not override
TORCH_CUDA_ARCH="${TORCH_CUDA_ARCH:-11.0a}"
SERVICE_NAME="${SERVICE_NAME:-vllm}"
SERVICE_USER="${SERVICE_USER:-$(whoami)}"
SERVICE_FILE="${SERVICE_FILE:-/etc/systemd/system/${SERVICE_NAME}.service}"
SERVE_MODEL="${SERVE_MODEL:-openai/gpt-oss-120b}"
VLLM_PORT="${VLLM_PORT:-8000}"

# NVFP4 / SM110 specifics
# SM110 patch enables the CUTLASS FP4 backend on Thor (40-60% faster than Marlin fallback).
# Triton attention backend works normally on SM110 — only Triton *MXFP4 GEMM* kernels
# are excluded (cap range (9,0)..(11,0) already excludes SM110=(11,0) upstream).
NVFP4_ENABLE="${NVFP4_ENABLE:-1}"   # 1 = apply SM110 + layernorm patches after install

# Versions — leave empty to auto-detect latest
VLLM_VERSION="${VLLM_VERSION:-}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.10.0}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-}"
NUMBA_VERSION="${NUMBA_VERSION:-0.61.2}"
CUDSS_VERSION="${CUDSS_VERSION:-0.7.1}"

# Derived paths (computed, not stored in config)
VENV_PATH="${BUILD_PATH}/${VENV_NAME}"
MODELS_DIR="${BUILD_PATH}/models"
# TPL_DIR and THORLLM_LIB are resolved relative to the running script (SELF_DIR/LIB_DIR)
# when available, falling back to the canonical install location.
_THORLLM_SHARE="${HOME}/.local/share/thorllm"
TPL_DIR="${TPL_DIR:-${SELF_DIR:+${SELF_DIR}/templates}}"
TPL_DIR="${TPL_DIR:-${_THORLLM_SHARE}/templates}"
THORLLM_LIB="${LIB_DIR:-${THORLLM_LIB:-${_THORLLM_SHARE}/lib}}"

# ── Load config from file ─────────────────────────────────────────────────────
config_load() {
    local cfg="${1:-${BUILD_PATH}/thorllm.conf}"
    [[ -f "${cfg}" ]] || return 0
    # shellcheck disable=SC1090
    source "${cfg}"
    # Recompute derived paths after loading
    VENV_PATH="${BUILD_PATH}/${VENV_NAME}"
    MODELS_DIR="${BUILD_PATH}/models"
}

# ── Write config to file ──────────────────────────────────────────────────────
config_save() {
    local cfg="${BUILD_PATH}/thorllm.conf"
    mkdir -p "${BUILD_PATH}"
    cat > "${cfg}" <<CONF
# thorllm configuration — auto-generated, safe to edit
# Re-run 'thorllm setup' to reconfigure interactively.

BUILD_PATH=${BUILD_PATH}
CACHE_ROOT=${CACHE_ROOT}
CUDA_HOME=${CUDA_HOME}
TORCH_CUDA_ARCH=${TORCH_CUDA_ARCH}
SERVICE_NAME=${SERVICE_NAME}
SERVICE_USER=${SERVICE_USER}
SERVE_MODEL=${SERVE_MODEL}

# Pinned versions (leave empty to always use latest)
VLLM_VERSION=${VLLM_VERSION}
TORCH_VERSION=${TORCH_VERSION}
TORCHVISION_VERSION=${TORCHVISION_VERSION}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}
FLASHINFER_VERSION=${FLASHINFER_VERSION}
NUMBA_VERSION=${NUMBA_VERSION}
CUDSS_VERSION=${CUDSS_VERSION}
CONF
    chmod 600 "${cfg}"
}

# ── Show current config ───────────────────────────────────────────────────────
config_show() {
    echo ""
    echo -e "Current configuration:"
    echo ""
    printf "  %-30s%s\n" "BUILD_PATH"        "${BUILD_PATH}"
    printf "  %-30s%s\n" "CACHE_ROOT"        "${CACHE_ROOT}"
    printf "  %-30s%s\n" "CUDA_HOME"         "${CUDA_HOME}"
    printf "  %-30s%s\n" "TORCH_CUDA_ARCH"   "${TORCH_CUDA_ARCH}"
    printf "  %-30s%s\n" "SERVICE_USER"      "${SERVICE_USER}"
    printf "  %-30s%s\n" "SERVE_MODEL"       "${SERVE_MODEL}"
    printf "  %-30s%s\n" "VLLM_PORT"         "${VLLM_PORT:-8000}"
    printf "  %-30s%s\n" "VLLM_VERSION"      "${VLLM_VERSION:-auto}"
    printf "  %-30s%s\n" "TORCH_VERSION"     "${TORCH_VERSION}"
    printf "  %-30s%s\n" "TORCHVISION_VERSION" "${TORCHVISION_VERSION}"
    printf "  %-30s%s\n" "TORCHAUDIO_VERSION"  "${TORCHAUDIO_VERSION}"
    printf "  %-30s%s\n" "FLASHINFER_VERSION" "${FLASHINFER_VERSION:-auto}"
    printf "  %-30s%s\n" "NUMBA_VERSION"      "${NUMBA_VERSION}"
    printf "  %-30s%s\n" "CUDSS_VERSION"      "${CUDSS_VERSION}"
    printf "  %-30s%s\n" "HF_TOKEN"           "${HF_TOKEN:+(set)}"
    echo ""
    print_footer
}

# ── Export all env vars needed by templates ───────────────────────────────────
config_export() {
    export BUILD_PATH VENV_NAME VENV_PATH CACHE_ROOT CUDA_HOME TORCH_CUDA_ARCH
    export SERVICE_NAME SERVICE_USER SERVICE_FILE SERVE_MODEL
    export MODELS_DIR TPL_DIR
    export VLLM_VERSION TORCH_VERSION TORCHVISION_VERSION TORCHAUDIO_VERSION
    export FLASHINFER_VERSION NUMBA_VERSION CUDSS_VERSION
    # Derived cache paths
    export VLLM_CACHE_ROOT="${CACHE_ROOT}"
    export VLLM_ASSETS_CACHE="${CACHE_ROOT}/assets"
    export VLLM_TUNED_CONFIG_FOLDER="${CACHE_ROOT}/kernels"
    export FLASHINFER_JIT_DIR="${CACHE_ROOT}/flashinfer"
    export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
    export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
    export TORCH_COMPILE_CACHE_DIR="${CACHE_ROOT}/torch_compile"
    export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
    export TIKTOKEN_ENCODINGS_BASE="${CACHE_ROOT}/tiktoken"
    export TIKTOKEN_RS_CACHE_DIR="${CACHE_ROOT}/tiktoken"
    export HF_DOWNLOAD_DIR="${CACHE_ROOT}/huggingface"

    # ── Triton toolchain paths (all point into CUDA_HOME) ────────────────────
    # Required for Triton to find CUDA components during kernel JIT compilation.
    export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"
    export TRITON_PTXAS_BLACKWELL_PATH="${CUDA_HOME}/bin/ptxas"
    export TRITON_CUDART_PATH="${CUDA_HOME}/include"
    export TRITON_CUDACRT_PATH="${CUDA_HOME}/include"
    export TRITON_CUPTI_PATH="${CUDA_HOME}/include"
    export TRITON_NVDISASM_PATH="${CUDA_HOME}/bin/nvdisasm"
    export TRITON_CUOBJDUMP_PATH="${CUDA_HOME}/bin/cuobjdump"

    # ── Build paths ───────────────────────────────────────────────────────────
    # Linker stub path — needed when compiling CUDA extensions
    export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH:-}"

    # ── tiktoken cache (three vars cover all tiktoken versions) ──────────────
    export TIKTOKEN_CACHE_DIR="${CACHE_ROOT}/tiktoken"
}
