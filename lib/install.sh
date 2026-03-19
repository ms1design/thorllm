#!/bin/bash
# lib/install.sh — vLLM installer for Jetson Thor
# Sources: common.sh, config.sh (must be loaded first)
# =============================================================================

# Ensure LIB_DIR is set when this file is sourced standalone (e.g. from wizard)
LIB_DIR="${LIB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# ── Preflight ─────────────────────────────────────────────────────────────────
preflight() {
    step "Preflight checks"

    [[ "$(uname -m)" == "aarch64" ]] || die "Jetson Thor is aarch64. Current arch: $(uname -m)"

    if ! command -v nvcc &>/dev/null; then
        warn "nvcc not on PATH — trying ${CUDA_HOME}/bin"
        export PATH="${CUDA_HOME}/bin:${PATH}"
    fi
    nvcc --version 2>/dev/null | head -1 \
        || die "CUDA toolkit not found.\n  Run: sudo apt-get install -y cuda-toolkit-13-0"

    local cuda_ver
    cuda_ver=$(nvcc --version 2>/dev/null | awk '/release/{print $6}' | tr -d 'V,')
    # JetPack ships CUDA 13.0 or 13.1 — both work with cu130 wheels
    [[ "${cuda_ver}" == 13.* ]] \
        || warn "Expected CUDA 13.x, found ${cuda_ver}. cu130 wheels may not match."

    require_cmd jq curl git wget

    success "Preflight passed (CUDA ${cuda_ver})."
}

# ── cuDSS ─────────────────────────────────────────────────────────────────────
install_cudss() {
    step "cuDSS ${CUDSS_VERSION}"

    if dpkg -l cudss 2>/dev/null | grep -q "^ii"; then
        success "cuDSS already installed — skipping."
        return
    fi

    local deb="cudss-local-repo-ubuntu2404-${CUDSS_VERSION}_${CUDSS_VERSION}-1_arm64.deb"
    local url="https://developer.download.nvidia.com/compute/cudss/${CUDSS_VERSION}/local_installers/${deb}"
    wget --quiet --show-progress --no-check-certificate "${url}" -O "/tmp/${deb}"
    sudo dpkg -i "/tmp/${deb}"
    sudo cp /var/cudss-local-*/cudss-*-keyring.gpg /usr/share/keyrings/ 2>/dev/null || true
    sudo apt-get update -qq
    sudo apt-get -y install cudss
    rm -f "/tmp/${deb}"
    success "cuDSS ${CUDSS_VERSION} installed."
}

# ── nvpl_slim (ARM Performance Libraries) ────────────────────────────────────
# Required for NVFP4 matrix ops on aarch64. Public distribution via apt.
install_nvpl() {
    step "nvpl (ARM Performance Libraries)"
    if dpkg -l libnvpl-blas-dev 2>/dev/null | grep -q "^ii"; then
        success "nvpl already installed — skipping."
        return
    fi
    # Available from the CUDA apt repo added during cuda-toolkit-13-0 install
    sudo apt-get install -y --no-install-recommends         libnvpl-blas-dev libnvpl-lapack-dev libnvpl-common-dev 2>/dev/null         && success "nvpl installed via apt."         || warn "nvpl not available via apt — skipping. NVFP4 GEMM may use fallback kernels."
}

# ── uv ────────────────────────────────────────────────────────────────────────
install_uv() {
    step "uv"
    if ! command -v uv &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    fi
    success "uv $(uv --version 2>&1 | awk '{print $2}') ready."
}

# ── Python venv ───────────────────────────────────────────────────────────────
create_venv() {
    step "Python venv (3.12)"
    sudo apt-get install -y --no-install-recommends python3-dev python3.12-dev 2>/dev/null || true
    mkdir -p "${BUILD_PATH}"
    cd "${BUILD_PATH}" || die "Cannot cd to BUILD_PATH: ${BUILD_PATH}"
    export UV_VENV_CLEAR=1
    uv venv "${VENV_NAME}" --python 3.12
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"
    success "venv: $(python --version)"
}

# ── CUDA env ──────────────────────────────────────────────────────────────────
set_cuda_env() {
    export CUDA_HOME
    export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"
    export TRITON_PTXAS_BLACKWELL_PATH="${TRITON_PTXAS_PATH}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    export CPATH="${CUDA_HOME}/include:${CPATH:-}"
    export C_INCLUDE_PATH="${CUDA_HOME}/include:${C_INCLUDE_PATH:-}"
    export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"
    export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
    export NVRTC_LIBRARY="${CUDA_HOME}/lib64/libnvrtc.so"
    export NVRTC_INCLUDE_DIR="${CUDA_HOME}/include"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH}"
    export DG_JIT_USE_NVRTC=1
    export USE_CUDSS=1 USE_CUDNN=1 USE_CUFILE=1 USE_CUSPARSELT=1
    info "CUDA env: CUDA_HOME=${CUDA_HOME}, ARCH=${TORCH_CUDA_ARCH}"
}

# ── PyTorch ───────────────────────────────────────────────────────────────────
install_pytorch() {
    step "PyTorch ${TORCH_VERSION}+cu130"
    uv pip install --force-reinstall \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu130

    python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
assert '13.0' in torch.version.cuda, f'Expected CUDA 13.0, got {torch.version.cuda}'
print(f'  torch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}')
" || die "PyTorch CUDA validation failed."
    success "PyTorch validated."
}

# ── vLLM (pre-compiled wheel) ─────────────────────────────────────────────────
_resolve_vllm_version() {
    if [[ -z "${VLLM_VERSION}" ]]; then
        info "Detecting latest vLLM release…"
        VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest \
            | jq -r .tag_name | sed 's/^v//')
        [[ -n "${VLLM_VERSION}" ]] || die "Could not detect latest vLLM version (GitHub API error?)"
    fi
    info "Target vLLM: ${VLLM_VERSION}"
}

install_vllm_wheel() {
    step "vLLM (pre-compiled wheel)"
    local arch; arch=$(uname -m)
    _resolve_vllm_version

    local wheel_url="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu130-cp38-abi3-manylinux_2_35_${arch}.whl"
    info "Checking: ${wheel_url}"
    local http_code
    http_code=$(curl -o /dev/null -s -w "%{http_code}" -L "${wheel_url}" 2>/dev/null || echo "000")

    if [[ "${http_code}" != "200" ]]; then
        warn "cu130 wheel not found (HTTP ${http_code}) — trying nightly index…"
        uv pip install vllm \
            --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
            --index-strategy unsafe-best-match \
        || return 1
    else
        uv pip install "${wheel_url}" \
            --extra-index-url https://download.pytorch.org/whl/cu130 \
            --index-strategy unsafe-best-match
    fi
    success "vLLM installed from pre-compiled wheel."
}

# ── vLLM (source build fallback) ──────────────────────────────────────────────
build_vllm_from_source() {
    warn "Source build — this takes 60–90 minutes on Thor."
    local nprocs; nprocs=$(nproc)
    sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true
    _resolve_vllm_version

    cd "${BUILD_PATH}" || die "Cannot cd to BUILD_PATH: ${BUILD_PATH}"
    [[ -d vllm ]] && rm -rf vllm
    git clone --recursive --depth=1 --branch "v${VLLM_VERSION}" \
        https://github.com/vllm-project/vllm.git
    cd vllm || die "Cannot cd into vllm source directory"

    python3 use_existing_torch.py 2>/dev/null || warn "use_existing_torch.py not present — skipping"
    uv pip install -r requirements/build.txt -r requirements/common.txt
    python3 -m setuptools_scm

    export SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}"
    export MAX_JOBS=${nprocs}
    export NVCC_THREADS=1
    export CUDA_NVCC_FLAGS="-Xcudafe --threads=1"
    export MAKEFLAGS='-j2'
    export CMAKE_BUILD_PARALLEL_LEVEL=${nprocs}
    export NINJAFLAGS='-j2'
    export CMAKE_ARGS="-DCAFFE2_USE_CUDNN=ON -DUSE_CUDNN=ON -DUSE_CUSPARSELT=ON -DUSE_CUDSS=ON -DUSE_CUFILE=ON"

    uv build --wheel --no-build-isolation -v --out-dir ./dist .
    uv pip install dist/vllm*.whl -v
    success "vLLM built from source."
}

# ── Dependencies ──────────────────────────────────────────────────────────────
install_deps() {
    step "Dependencies"

    # Probe vLLM's pinned flashinfer version
    local pinned_fi
    pinned_fi=$(python -c "
import importlib.metadata, re
try:
    req = [r for r in importlib.metadata.requires('vllm') or []
           if 'flashinfer' in r and 'extra' not in r][0]
    m = re.search(r'==([0-9.]+)', req)
    print(m.group(1) if m else '')
except Exception:
    print('')
" 2>/dev/null || echo "")

    if [[ -z "${FLASHINFER_VERSION}" ]]; then
        if [[ -n "${pinned_fi}" ]]; then
            FLASHINFER_VERSION="${pinned_fi}"
            info "Using vLLM-pinned flashinfer: ${FLASHINFER_VERSION}"
        else
            FLASHINFER_VERSION=$(curl -s https://flashinfer.ai/whl/cu130/ 2>/dev/null \
                | grep -oP 'flashinfer_python-\K[0-9]+\.[0-9]+\.[0-9]+' \
                | sort -V | tail -1 || echo "")
        fi
    fi

    # Pin apache-tvm-ffi — flashinfer pulls this in and >=0.1.8 breaks the import
    uv pip install "apache-tvm-ffi<=0.1.7" --prerelease=allow

    info "Installing triton, xgrammar, compressed-tensors…"
    uv pip install triton xgrammar compressed-tensors

    info "Installing flashinfer (${FLASHINFER_VERSION:-latest})…"
    if [[ -n "${FLASHINFER_VERSION}" ]]; then
        uv pip install \
            "flashinfer-python==${FLASHINFER_VERSION}" \
            "flashinfer-cubin==${FLASHINFER_VERSION}" \
            --extra-index-url https://flashinfer.ai/whl/cu130
        uv pip install "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
            --index-url https://flashinfer.ai/whl/cu130
    else
        uv pip install flashinfer-python flashinfer-cubin \
            --extra-index-url https://flashinfer.ai/whl/cu130 --prerelease=allow
        uv pip install flashinfer-jit-cache \
            --index-url https://flashinfer.ai/whl/cu130 --prerelease=allow
    fi

    uv pip install "numba==${NUMBA_VERSION}" nvidia-cutlass-dsl
    success "Dependencies installed."
}

# ── Cache dirs ────────────────────────────────────────────────────────────────
create_cache_dirs() {
    info "Creating cache directories under ${CACHE_ROOT}…"
    mkdir -p \
        "${CACHE_ROOT}" \
        "${CACHE_ROOT}/assets" \
        "${CACHE_ROOT}/kernels" \
        "${CACHE_ROOT}/flashinfer" \
        "${CACHE_ROOT}/triton" \
        "${CACHE_ROOT}/inductor" \
        "${CACHE_ROOT}/torch_compile" \
        "${CACHE_ROOT}/xdg" \
        "${CACHE_ROOT}/tiktoken" \
        "${CACHE_ROOT}/huggingface"
}

# ── NVFP4 patches ────────────────────────────────────────────────────────────
apply_nvfp4_patches() {
    step "NVFP4 patches (SM110 + layernorm)"

    local patch_dir="${LIB_DIR}/../patches"
    local venv_python="${VENV_PATH}/bin/python"

    if [[ ! -f "${patch_dir}/patch_sm110.py" ]]; then
        warn "Patch scripts not found at ${patch_dir} — skipping NVFP4 patches."
        warn "Clone the full thorllm repo to get patches/."
        return
    fi

    info "Applying SM110 capability family patch (enables CUTLASS NVFP4 on Thor)…"
    "${venv_python}" "${patch_dir}/patch_sm110.py"         && success "SM110 patch applied."         || warn "SM110 patch failed — NVFP4 will fall back to Marlin (slower)."

    info "Applying RMSNormGated layernorm patch…"
    "${venv_python}" "${patch_dir}/patch_layernorm.py"         && success "Layernorm patch applied."         || warn "Layernorm patch failed — NVFP4 gated-activation models may fail to load."
}

# ── Tiktoken encoding pre-fetch ───────────────────────────────────────────────
# Pre-download the tiktoken encoding file so the first vllm serve doesn't
# make a network call. tiktoken-rs expects the file named by its content hash.
prefetch_tiktoken() {
    local cache_dir="${CACHE_ROOT}/tiktoken"
    local url="https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    local hash_name="fb374d419588a4632f3f557e76b4b70aebbca790"
    local dest="${cache_dir}/${hash_name}"

    mkdir -p "${cache_dir}"
    if [[ -f "${dest}" ]]; then
        success "tiktoken encoding already cached — skipping."
        return
    fi

    info "Pre-fetching tiktoken o200k_base encoding…"
    wget --quiet --show-progress "${url}" -O "${dest}"         && success "tiktoken encoding cached: ${dest}"         || warn "tiktoken pre-fetch failed — will download on first serve."
}

# ── Main install orchestrator ─────────────────────────────────────────────────
run_install() {
    local update="${1:-0}"

    step "vLLM installer for Jetson Thor"
    sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true

    preflight
    create_cache_dirs
    install_cudss
    install_nvpl
    install_uv
    create_venv
    set_cuda_env
    install_pytorch

    if [[ "${update}" == "1" ]]; then
        VLLM_VERSION=""  # Force re-resolve to latest
    fi

    info "Attempting pre-compiled wheel (fast path)…"
    if install_vllm_wheel; then
        success "Pre-compiled wheel installed."
    else
        warn "Wheel unavailable — falling back to source build."
        build_vllm_from_source
    fi

    install_deps

    source "${LIB_DIR}/validate.sh"
    run_validate

    # Apply NVFP4 patches if enabled (default: yes)
    if [[ "${NVFP4_ENABLE:-1}" == "1" ]]; then
        apply_nvfp4_patches
    fi

    prefetch_tiktoken

    source "${LIB_DIR}/service.sh"
    write_env_file
    write_model_config "${SERVE_MODEL}"
    write_service_files
    write_activation_script

    config_save

    step "Installation complete"
    echo ""
    echo "  Activate:    source ${BUILD_PATH}/activate_vllm.sh"
    echo "  Start:       thorllm start"
    echo "  Watch logs:  thorllm logs -f"
    echo ""
}
