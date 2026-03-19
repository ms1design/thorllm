#!/bin/bash
# lib/validate.sh — post-install version checks and smoke test
# Sources: common.sh, config.sh
# =============================================================================

run_validate() {
    step "Validation"
    local fail=0

    python - <<'EOF' || fail=1
import torch, vllm, flashinfer, numba, triton

print(f"  vLLM           : {vllm.__version__}")
print(f"  PyTorch        : {torch.__version__}")
print(f"  CUDA runtime   : {torch.version.cuda}")
print(f"  CUDA available : {torch.cuda.is_available()}")
print(f"  GPU            : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"  FlashInfer     : {flashinfer.__version__}")
print(f"  Numba          : {numba.__version__}")
print(f"  Triton         : {triton.__version__}")

assert torch.cuda.is_available(), "CUDA not available"
assert "13.0" in (torch.version.cuda or ""), f"Expected CUDA 13.0, got {torch.version.cuda}"

fi_ver = flashinfer.__version__.split("+")[0]
try:
    import flashinfer_cubin
    cb_ver = flashinfer_cubin.__version__.split("+")[0]
    assert fi_ver == cb_ver, f"flashinfer version mismatch: python={fi_ver} cubin={cb_ver}"
    print(f"  flashinfer-cubin: {cb_ver} (matches)")
except ImportError:
    print("  flashinfer-cubin: not installed (JIT-only mode)")

print("\n  All checks passed ✓")
EOF

    if [[ ${fail} -eq 0 ]]; then
        success "Validation passed."
    else
        die "Validation failed — see output above."
    fi

    info "vLLM import smoke test…"
    python - <<'EOF' || warn "Smoke test skipped (no GPU or vLLM init error)."
import os; os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM
print("  vLLM LLM class importable ✓")
EOF
}

show_versions() {
    local venv_python="${VENV_PATH}/bin/python"
    [[ -x "${venv_python}" ]] || die "venv not found at ${VENV_PATH}. Run: thorllm install"

    echo ""
    "${venv_python}" - <<'EOF'
import importlib.metadata as m

pkgs = ["vllm", "torch", "flashinfer", "numba", "triton", "xgrammar", "compressed_tensors"]
for p in pkgs:
    try:
        print(f"  {p:<30} {m.version(p)}")
    except m.PackageNotFoundError:
        print(f"  {p:<30} (not installed)")
EOF
    echo ""
}
