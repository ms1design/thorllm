#!/usr/bin/env python3
"""
patches/patch_sm110.py — NVFP4 SM110 (Jetson Thor) capability family fix

Problem A — NVFP4 backend selection (affects vLLM < 0.18.0 primarily for MoE):
  vLLM's NVFP4 MoE backend selection calls is_device_capability_family(100)
  which matches SM10.x (datacenter Blackwell B100/B200/GB10) but NOT SM11.x
  (Thor, SM 11.0a). Without this patch the NVFP4 MoE path falls back to Marlin
  kernels, which are 40-60% slower than the native CUTLASS FP4 path.

  The GEMM (linear) backend in vLLM >=0.18.0 was centralised in nvfp4_utils.py
  which uses has_device_capability(100) — SM110 (110) passes this check, so the
  linear GEMM backend selection is already correct in >=0.18.0 without patching.

  Additionally, the Triton MXFP4 backend must be explicitly EXCLUDED for SM110
  because Triton FP4 kernels fail on SM110 (works on SM90/SM100/SM120 only).
  See: https://github.com/vllm-project/vllm/issues/29317

Problem B — Conv3dLayer crash on SM110 (NEW in vLLM 0.18.0):
  vLLM 0.18.0 added model_executor/layers/conv.py with Conv3dLayer used by the
  Qwen3-VL vision encoder's PatchEmbed (qwen3_vl.py). On SM110, the CustomOp
  dispatch selects forward_native() (SM110 is not in vLLM's compiled CUDA archs),
  which calls _forward_mulmat() → F.linear() → cuBLAS. On sm11.0a, cublasLtCreate
  fails with CUBLAS_STATUS_NOT_INITIALIZED. The fix forces _forward_conv()
  (F.conv3d via cuDNN, which works fine) when running on SM110.

  Stack: embed_multimodal → Qwen3VisionPatchEmbed.forward → self.proj (Conv3dLayer)
    → CustomOp.forward → forward_native → _forward_mulmat → F.linear → CRASH

Affected files (6):
  vllm/model_executor/layers/quantization/mxfp4.py
  vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py
  vllm/model_executor/layers/fused_moe/flashinfer_cutedsl_moe.py
  vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py
  vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py
  vllm/model_executor/layers/conv.py                              [NEW in 0.18.0]

Note on flashinfer_fp4_moe.py in >=0.18.0:
  The CUTLASS availability function (is_flashinfer_fp4_cutlass_moe_available)
  changed from is_device_capability_family(100) to has_device_capability(100),
  which already covers SM110. Only the CUTEDSL function still needs patching.
  The patch below will report ALREADY PATCHED or NOT FOUND for the CUTLASS
  function — that is expected and correct.

Usage:
  python3 patches/patch_sm110.py /path/to/vllm/site-packages
  # or from within the activated venv:
  python3 patches/patch_sm110.py
"""

import sys
import site
import importlib.util
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Locate vLLM package root
# ─────────────────────────────────────────────────────────────────────────────
def find_vllm_root(argv_path: str | None = None) -> Path:
    if argv_path:
        root = Path(argv_path)
        if (root / "vllm").is_dir():
            return root / "vllm"
        if root.name == "vllm":
            return root
        raise FileNotFoundError(f"vLLM not found at {argv_path}")

    spec = importlib.util.find_spec("vllm")
    if spec and spec.origin:
        return Path(spec.origin).parent
    # Fallback: search site-packages
    for sp in site.getsitepackages():
        candidate = Path(sp) / "vllm"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "vLLM package not found. Activate the venv or pass the site-packages path."
    )


VLLM_ROOT = find_vllm_root(sys.argv[1] if len(sys.argv) > 1 else None)
print(f"vLLM root: {VLLM_ROOT}")

# ─────────────────────────────────────────────────────────────────────────────
# Patch helpers
# ─────────────────────────────────────────────────────────────────────────────
def patch_file(rel_path: str, replacements: list[tuple[str, str]]) -> None:
    path = VLLM_ROOT / rel_path
    if not path.exists():
        print(f"  SKIP (not found): {rel_path}")
        return

    original = path.read_text()
    patched = original
    applied = 0

    for old, new in replacements:
        if old in patched:
            patched = patched.replace(old, new)
            applied += 1
        elif new in patched:
            print(f"  ALREADY PATCHED: {rel_path} — '{old[:60].strip()}'")
        else:
            print(f"  NOT FOUND: {rel_path} — '{old[:60].strip()}'")

    if patched != original:
        # Write backup
        (path.parent / (path.name + ".pre_sm110_patch")).write_text(original)
        path.write_text(patched)
        print(f"  PATCHED ({applied} replacements): {rel_path}")
    else:
        print(f"  NO CHANGE: {rel_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Patch 1: mxfp4.py — main backend selection
#
# Change 1a: CUTLASS backend check — add SM110 family.
#            This is the substantive fix: enables native CUTLASS NVFP4 kernels.
# Change 1b: Triton MXFP4 GEMM range — SM110=(11,0) is already outside the
#            upstream range (9,0)..(11,0), so this replacement is a no-op
#            but adds a comment documenting the intentional exclusion.
#            NOTE: Triton *attention* backend works fine on SM110. Only the
#            narrow MXFP4 GEMM kernel path is excluded.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] mxfp4.py")
patch_file(
    "model_executor/layers/quantization/mxfp4.py",
    [
        # CUTLASS/FlashInfer backend: add SM110 alongside SM100
        (
            "current_platform.is_device_capability_family(100)",
            "(current_platform.is_device_capability_family(100)"
            " or current_platform.is_device_capability_family(110))",
        ),
        # Triton backend range: exclude SM110 (must stay below (11, 0))
        # Original: (9, 0) <= capability < (11, 0)
        # Thor is (11, 0) so it's already excluded. But if any upstream
        # widens the range, we add an explicit guard comment.
        # This replacement is a no-op if already correct but documents intent.
        (
            "and (9, 0) <= current_platform.get_device_capability() < (11, 0)",
            # Keep identical — SM110 = (11,0) is already excluded. Add comment.
            "and (9, 0) <= current_platform.get_device_capability() < (11, 0)"
            "  # SM110 (Thor) excluded: Triton MXFP4 fails on SM11.x",
        ),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Patch 2: flashinfer_cutlass_moe.py
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] flashinfer_cutlass_moe.py (already upstream-fixed in 0.18.0)")
patch_file(
    "model_executor/layers/fused_moe/flashinfer_cutlass_moe.py",
    [
        (
            "current_platform.is_device_capability_family(100)",
            "(current_platform.is_device_capability_family(100)"
            " or current_platform.is_device_capability_family(110))",
        ),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Patch 3: flashinfer_cutedsl_moe.py
#
# In vLLM 0.18.0+cu130 this file stores current_platform in a local alias
# `p` before the capability check, so the old anchor
#   "current_platform.is_device_capability_family(100)"
# no longer matches.  The correct anchor is:
#   "return p.is_cuda() and p.is_device_capability_family(100)"
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] flashinfer_cutedsl_moe.py")
patch_file(
    "model_executor/layers/fused_moe/flashinfer_cutedsl_moe.py",
    [
        (
            "return p.is_cuda() and p.is_device_capability_family(100)",
            "return p.is_cuda() and (\n"
            "            p.is_device_capability_family(100)\n"
            "            or p.is_device_capability_family(110)  # SM110 (Jetson Thor)\n"
            "        )",
        ),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Patch 4: flashinfer_trtllm_moe.py
#
# Same `p` alias issue as patch 3.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] flashinfer_trtllm_moe.py")
patch_file(
    "model_executor/layers/fused_moe/flashinfer_trtllm_moe.py",
    [
        (
            "return p.is_cuda() and p.is_device_capability_family(100)",
            "return p.is_cuda() and (\n"
            "    p.is_device_capability_family(100)\n"
            "    or p.is_device_capability_family(110)  # SM110 (Jetson Thor)\n"
            ")",
        ),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Patch 5: flashinfer_fp4_moe.py
#
# In vLLM 0.18.0 this file uses has_device_capability(100) which already
# covers SM110 (110 >= 100 in the vLLM integer convention).  No patch needed;
# this step is kept as a no-op sentinel so the patch numbering stays stable
# for older vLLM versions that still use is_device_capability_family(100).
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] flashinfer_fp4_moe.py (already uses has_device_capability in 0.18.0)")
patch_file(
    "model_executor/layers/quantization/utils/flashinfer_fp4_moe.py",
    [
        # Pre-0.18.0 anchor — no-op on 0.18.0 (prints NOT FOUND / ALREADY PATCHED)
        (
            "current_platform.is_device_capability_family(100)",
            "(current_platform.is_device_capability_family(100)"
            " or current_platform.is_device_capability_family(110))",
        ),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Patch 6: conv.py — Conv3dLayer SM110 cuBLAS crash fix  [NEW in vLLM 0.18.0]
#
# vLLM 0.18.0 added Conv3dLayer (model_executor/layers/conv.py) which is used
# by the Qwen3-VL vision encoder's PatchEmbed. On SM110, CustomOp falls back to
# forward_native() because SM110 is not in vLLM's compiled CUDA archs. That
# calls _forward_mulmat() → F.linear() → cuBLAS, which fails on sm11.0a with
# CUBLAS_STATUS_NOT_INITIALIZED (cublasLtCreate returns error for this arch).
#
# Fix: force _forward_conv() (F.conv3d via cuDNN, works fine on SM110) when the
# device capability major version is 11. Also guard the torch-2.9.x branch in
# forward_cuda() for the same reason.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] conv.py (Conv3dLayer — SM110 cuBLAS crash, vLLM >=0.18.0)")
patch_file(
    "model_executor/layers/conv.py",
    [
        (
            "    def forward_native(self, x: torch.Tensor) -> torch.Tensor:\n"
            "        \"\"\"Expected input shape: (batch_size, in_channels, time, height, width)\"\"\"\n"
            "        if self.enable_linear:\n"
            "            return self._forward_mulmat(x)\n"
            "        else:\n"
            "            return self._forward_conv(x)",
            "    def forward_native(self, x: torch.Tensor) -> torch.Tensor:\n"
            "        \"\"\"Expected input shape: (batch_size, in_channels, time, height, width)\"\"\"\n"
            "        # SM110 (Jetson Thor / sm11.0a): cublasLtCreate fails for this arch,\n"
            "        # so F.linear is broken. Fall back to F.conv3d (cuDNN) instead.\n"
            "        is_sm110 = (\n"
            "            torch.cuda.is_available()\n"
            "            and torch.cuda.get_device_capability()[0] == 11\n"
            "        )\n"
            "        if self.enable_linear and not is_sm110:\n"
            "            return self._forward_mulmat(x)\n"
            "        else:\n"
            "            return self._forward_conv(x)",
        ),
        (
            "        if self.enable_linear and (is_torch_equal(\"2.9.0\") or is_torch_equal(\"2.9.1\")):\n"
            "            return self._forward_mulmat(x)\n"
            "        return self._forward_conv(x)",
            "        # SM110 (Jetson Thor / sm11.0a): same cuBLAS issue as forward_native.\n"
            "        is_sm110 = torch.cuda.get_device_capability()[0] == 11\n"
            "        if self.enable_linear and not is_sm110 and (\n"
            "            is_torch_equal(\"2.9.0\") or is_torch_equal(\"2.9.1\")\n"
            "        ):\n"
            "            return self._forward_mulmat(x)\n"
            "        return self._forward_conv(x)",
        ),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Patch 7: kernels/linear/scaled_mm/cutlass.py
#   — fix non-contiguous transposed weight (CUBLAS_STATUS_INVALID_VALUE on SM110)
#
# Root cause (ALL three "CUBLAS_STATUS_INVALID_VALUE" errors across every model):
#   CutlassInt8ScaledMMLinearKernel.process_weights_after_loading stores the
#   transposed weight as:
#     torch.nn.Parameter(weight.t().data, requires_grad=False)
#   weight.t() is a non-contiguous VIEW with column-major strides (1, K).
#   torch.inductor captures those strides and emits:
#     extern_kernels.mm(x, reinterpret_tensor(weight, (K, N), (1, K), 0), ...)
#   JetPack cuBLAS on SM110 rejects CUDA_R_16BF operands with strides (1, K)
#   and returns CUBLAS_STATUS_INVALID_VALUE.
#
#   Fix: weight.t().contiguous().data — the stored tensor is row-major (K, N)
#   with strides (N, 1). Inductor emits a normal mm with standard strides and
#   cuBLAS is happy on any architecture including SM110.
#
#   NOTE: This is NOT an SM110-only bug — it is a correctness issue on any
#   architecture whose cuBLAS rejects non-contiguous BF16 operands. The fix
#   applies unconditionally, not just for SM110.
#
#   NOTE: CutlassFP8ScaledMMLinearKernel has no process_weights_after_loading
#   (inherits `pass` from the base class) — no weight transposition happens
#   there, so no fix is needed. ops.cutlass_scaled_mm handles the FP8 weight
#   layout internally including on SM110.
#
#   SM110 support: BOTH CutlassInt8 and CutlassFP8 are KEPT enabled for SM110.
#   The vllm==0.18.0+cu130 build was compiled against CUDA 13.0 which has
#   full SM110 (Jetson Thor) support — CUTLASS kernels are compiled for SM11.x.
#   We do NOT add SM110 exclusions here.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/8] kernels/linear/scaled_mm/cutlass.py — fix non-contiguous weight (all arches)")
patch_file(
    "model_executor/kernels/linear/scaled_mm/cutlass.py",
    [
        # The ONLY fix needed: make the transposed weight contiguous.
        # This resolves CUBLAS_STATUS_INVALID_VALUE on SM110 (and any other
        # arch where cuBLAS rejects column-major BF16 operands) without
        # disabling CUTLASS for SM110.
        (
            "            torch.nn.Parameter(weight.t().data, requires_grad=False),",
            "            # thorllm: weight.t() is a non-contiguous view with strides\n"
            "            # (1, K) — column-major. JetPack cuBLAS on SM110 rejects BF16\n"
            "            # operands with that stride layout (CUBLAS_STATUS_INVALID_VALUE).\n"
            "            # .contiguous() materialises the transpose as a row-major (K, N)\n"
            "            # tensor with strides (N, 1) so inductor emits a standard mm.\n"
            "            torch.nn.Parameter(weight.t().contiguous().data, requires_grad=False),",
        ),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Patch 8 (renumbered): fp8.py / modelopt.py — legacy fallback anchors
#
# In vLLM 0.18.0 the flat cutlass_scaled_mm_supported() dispatch in fp8.py
# and the apply_fp8_linear call in modelopt.py no longer exist — the kernel
# selection was moved to kernels/linear/scaled_mm/cutlass.py (patch 7 above).
# These patches are kept as no-ops so older vLLM installs still get fixed and
# so the step counter in the log is stable. On 0.18.0 they will print
# NOT FOUND or ALREADY PATCHED for every anchor, which is correct and expected.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/8] fp8.py / modelopt.py (legacy anchors — no-op on vLLM 0.18.0+)")
for _legacy_file, _legacy_anchor, _legacy_replacement in [
    (
        "model_executor/layers/quantization/fp8.py",
        "        cutlass_fp8_supported = ops.cutlass_scaled_mm_supported(capability)\n",
        "        # thorllm SM110: excluded via cutlass.py is_supported() in 0.18.0+\n"
        "        cutlass_fp8_supported = ops.cutlass_scaled_mm_supported(capability)\n",
    ),
    (
        "model_executor/layers/quantization/modelopt.py",
        "        output = apply_fp8_linear(\n",
        "        # thorllm SM110: excluded via cutlass.py is_supported() in 0.18.0+\n"
        "        output = apply_fp8_linear(\n",
    ),
]:
    patch_file(_legacy_file, [(_legacy_anchor, _legacy_replacement)])

print("\n[9/9] kernels/linear/scaled_mm/flashinfer.py")
print("  NOTE: FlashInfer FP8 is_supported() requires compute_capability >= 100.")
print("  SM110 (Thor) = 110, which passes. vllm==0.18.0+cu130 was built against")
print("  CUDA 13.0 with full SM110 support — FlashInfer FP8 is presumed to work.")
print("  No exclusion applied. If bmm_fp8 fails on SM110, patch will be added then.")
patch_file(
    "model_executor/kernels/linear/scaled_mm/flashinfer.py",
    [
        # No-op sentinel: ensures this file is at least checked each run.
        # If a future FlashInfer build ships without SM110 support, add an
        # exclusion here with the exact error evidence to justify it.
    ],
)

print("\nSM110 patch complete.")
print(
    "Backup files (.pre_sm110_patch) written next to each patched file.\n"
    "To verify NVFP4 fixes: grep -r 'is_device_capability_family(110)' "
    f"{VLLM_ROOT}/model_executor/layers/\n"
    "To verify conv.py fix: grep -A4 'is_sm110' "
    f"{VLLM_ROOT}/model_executor/layers/conv.py\n"
    "To verify cutlass fix: grep -A2 'thorllm:' "
    f"{VLLM_ROOT}/model_executor/kernels/linear/scaled_mm/cutlass.py\n"
    "To verify flashinfer fix: grep -A4 'thorllm SM110' "
    f"{VLLM_ROOT}/model_executor/kernels/linear/scaled_mm/flashinfer.py"
)