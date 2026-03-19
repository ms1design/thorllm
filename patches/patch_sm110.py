#!/usr/bin/env python3
"""
patches/patch_sm110.py — NVFP4 SM110 (Jetson Thor) capability family fix

Problem:
  vLLM's NVFP4 backend selection calls is_device_capability_family(100) which
  matches SM10.x (datacenter Blackwell B100/B200/GB10) but NOT SM11.x (Thor,
  SM 11.0a). Without this patch the NVFP4 path falls back to Marlin kernels,
  which are 40-60% slower than the native CUTLASS FP4 path.

  Additionally, the Triton MXFP4 backend must be explicitly EXCLUDED for SM110
  because Triton FP4 kernels fail on SM110 (works on SM90/SM100/SM120 only).

Affected files (5):
  vllm/model_executor/layers/quantization/mxfp4.py
  vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py
  vllm/model_executor/layers/fused_moe/flashinfer_cutedsl_moe.py
  vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py
  vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py

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
print("\n[1/5] mxfp4.py")
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
print("\n[2/5] flashinfer_cutlass_moe.py")
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
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] flashinfer_cutedsl_moe.py")
patch_file(
    "model_executor/layers/fused_moe/flashinfer_cutedsl_moe.py",
    [
        (
            "current_platform.is_device_capability_family(100)",
            "(current_platform.is_device_capability_family(100)"
            " or current_platform.is_device_capability_family(110))",
        ),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Patch 4: flashinfer_trtllm_moe.py
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] flashinfer_trtllm_moe.py")
patch_file(
    "model_executor/layers/fused_moe/flashinfer_trtllm_moe.py",
    [
        (
            "current_platform.is_device_capability_family(100)",
            "(current_platform.is_device_capability_family(100)"
            " or current_platform.is_device_capability_family(110))",
        ),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Patch 5: flashinfer_fp4_moe.py
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] flashinfer_fp4_moe.py")
patch_file(
    "model_executor/layers/quantization/utils/flashinfer_fp4_moe.py",
    [
        (
            "current_platform.is_device_capability_family(100)",
            "(current_platform.is_device_capability_family(100)"
            " or current_platform.is_device_capability_family(110))",
        ),
    ],
)

print("\nSM110 patch complete.")
print(
    "Backup files (.pre_sm110_patch) written next to each patched file.\n"
    "To verify: grep -r 'is_device_capability_family(110)' "
    f"{VLLM_ROOT}/model_executor/layers/"
)
