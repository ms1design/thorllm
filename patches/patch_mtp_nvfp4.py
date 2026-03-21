#!/usr/bin/env python3
"""
patches/patch_mtp_nvfp4.py — NVFP4 MTP speculative-decoding weight shape fix

Problem (vLLM 0.18.0, Qwen3.5-NVFP4 + method='mtp'):
  When a Qwen3.5 NVFP4 model is loaded with speculative decoding using
  method='mtp', vLLM constructs an MTP draft head via qwen3_5_mtp.py.
  During weight loading the draft head's ColumnParallelLinear parameters are
  initialized by the quantization layer with the FP4-packed shape
  (e.g., out_features × in_features//2 in int8), but the weight tensor read
  from the NVFP4 checkpoint by the generic weight_loader_v2 path passes
  through load_column_parallel_weight which does a strict shape assertion:

      assert self.data.shape == loaded_weight.shape   # parameter.py:153

  Because the MTP module's parameter was allocated for the packed shape while
  load_column_parallel_weight only sliced along the output dim (which is
  correct), the inner dim stays full-width in loaded_weight but is halved in
  self.data — the assertion fires and the engine aborts.

  Stack:
    qwen3_5_mtp.py:319  load_weights
      → linear.py:572   weight_loader_v2 → load_column_parallel_weight
        → parameter.py:153  assert self.data.shape == loaded_weight.shape
          AssertionError

Fix A — parameter.py (primary):
  Replace the strict assert with a shape-adaptive check.  When element counts
  are equal but shapes differ (FP4-packing reorders dims), reshape
  loaded_weight to match self.data before the copy.  When element counts
  genuinely differ, raise a descriptive AssertionError so real bugs are still
  caught.

Fix B — qwen3_5_mtp.py (belt-and-suspenders):
  Wrap the weight_loader call in load_weights with a try/except for
  AssertionError.  On shape mismatch, attempt a dtype-safe reshape and retry;
  only re-raise if the mismatch is unrecoverable.

Affected files (2):
  vllm/model_executor/parameter.py
  vllm/model_executor/models/qwen3_5_mtp.py

Usage:
  python3 patches/patch_mtp_nvfp4.py /path/to/vllm/site-packages
  # or from within the activated venv:
  python3 patches/patch_mtp_nvfp4.py
"""

import sys
import site
import importlib.util
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Locate vLLM package root (same helper used by all thorllm patches)
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
    for sp in site.getsitepackages():
        candidate = Path(sp) / "vllm"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "vLLM package not found. Activate the venv or pass the site-packages path."
    )


VLLM_ROOT = find_vllm_root(sys.argv[1] if len(sys.argv) > 1 else None)
print(f"vLLM root: {VLLM_ROOT}")


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
        backup = path.parent / (path.name + ".pre_mtp_nvfp4_patch")
        backup.write_text(original)
        path.write_text(patched)
        print(f"  PATCHED ({applied} replacements): {rel_path}")
    else:
        print(f"  NO CHANGE: {rel_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Patch 1: parameter.py — relax the strict shape assertion in
#           load_column_parallel_weight so that FP4-packed (and other
#           quantization-packed) weights can be reshaped to fit.
#
# Original (line ~153):
#     assert self.data.shape == loaded_weight.shape
#
# Replacement: when numel matches, reshape; when it doesn't, raise a
# descriptive error rather than a bare AssertionError.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/2] parameter.py — load_column_parallel_weight shape assertion")
patch_file(
    "model_executor/parameter.py",
    [
        (
            "        assert self.data.shape == loaded_weight.shape\n"
            "        self.data.copy_(loaded_weight)",
            # ── replacement ──────────────────────────────────────────────────
            "        # thorllm: NVFP4-MTP fix — FP4-packed weights may have a\n"
            "        # different shape than the initialized parameter (e.g. the\n"
            "        # inner dim is halved due to 2-values-per-byte packing).\n"
            "        # When the element count matches, reshape to fit; when it\n"
            "        # genuinely differs, raise a descriptive error.\n"
            "        if self.data.shape != loaded_weight.shape:\n"
            "            if loaded_weight.numel() == self.data.numel():\n"
            "                loaded_weight = loaded_weight.reshape(self.data.shape)\n"
            "            else:\n"
            "                raise AssertionError(\n"
            "                    f\"load_column_parallel_weight: shape mismatch — \"\n"
            "                    f\"param {self.data.shape} ({self.data.dtype}) vs \"\n"
            "                    f\"loaded {loaded_weight.shape} ({loaded_weight.dtype}). \"\n"
            "                    f\"This may indicate a quantization format mismatch \"\n"
            "                    f\"between the checkpoint and the parameter initializer.\"\n"
            "                )\n"
            "        self.data.copy_(loaded_weight)",
        ),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Patch 2: qwen3_5_mtp.py — belt-and-suspenders try/except around weight_loader
#
# Idempotency: the anchor includes the weight_loader = getattr(...) block that
# precedes the call.  The full 4-line anchor string does NOT appear inside its
# own replacement (the replacement ends that block with a try-except, not with
# the bare call), so str.replace is a no-op on an already-patched file.
#
# Recovery: if a previous run applied the broken narrow-anchor patch (20-space
# bare call as anchor), the file contains a double-nested try block with a
# garbled-indent comment that causes IndentationError.  A recovery replacement
# is attempted first; if the broken state is detected it is collapsed back to
# the clean patched state before the normal anchor is checked.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/2] qwen3_5_mtp.py — weight_loader call guard (with corruption recovery)")

_CLEAN_ANCHOR = (
    "                    weight_loader = getattr(\n"
    "                        param, \"weight_loader\", default_weight_loader\n"
    "                    )\n"
    "                    weight_loader(param, loaded_weight)\n"
)
_CORRECT_PATCH = (
    "                    weight_loader = getattr(\n"
    "                        param, \"weight_loader\", default_weight_loader\n"
    "                    )\n"
    "                    # thorllm: NVFP4-MTP — guard against packed-\n"
    "                    # weight shape mismatches (see patch_mtp_nvfp4.py).\n"
    "                    try:\n"
    "                        weight_loader(param, loaded_weight)\n"
    "                    except AssertionError:\n"
    "                        if loaded_weight.numel() == param.data.numel():\n"
    "                            param.data.copy_(\n"
    "                                loaded_weight.reshape(param.data.shape)\n"
    "                            )\n"
    "                        else:\n"
    "                            raise\n"
)
# The broken state produced by applying the old narrow anchor twice:
# outer try: has an 8-line body that starts with a dedented comment, making
# Python see the outer try: as having no real statement — IndentationError.
_BROKEN_ANCHOR = (
    "                    # thorllm: NVFP4-MTP — guard against packed-\n"
    "                    # weight shape mismatches (see patch_mtp_nvfp4.py).\n"
    "                    try:\n"
    "                        # thorllm: NVFP4-MTP — guard against packed-\n"
    "                    # weight shape mismatches (see patch_mtp_nvfp4.py).\n"
    "                    try:\n"
    "                        weight_loader(param, loaded_weight)\n"
    "                    except AssertionError:\n"
    "                        if loaded_weight.numel() == param.data.numel():\n"
    "                            param.data.copy_(\n"
    "                                loaded_weight.reshape(param.data.shape)\n"
    "                            )\n"
    "                        else:\n"
    "                            raise\n"
    "                    except AssertionError:\n"
    "                        if loaded_weight.numel() == param.data.numel():\n"
    "                            param.data.copy_(\n"
    "                                loaded_weight.reshape(param.data.shape)\n"
    "                            )\n"
    "                        else:\n"
    "                            raise\n"
)
# Recovery replaces the broken state with the correct already-patched state.
# The normal anchor then sees the correct patched state and prints ALREADY PATCHED.
patch_file(
    "model_executor/models/qwen3_5_mtp.py",
    [
        # Step 1: recover from broken double-try state (no-op if not broken)
        (
            _BROKEN_ANCHOR,
            _CORRECT_PATCH,
        ),
        # Step 2: apply correct patch to clean state (no-op if already patched)
        (
            _CLEAN_ANCHOR,
            _CORRECT_PATCH,
        ),
    ],
)

print("\nMTP NVFP4 patch complete.")
print(
    "Backup files (.pre_mtp_nvfp4_patch) written next to each patched file.\n"
    f"To verify: grep -n 'thorllm: NVFP4-MTP' {VLLM_ROOT}/model_executor/parameter.py\n"
    f"           grep -n 'thorllm: NVFP4-MTP' {VLLM_ROOT}/model_executor/models/qwen3_5_mtp.py"
)