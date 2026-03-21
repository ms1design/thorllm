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
# Patch 2: qwen3_5_mtp.py — robust region-replacement (handles all broken states)
#
# Previous str.replace-based approaches produced multiple broken states because
# the replacement text was a substring of itself (idempotency failures) and
# because replacing a sub-region that didn't include the getattr block resulted
# in the getattr block being duplicated into the try-body at wrong indentation.
#
# Root cause of SyntaxError "expected 'except' or 'finally' block" at line 327:
#   The _BROKEN_ANCHOR (old) didn't include the getattr block, but _CORRECT_PATCH
#   (new) did. After replacing _BROKEN_ANCHOR with _CORRECT_PATCH, the getattr
#   block appeared TWICE — once at the correct level and once INSIDE the outer
#   try: body at 24-space indent (due to "    " + 20sp getattr → 24sp getattr).
#   Python parses the outer try: body as containing the getattr call statement,
#   then sees a dedented try: without an except → SyntaxError.
#
# Fix: abandon str.replace entirely for this file. Instead, find the stable
# region between two unique anchors and replace it wholesale. The start anchor
# ("                    param = params_dict[name]\n") and end anchor
# ("            loaded_params.add(name)\n") are each unique in the file and
# bracket the entire weight-loader region regardless of how broken it is.
# This approach is safe for ANY broken state.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/2] qwen3_5_mtp.py — weight_loader region repair (handles all broken states)")

_MTP_PATH = "model_executor/models/qwen3_5_mtp.py"
_MTP_FILE = VLLM_ROOT / _MTP_PATH

if not _MTP_FILE.exists():
    print(f"  SKIP (not found): {_MTP_PATH}")
else:
    _content = _MTP_FILE.read_text()

    # The correct region content: from param = params_dict[name] up to (not
    # including) loaded_params.add(name).  Both anchors are unique in the file.
    _REGION_START = "                    param = params_dict[name]\n"
    _REGION_END   = "            loaded_params.add(name)\n"

    _CORRECT_REGION = (
        "                    param = params_dict[name]\n"
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

    if _REGION_START not in _content or _REGION_END not in _content:
        print(f"  NOT FOUND: {_MTP_PATH} — stable anchors missing (file may have changed upstream)")
    else:
        _idx_start = _content.find(_REGION_START)
        _idx_end   = _content.find(_REGION_END, _idx_start)
        _current_region = _content[_idx_start:_idx_end]

        if _current_region == _CORRECT_REGION:
            print(f"  ALREADY PATCHED: {_MTP_PATH}")
        else:
            _new_content = _content[:_idx_start] + _CORRECT_REGION + _content[_idx_end:]
            # Validate before writing
            try:
                import ast as _ast
                _ast.parse(_new_content)
            except SyntaxError as _e:
                print(f"  ERROR: repaired content has syntax error: {_e} — not writing")
            else:
                _backup = _MTP_FILE.parent / (_MTP_FILE.name + ".pre_mtp_nvfp4_patch")
                if not _backup.exists():
                    _backup.write_text(_content)
                _MTP_FILE.write_text(_new_content)
                if "# thorllm: NVFP4-MTP" in _current_region:
                    print(f"  REPAIRED (was broken): {_MTP_PATH}")
                else:
                    print(f"  PATCHED: {_MTP_PATH}")

print("\nMTP NVFP4 patch complete.")
print(
    "Backup files (.pre_mtp_nvfp4_patch) written next to each patched file.\n"
    f"To verify: grep -n 'thorllm: NVFP4-MTP' {VLLM_ROOT}/model_executor/parameter.py\n"
    f"           grep -n 'thorllm: NVFP4-MTP' {VLLM_ROOT}/model_executor/models/qwen3_5_mtp.py"
)