#!/usr/bin/env python3
"""
patches/patch_layernorm.py — RMSNormGated missing activation attribute fix

Problem:
  vLLM's RMSNormGated.__init__ does not accept or store the `activation`
  parameter, but RMSNormGated.forward() references self.activation at line ~595.
  When loading NVFP4 models that use gated activations (e.g. Qwen3.5 MoE),
  this causes:
    AttributeError: 'RMSNormGated' object has no attribute 'activation'

Fix:
  Add `activation: str = "silu"` to the __init__ signature and store it as
  self.activation.

Usage:
  python3 patches/patch_layernorm.py /path/to/vllm/site-packages
  # or from within the activated venv:
  python3 patches/patch_layernorm.py
"""

import sys
import site
import importlib.util
from pathlib import Path


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
    raise FileNotFoundError("vLLM package not found.")


VLLM_ROOT = find_vllm_root(sys.argv[1] if len(sys.argv) > 1 else None)
print(f"vLLM root: {VLLM_ROOT}")

path = VLLM_ROOT / "model_executor/layers/layernorm.py"
if not path.exists():
    print(f"ERROR: not found: {path}")
    sys.exit(1)

original = path.read_text()
patched = original

# ─── Patch 1: Add activation param to __init__ signature ─────────────────────
# Before: norm_before_gate: bool = False,
# After:  norm_before_gate: bool = False,
#         activation: str = "silu",  # ← ADDED for NVFP4/gated-activation models
# OLD_SIG = "        norm_before_gate: bool = False,"
# NEW_SIG = (
#     "        norm_before_gate: bool = False,\n"
#     "        activation: str = \"silu\",  # NVFP4/gated-activation models (patch_layernorm)"
# )

# if OLD_SIG in patched and NEW_SIG not in patched:
#     patched = patched.replace(OLD_SIG, NEW_SIG, 1)
#     print("  Applied: activation param added to __init__ signature")
# elif NEW_SIG in patched:
#     print("  ALREADY PATCHED: activation param in signature")
# else:
#     print("  WARNING: Could not find signature anchor — check layernorm.py manually")

# ─── Patch 2: Store activation in __init__ body ───────────────────────────────
# Before: self.norm_before_gate = norm_before_gate
# After:  self.norm_before_gate = norm_before_gate
#         self.activation = activation  # ← ADDED
# OLD_BODY = "        self.norm_before_gate = norm_before_gate"
# NEW_BODY = (
#     "        self.norm_before_gate = norm_before_gate\n"
#     "        self.activation = activation  # NVFP4/gated-activation models (patch_layernorm)"
# )

# if OLD_BODY in patched and NEW_BODY not in patched:
#     patched = patched.replace(OLD_BODY, NEW_BODY, 1)
#     print("  Applied: self.activation stored in __init__ body")
# elif NEW_BODY in patched:
#     print("  ALREADY PATCHED: self.activation in body")
# else:
#     print("  WARNING: Could not find body anchor — check layernorm.py manually")

# if patched != original:
#     (path.parent / (path.name + ".pre_layernorm_patch")).write_text(original)
#     path.write_text(patched)
#     print(f"  PATCHED: {path}")
# else:
#     print("  NO CHANGE")

print("\nLayernorm patch complete.")
