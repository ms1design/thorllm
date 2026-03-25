#!/usr/bin/env python3
"""
patches/patch_fa4_sm110.py — Enable FA4 (FlashAttention 4) on SM110 (Jetson Thor)

Problem (vLLM >= 0.18.0):
  fa_utils.py get_flash_attn_version() maps capability majors like this:
    major == 9  → FA3  (Hopper, SM90)
    major == 10 → FA4  (Blackwell, SM100)
    else        → FA2   ← SM110 (major=11) falls here!

  flash_attn_interface.py _is_fa4_supported() already explicitly includes
  is_device_capability_family(110), meaning FA4 kernels ARE compiled for SM110.
  The routing logic is simply missing the SM110 case.

Fix:
  Extend the major==10 branch to cover major==11 so Thor auto-selects FA4.
  FA3 is intentionally NOT added for SM110 — _is_fa3_supported() is gated to
  is_device_capability_family(90) (Hopper-only kernel ISA).

Affected file:
  vllm/v1/attention/backends/fa_utils.py

Usage:
  python3 patches/patch_fa4_sm110.py /path/to/vllm/site-packages
  # or from within the activated venv:
  python3 patches/patch_fa4_sm110.py
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
# Patch: fa_utils.py — extend FA4 routing to SM110
#
# get_flash_attn_version() in vLLM 0.18.0:
#
#   if device_capability.major == 9 and is_fa_version_supported(3):
#       fa_version = 3      # Hopper (SM90): prefer FA3
#   elif device_capability.major == 10 and is_fa_version_supported(4):
#       fa_version = 4      # Blackwell (SM100): prefer FA4
#   else:
#       fa_version = 2      # SM110 (Thor) falls here despite FA4 support!
#
# The fix: match major==11 alongside major==10 for FA4.
# ─────────────────────────────────────────────────────────────────────────────

FA_UTILS_REL = "v1/attention/backends/fa_utils.py"
FA_UTILS = VLLM_ROOT / FA_UTILS_REL

print(f"\n[1/1] {FA_UTILS_REL}")

if not FA_UTILS.exists():
    print(f"  SKIP (not found): {FA_UTILS_REL}")
    print("  NOTE: This file was added in vLLM 0.18.0. Older vLLM versions are unaffected.")
    sys.exit(0)

original = FA_UTILS.read_text()

# Anchor 1: original unpatched (vLLM 0.18.0)
OLD_1 = (
    "        elif device_capability.major == 10 and is_fa_version_supported(4):\n"
    "            # Blackwell (SM100+, restrict to SM100 for now): prefer FA4\n"
    "            fa_version = 4\n"
)
NEW_CORRECT = (
    "        elif (\n"
    "            device_capability.major in (10, 11)  # thorllm: SM100 Blackwell + SM110 Thor\n"
    "            and is_fa_version_supported(4)\n"
    "        ):\n"
    "            # Blackwell (SM100) and Jetson Thor (SM110): prefer FA4\n"
    "            # FA4 kernels include SM110 per _is_fa4_supported() in flash_attn_interface.py\n"
    "            fa_version = 4\n"
)

# Anchor 2: idempotency check — already patched
ALREADY_PATCHED_MARKER = (
    "device_capability.major in (10, 11)  # thorllm: SM100 Blackwell + SM110 Thor"
)

patched = original

if ALREADY_PATCHED_MARKER in patched:
    print(f"  ALREADY PATCHED: {FA_UTILS_REL}")
elif OLD_1 in patched:
    # Write backup
    backup = FA_UTILS.parent / (FA_UTILS.name + ".pre_fa4_sm110_patch")
    backup.write_text(original)
    patched = patched.replace(OLD_1, NEW_CORRECT)
    FA_UTILS.write_text(patched)
    print(f"  PATCHED: {FA_UTILS_REL}")
    print(f"  BACKUP:  {backup.name}")
    print(f"  EFFECT:  SM110 (Jetson Thor / major=11) now auto-selects FA4 instead of FA2")
else:
    # Upstream may have changed the anchor text
    # Search for the broader pattern to give a more helpful error
    if "device_capability.major == 10" in patched:
        print(f"  NOT FOUND: exact anchor missing but 'major == 10' detected.")
        print(f"  The surrounding context may have changed. Check manually:")
        lines = patched.splitlines()
        for i, line in enumerate(lines):
            if "major == 10" in line:
                start = max(0, i - 2)
                end = min(len(lines), i + 4)
                print(f"  Lines {start+1}–{end}:")
                for l in lines[start:end]:
                    print(f"    {l}")
                print()
    else:
        print(f"  NOT FOUND: 'device_capability.major == 10' not in {FA_UTILS_REL}")
        print(f"  File may have been significantly refactored upstream.")

print("\nFA4 SM110 patch complete.")
print(
    f"To verify: grep -n 'SM110 Thor' {VLLM_ROOT}/v1/attention/backends/fa_utils.py"
)