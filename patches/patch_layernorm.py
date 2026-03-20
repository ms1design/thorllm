#!/usr/bin/env python3
"""
patches/patch_layernorm.py — RMSNormGated missing activation attribute fix

Problem (vLLM < 0.16.0):
  RMSNormGated.__init__ does not accept or store the `activation` parameter,
  but forward() references self.activation.  Loading NVFP4 gated-activation
  models causes:
    AttributeError: 'RMSNormGated' object has no attribute 'activation'

Fix:
  Add `activation: str = "silu"` to the __init__ signature and store it as
  self.activation — only when the attribute is not already present.

Version safety:
  - vLLM >= 0.16.0 already has `activation` in __init__; the patch detects
    this and exits cleanly without modifying the file.
  - Checks are idempotent: running the patch twice is safe.

Usage:
  python3 patches/patch_layernorm.py               # uses active venv
  python3 patches/patch_layernorm.py /path/to/site-packages
"""

import ast
import inspect
import sys
import site
import importlib.util
from pathlib import Path


# ── Locate vLLM ──────────────────────────────────────────────────────────────

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


# ── Version detection ─────────────────────────────────────────────────────────

def get_vllm_version(vllm_root: Path) -> tuple[int, int, int]:
    """Return (major, minor, patch) parsed from vllm/version.py or __version__."""
    version_py = vllm_root / "version.py"
    if version_py.exists():
        txt = version_py.read_text()
        for line in txt.splitlines():
            if "__version__" in line and "=" in line:
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                parts = v.split(".")
                try:
                    return int(parts[0]), int(parts[1]), int(parts[2].split("+")[0])
                except Exception:
                    pass
    # Fallback: try importlib.metadata
    try:
        import importlib.metadata
        v = importlib.metadata.version("vllm")
        parts = v.split(".")
        return int(parts[0]), int(parts[1]), int(parts[2].split("+")[0])
    except Exception:
        return (0, 0, 0)


# ── Introspect RMSNormGated ───────────────────────────────────────────────────

def rmsnorm_gated_has_activation(path: Path) -> bool | None:
    """
    Parse layernorm.py and return:
      True   — RMSNormGated.__init__ already has 'activation' param
      False  — it does not
      None   — class not found in this file (may have been moved)
    """
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "RMSNormGated":
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                args = item.args
                all_args = [a.arg for a in args.args + args.kwonlyargs]
                if "activation" in all_args:
                    return True
                # Also check for self.activation assignment in body
                for stmt in ast.walk(item):
                    if (
                        isinstance(stmt, ast.Assign)
                        and any(
                            isinstance(t, ast.Attribute)
                            and t.attr == "activation"
                            for t in stmt.targets
                        )
                    ):
                        return True
                return False
    return None  # class not found


# ── Main patch logic ──────────────────────────────────────────────────────────

def main() -> None:
    vllm_root = find_vllm_root(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"vLLM root: {vllm_root}")

    major, minor, patch = get_vllm_version(vllm_root)
    print(f"vLLM version: {major}.{minor}.{patch}")

    # Version guard: >= 0.16.0 already has the fix upstream
    if (major, minor) >= (0, 16):
        print("  SKIP: vLLM >= 0.16.0 already includes activation in RMSNormGated.")
        print("Layernorm patch not needed for this version.")
        return

    layernorm_path = vllm_root / "model_executor/layers/layernorm.py"
    if not layernorm_path.exists():
        print(f"ERROR: not found: {layernorm_path}")
        sys.exit(1)

    # Runtime introspection: check if already patched
    has_activation = rmsnorm_gated_has_activation(layernorm_path)
    if has_activation is True:
        print("  ALREADY PATCHED: RMSNormGated already has 'activation' attribute.")
        print("Layernorm patch complete (no-op).")
        return
    if has_activation is None:
        print("  WARNING: RMSNormGated class not found in layernorm.py.")
        print("  This may mean the class was renamed or moved. Skipping patch.")
        return

    # Apply the patch
    original = layernorm_path.read_text()
    patched = original

    # ── Patch 1: Add 'activation' param to __init__ signature ────────────────
    # We look for the last param before the closing paren of RMSNormGated.__init__
    # Common anchor: norm_before_gate is the last param in old vLLM
    OLD_SIG = "        norm_before_gate: bool = False,"
    NEW_SIG = (
        "        norm_before_gate: bool = False,\n"
        "        activation: str = \"silu\",  # thorllm/patch_layernorm: NVFP4 gated models"
    )

    if OLD_SIG in patched:
        patched = patched.replace(OLD_SIG, NEW_SIG, 1)
        print("  Applied: activation param added to __init__ signature")
    else:
        print("  WARNING: Could not find signature anchor 'norm_before_gate'.")
        print("  Manual inspection of layernorm.py may be required.")
        sys.exit(1)

    # ── Patch 2: Store activation in __init__ body ───────────────────────────
    OLD_BODY = "        self.norm_before_gate = norm_before_gate"
    NEW_BODY = (
        "        self.norm_before_gate = norm_before_gate\n"
        "        self.activation = activation  # thorllm/patch_layernorm: NVFP4 gated models"
    )

    if OLD_BODY in patched:
        patched = patched.replace(OLD_BODY, NEW_BODY, 1)
        print("  Applied: self.activation stored in __init__ body")
    else:
        print("  WARNING: Could not find body anchor 'self.norm_before_gate'.")
        sys.exit(1)

    if patched != original:
        # Write backup
        backup = layernorm_path.with_suffix(".py.pre_layernorm_patch")
        backup.write_text(original)
        layernorm_path.write_text(patched)
        print(f"  PATCHED: {layernorm_path}")
        print(f"  BACKUP:  {backup}")
    else:
        print("  NO CHANGE")

    print("\nLayernorm patch complete.")


if __name__ == "__main__":
    main()
