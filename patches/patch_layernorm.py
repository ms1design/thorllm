#!/usr/bin/env python3
"""
patches/patch_layernorm.py — RMSNormGated missing activation attribute fix

Problem (vLLM < 0.16.0 only):
  RMSNormGated.__init__ does not accept or store the `activation` parameter,
  but forward() references self.activation, causing AttributeError on NVFP4
  gated-activation models.

v0.17.1+ already includes activation in __init__ (confirmed in source).
This patch MUST NOT be applied to those versions.

The patch uses AST introspection as the primary check (looks at actual source)
and importlib.metadata as secondary. Both must agree before patching.

Usage:
  python3 patches/patch_layernorm.py               # uses active venv
  python3 patches/patch_layernorm.py /path/to/site-packages
"""
import ast
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


def get_vllm_version() -> tuple[int, int, int]:
    try:
        import importlib.metadata
        v = importlib.metadata.version("vllm")
        parts = v.split(".")
        return int(parts[0]), int(parts[1]), int(parts[2].split("+")[0])
    except Exception:
        return (0, 0, 0)


def rmsnorm_gated_init_has_activation(src: str) -> bool | None:
    """
    Parse the source and check whether RMSNormGated.__init__ already has
    an 'activation' parameter.

    Returns:
      True  — already has it (no patch needed)
      False — missing (patch needed)
      None  — class/method not found
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "RMSNormGated":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                continue
            # Check argument names
            arg_names = [a.arg for a in item.args.args + item.args.kwonlyargs]
            return "activation" in arg_names
    return None


def main() -> None:
    vllm_root = find_vllm_root(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"vLLM root: {vllm_root}")

    major, minor, patch_ver = get_vllm_version()
    print(f"vLLM version: {major}.{minor}.{patch_ver}")

    layernorm_path = vllm_root / "model_executor/layers/layernorm.py"
    if not layernorm_path.exists():
        print(f"ERROR: not found: {layernorm_path}")
        sys.exit(1)

    src = layernorm_path.read_text()

    # Primary check: read the actual source
    ast_result = rmsnorm_gated_init_has_activation(src)
    print(f"AST check — activation already in __init__: {ast_result}")

    if ast_result is True:
        print("SKIP: RMSNormGated already has 'activation' in __init__ (confirmed by AST).")
        print("Layernorm patch not needed.")
        return

    if ast_result is None:
        print("WARNING: RMSNormGated not found in layernorm.py — may have moved.")
        print("Skipping patch.")
        return

    # ast_result is False — need to patch
    # Secondary check: version guard (belt-and-suspenders)
    if (major, minor) >= (0, 16):
        # vLLM >= 0.16.0 should already have it; AST says otherwise — warn and skip
        print(f"WARNING: vLLM {major}.{minor} >= 0.16.0 but AST says activation missing.")
        print("This is unexpected. Skipping patch to avoid corrupting the install.")
        print("Please report this at https://github.com/ms1design/thorllm")
        return

    print("Applying layernorm patch...")
    original = src

    # ── Patch 1: Add activation param to __init__ signature ──────────────────
    OLD_SIG = "        norm_before_gate: bool = False,"
    NEW_SIG = (
        '        norm_before_gate: bool = False,\n'
        '        activation: str = "silu",  # thorllm: NVFP4 gated-activation models'
    )
    if OLD_SIG in src and NEW_SIG not in src:
        src = src.replace(OLD_SIG, NEW_SIG, 1)
        print("  Applied: activation param to __init__ signature")
    else:
        print("  WARNING: Could not find 'norm_before_gate' anchor")
        sys.exit(1)

    # ── Patch 2: Store self.activation in body ────────────────────────────────
    OLD_BODY = "        self.norm_before_gate = norm_before_gate"
    NEW_BODY = (
        "        self.norm_before_gate = norm_before_gate\n"
        "        self.activation = activation  # thorllm: NVFP4 gated-activation models"
    )
    if OLD_BODY in src and NEW_BODY not in src:
        src = src.replace(OLD_BODY, NEW_BODY, 1)
        print("  Applied: self.activation stored in body")
    else:
        print("  WARNING: Could not find 'self.norm_before_gate' anchor")
        sys.exit(1)

    # Write backup + patched file
    backup = layernorm_path.with_suffix(".py.pre_layernorm_patch")
    backup.write_text(original)
    layernorm_path.write_text(src)
    print(f"  PATCHED: {layernorm_path}")
    print(f"  BACKUP:  {backup}")
    print("Layernorm patch complete.")


if __name__ == "__main__":
    main()
