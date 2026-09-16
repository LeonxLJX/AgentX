# Static import checker: verifies that every intra-project import in AgentX
# resolves to a real module / name. Run from the repo root.
"""Quick static verification (no third-party deps required)."""

import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def package_files():
    for base in ("agentx", "examples", "tests"):
        for dirpath, _, filenames in os.walk(ROOT / base):
            for fn in filenames:
                if fn.endswith(".py"):
                    yield Path(dirpath) / fn


def module_name_for(path: Path) -> str:
    rel = path.relative_to(ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def resolve_top_level(pkg: str):
    """Map a dotted package name to a file path under ROOT if it exists."""
    parts = pkg.split(".")
    current = ROOT
    for i, part in enumerate(parts):
        current = current / part
        if current.is_dir():
            if not (current / "__init__.py").exists():
                return None
            continue
        if current.with_suffix(".py").exists():
            return current.with_suffix(".py")
        return None
    return current / "__init__.py" if (current / "__init__.py").exists() else None


errors = []
files = list(package_files())
mod_by_path = {p: module_name_for(p) for p in files}
name_to_path = {module_name_for(p): p for p in files}

for path in files:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("agentx") or alias.name in {"examples", "tests"}:
                    top = alias.name.split(".")[0]
                    if not any(p.startswith(top + ".") or p == top for p in name_to_path):
                        errors.append(f"{path.name}: import '{alias.name}' not found")
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:  # relative import
                base_parts = module_name_for(path).split(".")
                if node.level > len(base_parts):
                    errors.append(f"{path.name}: relative import level {node.level} out of range")
                    continue
                base = ".".join(base_parts[: len(base_parts) - node.level])
                module = f"{base}.{node.module}" if node.module else base
                target = name_to_path.get(module)
                if target is None:
                    errors.append(f"{path.name}: relative import '{module}' not found")
                    continue
                # Verify imported names exist (best-effort: __all__ or defs).
                src = target.read_text(encoding="utf-8")
                tree2 = ast.parse(src)
                defined = set()
                all_names = None
                for n in ast.walk(tree2):
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        defined.add(n.name)
                    elif isinstance(n, ast.Assign):
                        for t in n.targets:
                            if isinstance(t, ast.Name):
                                defined.add(t.id)
                    elif isinstance(n, ast.ImportFrom) and n.module == "__future__":
                        continue
                    elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "__all__":
                        try:
                            all_names = {e.value for e in n.value.elts if isinstance(e, ast.Constant)}
                        except Exception:
                            pass
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    if all_names is not None and alias.name not in all_names:
                        errors.append(f"{path.name}: '{alias.name}' not in {module}.__all__")
                    elif alias.name not in defined:
                        errors.append(f"{path.name}: '{alias.name}' not defined in {module}")
            elif node.module and node.module.startswith("agentx"):
                target = name_to_path.get(node.module)
                if target is None:
                    errors.append(f"{path.name}: import from '{node.module}' not found")

if errors:
    print(f"FAILED: {len(errors)} error(s)")
    for e in errors[:50]:
        print("  -", e)
    sys.exit(1)
print(f"OK: {len(files)} files, all intra-project imports resolve")
