"""Regression guard: the YAML runner must call the API that actually exists.

`run_from_config.py` drifted away from `ZoteroRAG` unnoticed - it called
`index_exists()`, `load_index()`, `build_index()` and read `rag.index_path`,
none of which were ever defined, and it passed constructor keywords the class
does not take. Nothing caught it because the module is only exercised by
running it. This reads both files as source, so the check costs no torch import.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "zotero_rag" / "run_from_config.py"
PIPELINE = ROOT / "zotero_rag" / "pipeline.py"


def _zotero_rag_class() -> ast.ClassDef:
    for node in ast.walk(ast.parse(PIPELINE.read_text())):
        if isinstance(node, ast.ClassDef) and node.name == "ZoteroRAG":
            return node
    raise AssertionError("ZoteroRAG class not found")


def _members() -> set[str]:
    """Method names plus every `self.<x>` the constructor assigns."""
    cls = _zotero_rag_class()
    names = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(cls):
        if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Name) and node.value.id == "self"):
            names.add(node.attr)
    return names


def _constructor_parameters() -> set[str]:
    for node in _zotero_rag_class().body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return {arg.arg for arg in node.args.args} - {"self"}
    raise AssertionError("ZoteroRAG.__init__ not found")


def _runner_tree() -> ast.Module:
    return ast.parse(RUNNER.read_text())


def test_runner_only_touches_members_zotero_rag_defines():
    used = {node.attr for node in ast.walk(_runner_tree())
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == "rag"}
    assert used, "the runner no longer touches the rag object at all"
    missing = sorted(used - _members())
    assert not missing, f"run_from_config calls members ZoteroRAG lacks: {missing}"


def test_runner_only_passes_keywords_the_constructor_accepts():
    unknown = set()
    for node in ast.walk(_runner_tree()):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "ZoteroRAG"):
            unknown |= {kw.arg for kw in node.keywords if kw.arg}
    assert unknown, "the runner no longer builds a ZoteroRAG"
    unknown -= _constructor_parameters()
    assert not unknown, f"run_from_config passes unknown keywords: {sorted(unknown)}"
