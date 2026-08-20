"""Convention guard for the Streamlit entry point.

Nothing imports `app.py` - it is run, not imported - so a leading underscore
draws a public/private line that has no other side. Keeping the module free of
them stops the two conventions from coexisting again.
"""

import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "zotero_rag" / "app.py"


def test_no_function_is_underscore_prefixed():
    tree = ast.parse(APP.read_text())
    underscored = [node.name for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name.startswith("_")]
    assert not underscored, f"app.py is imported by nobody: {underscored}"
