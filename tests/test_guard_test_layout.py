"""Convention guard: the test suite carries exactly two kinds of file.

`test_<module>.py` mirrors one source module and is named after its stem;
`test_guard_<subject>.py` pins an invariant that spans several modules and so
mirrors none. Without the marker the two schemes look alike, and a guard ends
up named after whichever module it happens to touch first.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = sorted(ROOT.glob("tests/test_*.py"))
MODULE_STEMS = {path.stem for directory in ("zotero_rag", "benchmark")
                for path in (ROOT / directory).glob("*.py")}


def test_every_test_file_mirrors_a_module_or_is_marked_a_guard():
    assert TESTS, "no test files found"
    stray = [path.name for path in TESTS
             if not path.stem.startswith("test_guard_")
             and path.stem.removeprefix("test_") not in MODULE_STEMS]
    assert not stray, f"neither a module mirror nor a guard: {stray}"
