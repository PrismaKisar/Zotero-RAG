"""Convention guard: the test suite carries exactly two kinds of file.

`test_<module>.py` mirrors one source module and sits at the same path under
`tests/` (`zotero_rag/ingest/zotero_db.py` -> `tests/zotero_rag/ingest/test_zotero_db.py`);
`guards/test_guard_<subject>.py` pins an invariant that spans several modules and
so mirrors none. Without the marker the two schemes look alike, and a guard ends
up named after whichever module it happens to touch first.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"


def mirrored_module(test: Path) -> Path:
    relative = test.relative_to(TESTS)
    return ROOT / relative.parent / (relative.name.removeprefix("test_"))


def test_every_test_file_mirrors_a_module_or_is_marked_a_guard():
    tests = sorted(TESTS.rglob("test_*.py"))
    assert tests, "no test files found"
    stray = [str(test.relative_to(TESTS)) for test in tests
             if not (test.parent.name == "guards" and test.stem.startswith("test_guard_"))
             and not mirrored_module(test).exists()]
    assert not stray, f"neither a module mirror nor a guard: {stray}"
