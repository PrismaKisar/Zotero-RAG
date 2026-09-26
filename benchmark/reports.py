"""The formats the studies share: per-question JSONL dumps in, Markdown tables out.

Kept apart from any one study so that reading a dump does not mean importing the
ablation harness, and a report table looks the same whichever study wrote it.
"""

import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """One record per line of ``path``."""
    return [json.loads(line) for line in path.open()]


def markdown_table(header: list[str], rows: list[list]) -> str:
    """A Markdown table; floats to three decimals, everything else as ``str``."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v)
                                for v in row) + " |" for row in rows]
    return "\n".join(lines)
