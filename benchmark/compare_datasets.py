"""Compare the two benchmark datasets (QASPER and QASA) along the axes the
methodology chapter needs: information density, kind of information, question
difficulty and level of abstraction.

Everything is computed from artefacts already produced by the pipeline:
  <out>/golden_set_aligned.jsonl   questions + evidence + chunk alignment
  <out>/chunks/<paper_id>.json     the chunked full text of each paper

Usage:
  python -m benchmark.compare_datasets \
      --dataset QASPER=benchmark_out --dataset QASA=benchmark_out_qasa \
      --out benchmark_out/dataset_comparison.md
"""

import argparse
import json
import math
import re
import statistics as st
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
SENT_RE = re.compile(r"[.!?]+\s")
# rough markers of formal / notation-heavy prose
MATH_RE = re.compile(r"\\\\[a-zA-Z]+|[=<>≤≥±∑∏∈θαβγλμσ]|\$")
CITE_RE = re.compile(r"\(\s*[A-Z][A-Za-z'-]+\s+et\s+al\.?,?\s*\d{4}|\[\d+(,\s*\d+)*\]")
NUM_RE = re.compile(r"\b\d+([.,]\d+)?%?\b")

YESNO = ("is", "are", "was", "were", "do", "does", "did", "can", "could",
         "has", "have", "had", "will", "would", "should")
STOP = frozenset("""a an the of to in for on and or is are was were be been this that these those
with as by from at it its their they we our what which how why who when where does do did
can could not no than then there here such using used use paper authors author study""".split())  # noqa: SIM905 - a 55-element list literal is unreadable


def words(text):
    return WORD_RE.findall(text.lower())


def syllables(word):
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups) - (1 if word.endswith("e") and len(groups) > 1 else 0)
    return max(n, 1)


def flesch_kincaid(text):
    """FK grade level; a coarse but standard readability proxy."""
    ws = words(text)
    sents = max(len(SENT_RE.findall(text)) + 1, 1)
    if not ws:
        return None
    return 0.39 * len(ws) / sents + 11.8 * sum(map(syllables, ws)) / len(ws) - 15.59


def q_type(question):
    ws = words(question)
    if not ws:
        return "other"
    for w in ws[:3]:
        if w in ("why",):
            return "why (causal)"
        if w in ("how",):
            return "how (procedural)"
        if w in ("what", "which", "who", "when", "where"):
            return "what/which (factual)"
        if w in YESNO:
            return "yes/no (verification)"
    return "other"


def jaccard(a, b):
    a, b = set(a) - STOP, set(b) - STOP
    return len(a & b) / len(a | b) if a | b else 0.0


def coverage(q, ev):
    q, ev = set(q) - STOP, set(ev)
    return len(q & ev) / len(q) if q else 0.0


def mann_whitney_p(x, y):
    """Two-sided Mann-Whitney U with normal approximation (no scipy)."""
    n1, n2 = len(x), len(y)
    if n1 < 10 or n2 < 10:
        return None
    pairs = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, pairs) if g == 0)
    u = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mu) / sigma if sigma else 0.0
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def cliffs_delta(x, y):
    """Non-parametric effect size in [-1, 1]; robust to the skew here."""
    ys = sorted(y)
    import bisect
    gt = sum(bisect.bisect_left(ys, v) for v in x)
    lt = sum(len(ys) - bisect.bisect_right(ys, v) for v in x)
    return (gt - lt) / (len(x) * len(ys)) if x and ys else 0.0


def load(out_dir):
    out_dir = Path(out_dir)
    records = [json.loads(line) for line in
               (out_dir / "golden_set_aligned.jsonl").read_text().splitlines() if line.strip()]
    chunks = {p.stem: json.loads(p.read_text()) for p in (out_dir / "chunks").glob("*.json")}
    return records, chunks


def analyse(records, chunks):
    """Return {metric: list_of_values} plus categorical counters."""
    m = {k: [] for k in (
        "q_words", "ev_spans", "ev_words", "ev_words_total", "aligned_chunks",
        "align_overlap", "spread", "position", "q_ev_jaccard", "q_ev_coverage",
        "paper_chunks", "paper_words", "chunk_words", "ttr", "fk_grade",
        "math_density", "cite_density", "num_density")}
    cats = {"q_type": Counter(), "extractive": Counter()}

    for texts in chunks.values():
        text = " ".join(texts)
        ws = words(text)
        if not ws:
            continue
        m["paper_chunks"].append(len(texts))
        m["paper_words"].append(len(ws))
        m["chunk_words"] += [len(words(t)) for t in texts]
        m["ttr"].append(len(set(ws)) / len(ws))
        fk = flesch_kincaid(text)
        if fk is not None:
            m["fk_grade"].append(fk)
        per_1k = 1000 / len(ws)
        m["math_density"].append(len(MATH_RE.findall(text)) * per_1k)
        m["cite_density"].append(len(CITE_RE.findall(text)) * per_1k)
        m["num_density"].append(len(NUM_RE.findall(text)) * per_1k)

    for r in records:
        qw = words(r["question"])
        m["q_words"].append(len(qw))
        cats["q_type"][q_type(r["question"])] += 1
        cats["extractive"]["short extractive span" if r.get("gold_spans")
                           else "free-form (no short span)"] += 1

        ev = r["evidence"]
        m["ev_spans"].append(len(ev))
        ev_words = [words(e) for e in ev]
        m["ev_words"] += [len(e) for e in ev_words]
        flat = [w for e in ev_words for w in e]
        m["ev_words_total"].append(len(flat))
        m["q_ev_jaccard"].append(jaccard(qw, flat))
        m["q_ev_coverage"].append(coverage(qw, flat))

        idx = sorted({c["chunk_index"] for hits in r["aligned_chunks"].values() for c in hits})
        m["aligned_chunks"].append(len(idx))
        m["align_overlap"] += [c["overlap"] for hits in r["aligned_chunks"].values() for c in hits]
        n_chunks = len(chunks.get(r["paper_id"], []))
        if idx and n_chunks:
            m["spread"].append((idx[-1] - idx[0]) / n_chunks)
            m["position"].append(st.mean(idx) / n_chunks)
    return m, cats


def fmt(vals):
    if not vals:
        return "n/a"
    return f"{st.mean(vals):.2f} / {st.median(vals):.2f}"


ROWS = [
    ("Papers analysed", "paper_chunks", "count"),
    ("Chunks per paper", "paper_chunks", "dist"),
    ("Words per paper", "paper_words", "dist"),
    ("Words per chunk", "chunk_words", "dist"),
    ("Type-token ratio (paper)", "ttr", "dist"),
    ("Flesch-Kincaid grade", "fk_grade", "dist"),
    ("Math/notation tokens per 1k words", "math_density", "dist"),
    ("Citations per 1k words", "cite_density", "dist"),
    ("Numbers per 1k words", "num_density", "dist"),
    ("Questions", "q_words", "count"),
    ("Question length (words)", "q_words", "dist"),
    ("Evidence spans per question", "ev_spans", "dist"),
    ("Words per evidence span", "ev_words", "dist"),
    ("Evidence words per question", "ev_words_total", "dist"),
    ("Chunks to retrieve per question", "aligned_chunks", "dist"),
    ("Alignment overlap", "align_overlap", "dist"),
    ("Evidence spread (frac. of doc)", "spread", "dist"),
    ("Evidence position (0=start, 1=end)", "position", "dist"),
    ("Question-evidence Jaccard", "q_ev_jaccard", "dist"),
    ("Question term coverage in evidence", "q_ev_coverage", "dist"),
]


def report(datasets):
    names = list(datasets)
    lines = ["# QASPER vs QASA: dataset comparison", "",
             ("Values are **mean / median** unless stated otherwise. "
              "`p` is a two-sided Mann-Whitney U test, `d` Cliff's delta "
              "(|d|: 0.15 negligible, 0.33 small, 0.47 medium)."), "",
             "| Metric | " + " | ".join(names) + " | p | d |",
             "|---|" + "---|" * (len(names) + 2)]

    for label, key, kind in ROWS:
        vals = [datasets[n][0][key] for n in names]
        if kind == "count":
            cells = [str(len(v)) for v in vals]
            p = d = ""
        else:
            cells = [fmt(v) for v in vals]
            if len(vals) == 2 and vals[0] and vals[1]:
                pv = mann_whitney_p(vals[0], vals[1])
                p = "n/a" if pv is None else ("<0.001" if pv < 0.001 else f"{pv:.3f}")
                d = f"{cliffs_delta(vals[0], vals[1]):+.2f}"
            else:
                p = d = ""
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {p} | {d} |")

    for cat, title in (("q_type", "Question form"), ("extractive", "Answer form")):
        keys = sorted({k for n in names for k in datasets[n][1][cat]})
        lines += ["", f"## {title}", "", "| Category | " + " | ".join(names) + " |",
                  "|---|" + "---|" * len(names)]
        for k in keys:
            cells = []
            for n in names:
                c = datasets[n][1][cat]
                tot = sum(c.values()) or 1
                cells.append(f"{c[k]} ({100 * c[k] / tot:.1f}%)")
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", action="append", required=True,
                    metavar="NAME=DIR", help="repeatable, e.g. QASPER=benchmark_out")
    ap.add_argument("--out", type=Path, help="write the markdown report here")
    ap.add_argument("--json", type=Path, help="also dump raw metric values")
    args = ap.parse_args()

    datasets = {}
    for spec in args.dataset:
        name, _, path = spec.partition("=")
        datasets[name] = analyse(*load(path))

    md = report(datasets)
    print(md)
    if args.out:
        args.out.write_text(md)
    if args.json:
        args.json.write_text(json.dumps(
            {n: {"metrics": m, "categories": {k: dict(v) for k, v in c.items()}}
             for n, (m, c) in datasets.items()}, indent=2))


if __name__ == "__main__":
    main()
