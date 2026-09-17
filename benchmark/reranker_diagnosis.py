"""Why the reranker does not beat retrieval's order, measured offline.

The campaigns found the cross-encoder useful as a filter (it roughly doubles
attribution precision) and no better than reciprocal-rank fusion as a ranking:
recall@10 on its output fell below recall@10 on retrieval's, the more so the
more retrieval found. That reading came from the pipeline's own lists, where
the ordering and the rerank threshold act together, on a 120-question sample,
with a tie order that was not reproducible. This isolates the ordering.

Every question's hybrid top-LIMIT is rebuilt offline and exactly, then ordered
three ways on the *same* candidates, with no threshold:

- ``hybrid``: retrieval's own RRF order, the reference;
- ``<model>``: the cross-encoder's score alone, as the pipeline ships it;
- ``<model>+rrf``: RRF of the two orders, the usual way to combine a reranker
  with a first stage instead of replacing it.

Two scopes: ``library`` searches every paper, as deployed; ``paper`` only the
question's own paper, which removes the confusion between papers and leaves the
choice of passage within one document, the reranker's job proper.

Metrics are over the questions whose candidates hold at least one gold chunk:
elsewhere no order can score, and those rows would only dilute the difference.
The disagreements file lists questions where exactly one of retrieval and a
model puts gold first, with the chunk each put first, so the cases can be read:
a reranker preferring a non-gold chunk that does answer the question is a gap
in the labels, not a ranking error.

Usage:
  python -m benchmark.reranker_diagnosis --golden-dir output_qasa
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# appended, never prepended: the repo root must keep priority over this directory
sys.path.append(str(Path(__file__).resolve().parent.parent / "zotero_rag"))

from benchmark.ablation import load_jsonl
from benchmark.paraphrase_retrieval import LIMIT, HybridSearch, rrf
from benchmark.retrieval_metrics import bootstrap_ci, recall_at_k, reciprocal_rank
from benchmark.retrieval_score_distribution import gold_ids

KS = (1, 3, 5, 10)
SCOPES = ("library", "paper")
DEFAULT_MODELS = ("BAAI/bge-reranker-base", "BAAI/bge-reranker-v2-m3")
MAX_LENGTH = 512       # the shipped reranker's window: a bigger model is not also a longer one
DISAGREEMENTS = 10     # per model and direction, enough to read by hand
EXCERPT = 500          # characters of a chunk shown in the disagreements file
# Listwise models read the query and every candidate in one context, so they
# load through their own remote code rather than as a CrossEncoder.
LISTWISE = ("jinaai/jina-reranker-v3",)


def short(model: str) -> str:
    return model.split("/")[-1]


def listwise_scores(results: list[dict], n: int) -> list[float]:
    """A listwise model's results, which come best first, back in input order."""
    scores = [0.0] * n
    for result in results:
        scores[result["index"]] = float(result["relevance_score"])
    return scores


def ranked_orders(hybrid: list, scores: dict[str, list[float]]) -> dict[str, list]:
    """Every order over the same candidates: retrieval's, each model's, each fused.

    A model's equal scores keep retrieval's order (the sort is stable), so a tie
    never counts as the reranker's choice.
    """
    orders = {"hybrid": list(hybrid)}
    for model, model_scores in scores.items():
        by_model = [c for _, c in sorted(zip(model_scores, hybrid, strict=True),
                                         key=lambda pair: -pair[0])]
        orders[model] = by_model
        orders[f"{model}+rrf"] = [c for c, _ in rrf([list(hybrid), by_model])]
    return orders


def order_scores(ranked: list, gold: set) -> dict[str, float]:
    row = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in KS}
    row["mrr"] = reciprocal_rank(ranked, gold)
    return row


def disagreements(cases: list[dict], model: str, limit: int = DISAGREEMENTS) -> dict:
    """Questions where exactly one of retrieval and ``model`` puts gold first."""
    out = {"hybrid": [], model: []}
    for case in cases:
        hybrid_hit = case["orders"]["hybrid"][0] in case["gold"]
        model_hit = case["orders"][model][0] in case["gold"]
        if hybrid_hit != model_hit:
            winner = "hybrid" if hybrid_hit else model
            out[winner].append(case)
    return {"counts": {k: len(v) for k, v in out.items()},
            **{k: v[:limit] for k, v in out.items()}}


def report(name: str, rows: dict, judged: dict, models: list[str]) -> str:
    """One table per scope: mean per order, paired deltas against retrieval's order."""
    header = ["order", *(f"recall@{k}" for k in KS), "mrr",
              "Δ recall@1", "Δ recall@3", "Δ mrr"]
    out = [f"# Reranker as a ranking, offline: {name}",
           (f"Same hybrid top-{LIMIT} per question, no threshold. Δ is the paired mean "
            "difference against `hybrid`, with its 95% bootstrap interval.")]
    for scope in SCOPES:
        base = rows[(scope, "hybrid")]
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for order in ["hybrid", *(o for m in models for o in (m, f"{m}+rrf"))]:
            cells = [f"{np.mean([r[m] for r in rows[(scope, order)]]):.3f}"
                     for m in (*(f"recall@{k}" for k in KS), "mrr")]
            for metric in ("recall@1", "recall@3", "mrr"):
                if order == "hybrid":
                    cells.append("")
                    continue
                deltas = [a[metric] - b[metric]
                          for a, b in zip(rows[(scope, order)], base, strict=True)]
                low, high = bootstrap_ci(deltas)
                cells.append(f"{np.mean(deltas):+.3f} [{low:+.3f}, {high:+.3f}]")
            lines.append(f"| {short(order)} | " + " | ".join(cells) + " |")
        out += [f"## scope: {scope}",
                (f"{judged[scope]['judged']} of {judged[scope]['total']} questions have "
                 f"gold among the {LIMIT} candidates."),
                "\n".join(lines)]
    return "\n\n".join(out) + "\n"


def render_disagreements(name: str, cases: list[dict], models: list[str]) -> str:
    out = [f"# Rank-1 disagreements, scope paper: {name}"]
    for model in models:
        split = disagreements(cases, model)
        out.append(f"## {short(model)}: gold first only for hybrid {split['counts']['hybrid']}, "
                   f"only for the model {split['counts'][model]}")
        for winner in ("hybrid", model):
            for case in split[winner]:
                top = {o: case["orders"][o][0] for o in ("hybrid", model)}
                out.append("\n".join([
                    f"### {case['question_id']} (gold first: {short(winner)})",
                    f"**Q:** {case['question']}",
                    f"**Evidence:** {' / '.join(e[:EXCERPT] for e in case['evidence'])}",
                    *(f"**{short(o)} #1 {'(gold)' if top[o] in case['gold'] else '(not gold)'}:** "
                      f"{case['texts'][top[o]][:EXCERPT]}" for o in top)]))
    return "\n\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", default="output_qasa",
                        help="holds golden_set_aligned.jsonl and chunks/")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--tag", default="",
                        help="suffix of the output files, so a run on other models keeps earlier ones")
    args = parser.parse_args()

    from device import resolve_device  # heavy imports (torch)
    from sentence_transformers import CrossEncoder

    golden_dir = Path(args.golden_dir)
    search = HybridSearch(golden_dir / "chunks")
    text_of = search.text_of
    device = resolve_device(None)

    def load(model):
        if model.startswith(LISTWISE):
            from transformers import AutoModel
            return AutoModel.from_pretrained(model, dtype="auto",
                                             trust_remote_code=True).to(device).eval()
        return CrossEncoder(model, device=device, max_length=MAX_LENGTH)

    rerankers = {m: load(m) for m in args.models}
    cache = {m: {} for m in args.models}

    def model_scores(model, question, candidates):
        if model.startswith(LISTWISE):
            # a candidate's score depends on the others in the list: nothing to reuse
            results = rerankers[model].rerank(question, [text_of[c] for c in candidates])
            return listwise_scores(results, len(candidates))
        seen = cache[model]
        todo = [c for c in candidates if (question, c) not in seen]
        if todo:
            predicted = rerankers[model].predict([[question, text_of[c]] for c in todo],
                                                 show_progress_bar=False)
            seen.update({(question, c): float(s) for c, s in zip(todo, predicted, strict=True)})
        return [seen[(question, c)] for c in candidates]

    rows, cases = defaultdict(list), []
    judged = {scope: {"judged": 0, "total": 0} for scope in SCOPES}
    for record in load_jsonl(golden_dir / "golden_set_aligned.jsonl"):
        gold = {g for g in gold_ids(record) if g in text_of}
        if not gold:
            continue
        question = record["question"]
        for scope in SCOPES:
            hits = search.top(question, paper=record["paper_id"] if scope == "paper" else None)
            judged[scope]["total"] += 1
            if not gold & set(hits):
                continue
            judged[scope]["judged"] += 1
            orders = ranked_orders(hits, {m: model_scores(m, question, hits) for m in args.models})
            for order, ranked in orders.items():
                rows[(scope, order)].append(order_scores(ranked, gold))
            if scope == "paper":
                cases.append({"question_id": record["question_id"], "question": question,
                              "evidence": record["evidence"], "gold": gold,
                              "orders": orders, "texts": text_of})

    name = golden_dir.name
    (golden_dir / f"reranker_disagreements{args.tag}.md").write_text(
        render_disagreements(name, cases, args.models))
    out_file = golden_dir / f"reranker_diagnosis{args.tag}.md"
    out_file.write_text(report(name, rows, judged, args.models))
    print(f"{judged['library']['total']} questions -> {out_file}")


if __name__ == "__main__":
    main()
