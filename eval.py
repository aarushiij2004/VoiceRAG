"""
eval.py -- Measure retrieval quality instead of guessing.

For every in-scope question in eval/questions.json we know an "evidence phrase" that
the correct passage contains. A retrieved chunk counts as a HIT if it contains that
phrase. From the rank of the first hit we compute:

    hit@k  fraction of questions with a hit in the top k chunks
    MRR    mean of 1/rank of the first hit (0 if none in the top 10)

Out-of-scope questions test the relevance gate: refusal rate should be high,
false-refusal rate on in-scope questions should be low.

    python eval.py --modes bm25                         # no embedding model needed
    python eval.py --modes bm25 dense hybrid            # full comparison
    python eval.py --modes hybrid --chunk-sizes 100 200 400
    python eval.py --modes dense --calibrate            # choose MIN_SIMILARITY
"""

import argparse
import json
from pathlib import Path

from config import CHUNK_SIZE, EMBEDDING_MODEL, MIN_SIMILARITY
from documents import build_chunks, load_documents
from retrieve import Retriever
from text_utils import normalize_for_match

KS = (1, 3, 5)
MRR_CUTOFF = 10


def load_questions(path) -> list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def first_hit_rank(chunks: list, evidence: str):
    """1-based rank of the first chunk containing the evidence phrase, or None."""
    needle = normalize_for_match(evidence)
    for rank, chunk in enumerate(chunks, start=1):
        if needle in normalize_for_match(chunk["text"]):
            return rank
    return None


def compute_metrics(ranks: list, ks=KS, cutoff: int = MRR_CUTOFF) -> dict:
    """ranks: first-hit rank per question (None = never retrieved)."""
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    metrics = {"n": n}
    for k in ks:
        metrics[f"hit@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n
    metrics["mrr"] = sum(1.0 / r for r in ranks if r is not None and r <= cutoff) / n
    return metrics


def threshold_sweep(in_scope_sims: list, oos_sims: list, thresholds) -> list:
    """For each threshold: how many OOS questions are refused vs in-scope wrongly refused."""
    rows = []
    for t in thresholds:
        rows.append({
            "threshold": round(t, 2),
            "oos_refused": sum(1 for s in oos_sims if s < t) / max(len(oos_sims), 1),
            "false_refusals": sum(1 for s in in_scope_sims if s < t) / max(len(in_scope_sims), 1),
        })
    return rows


def evaluate(retriever: Retriever, questions: list) -> dict:
    in_scope = [q for q in questions if q.get("in_scope", True)]
    oos = [q for q in questions if not q.get("in_scope", True)]

    ranks, by_type, in_relevant, in_sims = [], {}, [], []
    misses = []
    for q in in_scope:
        res = retriever.query(q["question"], top_k=MRR_CUTOFF)
        rank = first_hit_rank(res.chunks, q["evidence"])
        ranks.append(rank)
        by_type.setdefault(q.get("type", "direct"), []).append(rank)
        in_relevant.append(res.relevant)
        if res.top_similarity is not None:
            in_sims.append(res.top_similarity)
        if rank is None or rank > 3:
            misses.append({"question": q["question"], "first_hit_rank": rank})

    oos_relevant, oos_sims = [], []
    for q in oos:
        res = retriever.query(q["question"], top_k=MRR_CUTOFF)
        oos_relevant.append(res.relevant)
        if res.top_similarity is not None:
            oos_sims.append(res.top_similarity)

    return {
        "overall": compute_metrics(ranks),
        "by_type": {t: compute_metrics(r) for t, r in by_type.items()},
        "gate": {
            "oos_refused": (sum(1 for r in oos_relevant if not r) / len(oos_relevant)) if oos else None,
            "false_refusals": (sum(1 for r in in_relevant if not r) / len(in_relevant)) if in_relevant else None,
        },
        "in_scope_sims": in_sims,
        "oos_sims": oos_sims,
        "misses": misses,
    }


def _pct(x):
    return "  -  " if x is None else f"{100 * x:5.1f}"


def print_table(rows: list):
    header = (f"{'chunk':>5} {'mode':<7} | {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>6} | "
              f"{'para hit@3':>10} | {'OOS refused':>11} {'false refusal':>13}")
    print(header)
    print("-" * len(header))
    for r in rows:
        o, p, g = r["overall"], r["by_type"].get("paraphrase", {}), r["gate"]
        print(f"{r['chunk_size']:>5} {r['mode']:<7} | {_pct(o['hit@1']):>6} {_pct(o['hit@3']):>6} "
              f"{_pct(o['hit@5']):>6} {o['mrr']:>6.3f} | {_pct(p.get('hit@3')):>10} | "
              f"{_pct(g['oos_refused']):>11} {_pct(g['false_refusals']):>13}")
    print("\nAll numbers are percentages except MRR. 'para' = paraphrased questions that avoid the notes' keywords.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--modes", nargs="+", default=["bm25"], choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=[CHUNK_SIZE])
    ap.add_argument("--overlap-ratio", type=float, default=0.2, help="overlap = ratio x chunk size")
    ap.add_argument("--min-similarity", type=float, default=MIN_SIMILARITY)
    ap.add_argument("--calibrate", action="store_true", help="print a MIN_SIMILARITY sweep (dense/hybrid)")
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("--out", help="write full results to this JSON file")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    docs = load_documents()
    if not docs:
        raise SystemExit("No documents in documents/ to evaluate against.")

    embedder = None
    if any(m in ("dense", "hybrid") for m in args.modes):
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model {EMBEDDING_MODEL}...")
        embedder = SentenceTransformer(EMBEDDING_MODEL)

    rows = []
    for size in args.chunk_sizes:
        chunks = build_chunks(docs, size, round(args.overlap_ratio * size))
        for mode in args.modes:
            retriever = Retriever.build_in_memory(chunks, embedder, mode=mode,
                                                  min_similarity=args.min_similarity)
            result = evaluate(retriever, questions)
            result.update({"chunk_size": size, "mode": mode, "num_chunks": len(chunks)})
            rows.append(result)

    print(f"\n{sum(1 for q in questions if q.get('in_scope', True))} in-scope / "
          f"{sum(1 for q in questions if not q.get('in_scope', True))} out-of-scope questions\n")
    print_table(rows)

    if args.show_misses:
        for r in rows:
            if r["misses"]:
                print(f"\nMisses (rank > 3) for chunk={r['chunk_size']} mode={r['mode']}:")
                for m in r["misses"]:
                    print(f"  rank={m['first_hit_rank']}: {m['question']}")

    if args.calibrate:
        for r in rows:
            if r["in_scope_sims"]:
                print(f"\nThreshold sweep for chunk={r['chunk_size']} mode={r['mode']} "
                      "(best dense cosine of each question):")
                print(f"{'threshold':>9} {'OOS refused':>12} {'false refusals':>15}")
                for row in threshold_sweep(r["in_scope_sims"], r["oos_sims"],
                                           [x / 100 for x in range(10, 55, 5)]):
                    print(f"{row['threshold']:>9.2f} {100*row['oos_refused']:>11.0f}% "
                          f"{100*row['false_refusals']:>14.0f}%")
        print("\nPick the highest threshold that keeps false refusals near 0%.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nSaved full results -> {args.out}")


if __name__ == "__main__":
    main()
