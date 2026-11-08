"""Evaluate retrieval and out-of-scope rejection on the hand-written benchmark.

Half the benchmark calibrates the rejection threshold; the other half reports results.

    python scripts/evaluate_retrieval.py [--passage-embeddings PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medrag import config, data, metrics  # noqa: E402
from medrag.retrieval import Retriever  # noqa: E402

KS = (1, 3, 5)
DEPTH = 10
SEED = 42
ORIGINAL_THRESHOLD = 0.50


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = np.random.default_rng(SEED).permutation(len(frame))
    half = len(frame) // 2
    return frame.iloc[order[:half]], frame.iloc[order[half:]]


def ranks_for(positions: np.ndarray, kb: pd.DataFrame, focus: str, qtype: str | None):
    hits = kb["focus"].to_numpy()[positions] == focus
    if qtype is not None:
        hits &= kb["question_qtype"].to_numpy()[positions] == qtype
    return metrics.first_hit_rank(hits.tolist())


def retrieval_summary(all_positions, questions: pd.DataFrame, kb: pd.DataFrame) -> dict:
    """Score against the exact entry type (strict) and against the right condition (topic)."""
    summary: dict = {"n": len(questions)}
    for name, use_qtype in (("strict", True), ("topic", False)):
        ranks = [
            ranks_for(pos, kb, row.focus, row.qtype if use_qtype else None)
            for pos, row in zip(all_positions, questions.itertuples(), strict=True)
        ]
        scores = {f"hit@{k}": metrics.hit_rate(ranks, k) for k in KS}
        scores["mrr"] = metrics.mean_reciprocal_rank(ranks)
        low, high = metrics.wilson_interval(round(scores["hit@3"] * len(ranks)), len(ranks))
        scores["hit@3_ci95"] = [low, high]
        summary[name] = scores
    return summary


def dense_system(retriever: Retriever, in_scope: pd.DataFrame, out_of_scope: pd.DataFrame) -> dict:
    def search(questions):
        hits = [retriever.search(q, DEPTH) for q in questions]
        return np.array([s[0] for s, _ in hits]), [p for _, p in hits]

    dev_in, test_in = split(in_scope)
    dev_out, test_out = split(out_of_scope)
    dev_in_scores, _ = search(dev_in["question"])
    dev_out_scores, _ = search(dev_out["question"])
    test_in_scores, test_positions = search(test_in["question"])
    test_out_scores, _ = search(test_out["question"])

    threshold = metrics.calibrate_threshold(dev_in_scores, dev_out_scores)
    tuned = metrics.rejection_rates(test_in_scores, test_out_scores, threshold)
    shipped = metrics.rejection_rates(test_in_scores, test_out_scores, config.DEFAULT_THRESHOLD)
    original = metrics.rejection_rates(test_in_scores, test_out_scores, ORIGINAL_THRESHOLD)
    return {
        "retrieval_test": retrieval_summary(test_positions, test_in, retriever.documents),
        "threshold": threshold,
        "test_counts": {"in_scope": len(test_in), "out_of_scope": len(test_out)},
        "tuned": {"in_scope_answered": tuned[0], "out_of_scope_rejected": tuned[1]},
        "at_app_default": {"in_scope_answered": shipped[0], "out_of_scope_rejected": shipped[1]},
        "at_0.50": {"in_scope_answered": original[0], "out_of_scope_rejected": original[1]},
        "score_range_test": {
            "in_scope": [float(test_in_scores.min()), float(test_in_scores.max())],
            "out_of_scope": [float(test_out_scores.min()), float(test_out_scores.max())],
        },
    }


def tfidf_system(kb: pd.DataFrame, in_scope: pd.DataFrame) -> dict:
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform(kb["text"])
    _, test_in = split(in_scope)
    similarity = (vectorizer.transform(test_in["question"]) @ matrix.T).toarray()
    positions = list(np.argsort(-similarity, axis=1)[:, :DEPTH])
    return {"retrieval_test": retrieval_summary(positions, test_in, kb)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passage-embeddings", type=Path, help="embeddings built with a prefix")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "retrieval_eval.json")
    args = parser.parse_args()

    kb = data.load_knowledge_base(ROOT / config.DATA_PATH)
    in_scope = pd.read_csv(ROOT / "evaluation" / "in_scope.csv")
    out_of_scope = pd.read_csv(ROOT / "evaluation" / "out_of_scope.csv")
    known = set(zip(kb["focus"], kb["question_qtype"], strict=True))
    missing = [r for r in in_scope.itertuples() if (r.focus, r.qtype) not in known]
    if missing:
        raise SystemExit(f"Benchmark labels not in the knowledge base: {missing}")

    encoder = SentenceTransformer(config.EMBED_MODEL_NAME)
    systems = {"tfidf_baseline": tfidf_system(kb, in_scope)}
    embedding_files = {"dense_no_prefix": ROOT / config.EMBEDDINGS_CACHE}
    if args.passage_embeddings:
        embedding_files["dense_passage_prefix"] = args.passage_embeddings
    for name, path in embedding_files.items():
        retriever = Retriever(encoder, kb, np.load(path, allow_pickle=False))
        systems[name] = dense_system(retriever, in_scope, out_of_scope)

    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(systems, indent=2) + "\n")
    print(json.dumps(systems, indent=2))


if __name__ == "__main__":
    main()
