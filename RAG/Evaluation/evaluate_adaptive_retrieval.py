#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive Retrieval Evaluation
"""

import json
import time
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)

def get_retriever():

    from RAG.Retrieval.adaptive_retriever import AdaptiveRetriever

    persist_directory = "RAG/ProcessedDocuments/chroma_db"
    model_name = "all-MiniLM-L6-v2"
    logger.info(f"Loading vector store from {persist_directory}")
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embedding_function)

    return AdaptiveRetriever(
    vectorstore=vectorstore,
    k_init=50,
    pool_cap=20
    )
    raise RuntimeError("Implement get_retriever() to return your retriever instance.")


def load_testset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Test set at {path} is not a list.")
    return data


def doc_to_id(doc: Any) -> str:
    try:
        meta = getattr(doc, "metadata", None)
        if isinstance(meta, dict):
            t = meta.get("type")
            n = meta.get("name")
            if t and n:
                return f"{t}::{n}"
    except Exception:
        pass
    if isinstance(doc, dict):
        t = doc.get("type")
        n = doc.get("name")
        if t and n:
            return f"{t}::{n}"
    return str(doc)


def precision_recall_f1(selected: List[str], gold: List[str]) -> Tuple[float, float, float]:
    sel, gol = set(selected), set(gold)
    tp = len(sel & gol)
    fp = len(sel - gol)
    fn = len(gol - sel)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def mrr(selected: List[str], gold: List[str]) -> float:
    gold_set = set(gold)
    for rank, doc_id in enumerate(selected, start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def extract_score(doc):
    # 1) direct attributes or dict keys commonly seen
    for key in ("prob", "score", "similarity", "sim", "ce_prob", "ce_score"):
        # attribute style
        if hasattr(doc, key):
            try:
                v = getattr(doc, key)
                if isinstance(v, (int, float)):
                    return float(v)
            except Exception:
                pass
        # dict style
        if isinstance(doc, dict) and key in doc:
            try:
                v = doc[key]
                if isinstance(v, (int, float)):
                    return float(v)
            except Exception:
                pass

    # 2) LangChain Document metadata
    meta = getattr(doc, "metadata", None)
    if isinstance(meta, dict):
        for key in ("prob", "score", "similarity", "sim", "ce_prob", "ce_score"):
            if key in meta and isinstance(meta[key], (int, float)):
                return float(meta[key])

    # nothing found
    return None

def evaluate_queries(
    retriever,
    items: List[Dict[str, Any]],
    stage: str = "final",
    fixed_k: Optional[int] = None,
    ood_fp_mode: str = "nonempty",      # "nonempty" | "score"
    ood_min_score: Optional[float] = None,  # used when ood_fp_mode == "score"
):
    results = []

    for ex in items:
        question = ex.get("question", "")
        groundtruth = list(ex.get("ground_truth_docs", []))
        is_ood = len(groundtruth) == 0

        t0 = time.time()
        step1 = retriever.step1_wide_retrieval(question)
        t1 = time.time()
        step2 = retriever.step2_rerank(step1, question)
        t2 = time.time()
        step3 = retriever.step3_adaptive_selection(step2, question)
        t3 = time.time()

        # Choose stage output to evaluate
        if stage == "wide":
            stage_docs = step1
        elif stage == "reranked":
            stage_docs = step2
        else:
            stage_docs = step3

        # Keep selected docs and ids
        selected_docs = list(stage_docs)
        selected_ids = [doc_to_id(d) for d in selected_docs]
        if fixed_k is not None:
            selected_docs = selected_docs[:fixed_k]
            selected_ids = selected_ids[:fixed_k]

        # Standard metrics
        prec, rec, f1 = precision_recall_f1(selected_ids, groundtruth)
        rr = mrr(selected_ids, groundtruth) if groundtruth else 0.0

        gold_set = set(groundtruth)
        ranks_of_matches = [i + 1 for i, sid in enumerate(selected_ids) if sid in gold_set]
        rank_of_first_match = ranks_of_matches[0] if ranks_of_matches else None

        sims = []
        for d in selected_docs:
            s = extract_score(d)  # looks for prob/score/similarity/ce_prob (incl. in metadata)
            if isinstance(s, (int, float)):
                sims.append(float(s))
        avg_similarity = (sum(sims) / len(sims)) if sims else None
        print(f"avg_similarity: {avg_similarity}")

        if is_ood:
            if ood_fp_mode == "nonempty":
                ood_fp = 1 if len(selected_docs) > 0 else 0
            elif ood_fp_mode == "score":
                # consider FP only if any selected doc has score >= threshold
                # (default threshold must be provided by caller)
                passed = False
                if selected_docs and (ood_min_score is not None):
                    for d in selected_docs:
                        sc = extract_score(d)
                        if sc is not None and sc >= ood_min_score:
                            passed = True
                            break
                ood_fp = 1 if passed else 0
            else:
                raise ValueError("ood_fp_mode must be 'nonempty' or 'score'")
        else:
            ood_fp = 0

        results.append({
            "question": question,
            "gold": groundtruth,
            "selected": selected_ids,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "mrr": rr,
            "rank_of_first_match": rank_of_first_match,
            "ranks_of_matches": ranks_of_matches,
            "avg_similarity": avg_similarity,
            "ood_fp": ood_fp,
            "is_ood": int(is_ood),
            "n_selected": len(selected_ids),
            "t_total_ms": (t3 - t0) * 1000,
        })

    return results


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        return {}

    def avg(key: str) -> float:
        return sum(r[key] for r in results) / n

    # OOD-only pool
    ood = [r for r in results if r["is_ood"] == 1]
    if len(ood) > 0:
        ood_fp_rate = sum(r["ood_fp"] for r in ood) / len(ood)
    else:
        ood_fp_rate = 0.0

    ranks = [r["rank_of_first_match"] for r in results if r["rank_of_first_match"] is not None]
    avg_rank_of_first_match = sum(ranks) / len(ranks) if ranks else None
    hit_rate = len(ranks) / n  # fraction of queries with ≥1 correct doc
    sim_values = [r["avg_similarity"] for r in results if r.get("avg_similarity") is not None]
    avg_avg_similarity = (sum(sim_values) / len(sim_values)) if sim_values else None

    return {
        "n": n,
        "precision": avg("precision"),
        "recall": avg("recall"),
        "f1": avg("f1"),
        "mrr": avg("mrr"),
        "avg_rank_of_first_match": avg_rank_of_first_match,
        "hit_rate": hit_rate,
        "avg_similarity": avg_avg_similarity,
        "ood_fp_rate": ood_fp_rate,
        "ood_fp_rate_all": sum(r["ood_fp"] for r in results) / n
    }


def run_evaluation(
    testset_paths: List[str],
    stage: str = "final",
    fixed_k: Optional[int] = None,
    ood_fp_mode: str = "nonempty",
    ood_min_score: Optional[float] = None,
) -> Dict[str, Any]:
    items = []
    for p in testset_paths:
        items.extend(load_testset(Path(p)))

    retriever = get_retriever()
    results = evaluate_queries(
        retriever,
        items,
        stage=stage,
        fixed_k=fixed_k,
        ood_fp_mode=ood_fp_mode,
        ood_min_score=ood_min_score,
    )
    agg = aggregate(results)

    print("\n===== Evaluation Results =====")
    print(
        f"N={agg['n']} | P={agg['precision']:.3f} | R={agg['recall']:.3f} | "
        f"F1={agg['f1']:.3f} | MRR={agg['mrr']:.3f} | "
        f"OOD FP Rate={agg['ood_fp_rate']:.3f}"
    )

    return {"aggregate": agg, "per_query": results}
