from __future__ import annotations
import numpy as np


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / len(relevant)


def reciprocal_rank(recommended: list[str], relevant: set[str]) -> float:
    for idx, sku in enumerate(recommended, start=1):
        if sku in relevant:
            return 1.0 / idx
    return 0.0


def evaluate_retrieval(model, validation_pairs: list[dict], top_k: int = 3) -> dict:
    p_vals, r_vals, rr_vals = [], [], []
    for row in validation_pairs:
        source_sku = row["source_sku"]
        rel = row["relevant_skus"]
        relevant_skus = set(rel if isinstance(rel, list) else str(rel).split("|"))
        preds = model.recommend_by_sku(source_sku, top_k=top_k)
        pred_skus = [p.sku for p in preds]
        p_vals.append(precision_at_k(pred_skus, relevant_skus, top_k))
        r_vals.append(recall_at_k(pred_skus, relevant_skus, top_k))
        rr_vals.append(reciprocal_rank(pred_skus, relevant_skus))
    return {
        "precision_at_k": float(np.mean(p_vals)) if p_vals else 0.0,
        "recall_at_k": float(np.mean(r_vals)) if r_vals else 0.0,
        "mrr": float(np.mean(rr_vals)) if rr_vals else 0.0,
        "n_eval_cases": len(validation_pairs),
    }
