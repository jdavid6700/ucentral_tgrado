from __future__ import annotations
import pandas as pd
from app.services.model_service import load_model


def recommend_by_sku(sku: str, top_k: int = 3) -> list[dict]:
    model = load_model()
    if hasattr(model, "predict") and not hasattr(model, "recommend_by_sku"):
        result = model.predict(pd.DataFrame([{"sku": sku, "top_k": top_k}]))
        return result.to_dict(orient="records")
    results = model.recommend_by_sku(sku=sku, top_k=top_k)
    return [{
        "sku": r.sku,
        "product_name": r.product_name,
        "business_unit": r.business_unit,
        "line": r.line,
        "collection": r.collection,
        "cluster_label": r.cluster_label,
        "distance": r.distance,
        "similarity_score": r.similarity_score,
        "explanation": r.explanation,
    } for r in results]
