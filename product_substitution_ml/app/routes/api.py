from flask import Blueprint, jsonify, request
from app.services.recommendation_service import recommend_by_sku

api_bp = Blueprint("api", __name__)


@api_bp.post("/recommend")
def recommend():
    payload = request.get_json(silent=True) or {}
    sku = str(payload.get("sku", "")).strip()
    top_k = int(payload.get("top_k", 3))
    if not sku:
        return jsonify({"error": "sku es obligatorio"}), 400
    try:
        data = recommend_by_sku(sku=sku, top_k=top_k)
        return jsonify({"sku": sku, "top_k": top_k, "recommendations": data})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
