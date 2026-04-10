from flask import Blueprint, render_template, request
from app.services.recommendation_service import recommend_by_sku

ui_bp = Blueprint("ui", __name__)


@ui_bp.get("/")
def home():
    return render_template("index.html")


@ui_bp.post("/recommend")
def recommend_form():
    sku = request.form.get("sku", "").strip()
    top_k = int(request.form.get("top_k", 3))
    try:
        recommendations = recommend_by_sku(sku, top_k)
        return render_template("result.html", sku=sku, recommendations=recommendations, error=None)
    except Exception as exc:
        return render_template("result.html", sku=sku, recommendations=[], error=str(exc))
