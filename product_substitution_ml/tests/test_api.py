import joblib
from pathlib import Path
from app import create_app
from src.substitution.config import settings


def test_health_endpoint():
    app = create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_api_recommend_endpoint(fitted_model):
    model_path = Path(settings.models_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_model, model_path / "ranker.joblib")
    app = create_app()
    client = app.test_client()
    resp = client.post("/api/v1/recommend", json={"sku": "SKU_OLD_1", "top_k": 2})
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["sku"] == "SKU_OLD_1"
    assert len(payload["recommendations"]) >= 1
