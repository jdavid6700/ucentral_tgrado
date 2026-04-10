from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import joblib
import mlflow.pyfunc
from src.substitution.config import settings


@lru_cache(maxsize=1)
def load_model():
    try:
        model_uri = f"models:/{settings.model_name}/{settings.model_stage}"
        return mlflow.pyfunc.load_model(model_uri)
    except Exception:
        model_path = Path(settings.models_dir) / "ranker.joblib"
        if not model_path.exists():
            raise FileNotFoundError("No se encontró el modelo. Ejecuta primero scripts/train.py")
        return joblib.load(model_path)
