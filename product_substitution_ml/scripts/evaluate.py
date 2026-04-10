from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from src.substitution.config import settings
from src.substitution.evaluation.reporting import save_json_report


def main() -> None:
    model_path = Path(settings.models_dir) / "ranker.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Primero debes entrenar el modelo.")
    model = joblib.load(model_path)
    report = {
        "rows_catalog": int(len(model.catalog_df)),
        "n_clusters": int(model.catalog_df["cluster_label"].nunique()),
        "cluster_distribution": model.catalog_df.groupby("cluster_label").size().to_dict(),
        "current_season_rows": int((model.catalog_df["is_current_season"] == 1).sum()),
        "historical_rows": int((model.catalog_df["is_current_season"] == 0).sum()),
    }
    save_json_report(report, str(Path(settings.reports_dir) / "best_model_report.json"))
    print(report)


if __name__ == "__main__":
    main()
