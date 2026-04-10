from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.substitution.config import settings
from src.substitution.evaluation.reporting import save_json_report


def main() -> None:
    base_path = Path(settings.processed_data_dir) / "base_consolidada_modelado.csv"
    if not base_path.exists():
        raise FileNotFoundError("No existe base consolidada. Ejecuta train.py o build_consolidated_base.py")
    df = pd.read_csv(base_path)
    report = {
        "rows": int(len(df)),
        "distinct_lines": sorted(df["LINEA"].dropna().astype(str).unique().tolist())[:50],
        "distinct_business_units": sorted(df["UNIDAD DE NEGOCIO"].dropna().astype(str).unique().tolist())[:50],
        "distinct_clusters_if_trained": "Ver artifacts/reports/best_model_report.json después de evaluate.py",
    }
    save_json_report(report, str(Path(settings.reports_dir) / "dataset_summary.json"))
    print(report)


if __name__ == "__main__":
    main()
