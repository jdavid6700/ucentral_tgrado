from __future__ import annotations
import argparse
from pathlib import Path
import yaml

from src.substitution.config import settings
from src.substitution.data.loaders import load_raw_input
from src.substitution.data.preprocess import build_consolidated_model_base, persist_consolidated_outputs
from src.substitution.evaluation.reporting import save_json_report
from src.substitution.data.quality import build_data_quality_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    df_consolidado = load_raw_input(settings.raw_data_dir, config["raw_files"]["consolidado"])
    df_precios = load_raw_input(settings.raw_data_dir, config["raw_files"]["precios"])
    df_catalogo = load_raw_input(settings.raw_data_dir, config["raw_files"]["catalogo"])

    df_full = build_consolidated_model_base(
        df_consolidado=df_consolidado,
        df_precios=df_precios,
        df_catalogo=df_catalogo,
        target_season=config["target_season"],
        historical_line_filter=config.get("historical_line_filter"),
    )
    csv_path, xlsx_path = persist_consolidated_outputs(df_full, settings.processed_data_dir)
    save_json_report(build_data_quality_report(df_full), str(Path(settings.reports_dir) / "data_quality_report.json"))
    print({"csv_path": csv_path, "xlsx_path": xlsx_path, "rows": len(df_full)})


if __name__ == "__main__":
    main()
