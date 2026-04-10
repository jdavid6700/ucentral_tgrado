from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import pandas as pd
import time 

from src.substitution.config import settings
from src.substitution.data.loaders import load_raw_input
from src.substitution.data.preprocess import build_consolidated_model_base, persist_consolidated_outputs
from src.substitution.data.quality import build_data_quality_report
from src.substitution.evaluation.reporting import save_json_report, save_dataframe
from src.substitution.training.trainer import Trainer
from src.substitution.utils.validation import validate_catalog_schema
from src.substitution.models.heuristic_ranker import HeuristicRanker


def load_validation_pairs(path: str | None) -> list[dict]:
    if not path:
        return []
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    return df.to_dict(orient="records")


def main() -> None:
    print("\n" + "="*50)
    print(" INICIANDO PIPELINE DE ENTRENAMIENTO MLOPS")
    print("="*50 + "\n")
    start_time = time.time()

    # --- PASO 1: Configuración ---
    print(" [1/9] Leyendo argumentos y configuración...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--validation", required=False)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    print(" [1/9] Configuración cargada correctamente.\n")

    # --- PASO 2: Carga de Datos ---
    print(" [2/9] Cargando datos crudos desde Excel/CSV...")
    df_consolidado = load_raw_input(settings.raw_data_dir, config["raw_files"]["consolidado"])
    df_precios = load_raw_input(settings.raw_data_dir, config["raw_files"]["precios"])
    df_catalogo = load_raw_input(settings.raw_data_dir, config["raw_files"]["catalogo"])
    print(f" [2/9] Datos cargados: Consolidado ({len(df_consolidado)} filas), Precios ({len(df_precios)} filas), Catálogo ({len(df_catalogo)} filas).\n")

    # --- PASO 3: Preprocesamiento ---
    print(" [3/9] Construyendo base consolidada para modelado...")
    df = build_consolidated_model_base(
        df_consolidado=df_consolidado,
        df_precios=df_precios,
        df_catalogo=df_catalogo,
        target_season=config["target_season"],
        historical_line_filter=config.get("historical_line_filter"),
    )
    print(f" [3/9] Base consolidada creada con {len(df)} registros.\n")

    # --- PASO 4: Validación y Guardado ---
    print(" [4/9] Validando esquema y guardando archivos...")
    validate_catalog_schema(df)
    csv_path, xlsx_path = persist_consolidated_outputs(df, settings.processed_data_dir)
    print(" [4/9] Archivos guardados en data/processed/.\n")

    # --- PASO 5: Calidad de Datos ---
    print(" [5/9] Generando reporte de calidad de datos...")
    quality_report = build_data_quality_report(df)
    save_json_report(quality_report, str(Path(settings.reports_dir) / "data_quality_report.json"))
    print(" [5/9] Reporte de calidad guardado en formato JSON.\n")

    # --- PASO 6: Entrenamiento y MLflow ---
    print(" [6/9] Cargando pares de validación y entrenando el modelo...")
    validation_pairs = load_validation_pairs(args.validation)
    if not validation_pairs:
        print("⚠️  Aviso: No se proporcionaron pares de validación. Las métricas serán 0.0.")
    
    trainer = Trainer(settings, config)
    print(f"   ➤ Iniciando optimización con {args.trials} trials. Esto puede tomar unos minutos...")
    model, best_params, metrics, run_id = trainer.tune_and_train(df, validation_pairs, n_trials=args.trials)
    print(" [6/9] Entrenamiento finalizado y registrado en MLflow.\n")

    # --- PASO 7: Muestra Heurística ---
    print(" [7/9] Generando muestra del modelo heurístico...")
    source_pool = df[df["is_current_season"] == 0].copy()
    candidate_pool = df[df["is_current_season"] == 1].copy()
    heur = HeuristicRanker()
    heuristic_rows = []
    for sku in source_pool["sku"].head(20):
        recs = heur.recommend(sku, source_pool, candidate_pool, top_k=3)
        for i, rec in enumerate(recs, start=1):
            heuristic_rows.append({
                "source_sku": sku, "rank": i, "candidate_sku": rec.sku,
                "candidate_name": rec.product_name, "score": rec.score, "detail": rec.detail,
            })
    if heuristic_rows:
        save_dataframe(pd.DataFrame(heuristic_rows), str(Path(settings.reports_dir) / "heuristic_sample.csv"))
    print(" [7/9] Muestra heurística guardada.\n")

    # --- PASO 8: Muestra Modelo Vectorial ---
    print(" [8/9] Generando predicciones de muestra con el modelo final...")
    sample_rows = []
    for sku in source_pool["sku"].head(20):
        try:
            recs = model.recommend_by_sku(sku, top_k=config["model"]["top_k_default"])
            for i, rec in enumerate(recs, start=1):
                sample_rows.append({
                    "source_sku": sku, "rank": i, "candidate_sku": rec.sku,
                    "candidate_name": rec.product_name, "cluster_label": rec.cluster_label,
                    "distance": rec.distance, "similarity_score": rec.similarity_score,
                    "explanation": str(rec.explanation),
                })
        except Exception as e:
            print(f"   ➤ Error prediciendo para SKU {sku}: {e}")
            continue
    if sample_rows:
        save_dataframe(pd.DataFrame(sample_rows), str(Path(settings.reports_dir) / "vector_model_sample.csv"))
    print(" [8/9] Muestra del modelo guardada.\n")

    # --- PASO 9: Resumen Final ---
    print(" [9/9] ¡PROCESO COMPLETADO EXITOSAMENTE! ")
    elapsed_time = time.time() - start_time
    print(f"⏱️  Tiempo total: {elapsed_time/60:.2f} minutos")
    print("\n RESULTADOS DEL MODELO:")
    print(f" ➤ MLflow Run ID: {run_id}")
    print(f" ➤ Mejores Hiperparámetros: {best_params}")
    print(f" ➤ Métricas: {metrics}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()