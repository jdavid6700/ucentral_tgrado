from __future__ import annotations
from pathlib import Path
import joblib
import mlflow
import optuna
from src.substitution.logger import get_logger
from src.substitution.training.objectives import objective
from src.substitution.training.pipelines import build_representation_pipeline
from src.substitution.models.vector_ranker import VectorRanker
from src.substitution.evaluation.metrics import evaluate_retrieval
from src.substitution.evaluation.reporting import save_json_report
from src.substitution.mlops.mlflow_utils import configure_mlflow, log_pyfunc_model

logger = get_logger(__name__)


class Trainer:
    def __init__(self, settings, config: dict):
        self.settings = settings
        self.config = config
        configure_mlflow(settings)

    def tune_and_train(self, train_df, validation_pairs, n_trials: int = 10):
        mlflow.set_experiment(self.settings.mlflow_experiment_name)

        if validation_pairs:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda t: objective(t, train_df, validation_pairs, self.config), n_trials=n_trials)
            best_params = study.best_params
        else:
            best_params = {
                "n_components": min(self.config["model"]["n_components_default"], max(2, len(train_df) - 1)),
                "max_features": 2000,
                "metric": self.config["model"]["default_metric"],
                "n_clusters": self.config["model"]["n_clusters_default"],
            }

        with mlflow.start_run(run_name="vector_ranker_best") as run:
            mlflow.log_params(best_params)
            pipeline = build_representation_pipeline(
                text_col=self.config["text_column"],
                categorical_cols=self.config["categorical_columns"],
                numeric_cols=self.config["numeric_columns"],
                n_components=best_params["n_components"],
                max_features=best_params["max_features"],
            )
            model = VectorRanker(
                representation_pipeline=pipeline,
                metric=best_params["metric"],
                n_neighbors=self.config["model"]["n_neighbors"],
                candidate_filters=self.config["candidate_filters"],
                n_clusters=best_params["n_clusters"],
            )
            model.fit(train_df)
            metrics = evaluate_retrieval(model, validation_pairs, top_k=self.config["model"]["top_k_default"]) if validation_pairs else {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "n_eval_cases": 0,
            }
            cluster_distribution = train_df.assign(cluster_label=model.catalog_df["cluster_label"]).groupby("cluster_label").size().to_dict()
            metrics["n_clusters"] = int(model.catalog_df["cluster_label"].nunique())

            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_dict(cluster_distribution, "cluster_distribution.json")

            models_dir = Path(self.settings.models_dir)
            models_dir.mkdir(parents=True, exist_ok=True)
            local_model_path = models_dir / "ranker.joblib"
            joblib.dump(model, local_model_path)
            mlflow.log_artifact(str(local_model_path), artifact_path="local_model")
            log_pyfunc_model(model)

            report_path = Path(self.settings.reports_dir) / "training_summary.json"
            save_json_report({
                "run_id": run.info.run_id,
                "best_params": best_params,
                "metrics": metrics,
                "cluster_distribution": cluster_distribution,
            }, str(report_path))
            return model, best_params, metrics, run.info.run_id
