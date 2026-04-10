from __future__ import annotations
import optuna
from src.substitution.training.pipelines import build_representation_pipeline
from src.substitution.models.vector_ranker import VectorRanker
from src.substitution.evaluation.metrics import evaluate_retrieval


def objective(trial: optuna.Trial, train_df, validation_pairs, config: dict):
    max_components = min(config["model"]["max_svd_components"], max(2, min(len(train_df) - 1, 40)))
    min_components = min(config["model"]["min_svd_components"], max_components)
    n_components = trial.suggest_int("n_components", min_components, max_components)
    max_features = trial.suggest_int("max_features", 500, 2500, step=250)
    metric = trial.suggest_categorical("metric", ["cosine", "euclidean"])
    n_clusters = trial.suggest_int("n_clusters", 4, min(12, max(4, len(train_df) // 20 if len(train_df) >= 20 else 4)))

    pipeline = build_representation_pipeline(
        text_col=config["text_column"],
        categorical_cols=config["categorical_columns"],
        numeric_cols=config["numeric_columns"],
        n_components=n_components,
        max_features=max_features,
    )
    model = VectorRanker(
        representation_pipeline=pipeline,
        metric=metric,
        n_neighbors=config["model"]["n_neighbors"],
        candidate_filters=config["candidate_filters"],
        n_clusters=n_clusters,
    )
    model.fit(train_df)
    if validation_pairs:
        metrics = evaluate_retrieval(model, validation_pairs, top_k=config["model"]["top_k_default"])
        return metrics["precision_at_k"]
    return 0.0
