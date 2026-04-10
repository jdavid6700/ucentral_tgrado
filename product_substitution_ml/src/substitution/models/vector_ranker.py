from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from src.substitution.exceptions import ModelNotFittedError


@dataclass
class RankedNeighbor:
    sku: str
    product_name: str
    business_unit: str
    line: str
    collection: str
    cluster_label: int
    distance: float
    similarity_score: float
    explanation: dict


class VectorRanker:
    def __init__(
        self,
        representation_pipeline,
        metric: str = "cosine",
        n_neighbors: int = 15,
        candidate_filters: dict | None = None,
        n_clusters: int = 8,
    ):
        self.representation_pipeline = representation_pipeline
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.candidate_filters = candidate_filters or {}
        self.n_clusters = n_clusters
        self.nn = NearestNeighbors(metric=metric, n_neighbors=n_neighbors)
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.catalog_df: pd.DataFrame | None = None
        self.embeddings = None

    def fit(self, df: pd.DataFrame) -> "VectorRanker":
        self.catalog_df = df.reset_index(drop=True).copy()
        self.embeddings = self.representation_pipeline.fit_transform(self.catalog_df)
        if len(self.catalog_df) >= self.n_clusters:
            self.catalog_df["cluster_label"] = self.clusterer.fit_predict(self.embeddings)
        else:
            self.catalog_df["cluster_label"] = 0
        self.nn.fit(self.embeddings)
        return self

    def _ensure_fitted(self) -> None:
        if self.catalog_df is None or self.embeddings is None:
            raise ModelNotFittedError("El modelo debe entrenarse antes de predecir")

    def _similarity_from_distance(self, distance: float) -> float:
        if self.metric == "cosine":
            return max(0.0, 1.0 - float(distance))
        return 1.0 / (1.0 + float(distance))

    def _filter_candidates(self, source_row: pd.Series, indices: list[int]) -> list[int]:
        filtered = []
        for idx in indices:
            candidate = self.catalog_df.iloc[idx]
            if self.candidate_filters.get("exclude_same_sku", True) and candidate["sku"] == source_row["sku"]:
                continue
            if self.candidate_filters.get("same_line", False) and candidate["line"] != source_row["line"]:
                continue
            if self.candidate_filters.get("same_business_unit", False) and candidate["business_unit"] != source_row["business_unit"]:
                continue
            if self.candidate_filters.get("current_season_only", False) and int(candidate.get("is_current_season", 0)) != 1:
                continue
            if self.candidate_filters.get("positive_inventory_only", False) and float(candidate.get("inventory_available", 0.0)) <= 0:
                continue
            if self.candidate_filters.get("same_has_pc", False) and int(candidate.get("Tiene_PC", 0)) != int(source_row.get("Tiene_PC", 0)):
                continue
            if self.candidate_filters.get("exclude_same_name", False) and str(candidate.get("product_name", "")).strip().upper() == str(source_row.get("product_name", "")).strip().upper():
                continue
            filtered.append(idx)
        return filtered

    def _build_explanation(self, source: pd.Series, candidate: pd.Series) -> dict:
        return {
            "same_line": int(source.get("line", "") == candidate.get("line", "")),
            "same_business_unit": int(source.get("business_unit", "") == candidate.get("business_unit", "")),
            "same_collection": int(source.get("collection", "") == candidate.get("collection", "")),
            "same_base_color": int(source.get("base_color", "") == candidate.get("base_color", "")),
            "same_pattern_type": int(source.get("pattern_type", "") == candidate.get("pattern_type", "")),
            "same_has_pc": int(int(source.get("Tiene_PC", 0)) == int(candidate.get("Tiene_PC", 0))),
            "candidate_inventory": float(candidate.get("inventory_available", 0.0)),
            "candidate_cluster": int(candidate.get("cluster_label", 0)),
        }

    def recommend_by_sku(self, sku: str, top_k: int = 3) -> list[RankedNeighbor]:
        self._ensure_fitted()
        matches = self.catalog_df.index[self.catalog_df["sku"] == sku].tolist()
        if not matches:
            raise ValueError(f"SKU no encontrado: {sku}")
        src_idx = matches[0]
        source_row = self.catalog_df.iloc[src_idx]
        n_query = min(len(self.catalog_df), max(self.n_neighbors, top_k + 10))
        distances, indices = self.nn.kneighbors(self.embeddings[src_idx: src_idx + 1], n_neighbors=n_query)

        idx_list = indices[0].tolist()
        dist_list = distances[0].tolist()
        filtered_indices = self._filter_candidates(source_row, idx_list)

        results = []
        for idx in filtered_indices[:top_k]:
            candidate = self.catalog_df.iloc[idx]
            dist = float(dist_list[idx_list.index(idx)])
            results.append(RankedNeighbor(
                sku=str(candidate["sku"]),
                product_name=str(candidate["product_name"]),
                business_unit=str(candidate["business_unit"]),
                line=str(candidate["line"]),
                collection=str(candidate["collection"]),
                cluster_label=int(candidate.get("cluster_label", 0)),
                distance=round(dist, 6),
                similarity_score=round(self._similarity_from_distance(dist), 6),
                explanation=self._build_explanation(source_row, candidate),
            ))
        return results
