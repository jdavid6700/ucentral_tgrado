from __future__ import annotations
from sklearn.base import BaseEstimator, TransformerMixin


class SeriesFlattener(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "iloc"):
            return X.iloc[:, 0].fillna("").astype(str).tolist()
        return [str(x) for x in X]
