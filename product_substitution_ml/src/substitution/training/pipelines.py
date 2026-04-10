from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.substitution.features.text_vectorizers import SeriesFlattener


def build_representation_pipeline(
    text_col: str,
    categorical_cols: list[str],
    numeric_cols: list[str],
    n_components: int = 20,
    max_features: int = 3000,
    reducer_type: str = "svd",
):
    text_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
        ("flatten", SeriesFlattener()),
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("text", text_pipe, [text_col]),
        ("cat", cat_pipe, categorical_cols),
        ("num", num_pipe, numeric_cols),
    ], remainder="drop")

    if reducer_type == "svd":
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
    elif reducer_type == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        reducer = "passthrough"

    return Pipeline([
        ("preprocessor", preprocessor),
        reducer,
    ])
