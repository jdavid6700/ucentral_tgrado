from __future__ import annotations
from pathlib import Path
import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        engine = "xlrd" if suffix == ".xls" else None
        return pd.read_excel(file_path, engine=engine)
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    raise ValueError(f"Formato no soportado: {suffix}")
