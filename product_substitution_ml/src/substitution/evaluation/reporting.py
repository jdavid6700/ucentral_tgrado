from __future__ import annotations
from pathlib import Path
import json
import pandas as pd


def save_json_report(payload: dict, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if path_obj.suffix.lower() == ".csv":
        df.to_csv(path_obj, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(path_obj, index=False)
