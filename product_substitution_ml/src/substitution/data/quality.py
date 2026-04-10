from __future__ import annotations
import pandas as pd


def build_data_quality_report(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "distinct_skus": int(df["sku"].nunique()) if "sku" in df.columns else 0,
        "null_sku": int(df["sku"].isna().sum()) if "sku" in df.columns else 0,
        "duplicate_sku_rows": int(df["sku"].duplicated().sum()) if "sku" in df.columns else 0,
        "null_price": int(df["price"].isna().sum()) if "price" in df.columns else 0,
        "non_positive_inventory": int((df.get("inventory_available", 0) <= 0).sum()) if "inventory_available" in df.columns else 0,
        "current_season_rows": int((df.get("is_current_season", 0) == 1).sum()) if "is_current_season" in df.columns else 0,
        "historical_rows": int((df.get("is_current_season", 0) == 0).sum()) if "is_current_season" in df.columns else 0,
    }
