from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from src.substitution.features.cleaners import clasificar_binario, add_business_features
from src.substitution.utils.text import normalize_text


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _prepare_catalog_texts(df_catalogo: pd.DataFrame) -> pd.DataFrame:
    out = df_catalogo.copy()
    rename_map = {}
    for col in out.columns:
        if normalize_text(col) == "_SKUREFERENCECODE":
            rename_map[col] = "_SKUReferenceCode"
        if normalize_text(col) == "_PRODUCTDESCRIPTION":
            rename_map[col] = "_ProductDescription"
    out = out.rename(columns=rename_map)
    needed = ["_SKUReferenceCode", "_ProductDescription"]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"El catálogo no contiene columnas requeridas: {missing}")
    out["_SKUReferenceCode"] = out["_SKUReferenceCode"].astype(str).str.replace(".0", "", regex=False).str.strip()
    out["_ProductDescription"] = out["_ProductDescription"].fillna("").astype(str)
    return out[["_SKUReferenceCode", "_ProductDescription"]].drop_duplicates("_SKUReferenceCode")


def _prepare_prices(df_precios: pd.DataFrame) -> pd.DataFrame:
    out = df_precios.copy()
    out["REF. COLOR"] = out["REF. COLOR"].astype(str).str.strip()
    out["PVP CON DCTO LISTA"] = pd.to_numeric(out["PVP CON DCTO LISTA"], errors="coerce")
    out["TOTAL \nINV DISPONIBLE"] = pd.to_numeric(out["TOTAL \nINV DISPONIBLE"], errors="coerce")
    return out


def _build_base(df_base: pd.DataFrame, df_precios: pd.DataFrame, df_textos: pd.DataFrame, target_season: str) -> pd.DataFrame:
    df = df_base.copy()
    vol_cols = [
        "ALTO (sin unidad de medida) NUEVO",
        "ANCHO (sin unidad de medida) NUEVO",
        "PROFUNDO (sin unidad de medida) NUEVO",
    ]
    df = _coerce_numeric(df, vol_cols)
    df["volumen_calculado"] = (
        df["ALTO (sin unidad de medida) NUEVO"]
        * df["ANCHO (sin unidad de medida) NUEVO"]
        * df["PROFUNDO (sin unidad de medida) NUEVO"]
    )

    df["CODIGO PLM + COLOR"] = df["CODIGO PLM + COLOR"].astype(str).str.strip()
    prices = df_precios[["REF. COLOR", "PVP CON DCTO LISTA", "TOTAL \nINV DISPONIBLE"]].drop_duplicates("REF. COLOR")
    df = df.merge(prices, left_on="CODIGO PLM + COLOR", right_on="REF. COLOR", how="left")

    df["CODIGO PLM SKU"] = df["CODIGO PLM SKU"].astype(str).str.replace(".0", "", regex=False).str.strip()
    df = df.merge(df_textos, left_on="CODIGO PLM SKU", right_on="_SKUReferenceCode", how="left")
    df["Descripcion_Web"] = df["_ProductDescription"].fillna("")
    if "DESCRIPCIONES TECNICAS" not in df.columns:
        df["DESCRIPCIONES TECNICAS"] = ""
    df["DESCRIPCIONES TECNICAS"] = df["DESCRIPCIONES TECNICAS"].fillna("")
    df["Texto_Completo_NLP"] = (df["Descripcion_Web"].astype(str) + " " + df["DESCRIPCIONES TECNICAS"].astype(str)).str.strip()

    df["Tiene_PC"] = np.maximum(
        df.get("PC", pd.Series(0, index=df.index)).apply(clasificar_binario),
        df.get('PORTA TABLET (8", 9", 10", 11",12")', pd.Series(0, index=df.index)).apply(clasificar_binario),
    )
    df["Tiene_Ruedas"] = df.get("RUEDAS", pd.Series(0, index=df.index)).apply(clasificar_binario)
    df["Tiene_Organizador"] = df.get("ORGANIZADORES", pd.Series(0, index=df.index)).apply(clasificar_binario)

    df["TEMPORADA PORTAFOLIO"] = df["TEMPORADA PORTAFOLIO"].astype(str).str.strip()
    df["is_current_season"] = (df["TEMPORADA PORTAFOLIO"] == str(target_season)).astype(int)

    df["price"] = pd.to_numeric(df["PVP CON DCTO LISTA"], errors="coerce").fillna(120000.0)
    df["inventory_available"] = pd.to_numeric(df["TOTAL \nINV DISPONIBLE"], errors="coerce").fillna(0.0)
    df["sku"] = df["CODIGO PLM SKU"].astype(str).str.strip()
    df["product_name"] = df["DESCRIPCION DEL ARTICULO"].fillna("").astype(str).str.strip()
    df["business_unit"] = df["UNIDAD DE NEGOCIO"].fillna("").astype(str).str.strip()
    df["line"] = df["LINEA"].fillna("").astype(str).str.strip()
    df["collection"] = df["COLECCION"].fillna("").astype(str).str.strip()
    df["base_color"] = df["BASE COLOR (LA PRIMER LETRA EN MAYUSCULA)"].fillna("").astype(str).str.strip()
    df["pattern_type"] = df["TIPO COLOR SOLIDO/ESTAMPADO/COMBINADO NUEVO"].fillna("").astype(str).str.strip()
    df["size"] = df["TALLA"].fillna("").astype(str).str.strip()
    df["style"] = df["ESTILO DE VIDA"].fillna("").astype(str).str.strip()
    df["audience"] = df["AUDIENCIA"].fillna("").astype(str).str.strip()
    df["gender"] = df["GENERO COLOR"].fillna("").astype(str).str.strip()
    df["category"] = df["CATEGORIA (TRADUCCION A CATEGORIA VTEX)"].fillna("").astype(str).str.strip()
    df["material"] = df.get("COMPOSICIONES ESPAÑOL", "").fillna("").astype(str)
    df["useful_catalogo_digital"] = df.get(
        "USEFUL CATALOGO DIGITAL", pd.Series("", index=df.index)
    ).fillna("").astype(str)
    
    df["useful_impreso"] = df.get(
        "USEFUL IMPRESO (SEPARADOS POR COMAS UNICAMENTE CODIGO) ", pd.Series("", index=df.index)
    ).fillna("").astype(str)
    return df


def build_consolidated_model_base(
    df_consolidado: pd.DataFrame,
    df_precios: pd.DataFrame,
    df_catalogo: pd.DataFrame,
    target_season: str = "252",
    historical_line_filter: str | None = None,
) -> pd.DataFrame:
    df_consolidado = standardize_columns(df_consolidado)
    df_precios = standardize_columns(df_precios)
    df_catalogo = standardize_columns(df_catalogo)

    df_precios = _prepare_prices(df_precios)
    df_textos = _prepare_catalog_texts(df_catalogo)
    df_full = _build_base(df_consolidado, df_precios, df_textos, target_season)
    if historical_line_filter:
        mask_hist = (
            (df_full["TEMPORADA PORTAFOLIO"] != str(target_season))
            & (df_full["LINEA"].astype(str).str.upper() == str(historical_line_filter).upper())
        )
        current = df_full[df_full["TEMPORADA PORTAFOLIO"] == str(target_season)]
        hist = df_full[mask_hist]
        df_full = pd.concat([current, hist], ignore_index=True)
    df_full = add_business_features(df_full)
    df_full["text_blob"] = (
        df_full["product_name"].fillna("")
        + " " + df_full["collection"].fillna("")
        + " " + df_full["base_color"].fillna("")
        + " " + df_full["pattern_type"].fillna("")
        + " " + df_full["style_clean"].fillna("")
        + " " + df_full["Texto_Completo_NLP"].fillna("")
    ).str.strip()
    df_full["model_role"] = np.where(df_full["is_current_season"] == 1, "candidate_current", "historical_source")
    return df_full


def persist_consolidated_outputs(df_full: pd.DataFrame, output_dir: str) -> tuple[str, str]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "base_consolidada_modelado.csv"
    xlsx_path = outdir / "base_consolidada_modelado.xlsx"
    df_full.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df_full.to_excel(xlsx_path, index=False)
    return str(csv_path), str(xlsx_path)
