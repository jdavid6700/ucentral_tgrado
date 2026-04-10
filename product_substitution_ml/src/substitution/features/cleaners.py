from __future__ import annotations
import pandas as pd
from src.substitution.utils.text import normalize_text


def clasificar_binario(x) -> int:
    if pd.isna(x):
        return 0
    v = normalize_text(x)
    return 0 if v in ["NO APLICA", "NO", "0", "0 X 0 X 0", "NAN", "SIN RUEDAS", "FALSE", ""] else 1


def clean_gender(v) -> str:
    if pd.isna(v):
        return "UNISEX"
    v = normalize_text(v)
    if v in ["MASCULINO", "HOMBRE"]:
        return "HOMBRE"
    if v in ["SIN GENERO", "SIN GENERO ", "HOMBRE - MUJER", "UNISEX"]:
        return "UNISEX"
    if v in ["HEMBRA", "MACHO", "MACHO - HEMBRA"]:
        return "MASCOTA"
    if v in ["MUJER", "NINA", "NINO"]:
        return v
    return "UNISEX"


def clean_audience(v) -> str:
    if pd.isna(v) or normalize_text(v) == "NO APLICA":
        return "GENERAL"
    v = normalize_text(v)
    if v in ["ADULTO", "YOUNG ADULT", "16+", "18 A 25 ANOS", "+ 26 ANOS"]:
        return "ADULTO"
    if v in ["INFANTIL", "PREESCOLAR", "7 A 10 ANOS", "4 A 6 ANOS"]:
        return "INFANTIL"
    if v in ["13 A 17 ANOS", "TEENS"]:
        return "TEENS"
    if v in ["PERRO", "GATO", "PERRO-GATO"]:
        return "MASCOTA"
    return "GENERAL"


def clean_style(v) -> str:
    if pd.isna(v) or normalize_text(v) == "NO APLICA":
        return "GENERAL"
    v = normalize_text(v)
    if v in ["CASUAL", "NEW CASUAL", "MODA"]:
        return "CASUAL"
    if v in ["ESTUDIO"]:
        return "ESTUDIO"
    if v in ["VIAJE", "TRAVEL"]:
        return "VIAJE"
    if v in ["KIDS", "INFANTIL", "LICENCIAS", "LICENCIA", "JUGUETES"]:
        return "KIDS"
    if v in ["PRO", "PERFORMANCE"]:
        return "PRO"
    if v in ["PASEO", "OUTDOOR", "DEPORTIVO"]:
        return "OUTDOOR"
    return "GENERAL"


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["genre_clean"] = out["GENERO COLOR"].apply(clean_gender)
    out["audience_clean"] = out["AUDIENCIA"].apply(clean_audience)
    out["style_clean"] = out["ESTILO DE VIDA"].apply(clean_style)
    out["has_pc_text"] = out["Tiene_PC"].astype(int).map({1: "TIENE_PC", 0: "SIN_PC"})
    out["has_wheels_text"] = out["Tiene_Ruedas"].astype(int).map({1: "TIENE_RUEDAS", 0: "SIN_RUEDAS"})
    out["has_organizer_text"] = out["Tiene_Organizador"].astype(int).map({1: "TIENE_ORGANIZADOR", 0: "SIN_ORGANIZADOR"})
    out["nlp_text_length"] = out["Texto_Completo_NLP"].fillna("").astype(str).str.len().astype(float)
    return out
