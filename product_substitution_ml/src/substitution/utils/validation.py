from __future__ import annotations

REQUIRED_COLUMNS = [
    "sku",
    "product_name",
    "business_unit",
    "line",
    "collection",
    "base_color",
    "pattern_type",
    "GENERO COLOR",
    "AUDIENCIA",
    "ESTILO DE VIDA",
    "Texto_Completo_NLP",
    "price",
    "volumen_calculado",
    "inventory_available",
    "is_current_season",
    "Tiene_PC",
    "Tiene_Ruedas",
    "Tiene_Organizador",
    "text_blob",
]


def validate_catalog_schema(df) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"La base consolidada no contiene columnas requeridas: {missing}")
