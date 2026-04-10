import pandas as pd
import pytest
from src.substitution.training.pipelines import build_representation_pipeline
from src.substitution.models.vector_ranker import VectorRanker


@pytest.fixture
def sample_catalog():
    return pd.DataFrame([
        {
            "CODIGO PLM SKU": "SKU_OLD_1", "DESCRIPCION DEL ARTICULO": "MORRAL CONEL", "UNIDAD DE NEGOCIO": "BOLSOS",
            "LINEA": "MORRAL", "COLECCION": "CONEL", "BASE COLOR (LA PRIMER LETRA EN MAYUSCULA)": "NEGRO",
            "TIPO COLOR SOLIDO/ESTAMPADO/COMBINADO NUEVO": "SOLIDO", "GENERO COLOR": "UNISEX",
            "AUDIENCIA": "ADULTO", "ESTILO DE VIDA": "VIAJE", "Texto_Completo_NLP": "MORRAL VIAJE NEGRO PC",
            "Tiene_PC": 1, "Tiene_Ruedas": 0, "Tiene_Organizador": 1, "price": 150000,
            "volumen_calculado": 30.0, "inventory_available": 0.0, "is_current_season": 0,
            "sku": "SKU_OLD_1", "product_name": "MORRAL CONEL", "business_unit": "BOLSOS", "line": "MORRAL",
            "collection": "CONEL", "base_color": "NEGRO", "pattern_type": "SOLIDO", "style_clean": "VIAJE",
            "genre_clean": "UNISEX", "audience_clean": "ADULTO", "has_pc_text": "TIENE_PC",
            "has_wheels_text": "SIN_RUEDAS", "has_organizer_text": "TIENE_ORGANIZADOR",
            "text_blob": "MORRAL CONEL NEGRO SOLIDO VIAJE MORRAL VIAJE NEGRO PC",
        },
        {
            "CODIGO PLM SKU": "SKU_NEW_1", "DESCRIPCION DEL ARTICULO": "MORRAL CONEL 2.0", "UNIDAD DE NEGOCIO": "BOLSOS",
            "LINEA": "MORRAL", "COLECCION": "CONEL", "BASE COLOR (LA PRIMER LETRA EN MAYUSCULA)": "NEGRO",
            "TIPO COLOR SOLIDO/ESTAMPADO/COMBINADO NUEVO": "SOLIDO", "GENERO COLOR": "UNISEX",
            "AUDIENCIA": "ADULTO", "ESTILO DE VIDA": "VIAJE", "Texto_Completo_NLP": "MORRAL VIAJE NEGRO ORGANIZADOR PC",
            "Tiene_PC": 1, "Tiene_Ruedas": 0, "Tiene_Organizador": 1, "price": 160000,
            "volumen_calculado": 31.0, "inventory_available": 10.0, "is_current_season": 1,
            "sku": "SKU_NEW_1", "product_name": "MORRAL CONEL 2.0", "business_unit": "BOLSOS", "line": "MORRAL",
            "collection": "CONEL", "base_color": "NEGRO", "pattern_type": "SOLIDO", "style_clean": "VIAJE",
            "genre_clean": "UNISEX", "audience_clean": "ADULTO", "has_pc_text": "TIENE_PC",
            "has_wheels_text": "SIN_RUEDAS", "has_organizer_text": "TIENE_ORGANIZADOR",
            "text_blob": "MORRAL CONEL 2.0 NEGRO SOLIDO VIAJE MORRAL VIAJE NEGRO ORGANIZADOR PC",
        },
        {
            "CODIGO PLM SKU": "SKU_NEW_2", "DESCRIPCION DEL ARTICULO": "MORRAL TREK", "UNIDAD DE NEGOCIO": "BOLSOS",
            "LINEA": "MORRAL", "COLECCION": "TREK", "BASE COLOR (LA PRIMER LETRA EN MAYUSCULA)": "AZUL",
            "TIPO COLOR SOLIDO/ESTAMPADO/COMBINADO NUEVO": "COMBINADO", "GENERO COLOR": "UNISEX",
            "AUDIENCIA": "ADULTO", "ESTILO DE VIDA": "VIAJE", "Texto_Completo_NLP": "MORRAL VIAJE AZUL PC",
            "Tiene_PC": 1, "Tiene_Ruedas": 0, "Tiene_Organizador": 1, "price": 158000,
            "volumen_calculado": 29.0, "inventory_available": 8.0, "is_current_season": 1,
            "sku": "SKU_NEW_2", "product_name": "MORRAL TREK", "business_unit": "BOLSOS", "line": "MORRAL",
            "collection": "TREK", "base_color": "AZUL", "pattern_type": "COMBINADO", "style_clean": "VIAJE",
            "genre_clean": "UNISEX", "audience_clean": "ADULTO", "has_pc_text": "TIENE_PC",
            "has_wheels_text": "SIN_RUEDAS", "has_organizer_text": "TIENE_ORGANIZADOR",
            "text_blob": "MORRAL TREK AZUL COMBINADO VIAJE MORRAL VIAJE AZUL PC",
        },
    ])


@pytest.fixture
def fitted_model(sample_catalog):
    pipeline = build_representation_pipeline(
        text_col="text_blob",
        categorical_cols=["line", "business_unit", "collection", "base_color", "pattern_type", "genre_clean", "audience_clean", "style_clean", "has_pc_text", "has_wheels_text", "has_organizer_text"],
        numeric_cols=["price", "volumen_calculado", "inventory_available", "nlp_text_length"],
        n_components=2,
        max_features=50,
    )
    catalog = sample_catalog.copy()
    catalog["nlp_text_length"] = catalog["Texto_Completo_NLP"].str.len().astype(float)
    model = VectorRanker(
        representation_pipeline=pipeline,
        metric="cosine",
        n_neighbors=3,
        candidate_filters={"same_line": True, "same_business_unit": True, "current_season_only": True, "positive_inventory_only": True, "same_has_pc": True, "exclude_same_sku": True, "exclude_same_name": True},
        n_clusters=2,
    )
    return model.fit(catalog)
