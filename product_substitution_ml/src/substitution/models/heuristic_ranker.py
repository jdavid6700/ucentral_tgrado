from __future__ import annotations
from dataclasses import dataclass
import difflib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class HeuristicRecommendation:
    sku: str
    product_name: str
    score: float
    detail: str


class HeuristicRanker:
    def __init__(self):
        self.reglas_precio_linea = {
            "MORRAL": {"inflacion": 0.067, "iqr": 60000},
            "LONCHERA": {"inflacion": 0.125, "iqr": 20000},
        }
        self.reglas_fallback_color = {
            "MORRAL": ["NEGRO", "AZUL", "GRIS"],
            "LONCHERA": ["FUCSIA", "MORADO"],
        }

    @staticmethod
    def _similitud_nlp(texto_base, lista_textos):
        if not texto_base or str(texto_base).strip() == "":
            return [0.0] * len(lista_textos)
        docs = [str(texto_base)] + [str(t) for t in lista_textos]
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(docs)
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        except Exception:
            return [0.0] * len(lista_textos)

    def recommend(self, source_sku: str, df_source_pool: pd.DataFrame, df_candidates: pd.DataFrame, top_k: int = 3) -> list[HeuristicRecommendation]:
        match = df_source_pool[df_source_pool["sku"] == source_sku]
        if match.empty:
            raise ValueError(f"SKU no encontrado: {source_sku}")
        producto_viejo = match.iloc[0].to_dict()

        linea = str(producto_viejo.get("LINEA", "")).upper()
        uen = str(producto_viejo.get("UNIDAD DE NEGOCIO", "")).upper()
        pc = int(producto_viejo.get("Tiene_PC", 0))
        precio_antiguo = float(producto_viejo.get("price", 0) or 0)
        sku_viejo = str(producto_viejo.get("CODIGO PLM SKU", "")).strip()
        nombre_viejo = str(producto_viejo.get("DESCRIPCION DEL ARTICULO", "")).strip().upper()

        candidatos = df_candidates[
            (df_candidates["LINEA"].astype(str).str.upper() == linea)
            & (df_candidates["UNIDAD DE NEGOCIO"].astype(str).str.upper() == uen)
            & (df_candidates["Tiene_PC"] == pc)
            & (df_candidates["CODIGO PLM SKU"].astype(str).str.strip() != sku_viejo)
            & (df_candidates["DESCRIPCION DEL ARTICULO"].astype(str).str.strip().str.upper() != nombre_viejo)
        ].copy()

        if candidatos.empty:
            return []

        if precio_antiguo > 0:
            regla_pr = self.reglas_precio_linea.get(linea, {"inflacion": 0.10, "iqr": 30000})
            limite_inf = (precio_antiguo * (1 + regla_pr["inflacion"])) - (regla_pr["iqr"] / 2)
            limite_sup = (precio_antiguo * (1 + regla_pr["inflacion"])) + (regla_pr["iqr"] / 2)
            tmp = candidatos[(candidatos["price"] >= limite_inf) & (candidatos["price"] <= limite_sup)]
            if not tmp.empty:
                candidatos = tmp.copy()

        similitudes_nlp = self._similitud_nlp(producto_viejo.get("Texto_Completo_NLP", ""), candidatos["Texto_Completo_NLP"].tolist())
        volumen_viejo = float(producto_viejo.get("volumen_calculado", 1) or 1)

        results = []
        for idx, (_, row) in enumerate(candidatos.iterrows()):
            puntos = 0.0
            sku_cand = str(row["CODIGO PLM SKU"]).strip().upper()
            nombre_cand = str(row["DESCRIPCION DEL ARTICULO"]).upper()

            pts_bono = 100.0 if difflib.SequenceMatcher(None, nombre_viejo, nombre_cand).ratio() >= 0.75 else 0.0
            puntos += pts_bono

            useful_digital = str(producto_viejo.get("USEFUL CATALOGO DIGITAL", "")).upper()
            useful_impreso = str(producto_viejo.get("USEFUL IMPRESO (SEPARADOS POR COMAS UNICAMENTE CODIGO) ", "")).upper()
            pts_use = 50.0 if sku_cand in useful_digital or sku_cand in useful_impreso else 0.0
            puntos += pts_use

            pts_nlp = float(similitudes_nlp[idx]) * 20
            puntos += pts_nlp

            dif_vol = abs(float(row["volumen_calculado"] or 0) - volumen_viejo)
            pts_vol = max(0, 1 - (dif_vol / max(volumen_viejo, 1))) * 15
            puntos += pts_vol

            pts_rue = 15.0 if int(row["Tiene_Ruedas"]) == int(producto_viejo.get("Tiene_Ruedas", 0)) else 0.0
            puntos += pts_rue

            color_cand = str(row["BASE COLOR (LA PRIMER LETRA EN MAYUSCULA)"]).upper()
            color_old = str(producto_viejo.get("BASE COLOR (LA PRIMER LETRA EN MAYUSCULA)", "")).upper()
            if color_cand == color_old:
                pts_col = 10.0
            elif color_cand in self.reglas_fallback_color.get(linea, ["NEGRO", "AZUL", "GRIS"]):
                pts_col = 5.0
            else:
                pts_col = 0.0
            puntos += pts_col

            pts_estmp = 10.0 if str(row["TIPO COLOR SOLIDO/ESTAMPADO/COMBINADO NUEVO"]).upper() == str(producto_viejo.get("TIPO COLOR SOLIDO/ESTAMPADO/COMBINADO NUEVO", "")).upper() else 0.0
            puntos += pts_estmp

            pts_estilo = 10.0 if str(row["style_clean"]) == str(producto_viejo.get("style_clean", "")) else 0.0
            puntos += pts_estilo

            pts_ctx = 0.0
            if str(row["genre_clean"]) == str(producto_viejo.get("genre_clean", "")):
                pts_ctx += 5.0
            elif str(row["genre_clean"]) == "UNISEX":
                pts_ctx += 2.0
            if str(row["audience_clean"]) == str(producto_viejo.get("audience_clean", "")):
                pts_ctx += 5.0
            puntos += pts_ctx

            pts_colec = 5.0 if str(row["COLECCION"]).strip().upper() == str(producto_viejo.get("COLECCION", "")).strip().upper() else 0.0
            puntos += pts_colec

            pts_org = 5.0 if int(row["Tiene_Organizador"]) == int(producto_viejo.get("Tiene_Organizador", 0)) else 0.0
            puntos += pts_org

            detail = f"Bonos:[{round(pts_bono + pts_use, 0)}] | NLP:{round(pts_nlp,1)} Vol:{round(pts_vol,1)} Rue:{pts_rue} Col:{pts_col} Estm:{pts_estmp} EstL:{pts_estilo} Ctx:{pts_ctx} Colc:{pts_colec} Org:{pts_org}"
            results.append(HeuristicRecommendation(
                sku=str(row["CODIGO PLM SKU"]),
                product_name=str(row["DESCRIPCION DEL ARTICULO"]),
                score=round(puntos, 2),
                detail=detail,
            ))
        return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
