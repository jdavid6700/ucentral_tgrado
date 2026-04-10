# Product Substitution ML

Proyecto profesional para:
1. consolidar automáticamente `Matriz_Consolidado.xlsx`, `Lista_precio_252.xlsx` y `Catalogo.xls`
2. construir una base única de modelado
3. entrenar un modelo no supervisado con representación vectorial, reducción de dimensionalidad, vecinos cercanos y clusters
4. exponer recomendaciones mediante Flask

## Ejecución rápida

```bash
conda create -n ucentral_tgrado python=3.12.13 -y
conda activate ucentral_tgrado
pip install -r requirements.txt
copy .env.example .env
python scripts/train.py
python run.py
```

## Ubicación esperada de insumos

Coloca estos archivos en `data/raw/`:
- `Matriz_Consolidado.xlsx`
- `Lista_precio_252.xlsx`
- `Catalogo.xls`

## Artefactos generados

- `data/processed/base_consolidada_modelado.csv`
- `data/processed/base_consolidada_modelado.xlsx`
- `artifacts/models/ranker.joblib`
- `artifacts/reports/*.json`
