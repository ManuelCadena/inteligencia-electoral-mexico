# Síntesis del plan de investigación generado por AION Brain

AION Brain (modelo `claude-sonnet-5`, ruta `API-002`) recibió la siguiente tarea:

> Diseñar un plan de investigación académico para un proyecto de NLP y Deep Learning llamado "Inteligencia Electoral Mexico" que descargue datos electorales históricos de México (INE SICEE 1991-2024, locales 2015-2024, PREP, cómputos judiciales 2025) y entrene modelos de redes neuronales para predecir/segmentar resultados por entidad/municipio.

## 1. Objetivos priorizados

1. **OE1 — Repositorio estructurado y reproducible** con datos federales y locales.
2. **OE2 — Pipeline ETL automatizado** para descargar, limpiar y normalizar datos heterogéneos (CSV, XLS, PDF, JSON, API).
3. **OE3 — Entrenar y comparar** LSTM, Transformer tabular (TabTransformer / FT-Transformer) y MLP profundo.
4. **OE4 — Segmentación geopolítica** mediante autoencoder + clustering por municipio y sección.
5. **OE5 — Auditoría de sesgos** y limitaciones metodológicas.
6. **OE6 — Publicación** de datasets anonimizados y notebooks reproducibles.

## 2. Hipótesis centrales

- **H0:** Los resultados electorales municipales siguen patrones temporales y espaciales suficientemente estructurados para ser modelados con **RMSE < 5%** de participación relativa.
- **H1:** Los Transformers tabulares superan a LSTM y MLP en predicción multiclase del partido ganador.
- **H2:** Existe autocorrelación espacial significativa (**Moran's I > 0.3**) en votaciones municipales.
- **H3:** La volatilidad electoral es predecible con **AUC > 0.75** usando variables de contexto (ENIGH/CONAPO).
- **H4:** Variables textuales de redes sociales y noticias mejoran ≥ 3% en F1-macro para elecciones presidenciales.
- **H5:** Transfer learning de modelos federales a locales tiene degradación < 8%.

## 3. Arquitectura de carpetas recomendada

AION propuso una estructura más granular de la implementada en este repo:

```
data/raw/{federal,local,contexto,nlp}
data/processed/{electoral_federal_clean,features_municipio,series_temporales,embeddings}
data/interim/{unified_schema,geocoded,imputed}
src/{ingestion,etl,models,nlp,evaluation,visualization}
notebooks/01_eda_federal ... 08_resultados_finales
configs/{model_config.yaml,data_config.yaml,experiment_config.yaml}
tests/
```

La versión actual del repositorio es una implementación funcional mínima de esta arquitectura; puede extenderse a la estructura completa conforme crezca el proyecto.

## 4. Modelos y métricas recomendadas

| Modelo | Uso | Métricas |
|--------|-----|----------|
| LSTM / GRU | Series temporales de votación | RMSE, MAE, MAPE |
| Transformer tabular | Tablas heterogéneas (numéricas + categóricas) | Accuracy, F1-macro, log-loss |
| MLP profundo | Línea base tabular | RMSE, R² |
| Autoencoder + K-Means | Segmentación municipal | Silhouette, Davies-Bouldin |
| GNN (Graph Neural Network) | Autocorrelación espacial | Moran's I de residuos |
| BETO / RoBERTa-BNE | Sentimiento y tópicos (NLP) | F1-macro |

## 5. Riesgos metodológicos enfatizados por AION

- **Coaliciones cambiantes:** una sigla no representa la misma alianza en distintos procesos.
- **Reseccionamiento y redistritación:** mismos números de sección/distrito pueden cambiar de territorio.
- **Lista nominal dinámica:** usar tasas en lugar de conteos absolutos.
- **Data leakage:** validación temporal estricta; no incluir información futura.
- **Desbalance territorial:** ponderar por lista nominal o usar métricas por entidad.

## 6. Costo estimado del análisis AION

- **Modelo seleccionado:** `claude-sonnet-5`
- **Latencia total:** ~192 s
- **Costo:** USD 0.034974

## 7. Próximos pasos sugeridos por AION

1. Completar ingesta federal y local.
2. Construir schema unificado con catálogo normalizado de partidos.
3. Entrenar MLP de línea base y LSTM por entidad.
4. Incorporar variables de contexto (INEGI/CONAPO) y espaciales.
5. Ejecutar auditoría de sesgos y calibración de intervalos.
6. Publicar modelos y dataset anonimizado.
