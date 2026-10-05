# Metodología — Inteligencia Electoral México

## 1. Pregunta de investigación

¿Es posible predecir y explicar la distribución territorial del voto en México mediante modelos de deep learning entrenados con resultados electorales históricos oficiales?

Subpreguntas:

1. ¿Qué arquitectura (LSTM, Transformer tabular, MLP, GNN) ofrece mejor desempeño en forecasting por entidad?
2. ¿Cómo manejar el cambio de coaliciones y reseccionamiento para construir series temporales comparables?
3. ¿Qué variables (participación, lista nominal, volatilidad, margen) aportan más a la predicción?
4. ¿Es útil incorporar texto de actas/comunicados mediante embeddings de lenguaje?

## 2. Pipeline de datos

```
┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ SICEE / PREP │──▶│  download_sicee  │──▶│ raw/*.zip       │
└──────────────┘   └──────────────────┘   └────────┬────────┘
                                                   │
                          ┌──────────────────┐      │
                          │  build_series    │◀─────┘
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                          │ processed/series │
                          └────────┬─────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
   ┌────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ EDA / NLP      │   │ Modelos LSTM/MLP │   │ Evaluación y     │
   │ notebooks      │   │ notebooks        │   │ reportes         │
   └────────────────┘   └──────────────────┘   └──────────────────┘
```

## 3. Definición de variables

### Variables dependientes (a predecir)

- `VOTOS_NORMALIZADOS_ENTIDAD` = votos del partido/candidato en la entidad / total de votos emitidos en la entidad.
- `PARTICIPACION_ENTIDAD` = total votos emitidos / lista nominal.
- `MARGEN_VICTORIA` = diferencia porcentual entre primer y segundo lugar.
- `ALTERNANCIA` = variable binaria que indica si ganó un partido distinto al anterior.

### Variables independientes

- Rezagos temporales: resultados de los últimos 1-3 procesos en la misma unidad.
- Variables estructurales: lista nominal, secciones, casillas.
- Variables de competitividad: número de partidos, dispersión del voto.
- Variables temporales: año, tipo de elección (presidencial, intermedia, extraordinaria).
- Variables espaciales: vecindad con otras entidades/distritos.
- Variables textuales (futuro): embeddings de comunicados, nombres de candidatos, actas.

## 4. División train/validation/test

Dado que los datos son una serie temporal, **no se usa partición aleatoria**. Se emplea validación temporal:

- **Train:** procesos 1994-2012.
- **Validation:** procesos 2015-2018.
- **Test:** procesos 2021-2024.

Para modelos por entidad, la unidad de predicción es (año, entidad) y la ventana de entrada son los procesos anteriores.

## 5. Modelos candidatos

| Modelo | Librería sugerida | Input | Caso de uso |
|--------|-------------------|-------|-------------|
| MLP | scikit-learn / Keras | Tabla plana (features por entidad) | Línea base |
| LSTM | Keras / PyTorch | Serie temporal por entidad | Dependencias de largo plazo |
| BiLSTM | Keras / PyTorch | Serie temporal | Uso de información futura en secuencia |
| Transformer tabular | PyTorch Tabular / Keras | Tabla heterogénea (num+cat+embed) | Interacciones complejas |
| GNN | PyTorch Geometric / DGL | Grafo de entidades/distritos | Autocorrelación espacial |
| BNN | TensorFlow Probability / Pyro | Serie/tabular | Intervalos de incertidumbre |

## 6. Métricas de evaluación

- **Regresión:** RMSE, MAE, MAPE, R².
- **Clasificación (alternancia/ganador):** accuracy, F1, log-loss.
- **Calibración:** coverage de intervalos predictivos, Brier score.
- **Espacial:** Moran's I de residuos, mapas de error por entidad.

## 7. Riesgos metodológicos y mitigaciones

| Riesgo | Mitigación |
|--------|-----------|
| Coaliciones cambiantes | Conservar tres vistas: partido, coalición-proceso, candidatura. No comparar siglas entre procesos sin validación histórica. |
| Reseccionamiento | Agregar a entidad/municipio como unidad estable; documentar cambios de secciones. |
| Lista nominal variable | Usar tasas (participación, voto normalizado) en vez de conteos absolutos. |
| Data leakage | Validación temporal; no incluir información posterior a la fecha de predicción. |
| Desbalance territorial | Ponderar por lista nominal o usar métricas nivel entidad. |
| PREP ≠ definitivo | Usar SICEE/cómputos para entrenamiento; PREP solo para nowcasting. |

## 8. Reproducibilidad

- `requirements.txt` fija versiones mínimas.
- Scripts son deterministas salvo por orden de descarga concurrente.
- Semillas aleatorias fijas en notebooks (`random_state=42`).
- Catálogo `data/raw/catalog_*.csv` registra URL, timestamp y estado de cada archivo.

## 9. Próximos pasos sugeridos

1. Completar descarga federal y local.
2. Generar series normalizadas (`build_series.py`).
3. Entrenar MLP de línea base y evaluar.
4. Experimentar con LSTM por entidad y comparar contra ARIMA.
5. Incorporar features socioeconómicas (INEGI) y espaciales.
