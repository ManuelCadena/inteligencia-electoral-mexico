# Antecedentes académicos — Redes neuronales en análisis electoral y series temporales

## 1. Forecasting electoral con deep learning

El análisis de series temporales electorales ha migrado progresivamente de modelos econométricos clásicos (ARIMA, regresión espacial) a arquitecturas neuronales capaces de capturar dependencias de largo plazo y relaciones no lineales. La evidencia empírica sugiere que las **Long Short-Term Memory (LSTM)** y sus variantes bidireccionales (**BiLSTM**) superan consistentemente a ARIMA en precisión de pronóstico cuando se dispone de suficientes observaciones temporales [5].

| Modelo | Ventaja principal | Referencia clave |
|--------|-------------------|------------------|
| LSTM | Captura dependencias de largo plazo en votación histórica | Siami-Namini et al. (2019) [5] |
| BiLSTM | Usa información pasada y futura de la secuencia | Siami-Namini et al. (2019) [5] |
| EA-LSTM | Atención evolutiva sobre sub-ventanas temporales | Li et al. (2019) [3] |
| CNN + LSTM | Captura patrones locales y globales en series | Ahmed et al. (2022) [1] |
| Transformers tabulares | Integran variables heterogéneas (numéricas + categóricas + texto) | Torres et al. (2020) [2] |
| Redes bayesianas | Proporcionan intervalos de predicción calibrados | — |
| GNN | Modelan autocorrelación espacial entre secciones/distritos | — |

## 2. Aplicaciones específicas en procesos electorales

En el contexto latinoamericano y mexicano, la literatura se concentra en:

1. **Predicción de intención de voto** a partir de encuestas y sentimiento en redes sociales (LSTM + embeddings de texto) [2].
2. **Nowcasting de resultados** combinando PREP/cómputos con modelos jerárquicos bayesianos.
3. **Análisis territorial** usando auto-correlación espacial y modelos de efectos mixtos por entidad y sección.
4. **Detección de anomalías** en actas mediante clasificadores neuronales sobre imágenes de escrutinio.

## 3. Referencias citadas

[1] [A Review on Deep Sequential Models for Forecasting Time Series Data](https://consensus.app/papers/details/87d2378b6a7e508281c3c619b2b6f76e/?utm_source=unknown) — D. Ahmed et al., *Appl. Comput. Intell. Soft Comput.*, 2022. DOI: 10.1155/2022/6596397

[2] [Deep Learning for Time Series Forecasting: A Survey](https://consensus.app/papers/details/47d4ecf70fdd5ba98d4b31ca74792da8/?utm_source=unknown) — J. F. Torres et al., *Big Data*, 2020. DOI: 10.1089/big.2020.0159

[3] [EA-LSTM: Evolutionary Attention-based LSTM for Time Series Prediction](https://consensus.app/papers/details/8d1500115d7a56cd9a265b9d4ef27c22/?utm_source=unknown) — You-Ru Li et al., *Knowl. Based Syst.*, 2019. DOI: 10.1016/j.knosys.2019.05.028

[4] [Deep Learning with Long Short-Term Memory for Time Series Prediction](https://consensus.app/papers/details/ae64b4a64e255fa78ea4f6d9a6f0986b/?utm_source=unknown) — Yuxiu Hua et al., *IEEE Communications Magazine*, 2019. DOI: 10.1109/MCOM.2019.1800155

[5] [The Performance of LSTM and BiLSTM in Forecasting Time Series](https://consensus.app/papers/details/10e091085f9a59ba9ad40049a79a38e1/?utm_source=unknown) — Sima Siami-Namini et al., *IEEE Big Data*, 2019. DOI: 10.1109/BigData47090.2019.9005997

## 4. Fuentes complementarias no revisadas por pares

- INE — Resultados electorales: https://www.ine.mx/voto-y-elecciones/resultados-electorales/
- SICEE — Sistema de Consulta de la Estadística de las Elecciones: https://sicee.ine.mx/home
- PREP 2024 — Base de datos: https://prep2024.ine.mx/publicacion/nacional/base-datos
- Cómputos Judiciales 2025 — Base de datos: https://computospj2025.ine.mx/base-datos

## 5. Implicaciones para este proyecto

- **Línea base:** un MLP sobre variables tabulares (participación histórica, lista nominal, votos por partido, margen) para validar que la complejidad neuronal aporta valor.
- **Modelo temporal:** LSTM/BiLSTM por entidad con ventanas deslizantes y validación temporal estricta (sin mezclar años en train/test).
- **Heterogeneidad estructural:** usar embeddings para variables categóricas (partido, entidad, distrito) e incluir texto de nombres de candidatos o actas si se consiguen.
- **Incertidumbre:** reportar intervalos predictivos (MC Dropout o red neuronal bayesiana) y métricas de calibración.
- **Espacialidad:** evaluar GNN o features de vecindad para capturar patrones regionales.
