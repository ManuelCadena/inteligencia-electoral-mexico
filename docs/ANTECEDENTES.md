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

## 2. Estudios empíricos con datos económicos, sociales, textuales y espaciales

La literatura reciente combina cuatro familias de señales: (i) fundamentales económicos y encuestas, (ii) texto de redes sociales, discursos o conferencias de prensa, (iii) estructura geográfica y censal, y (iv) información temporal de campaña.

| Estudio | Modelo | Datos | Resultado destacado |
|---|---|---|---|
| Zolghadr et al. (2018) [6] | ANN y SVR | PIB, desempleo, aprobación presidencial, índices económicos | SVR predijo correctamente elecciones EE.UU. 2004, 2008 y 2012; aprobación fue la variable más significativa. |
| Brito et al. (2020) [7] | ANN | Posts de perfiles oficiales de candidatos + encuestas (Brasil 2018, EE.UU. 2016) | Mejor que encuestas en escenarios con muchos candidatos; robusto contra manipulación de volumen. |
| Liu et al. (2020) [8] | Regresión + sentimiento Twitter | Sentimiento geolocalizado de Twitter + variables económicas a nivel condado (Georgia, EE.UU.) | 81% de accuracy en predicción condado a condado. |
| Li et al. (2019) [9] | Hierarchical GCNN | Datos censales geoespaciales de Australia 2016 | Mejora sobre GLM y GCNN estándar; GCNN captura vecindad y estratos socioeconómicos. |
| Mancilla et al. (2025) [10] | ML + análisis topológico de datos + geostatística | Estratos socioeconómicos, educativos y de edad (Nuevo León 2021) | Predice correctamente al ganador de la gubernatura con modelo geoespacial. |
| Topîrceanu (2020) [11] | Modelo de atenuación temporal (TA) | Series de encuestas 1968-2016 EE.UU. | Error medio 2.8-3.28 pts; mejora 23-37% vs. métodos estadísticos y encuestadores. |
| Kennedy et al. (2017) [12] | Modelos de pooling global | 86 países, >500 elecciones, encuestas 146 rondas | 80-90% de elecciones predichas correctamente; encuestas globales robustas, evidencia débil de indicadores económicos puros. |
| Belcastro et al. (2020) [13] | IOM-NN (FFNN iterativo) | Posts de Twitter en elecciones Italia 2018 y EE.UU. 2016 | Polarización de usuarios cercana a encuestas reales. |
| Vendeville et al. (2020) [14] | Voter model con *stubborn nodes* | Resultados oficiales previos (UK, EE.UU.) | MAE 4.74% en votos populares. |

### Hallazgos transversales

- Los **fundamentales económicos** (PIB, inflación, desempleo) aportan señal, pero la **aprobación presidencial**, las **encuestas** y la **información temporal** suelen dominar.
- El **sentimiento de redes sociales** correlaciona con votos (r ~ 0.68-0.72 en India y EE.UU.), pero no es equivalente a un pronóstico; requiere corrección geográfica y demográfica.
- Los **modelos espaciales** (GCNN, geostatística, TDA) mejoran predicciones subnacionales al aprovechar vecindad y homofilia territorial.
- La **información textual** (discursos, mañaneras, tweets) es más útil para capturar *agenda setting* y *issue ownership* que para estimar directamente intención de voto.

## 3. Referencias citadas

[1] [A Review on Deep Sequential Models for Forecasting Time Series Data](https://consensus.app/papers/details/87d2378b6a7e508281c3c619b2b6f76e/?utm_source=unknown) — D. Ahmed et al., *Appl. Comput. Intell. Soft Comput.*, 2022. DOI: 10.1155/2022/6596397

[2] [Deep Learning for Time Series Forecasting: A Survey](https://consensus.app/papers/details/47d4ecf70fdd5ba98d4b31ca74792da8/?utm_source=unknown) — J. F. Torres et al., *Big Data*, 2020. DOI: 10.1089/big.2020.0159

[3] [EA-LSTM: Evolutionary Attention-based LSTM for Time Series Prediction](https://consensus.app/papers/details/8d1500115d7a56cd9a265b9d4ef27c22/?utm_source=unknown) — You-Ru Li et al., *Knowl. Based Syst.*, 2019. DOI: 10.1016/j.knosys.2019.05.028

[4] [Deep Learning with Long Short-Term Memory for Time Series Prediction](https://consensus.app/papers/details/ae64b4a64e255fa78ea4f6d9a6f0986b/?utm_source=unknown) — Yuxiu Hua et al., *IEEE Communications Magazine*, 2019. DOI: 10.1109/MCOM.2019.1800155

[5] [The Performance of LSTM and BiLSTM in Forecasting Time Series](https://consensus.app/papers/details/10e091085f9a59ba9ad40049a79a38e1/?utm_source=unknown) — Sima Siami-Namini et al., *IEEE Big Data*, 2019. DOI: 10.1109/BigData47090.2019.9005997

[6] [Modeling and forecasting US presidential election using learning algorithms](https://consensus.app/papers/details/f1a11d1e9eb8530daadc9965a188646f/?utm_source=unknown) — Mohammad Zolghadr et al., *Journal of Industrial Engineering International*, 2018. DOI: 10.1007/s40092-017-0238-2

[7] [Predicting Brazilian and U.S. Elections with Machine Learning and Social Media Data](https://consensus.app/papers/details/ad7c0ff9b57e512cb2116e71a0f8ac73/?utm_source=unknown) — Kellyton dos Santos Brito et al., *IJCNN*, 2020. DOI: 10.1109/ijcnn48605.2020.9207147

[8] [Can We Forecast Presidential Election Using Twitter Data? An Integrative Modelling Approach](https://consensus.app/papers/details/a4c03ad0d3fd50ea9eae447480642b32/?utm_source=unknown) — Ruowei Liu et al., *Annals of GIS*, 2020. DOI: 10.1080/19475683.2020.1829704

[9] [Deep Hierarchical Graph Convolution for Election Prediction from Geospatial Census Data](https://consensus.app/papers/details/651f34ee692e5423b20004015e2c0752/?utm_source=unknown) — Mike Li et al., *AAAI*, 2019. DOI: 10.1609/aaai.v33i01.3301647

[10] [Sub-spatial prediction of votes integrating socioeconomic, educational, and age strata with machine learning and topological data analysis](https://consensus.app/papers/details/38d96429ea2b521ca61de5adc2c6e6c1/?utm_source=unknown) — Salvador Mancilla et al., *Journal of Big Data*, 2025. DOI: 10.1186/s40537-025-01112-x

[11] [Electoral Forecasting Using a Novel Temporal Attenuation Model: Predicting the US Presidential Elections](https://consensus.app/papers/details/9827ba6df0af532db93953fbc6debe3b/?utm_source=unknown) — Alexandru Topîrceanu, *Expert Syst. Appl.*, 2021. DOI: 10.1016/j.eswa.2021.115289

[12] [Improving election prediction internationally](https://consensus.app/papers/details/cf5337e3d21f514989e123557a442cab/?utm_source=unknown) — Ryan Kennedy et al., *Science*, 2017. DOI: 10.1126/science.aal2887

[13] [Learning Political Polarization on Social Media Using Neural Networks](https://consensus.app/papers/details/4e23c443a3a75bbba6bebea10e366666a/?utm_source=unknown) — Loris Belcastro et al., *IEEE Access*, 2020. DOI: 10.1109/access.2020.2978950

[14] [Forecasting elections results via the voter model with stubborn nodes](https://consensus.app/papers/details/7c7c9c36f8d25de883ad2bc9d2ceb901/?utm_source=unknown) — Antoine Vendeville et al., *Applied Network Science*, 2020. DOI: 10.1007/s41109-020-00342-7

## 4. Fuentes complementarias no revisadas por pares

- INE — Resultados electorales: https://www.ine.mx/voto-y-elecciones/resultados-electorales/
- SICEE — Sistema de Consulta de la Estadística de las Elecciones: https://sicee.ine.mx/home
- PREP 2024 — Base de datos: https://prep2024.ine.mx/publicacion/nacional/base-datos
- Cómputos Judiciales 2025 — Base de datos: https://computospj2025.ine.mx/base-datos

## 5. Implicaciones para este proyecto

- **Línea base:** un MLP sobre variables tabulares (participación histórica, lista nominal, votos por partido, margen, variables económicas) para validar que la complejidad neuronal aporta valor.
- **Modelo temporal:** LSTM/GRU/BiLSTM por entidad/municipio con ventanas deslizantes; la validación debe ser estrictamente temporal (sin mezclar años en train/test).
- **Texto y NLP:** aplicar BETO/RoBERTa español sobre mañaneras/medios para topic modeling, *stance*, emociones, menciones geográficas y agenda-setting, no solo sentimiento genérico.
- **Espacialidad:** usar GNN (GCN, GraphSAGE, GAT) sobre grafos de municipios/distritos con vecindad geográfica o similitud socioeconómica.
- **Multimodalidad:** combinar MLP tabular + LSTM temporal + BERT textual + GNN espacial con *late fusion*; predecir simultáneamente votos, participación y probabilidad de ganador.
- **Composicionalidad:** emplear softmax/logistic-normal para asegurar que las proporciones de voto sumen 1.
- **Incertidumbre:** reportar intervalos predictivos (MC Dropout, redes bayesianas, ensembles) y métricas de calibración (Brier, coverage).
- **Data leakage:** cortar toda información posterior a la fecha de pronóstico (encuestas, noticias, datos macro revisados, límites geográficos actuales).
