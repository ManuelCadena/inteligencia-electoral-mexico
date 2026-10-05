# Oportunidades de modelado — Redes neuronales para elecciones mexicanas

Este documento traduce los antecedentes académicos (Consensus, Perplexity,
AION Brain) y el corpus del curso (Advanced Deep Learning, Deep Learning, NLP)
en una hoja de ruta concreta para construir modelos con los datasets del repo.

## 1. Preguntas de investigación operativas

1. ¿Se puede predecir la **proporción de voto por partido/candidato** a nivel
   entidad, distrito o municipio en una elección futura usando series temporales?
2. ¿Mejora la predicción al incorporar **variables económicas, discurso
   presidencial (mañaneras) y estructura geográfica**?
3. ¿Qué canal es más informativo: atención mediática, aprobación, bienestar
   económico o vecindad espacial?
4. ¿Es posible estimar **participación, abstención y probabilidad de
   alternancia** con incertidumbre calibrada?

## 2. Tipos de datos disponibles

| Fuente | Forma | Rango | Uso principal |
|---|---|---|---|
| SICEE federal | Panel (año, entidad, partido) | 1991-2024 | Votos, lista nominal, participación, margen |
| SICEE local/estatal/municipal | Panel (año, estado, municipio/distrito, partido) | 2015-2024 | Votos subnacionales, gobernador, diputados locales, ayuntamientos |
| ENCO/ICC | Serie mensual | 2001-2026 | Confianza del consumidor |
| Macroeconómicas | Mensual/anual | 1969-2026 | Desempleo, inflación, tipo de cambio, remesas |
| Remesas por entidad | Trimestral | 2003-2026 | Flujos por estado |
| Mañaneras (proyecto previo) | Corpus diario de textos | 2018-2025 | Tópicos, sentimiento/stance, atención |
| Google Trends / encuestas | Posible (no integrado) | 2004+ | Intención de voto, atención |

## 3. Arquitecturas candidatas, ordenadas por complejidad

### 3.1 MLP tabular (baseline)

**Input:**
- Lags de votos normalizados por partido (1-3 procesos previos).
- Variables macro anuales (ICC, desempleo, inflación, remesas, tipo de cambio).
- Variables estructurales: lista nominal, participación previa, margen, número de
  partidos, incumbencia, elección presidencial/intermedia.
- Embeddings categóricos: entidad, distrito, municipio, partido.

**Output:**
- Softmax sobre K partidos: proporción de voto en la unidad geográfica.
- Cabeza secundaria para participación (regresión).

**Razón:** comparar contra modelos más complejos; detectar overfitting; servir
como benchmark interpretable.

### 3.2 LSTM/GRU/BiLSTM temporal

**Input:**
- Secuencia multivariada por unidad geográfica:
  - votos normalizados de los últimos N procesos,
  - ICC, inflación, remesas, tipo de cambio,
  - atención/stance promedio de mañaneras en la ventana previa.

**Arquitectura:**
```
X_t ∈ R^(T x F) -> LSTM/GRU -> h_T -> Dropout -> MLP -> y
```

**Variantes:**
- **BiLSTM** para aprovechar información futura si se trata de análisis
  ex-post (no pronóstico real).
- **Sequence-to-sequence** para pronosticar toda la trayectoria de votos.
- **Attention temporal** (Transformer encoder pequeño) para ponderar qué
  procesos históricos importan más.

**Riesgo:** pocas elecciones por unidad geográfica; overfitting. Mitigar con
regularización, early stopping y validación temporal.

### 3.3 BETO / RoBERTa / XLM-R para texto (NLP del curso)

**Tareas de pre-entrenamiento/fine-tuning:**
- Clasificación de *stance*: apoyo/oposición/neutro hacia candidato/partido.
- Clasificación de tópico: economía, seguridad, salud, corrupción, programas
  sociales, elecciones.
- Extracción de entidades: estados, municipios, programas, nombres.
- Detección de *framing*: promesa, ataque, defensa, advertencia.

**Uso electoral:**
- Agregar embeddings diarios/semanales por entidad/municipio mencionado.
- Construir series de saliencia de temas e intensidad comunicativa.
- Incluir como features en LSTM/MLP.

**Nota:** para México el sentimiento genérico es insuficiente; importa más el
*stance* y la propiedad de temas (*issue ownership*).

### 3.4 GNN espacial

**Grafo:**
- Nodos: municipios o distritos.
- Aristas: adyacencia geográfica, similitud socioeconómica, migración/comunicación.

**Features de nodo:**
- Votos previos, variables económicas, remesas, pobreza, violencia, demografía.
- Embeddings de texto agregados por municipio/estado.

**Arquitectura:**
```
GCN / GraphSAGE / GAT -> Pooling jerárquico -> MLP -> y_municipio
```

**Oportunidad:** capturar difusión espacial de preferencias, efectos frontera,
corredores metropolitanos y homofilia socioeconómica.

### 3.5 Modelo multimodal (late fusion)

```
Texto:  BERT/roBERTa -> Pooling -> h_text
Serie:  LSTM/GRU     -> h_temp
Tabla:  MLP          -> h_tab
Grafo:  GNN          -> h_graph
                |
                v
      Concatenación -> MLP de fusión -> Heads:
                         - votos (softmax)
                         - participación
                         - winner probability
                         - uncertainty interval
```

**Consideraciones:**
- Usar *late fusion* primero; es más fácil de depurar y evita que una
  modalidad de alta dimensión domine.
- Aplicar *modality dropout* para robustez ante datos faltantes.
- Pérdida multi-tarea:
  - votos: compositional (Dirichlet/logistic-normal) o MSE/MAE sobre shares.
  - ganador: cross-entropy / Brier.
  - participación: MSE.
  - calibración: término adicional (PLAD, expected calibration error).

## 4. Aplicaciones directas del curso

| Curso | Tema | Aplicación electoral |
|---|---|---|
| Deep Learning | MLP, regularización, embeddings | Baseline tabular + embeddings geográficos/partidos |
| Deep Learning | RNN/LSTM/GRU | Series temporales de votos, aprobación, confianza |
| NLP | BERT, fine-tuning, stance/topic | Mañaneras: topic, stance, framing, NER |
| NLP | Word2Vec, TF-IDF, LDA, STM | Modelos de referencia y features explicables |
| Advanced Deep Learning | Transformers, atención, multimodalidad | Fusionar texto, series, tablas y grafo |
| Advanced Deep Learning | GNNs | Vecindad espacial entre municipios/distritos |
| Advanced Deep Learning | Uncertainty / Bayesian DL | Intervalos predictivos calibrados |

## 5. Estrategia de validación

- **División temporal:** entrenar en procesos 1994-2012, validar en 2015-2018,
  probar en 2021-2024. Para locales: ajustar ventanas según calendario.
- **Rolling origin:** pronosticar cada proceso usando solo información previa.
- **Validación espacial:** dejar fuera estados o municipios similares.
- **Ablation:** comparar MLP, LSTM, BERT-only, GNN-only y combinaciones.
- **Métricas:** RMSE/MAE de votos, log-loss/Brier de ganador, calibration,
  coverage de intervalos, MAE por región.

## 6. Riesgos y salvaguardas

| Riesgo | Salvaguarda |
|---|---|
| Data leakage temporal | Fecha de corte estricta; no usar encuestas/textos/data revisada posterior al pronóstico. |
| Coaliciones cambiantes | Mantener vistas por partido, coalición-proceso y candidato; no comparar siglas cruzadas sin validación. |
| Cambio de límites geográficos | Documentar reseccionamiento/redistritación; agregar a unidades estables. |
| Ecological fallacy | No inferir preferencias individuales desde agregados. |
| Sesgo de plataforma/texto | El discurso presidencial no representa a todos los votantes; usar como covariable, no ground truth. |
| Overfitting | Regularización, early stopping, validación temporal, pocos parámetros dados pocos ciclos electorales. |

## 7. Próximos pasos recomendados

1. Implementar **MLP baseline** sobre `serie_presidencia_entidad.csv` + features
   macro en un notebook de Colab.
2. Entrenar **LSTM** con ventana histórica de votos y variables económicas.
3. Generar features de mañaneras (topic, stance, volumen) y evaluar su
   aporte por ablación.
4. Construir **grafo de municipios** e implementar GNN baseline.
5. Integrar todo en un **modelo multimodal** con incertidumbre.
6. Comparar contra modelos estadísticos (ARIMA, regresión espacial, poll
   aggregation) y reportar calibración.
