# Propuesta de modelado neuronal para el estudio de procesos electorales en México

**Resumen ejecutivo.** Este documento presenta una propuesta de investigación para
modelar resultados electorales históricos y pronósticos en México mediante redes
neuronales. Se integran datos del Instituto Nacional Electoral (SICEE), variables
económicas y sociales, encuestas de aprobación, y texto masivo de conferencias
matutinas presidenciales (mañaneras). La propuesta se articula como una progresión
didáctica: se parte de un perceptrón multicapa (MLP) tabular, se agregan redes
recurrentes (LSTM/GRU) para series temporales, modelos Transformer/BERT para
texto, redes neuronales de grafos (GNN) para estructura espacial, y se cierra con
un modelo multimodal que fusiona las señales. Se enfatiza el riesgo de *data
leakage* temporal, la necesidad de validación cronológica y la distinción entre
asociación predictiva e inferencia causal.

---

## 1. Introducción y justificación

El estudio de elecciones mexicanas ha dependido tradicionalmente de encuestas,
modelos econométricos lineales y análisis geográfico descriptivo. Sin embargo, los
procesos electorales contemporáneos generan tres tipos de señal que los modelos
calsicos capturan de manera parcial:

1. **Señal temporal:** la evolución histórica del voto por partido, la aprobación
   presidencial, la confianza del consumidor y la inseguridad.
2. **Señal textual:** discursos presidenciales, conferencias de prensa, medios y
   redes sociales configuran la agenda pública y la atención ciudadana.
3. **Señal espacial:** municipios y secciones electorales vecinas comparten
   mercados laborales, medios de comunicación, inseguridad y patrones migratorios.

Las arquitecturas neuronales permiten aprender representaciones de estas señales
sin imponer funciones específicas *a priori*. No obstante, el pequeño número de
procesos electorales por unidad geográfica y el riesgo de filtrar información
futura hacen imprescindible un diseño disciplinado y una jerarquía de modelos que
vaya del simple al complejo.

**Preguntas de investigación.**

1. ¿Es posible predecir la proporción de voto por partido/candidato a nivel
   entidad, distrito o municipio integrando datos históricos, macroeconómicos y
   textuales?
2. ¿Qué aporta cada modalidad (tabular, temporal, textual, espacial) sobre un
   modelo que solo usa votos rezagados?
3. ¿Cómo se pueden generar pronósticos con intervalos de incertidumbre
   calibrados, respetando la cronología de publicación de cada variable?

---

## 2. Marco teórico: votos como problema de aprendizaje multimodal

### 2.1 Voto económico y aprobación

El modelo clásico de *voto económico* sostiene que los ciudadanos recompensan o
castigan al partido en el poder según el desempeño económico reciente [Fair,
1996; Markus, 1988; Mackuen et al., 1992]. La aprobación presidencial funciona
como un resumen de estas evaluaciones y suele ser uno de los predictores más
fuertes del desempeño electoral del partido gobernante [Zolghadr et al., 2018].
La evidencia, sin embargo, muestra que la relación es condicional: las
percepciones económicas y la aprobación se influyen mutuamente, y los efectos son
más fuertes cuando la economía se deteriora [Dickerson, 2016].

### 2.2 Agenda setting, atención y creencias

El proyecto previo *What Mexico Hears vs. What Mexico Believes* encontró que las
mañaneras predicen con más fuerza la atención temática (búsquedas en Google) que
la aprobación generalizada o la confianza del consumidor. Esto es coherente con
la literatura de *agenda setting*: el discurso político define qué temas son
salientes, pero no necesariamente cambia la evaluación afectiva global. Por
eso, en lugar de usar solo "sentimiento promedio", proponemos extraer **stance**
(hacia candidatos y temas), **salencia de tópicos** y **menciones geográficas**.

### 2.3 Dependencia espacial y difusión

El voto no se distribuye aleatoriamente en el territorio. Municipios vecinos
tienen similitud en ingresos, medios, inseguridad y historia partidista. Esta
autocorrelación espacial puede modelarse con **redes neuronales de grafos**
(GNN), que propagan información a través de aristas definidas por adyacencia o
similitud socioeconómica [Li et al., 2019; Mancilla et al., 2025].

---

## 3. Fuentes de datos y variables

### 3.1 Resultados electorales

| Fuente | Cobertura | Nivel | Variables |
|---|---|---|---|
| SICEE (INE) | Federales 1991-2024; locales 2015-2024 | Nacional, entidad, distrito, municipio, sección | Votos, lista nominal, participación, nulos, votos por partido/coalición/candidato |
| PREP / Cómputos | Eventos específicos (2018, 2024, judicial 2025) | Nacional y subnacional | Resultados preliminares y definitivos con marca temporal |

En el repositorio se han descargado **196 archivos ZIP** del SICEE, que se
procesaron en series *tidy* por nivel electoral.

### 3.2 Variables económicas y sociales

| Variable | Fuente | Frecuencia | Justificación |
|---|---|---|---|
| ICC (confianza del consumidor) | INEGI ENCO | Mensual | Captura evaluaciones prospectivas/retrospectivas de la economía |
| Inflación (INPC) | INEGI / OECD | Mensual | Costo de vida, erosión del poder adquisitivo |
| Desempleo | OECD / INEGI | Mensual | Condiciones del mercado laboral |
| Tipo de cambio | Banxico / FRED | Diario | Competitividad, precios importados, remesas |
| Remesas | Banxico | Mensual/trimestral | Ingresos familiares, especialmente en zonas receptoras |
| Salario mínimo real | CONASAMI / INEGI | Anual | Bienestar material del trabajador formal e informal |
| Pobreza / ingreso | CONEVAL / ENIGH | Periódica | Variación territorial del bienestar |
| Inseguridad / violencia | SESNSP / INEGI ENVIPE | Mensual/anual | Problema político saliente en muchas entidades |

Las variables con mayor señal esperada son, en orden heurístico: (1) aprobación
presidencial del partido en el poder, (2) evaluaciones económicas reales del
hogar, (3) inflación alimentaria y energética, (4) percepción de inseguridad, (5)
remesas por entidad, (6) ingreso/marginación municipal.

### 3.3 Encuestas de aprobación

No existe en México una serie pública, estandarizada e ininterrumpida de
aprobación presidencial 1991-2024. La literatura y la práctica periodística
combinan varias encuestadoras. Las más relevantes son:

| Fuente | Cobertura | Nivel | Observaciones |
|---|---|---|---|
| **Consulta Mitofsky** | Presidencial, gobernadores, presidentes municipales (desde 2000s) | Nacional, estatal, municipal selecto | Serie más institucional; datos históricos a veces como gráficas |
| **Reforma** | Presidencial y benchmark histórico (1990s-2020s) | Nacional | Archivo parcialmente bajo muro de pago |
| **El Financiero** | Presidencial reciente (Peña, AMLO) | Nacional | Serie mensual/cuatrimestral; histórico comparativo con Reforma |
| **BGC / Parametría / OPR** | Evaluaciones presidenciales | Nacional | Referenciadas en trabajo académico comparativo |
| **Latinobarómetro** | Actitudes democráticas, confianza, evaluación presidencial | Nacional, comparativa latinoamericana | Ondas anuales; no es seguimiento mensual |
| **Pew Research Center** | Confianza en el presidente, satisfacción con democracia | Nacional | Ondas esporádicas; útil para validación cruzada |

Para **gobernadores**, la fuente más consistente es el *Ranking de
gobernadores y gobernadoras* de Mitofsky, publicado frecuentemente en *El
Economista*. Para **presidentes municipales**, existen rankings de Mitofsky y
Demoscopia Digital, pero la cobertura es selectiva y reciente.

**Implicación metodológica:** la aprobación debe construirse como una **serie
armonizada de encuestas heterogéneas**, registrando para cada observación:
encuestadora, fechas de campo, tamaño de muestra, modo (teléfono, cara a cara,
online), redacción exacta de la pregunta y alcance geográfico. No es aceptable
mezclar distintas formulaciones como si fueran una sola variable.

### 3.4 Texto: mañaneras y medios

El corpus de mañaneras (2018-2025) contiene aproximadamente 1,611 conferencias.
Además se pueden incorporar:

- Transcripciones de debates y entrevistas.
- Titulares de periódicos digitales.
- Tendencias de búsqueda (Google Trends) como proxy de atención pública.

Del texto se extraerán: tópicos (economía, seguridad, salud, corrupción,
programas sociales), *stance* (apoyo/oposición/neutro hacia partidos/temas),
emociones, menciones geográficas y eventos (anuncios, promesas, ataques).

---

## 4. Módulos del modelo: una progresión didáctica

### 4.1 Lección 1 — Perceptrón multicapa (MLP): el punto de partida

**Concepto.** Un MLP aprende una función no lineal $f: \mathbb{R}^d \to
\mathbb{R}^k$ compuesta de capas:

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}, \qquad
\mathbf{a}^{[l]} = \sigma(\mathbf{z}^{[l]})
$$

 donde $\mathbf{W}^{[l]}$ y $\mathbf{b}^{[l]}$ son parámetros aprendidos y
$\sigma$ es una función no lineal (ReLU, sigmoide, tanh). El teorema de
aproximación universal garantiza que una red suficientemente amplia puede
aproximar cualquier función continua, aunque eso no implica generalización.

**Aplicación electoral.** En nuestro problema, la entrada puede ser:

- Votos normalizados del partido $i$ en los procesos $t-1, t-2, \dots, t-k$.
- Variables macro del año electoral (ICC, inflación, desempleo, remesas, tipo de
cambio).
- Variables estructurales: lista nominal, participación previa, margen, número
efectivo de partidos, incumbencia, elección presidencial/intermedia.
- *Embeddings* de entidad y partido.

La salida es un vector de proporciones de voto. Para asegurar que sumen 1 usamos
una capa **softmax**:

$$
\hat{p}_{i,g,t} = \frac{\exp(z_{i,g,t})}{\sum_{j} \exp(z_{j,g,t})}
$$

**Razón para empezar aquí.** El MLP es la línea base más interpretable y rápida.
Si un modelo mucho más complejo no mejora al MLP, probablemente estamos
sobreajustando.

**Regularización.** Dado el pequeño número de procesos electorales por entidad,
usaremos dropout, *early stopping*, normalización de entradas, peso de
decaimiento ($L_2$) y *batch normalization* con precaución.

### 4.2 Lección 2 — Redes recurrentes: LSTM y GRU para series temporales

**Concepto.** Un vector de votos $\mathbf{x}_t \in \mathbb{R}^d$ en cada año $t$
es una secuencia. Una red neuronal recurrente (RNN) mantiene un estado oculto
$\mathbf{h}_t$ que resume la historia:

$$
\mathbf{h}_t = \tanh(\mathbf{W}_x \mathbf{x}_t + \mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{b})
$$

El problema de las RNN simples es el **desvanecimiento o explosión del
gradiente**: los gradientes se atenúan o crecen exponencialmente al propagarse
hacia atrás en secuencias largas.

**LSTM.** La unidad de memoria a largo plazo introduce tres *puertas* que
regulan el flujo de información:

- **Puerta de olvido** $f_t$: qué parte del estado anterior se descarta.
- **Puerta de entrada** $i_t$: qué nueva información se almacena.
- **Puerta de salida** $o_t$: qué información se expone al siguiente paso.

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \\
\mathbf{i}_t &= \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \\
\tilde{\mathbf{C}}_t &= \tanh(\mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C) \\
\mathbf{C}_t &= \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{C}_t)
\end{aligned}
$$

**GRU.** Es una variante más ligera que fusiona las puertas de olvido y entrada
en una sola *puerta de actualización*.

**Aplicación electoral.** Para cada entidad $g$ construimos una secuencia
multivariada:

$$
\mathbf{x}_{g,t} = \left[ v_{g,t-1}, \text{ICC}_{t}, \text{inflación}_{t}, \text{desempleo}_{t}, \text{aprobación}_{t}, \text{inseguridad}_{g,t} \right]
$$

Una LSTM lee la secuencia y produce una representación $\mathbf{h}_{g,T}$, que se
pasa a un MLP de salida para predecir el vector de votos del siguiente proceso.

**BiLSTM.** Si el objetivo es análisis ex-post y no pronóstico real, una LSTM
bidireccional puede usar información futura y pasada. Para pronóstico, solo la
versión unidireccional es válida.

### 4.3 Lección 3 — Atención y Transformers: de las mañaneras a representaciones

**Concepto.** El mecanismo de **atención** calcula una suma ponderada de los
vectores de una secuencia, donde los pesos dependen de la similitud entre
posiciones:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right) V
$$

$Q$, $K$ y $V$ son proyecciones lineales de la entrada y se interpretan como
*query*, *key* y *value*. La división por $\sqrt{d_k}$ estabiliza el gradiente.

**BERT y modelos en español.** BERT (*Bidirectional Encoder Representations from
Transformers*) preentrena representaciones contextualizadas con dos tareas:

1. Predicción de palabras enmascaradas (MLM).
2. Predicción de la siguiente oración.

Para español usamos modelos como **BETO**, **RoBERTa español** o **XLM-R**.

**Tareas de fine-tuning.** En lugar de entrenar un clasificador de sentimiento
genérico, proponemos:

- **Clasificación de tópico:** a qué tema pertenece el párrafo.
- **Stance detection:** apoyo u oposición a un partido o política.
- **Named Entity Recognition (NER):** menciones de estados, municipios,
  programas, personas.
- **Regresión de saliencia:** asociar el texto con el incremento posterior en
  búsquedas Google.

**Agregación temporal.** Los embeddings se promedian por semana, mes o ventana
pre-electoral. La representación textual del periodo $t$ es:

$$
\mathbf{h}^{\text{text}}_t = \frac{1}{|\mathcal{D}_t|} \sum_{d \in \mathcal{D}_t} \text{BERT}_{\text{[CLS]}}(d)
$$

### 4.4 Lección 4 — Redes neuronales de grafos (GNN): el territorio como red

**Concepto.** Un grafo $G=(V,E)$ tiene nodos $V$ y aristas $E$. En un GCN
(Graph Convolutional Network), la representación de un nodo se actualiza
combinando su estado con el promedio de sus vecinos:

$$
\mathbf{H}^{(l+1)} = \sigma\left( \tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)
$$

Aquí $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ incluye lazos y
$\tilde{\mathbf{D}}$ es la matriz diagonal de grados.

**Aplicación electoral.** Definimos un nodo por municipio o sección. Las
features son:

- Votos previos, participación, nulos.
- Variables socioeconómicas (pobreza, remesas, ingreso, inseguridad).
- Representaciones textuales agregadas si el municipio fue mencionado en
  mañaneras.

Las aristas pueden ser:

1. **Adyacencia geográfica:** comparten límite administrativo.
2. **Similitud socioeconómica:** distancia euclidiana en ingreso, educación,
   marginación.
3. **Conectividad migratoria o de medios:** flujos de remesas, zonas
   metropolitanas.

**Variantes.** GraphSAGE muestrea vecinos y permite generalizar a nodos no
vistos. GAT (Graph Attention Network) aprende pesos de importancia para cada
vecino. Esto es útil porque no todos los vecinos políticos son igualmente
influyentes.

### 4.5 Lección 5 — Modelo multimodal: fusión de señales

**Concepto.** La información relevante vive en representaciones de distintos
tamaños y naturaleza. El *late fusion* concatena representaciones aprendidas por
separado:

$$
\mathbf{h} = \left[ \mathbf{h}^{\text{tab}}; \mathbf{h}^{\text{temp}}; \mathbf{h}^{\text{text}}; \mathbf{h}^{\text{graph}} \right]
$$

y luego se pasa a un MLP de fusión:

$$
\hat{\mathbf{y}} = \text{MLP}_{\text{fusion}}(\mathbf{h})
$$

**Multi-task learning.** Se pueden predecir simultáneamente:

- proporción de voto por partido (salida softmax);
- participación (regresión);
- probabilidad de ganador (cross-entropy);
- intervalos de incertidumbre (aleatoriedad en dropout o redes bayesianas).

La función de pérdida combinada es:

$$
\mathcal{L} = \lambda_1 \mathcal{L}_{\text{vote}} + \lambda_2 \mathcal{L}_{\text{turnout}} + \lambda_3 \mathcal{L}_{\text{winner}} + \lambda_4 \mathcal{L}_{\text{calibration}}
$$

**Uncertainty.** Técnicas como MC Dropout ejecutan la red varias veces con
dropout activado en inferencia para aproximar la distribución predictiva. Una
alternativa más principiosa son las **redes neuronales bayesianas**, aunque son
computacionalmente más exigentes.

---

## 5. Arquitectura propuesta

### 5.1 Flujo de datos

```
SICEE + macro + aprobación + mañaneras + grafo espacial
        |               |               |
        v               v               v
   Tabla tidy     Series temporales   Texto
        |               |               |
        v               v               v
      MLP            LSTM/GRU        BERT/RoBERTa
        |               |               |
        v               v               v
    h_tabular      h_temporal        h_texto
        \               |               /
         \              |              /
          \             v             /
           \       GNN espacial    /
            \           |          /
             \          v         /
              \   h_grafo       /
               \      |        /
                \     v       /
                 \  Fusión  /
                  \    |   /
                   \   v  /
                    MLP
                     |
         [votos, turnout, ganador, incertidumbre]
```

### 5.2 Pronóstico paso a paso

1. **Predecir para cada unidad geográfica $g$** (entidad, distrito o
   municipio) la proporción de voto $\hat{\mathbf{p}}_{g,t}$.
2. **Agregar** proporciones a nivel nacional o estatal respetando lista nominal
   o población.
3. **Asignar escaños** (distritos o RP) mediante una regla de asignación
   proporcional o mayoritario según el cargo.
4. **Reportar intervalos** de probabilidad para cada escenario.

---

## 6. Validación y evaluación

### 6.1 Estrategia de validación temporal

No se permite mezclar años. La partición cronológica sugerida para elecciones
federales es:

- **Entrenamiento:** 1994, 1997, 2000, 2003, 2006, 2009, 2012.
- **Validación:** 2015, 2018.
- **Prueba:** 2021, 2024.

Para locales, la partición se ajusta al calendario de cada estado.

### 6.2 Cortes de información

Cada variable debe estar disponible *antes* de la fecha de pronóstico. Ejemplos
de filtrado:

- Encuestas publicadas hasta un mes antes de la elección.
- Datos macro del mes anterior (no revisiones posteriores).
- Mañaneras hasta una semana antes.

### 6.3 Métricas

| Tarea | Métrica | Interpretación |
|---|---|---|
| Votos | MAE, RMSE del share | Error promedio en puntos porcentuales |
| Composición | KL-divergence | Qué tan cerca está el vector completo |
| Ganador | log-loss, Brier, accuracy | Calibración y acierto |
| Turnout | MAE/RMSE | Participación efectiva |
| Incertidumbre | Coverage, CRPS | Intervalos calibrados |

### 6.4 Comparadores obligatorios

Todo modelo neuronal debe superar o empatar con:

- Promedio del voto previo por partido.
- Regresión lineal con variables fundamentales.
- Random forest o gradient boosting.
- Modelo de aprobación + economía simple.

---

## 7. Riesgos metodológicos y éticos

1. **Fuga temporal.** Usar datos posteriores al pronóstico invalida el
   ejercicio. La solución es un *pipeline* de fechas estricto.
2. **Coaliciones cambiantes.** La misma sigla puede representar alianzas
   distintas. Se conservan tres vistas: partido, coalición-proceso y
   candidatura.
3. **Cambio de límites.** Reseccionamiento y creación de municipios rompen
   series históricas. Se usan agregaciones estables o tablas de correspondencia.
4. **Falacia ecológica.** Los agregados municipales no revelan comportamiento
   individual.
5. **Sesgo de plataforma.** El discurso presidencial y las redes no representan
   a todos los votantes.
6. **Causalidad vs. predicción.** Un modelo predictivo no prueba que la economía
   "cause" el voto sin estrategia de identificación.
7. **Privacidad y vigilancia política.** Evitar modelos individuales y usar
   agregados oficiales.

---

## 8. Resultados esperados y contribución

1. Un **repositorio reproducible** con descarga automática del SICEE, limpieza de
   series, variables exógenas y notebooks de modelos.
2. Una **jerarquía de modelos** (MLP → LSTM → BERT → GNN → multimodal) con
   evaluación cronológica.
3. Una **serie armonizada de aprobación presidencial** documentando fuentes,
   limitaciones y campos de cada encuesta.
4. **Código y notebooks** listos para Google Colab, con arquitecturas visuales y
   diagramas de flujo.
5. Un análisis de **importancia de variables** por modalidad y nivel geográfico.

---

## 9. Referencias

1. Zolghadr, M., et al. (2018). *Modeling and forecasting US presidential election
   using learning algorithms*. Journal of Industrial Engineering International.
   DOI: 10.1007/s40092-017-0238-2
2. Brito, K. S., et al. (2020). *Predicting Brazilian and U.S. Elections with
   Machine Learning and Social Media Data*. IJCNN.
   DOI: 10.1109/ijcnn48605.2020.9207147
3. Liu, R., et al. (2020). *Can We Forecast Presidential Election Using Twitter
   Data? An Integrative Modelling Approach*. Annals of GIS.
   DOI: 10.1080/19475683.2020.1829704
4. Li, M., et al. (2019). *Deep Hierarchical Graph Convolution for Election
   Prediction from Geospatial Census Data*. AAAI.
   DOI: 10.1609/aaai.v33i01.3301647
5. Mancilla, S., et al. (2025). *Sub-spatial prediction of votes integrating
   socioeconomic, educational, and age strata*. Journal of Big Data.
   DOI: 10.1186/s40537-025-01112-x
6. Kennedy, R., et al. (2017). *Improving election prediction internationally*.
   Science. DOI: 10.1126/science.aal2887
7. Fair, R. C. (1996). *Econometrics and Presidential Elections*. Journal of
   Economic Perspectives. DOI: 10.1257/jep.10.3.89
8. Markus, G. B. (1988). *The Impact of Personal and National Economic
   Conditions on the Presidential Vote*. American Journal of Political Science.
   DOI: 10.2307/2111314
9. Mackuen, M. B., et al. (1992). *Peasants or Bankers? The American Electorate
   and the U.S. Economy*. American Political Science Review.
   DOI: 10.2307/1964124
10. Dickerson, B. T. (2016). *Economic Perceptions, Presidential Approval, and
    Causality*. American Politics Research.
    DOI: 10.1177/1532673x15600787
11. Skoric, M., et al. (2020). *Electoral and Public Opinion Forecasts with
    Social Media Data: A Meta-Analysis*. Information.
    DOI: 10.3390/info11040187
12. Topîrceanu, A. (2021). *Electoral Forecasting Using a Novel Temporal
    Attenuation Model*. Expert Systems with Applications.
    DOI: 10.1016/j.eswa.2021.115289
13. Belcastro, L., et al. (2020). *Learning Political Polarization on Social
    Media Using Neural Networks*. IEEE Access.
    DOI: 10.1109/access.2020.2978950
14. Vendeville, A., et al. (2020). *Forecasting elections results via the voter
    model with stubborn nodes*. Applied Network Science.
    DOI: 10.1007/s41109-020-00342-7
