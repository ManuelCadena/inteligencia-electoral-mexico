# Estrategia de Investigación — Inteligencia Electoral México

> Documento rector del proyecto. Define (a) la pregunta de investigación,
> (b) la estrategia por la que cada tema de los cursos **CSCI E-89 Deep
> Learning** y **CSCI E-89b Intro NLP** se aplica al dataset canónico,
> (c) la estructura del paper que se entregará, y (d) la justificación y
> coherencia metodológica de cada decisión.
>
> Documentos base: `KB_SISTEMA_ELECTORAL_MEXICO.md` (qué significa cada dato),
> `METODOLOGIA.md` §10 (dataset canónico), `PLAN_ANALISIS_CURSOS.md` (mapa
> de análisis), `PROPUESTA_MODELADO_ELECTORAL_MEXICO.md` (marco teórico),
> `VARIABLES_EXOGENAS.md` (fuentes exógenas y de aprobación).

---

## 1. Pregunta de investigación

**Pregunta central.**
¿En qué medida los resultados electorales mexicanos —voto por partido,
participación y cambio de control político— pueden modelarse y pronosticarse
como una función de (i) la historia electoral de cada unidad territorial,
(ii) condiciones económico-sociales medibles, (iii) el discurso político
oficial y su atención territorial, y (iv) la estructura espacial del país,
usando exclusivamente información disponible antes de la jornada electoral?

**Subpreguntas operativas.**
- Q1: ¿Cuánta señal aporta la **persistencia electoral** frente a las
  covariables exógenas? (línea de base honesta)
- Q2: ¿La geografía (vecindad municipal) aporta información que las
  covariables tabulares no captan? (autocorrelación espacial, KB §6.2)
- Q3: ¿El discurso presidencial (mañaneras) contiene señal predictiva de
  votos, participación o alternancia, y por qué canal?
- Q4: ¿Un modelo multimodal calibrado supera a cada rama por separado y a
  las encuestas cuando existen?
- Q5: ¿Dónde se rompe el modelo? (elecciones anuladas, usos y costumbres,
  coaliciones cambiantes, redistritación) — el fracaso es también resultado.

**Lo que el proyecto NO pretende** (delimitación explícita):
- Inferencia causal: las relaciones son predictivas, no causales, salvo
  diseños específicos (coattails a la Gomberg et al. 2019).
- Perfilado individual: todos los análisis son agregados (ecológicos).
- Pronóstico en tiempo real: es investigación retrospectiva con cortes
  temporales simulados.

---

## 2. Datos y su lectura correcta (puente con el KB)

| Capa del dataset | Unidad | Regla de interpretación (KB §) |
|---|---|---|
| `canonical_long` (355K filas) | elección × geo × sigla de boleta | `PARTIDO_COALICION` es sigla de boleta, no partido; coaliciones desagregadas deben agregarse antes de shares (§4.1) |
| `canonical_wide` (11,739 filas) | elección × geo | unidad de análisis principal |
| Cargos MR/RP separados | SENADOR_MR/RP, DIP_LOCAL_MR/RP | fórmulas distintas: nunca comparar share MR con RP (§3.1) |
| Niveles geo | ENTIDAD / DISTRITO / MUNICIPIO | distrito federal no estable entre distritaciones (§7.1); municipio es la unidad más estable para panel |
| `FLAG_CONSISTENCIA` | 318/11,739 grupos | anulaciones e inconsistencias de fuente — excluir de entrenamiento, analizar aparte (§7.10) |
| Municipios Oaxaca usos y costumbres | ~418 | sin elección partidista; missing estructural, no aleatorio (§3.4) |
| Features exógenas | anuales, nacional + entidad | contemporáneas y `_LAG1`; anual es proxy — refinar a ventana pre-electoral (§10 METODOLOGÍA) |

**Cadena de anidación territorial** (KB §2): casilla ⊂ sección ⊂
distrito local/federal/municipio ⊂ entidad ⊂ circunscripción/nación.
Toda agregación del análisis debe respetar esta jerarquía.

---

## 3. Estrategia por tema de curso

### 3.1 CSCI E-89 Deep Learning

| Tema del curso | Aplicación al dataset | Rol en la investigación | Justificación metodológica |
|---|---|---|---|
| **W01 Intro NN / MLP** | A1 — MLP tabular sobre `canonical_wide`: shares por partido (softmax), participación y ganador. Inputs: shares anteriores + exógenas `_LAG1` + embeddings de entidad/cargo. | **Baseline neural obligatorio**: todo lo demás debe superarlo. | La persistencia electoral es el predictor dominante documentado (Kennedy 2017: encuestas/fundamentales; Vendeville 2020: voter model). Un MLP mide el techo de la información tabular sin estructura. |
| **W02 PyTorch** | Pipeline `Dataset/DataLoader` con máscaras de coalición y flag de consistencia. | Infraestructura. | Reproducibilidad y control de cardinalidad (lecciones del debug canónico). |
| **W03 AI-assisted coding** | Generación asistida de tests: shares∈[0,1], Σ=1, unicidad de clave. | Calidad. | El QA del dataset es parte del método (S1 M-CHEX del flujo de trabajo). |
| **W04 CNN** | A2 — CNN 2-D sobre grilla geográfica rasterizada (o conv 1-D sobre matriz año×partido por entidad). | Hipótesis espacial exploratoria. | El efecto vecindario (Hernández 2025) implica correlación local que una convolución espacial explota; contraste con GNN (C1). |
| **W05 Transfer/fine-tune** | A3 — pre-entrenar en federales (datos abundantes), fine-tune a gobernaturas (escasas). | Aborda el problema de n chico en locales. | Transferencia inter-nivel es plausible porque ambas elecciones comparten geografía y sistema de partidos; riesgo: calendarios desfasados (KB §6.3). |
| **W06 AE/VAE / manifold** | A4 — VAE del vector share-por-partido por municipio-año; latente 2-3D; clustering de "regímenes electorales". | **Análisis estructural clave**: valida la geografía electoral del KB como variedad de baja dimensión. | Si el mapa electoral es una variedad de dimensión baja, los clusters latentes deben recuperar las macro-regiones norte/Bajío/centro-sur (Skachkov 2020). Convierte el KB en hipótesis comprobable. |
| **W07 Transformers** | A5 — encoder-only temporal: secuencia de elecciones por entidad → siguiente elección. | Compara atención vs. recurrencia (A6). | Eventos no equiespaciados: la atención aprende qué ciclos pasados importan (p. ej. voto 2018→2024 vs. 2021→2024). |
| **W09 Seq2Seq / RNN** | A6 — LSTM/GRU: secuencia multivariada (shares + exógenas) → siguiente share. | Modelo temporal principal. | Diseño estándar de forecasting con pocas observaciones; regularización + pooling jerárquico obligatorios. |
| **W11 RAG** | A7 — RAG sobre KB + metodología + diccionario de datos. | Herramienta de trabajo: asistente que interpreta cada variable y regla. | Convierte el KB en accesible durante el análisis y la escritura. |
| **W12 Agents** | A8 — agente analista: tools = consultas al dataset + KB + gráficas. | Automatización de informes por elección. | — |
| **W13 Generative** | A9 — escenarios contrafácticos (difusión/GAN condicional) sobre vectores de shares dados features exógenas. | Exploración de sensibilidad, **marcada como ilustrativa**. | Escenarios "inflación +2pts" solo como visualización de la superficie aprendida; jamás como pronóstico. |
| **W14 Post-transformer** | A10 — Mamba/TSMixer sobre panel entidad×año vs. A5/A6. | Ablación arquitectónica de largo plazo. | Pocos puntos por serie pero muchas series paralelas: los mixers/SSM son candidatos naturalmente competitivos aquí. |

**Hilo conductor DL:** cada semana añade un *canal de estructura* distinto
(tabular → temporal → espacial → latente → generativo) sobre el mismo
problema, lo que permite una ablación limpia por tipo de inductive bias.

### 3.2 CSCI E-89b Intro NLP

Corpus: transcripciones de **mañaneras** (del proyecto final previo) +
terminología electoral. Regla absoluta: solo texto **publicado antes** de
cada elección entra como feature (fecha de publicación, no de referencia).

| Tema del curso | Aplicación | Rol en la investigación | Justificación metodológica |
|---|---|---|---|
| **W01 BoW + MLP** | N1 — TF-IDF agregado pre-electoral → MORENA-share/participación. | Baseline de texto honesto. | Si ni BoW aporta señal, BERT difícilmente la rescatará; test de sensatez. |
| **W02 RNN** | N2 — LSTM de sentimiento diario → serie temporal → rama del multimodal. | Covariable dinámica de texto. | El tono del discurso varía dentro del sexenio; solo una serie temporal lo respeta. |
| **W03 Tokenización** | N3 — BPE propio sobre corpus político mexicano; medir OOV de siglas/municipios vs. BETO. | Decisión de representación informada. | Vocabulario electoral (siglas, topónimos) está mal cubierto por tokenizers genéricos → medir, no suponer. |
| **W04 N-grams** | N4 — perplexity por sexenio; detección de cambio retórico Zedillo→AMLO. | Diagnóstico de concept drift textual. | Si el lenguaje cambia de régimen, los features de texto no son estacionarios → afecta la validez de N12. |
| **W05 Embeddings** | N5 — word2vec propio; embeddings de partido/entidad para A1. | Features densos. | Los embeddings capturan afinidades partido-región que las dummies no tienen. |
| **W06 t-SNE** | N6 — proyección de embeddings por época. | Visualización del drift. | Comunica el cambio semántico; pedagógico para el paper. |
| **W07 LDA** | N7 — topic shares pre-electorales (seguridad, economía, programas sociales) como features. | Canal de *issue ownership*. | Hipótesis clásica: la agenda del gobierno antes de votar modula el voto; LDA lo hace interpretable. |
| **W08 STM** | N8 — Structural Topic Model con covariables (año, aprobación) → prevalencia temática condicionada. | Versión correcta de N7 con metadata. | STM es el modelo adecuado para texto+contexto; estima el efecto del ciclo electoral sobre los temas. |
| **W09 Clasificación** | N9 — clasificador de stance/agenda (BETO fine-tuned): pro-oposición, económico, seguridad. | Feature de postura. | Stance informativo > sentimiento genérico en política (hallazgo Perplexity/Consensus). |
| **W10 NER** | N10 — menciones a entidades/municipios en mañaneras → `atención_presidencial` por geo. | **Hipótesis original fuerte**: ¿la atención discursiva territorial predice el voto? | Canal medible y novedoso: discurso→atención→voto. Solo posible con NER confiable + desambiguación de topónimos. |
| **W11 CRF/GAN** | N11 — CRF como baseline interpretable de NER; o augmentation de texto escaso. | Baseline secuencial. | CRF transparente vs. transformer: tabla de comparación en el paper. |
| **W12 Attention** | N12 — fine-tune BETO/RoBERTa-es: sentimiento, temas, regresión texto→aprobación; `[CLS]` → rama textual de C2. | Integración al multimodal. | BETO es el estándar español; XLM-R como robustez multilingüe. |

**Hilo conductor NLP:** cada semana produce una **representación de texto
progresivamente más rica** (BoW → n-gram → embedding → topics → stance →
contextual), y cada una se evalúa como feature pre-electoral del mismo
target — ablación por nivel de representación, no por modelo aislado.

### 3.3 Capstones integradores (donde convergen ambos cursos)

| # | Análisis | Integra | Pregunta que responde |
|---|---|---|---|
| C1 | **GNN municipal** (adyacencia geográfica/socioeconómica → GCN/GraphSAGE/GAT) | W04, W06, KB §6.2 | ¿La vecindad aporta señal que las covariables no? (Li et al. 2019; Hernández 2025) |
| C2 | **Multimodal late-fusion** = MLP + LSTM + BETO + GNN → softmax + incertidumbre | W01-W09, N01-N12 | ¿La fusión supera a cada rama? (arquitectura propuesta en OPORTUNIDADES) |
| C3 | **Forecasting de `PARTICIPACION`** — MLP/LSTM/Transformer/SSM | W01-W14 | Problema escalar, el experimento end-to-end más limpio; fija el harness. |
| C4 | **Forense de anomalías** — AE/Isolation sobre residuales de modelo + flags | W06, KB §7 | Detección de irregularidades a la Cantú (2014); explica `FLAG_CONSISTENCIA`. |
| C5 | **Nowcasting textual** — solo señales 90 días pre-elección | N02, N07, N12 | Réplica del diseño Brito et al. 2020 en México. |

---

## 4. Coherencia del análisis (por qué esta arquitectura y no otra)

1. **El problema es composicional + temporal + espacial + textual.**
   Ningún modelo único respeta las cuatro estructuras. La estrategia es
   *un modelo por estructura* + fusión, no un monolito.
2. **El objetivo es softmax composicional**, no regresiones por partido
   independientes (que no suman 1 y producen artefactos con coaliciones).
3. **La unidad estable de análisis es el municipio** (2,469) no el distrito
   (redistritación); para federales pre-2015 solo existe nivel entidad —
   por eso el panel municipal inicia en 2015 y se documenta explícitamente.
4. **La señal dominante conocida es la persistencia** → todo modelo se
   reporta contra baseline de persistencia y contra MLP; la contribución
   del proyecto es medir el *incremento* de cada canal (económico,
   espacial, textual, aprobación) sobre esa base.
5. **La aprobación es heterogénea e incompleta** (VARIABLES_EXOGENAS §2.5):
   entra como variable con metadatos de encuestadora y missingness
   indicator, nunca como serie lisa imputada.
6. **El fracaso es dato**: anulaciones, usos y costumbres y coaliciones
   cambiantes se modelan como flags/categorías, no se borran — C4 los
   convierte en objeto de estudio.
7. **Evaluación honesta**: split temporal 1994-2012/2015-18/2021-24;
   rolling-origin; intervalos calibrados; desagregación por entidad,
   tamaño municipal y urbano/rural — no un solo número nacional.

---

## 5. Estructura del paper a entregar

> Título tentativo: *"Modeling Mexican Elections as a Multimodal
> Spatio-Temporal Problem: Persistence, Economics, Discourse and Geography
> (1991–2024)"*

| § | Sección | Contenido | Fuentes de soporte |
|---|---|---|---|
| 1 | **Abstract** | Pregunta, datos (196 ZIPs SICEE, 355K obs), modelos, hallazgo principal, limitación clave. | — |
| 2 | **Introducción** | Motivación: México como caso de sistema mixto + transición partidista + discurso oficial diario único (mañaneras). Por qué la estructura del sistema (KB) obliga a un modelo por estructura. Contribuciones (3): dataset reproducible, benchmark multimodal, medición del canal discursivo territorial. | KB §1-3; Molinar & Weldon 2003 |
| 3 | **Antecedentes** | (a) electoral forecasting con NN (Zolghadr, Liu, Brito); (b) sistemas mixtos y geografía electoral mexicana (Weldon, Kerevel, Gómez Díaz, Skachkov); (c) texto político→voto (Belcastro, ElecBERT); (d) modelos espaciales (Li GCNN, Mancilla); (e) anomalías (Cantú). | ANTECEDENTES.md |
| 4 | **El sistema electoral mexicano** | Síntesis del KB: unidades, fórmulas, coaliciones, calendario, regímenes especiales. Qué implica para el diseño (niveles, composicionalidad, missing estructural). | KB completo |
| 5 | **Datos** | SICEE crudo → series → canónico; QA (0 dup, flags); exógenas; aprobación (heterogeneidad); corpus mañaneras. Tabla de cobertura por nivel/año. | METODOLOGIA §10, qa_report.json |
| 6 | **Marco teórico de modelos** | Una subsección pedagógica por arquitectura (MLP, LSTM/GRU, Transformer, BETO, GNN, late-fusion, incertidumbre) con la justificación de cada una *para este problema* — estilo lecture, como ya está en PROPUESTA. | PROPUESTA §modelos |
| 7 | **Diseño experimental** | Targets (share softmax, participación, ganador, alternancia); splits temporales; cortes de información; baselines; ablaciones (sin texto / sin geografía / sin economía / sin historia). | §4 de este documento |
| 8 | **Resultados** | Tabla maestra modelo × nivel × métrica; calibración; desagregación por región/tamaño/urbano-rural. | Notebooks A/N/C |
| 9 | **Análisis de errores y casos límite** | Anulaciones, usos y costumbres, coaliciones, redistritación; qué revelan los `FLAG_CONSISTENCIA`. | KB §7 |
| 10 | **Amenazas a la validez** | Falacia ecológica, sesgo de selección textual, leakage temporal/geográfico, drift, aprobación heterogénea, n chico por ciclo. | KB §7 + literatura |
| 11 | **Ética** | Agregación, no perfilado; licencias de datos; uso no electoral del modelo. | — |
| 12 | **Conclusiones y trabajo futuro** | Qué señal existe realmente; qué falta (sección/casilla, aprobación armónica, PREP). | — |
| — | **Apéndices** | Diccionario de datos, tablas de congresos locales, lista de siglas, reproducción (scripts + commits). | KB §4, DICCIONARIO |

**Regla editorial**: cada afirmación lleva etiqueta de evidencia
(peer-reviewed / preprint / oficial / propuesta propia) — misma convención
que el KB.

---

## 6. Cronograma de entregables

| Sprint | Entregable | Valida |
|---|---|---|
| 1 | `nb C3` forecasting participación (MLP vs LSTM) | harness + baseline honesto |
| 2 | `nb A4` VAE regional + `nb C1` GNN municipal | hipótesis espacial |
| 3 | `nb N7` LDA + `nb N12` BETO (corpus integrado) | canal textual |
| 4 | `nb C2` multimodal + calibración + `nb C5` nowcasting | tesis central del paper |
| 5 | `nb A5/A10` ablación arquitectural + `nb A7` RAG KB | robustez + herramientas |
| 6 | Paper v1 (§1-12) + apéndices | entrega |

Cada notebook cumple la regla visual-first: diagrama de arquitectura →
formas → código → predicción → error → interpretación con el KB.

---

## 7. Criterios de éxito del proyecto

1. **Dataset**: reproducible desde cero con 3 scripts; QA publicado.
2. **Benchmark**: tabla modelo × métrica con baselines honestos en test
   2021/2024 — cualquier ganancia debe medirse contra persistencia.
3. **Contribución científica**: cuantificar si (a) la geografía y (b) el
   discurso añaden señal *incremental* medible — aunque sea negativa.
4. **KB**: que cualquier lector pueda interpretar cada columna y cada
   anomalía del dataset sin conocimiento previo del sistema mexicano.
5. **Cobertura de cursos**: cada semana de DL y NLP tiene un análisis
   aplicado real, no ilustrativo.
