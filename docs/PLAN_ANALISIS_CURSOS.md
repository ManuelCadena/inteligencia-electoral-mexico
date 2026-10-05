# Plan de Análisis — Cobertura de Cursos con el Dataset Electoral

> Mapeo entre el temario de **CSCI E-89 Deep Learning** (14 semanas),
> **CSCI E-89b Intro NLP** (12 semanas) y análisis concretos sobre
> `data/processed/canonical/` (355K filas, 11,739 elección-geografías,
> 1991–2024) + corpus de mañaneras del proyecto final NLP.
> Regla: cada análisis produce (a) notebook reproducible, (b) visualización
> de arquitectura (regla visual-first), (c) métrica con split temporal,
> (d) interpretación con el KB (`docs/KB_SISTEMA_ELECTORAL_MEXICO.md`).

---

## 0. Convenciones de validación (aplican a TODO)

- **Split temporal fijo**: train 1994–2012 · val 2015–2018 · test 2021–2024.
- **Corte de información**: solo features disponibles antes de la elección
  (`_LAG1` cuando existan).
- **Filtro de calidad**: `FLAG_CONSISTENCIA=0` para entrenamiento; reportar
  sensibilidad incluyéndolos.
- **Baselines obligatorios**: persistencia (resultado anterior), media
  histórica por geografía, regresión lineal.
- **Métricas**: MAE/RMSE en `SHARE_VALIDO`; Brier/log-loss para ganador;
  macro-F1 para clasificación; cobertura de intervalos al 80/95%.
- **Objetivo composicional**: el vector de shares por partido debe sumar 1
  → salida softmax o capa composicional.

---

## 1. Deep Learning (CSCI E-89) — 14 semanas

| Semana | Tema | Análisis propuesto sobre el dataset | Justificación electoral |
|---|---|---|---|
| W01 Intro NN | Perceptrón/MLP | **A1. MLP baseline tabular**: predecir `SHARE_VALIDO` por partido en `canonical_wide` a partir de shares anteriores + exógenas + embeddings de entidad/cargo. | Voto es fuertemente serialmente correlacionado; baseline honesto que todo modelo debe superar. |
| W02 PyTorch | Framework | Implementar todo el pipeline en PyTorch `Dataset/DataLoader` con máscaras de coalición y `FLAG_CONSISTENCIA`. | — |
| W03 AI Coding | Tooling | Usar agente para generar suite de tests del pipeline (cardinalidad, shares∈[0,1], suma=1). | — |
| W04 CNN | Convoluciones | **A2. CNN 1-D sobre "imagen electoral"**: matriz año×partido por entidad (mapa de calor del voto) → conv sobre ejes (tiempo, partido); alternativa: CNN 2-D sobre grilla geográfica de municipios rasterizada. | CNN capta patrones locales norte-sur y efectos de vecindad (KB §6.2). |
| W05 Transfer/Fine-tune | Transfer learning | **A3. Pre-entrenar MLP en elecciones federales (muchas observaciones) → fine-tune a gobernaturas** (pocas observaciones por entidad). Medir transferencia inter-nivel. | Pocas gubernaturas por estado; el conocimiento federal transfiere. |
| W06 AE/VAE | Manifold | **A4. Autoencoder/VAE de perfiles electorales**: vector share-partido por municipio-año → espacio latente 2-3D; clustering de "regímenes electorales" (PRI rural, PAN urbano, MORENA popular). Comparar con regiones del KB §6. | Valida la geografía electoral como estructura de baja dimensión (manifold hypothesis). |
| W07 Transformers | Atención | **A5. Transformer de series temporales** (encoder-only) sobre secuencia de shares por entidad → siguiente elección. Comparar atención vs. LSTM (A6). | Elecciones son eventos discretos no equiespaciados; la atención aprende qué ciclos pasados pesan más. |
| W08 Speech | Audio (opcional) | Fuera de alcance del dataset; cubrir con mañaneras transcritas si se desea Whisper (ya en corpus NLP). | — |
| W09 Seq2Seq | Secuencias | **A6. LSTM/GRU/Seq2Seq**: secuencia de shares estatales → vector de shares de la siguiente elección por partido (seq2seq multivariado). | Marco clásico de forecasting electoral. |
| W10 LLMs/NLP | Ver sección NLP | — | — |
| W11 RAG | Retrieval | **A7. RAG sobre el KB + METODOLOGIA + DICCIONARIO**: asistente que responde "¿qué significa GEO_KEY_2 en diputados locales 2018?" citando el KB. | Convierte el KB en herramienta usable. |
| W12 Agents | Agentes | **A8. Agente analista electoral**: herramientas = query al dataset + RAG del KB + gráficos; produce mini-reportes por elección. | — |
| W13 Generative | Generación | **A9. GAN/difusión condicional (opcional)**: simular escenarios contrafácticos de voto dado un vector de features exógenas; evaluar calibración distributiva. | Escenarios "¿qué pasa si la inflación sube 2pts?" — marcar como simulación ilustrativa. |
| W14 Post-Transformer | State-space/MLP-mixers | **A10. Mamba/TSMixer sobre el panel entidad×año** vs. A5/A6 — ablación de arquitectura temporal. | Elecciones largas, pocas observaciones: los SSM son candidatos naturales. |

**Entregable DL:** notebook `04_dl_*.ipynb` por análisis + tabla comparativa
persistencia < lineal < MLP < LSTM < Transformer (test 2021/2024).

---

## 2. NLP (CSCI E-89b) — 12 semanas

Corpus disponible: transcripciones de **mañaneras** (proyecto final previo)
+ nombres de municipios/partidos/candidatos como vocabulario propio.
Ventana: alinear features de texto con la fecha de cada elección (solo
discurso previo → sin leakage).

| Semana | Tema | Análisis propuesto | Conexión electoral |
|---|---|---|---|
| W01 MLP/boW | Representación de texto | **N1. BoW/TF-IDF + MLP**: features léxicas de mañaneras agregadas pre-elección → participación o share MORENA. | Baseline de texto honesto. |
| W02 RNN | RNN | **N2. RNN/LSTM de sentimiento diario** del discurso → serie temporal que alimenta A6 (late fusion). | Sentimiento del discurso como covariable dinámica. |
| W03 Tokenización | Tokenizers | **N3. Entrenar BPE/tokenizer propio** sobre corpus mañaneras+terminología electoral; comparar OOV vs. tokenizer BETO. | Vocabulario político mexicano (siglas, municipios, cargos) mal cubierto por tokenizers genéricos. |
| W04 N-grams | Modelos de lenguaje | **N4. Lenguaje n-gram por periodo**: perplexity de discursos por sexenio; detectar cambio de régimen retórico Zedillo→AMLO. | Diagnóstico de concept drift textual. |
| W05 Embeddings | Word2Vec/GloVe | **N5. Embeddings propios (word2vec) sobre corpus político**; analogías (partido:región :: candidato:entidad); embeddings de entidad para A1. | Embeddings como features del MLP. |
| W06 t-SNE | Visualización | **N6. t-SNE/UMAP de embeddings** de términos electorales por época → clusters semánticos por sexenio. | Visual del drift semántico. |
| W07 LDA | Topic modeling | **N7. LDA sobre mañaneras** → shares temáticos pre-electorales como features exógenas (seguridad, economía, programas sociales). | Issue ownership: qué temas domina el gobierno antes de votar. |
| W08 STM | Structural TM | **N8. STM (Structural Topic Model)** con covariables (año, aprobación) → cómo cambia la prevalencia temática con el ciclo electoral. | STM es el modelo correcto para texto+metadata. |
| W09 Clasificación | Clasificación | **N9. Clasificador de stance/agenda**: etiquetar segmentos de mañaneras (pro-oposición, económico, seguridad) con BETO fine-tuned. | Stance presidencial hacia entidades antes de elecciones. |
| W10 NER | Entidades | **N10. NER político**: extraer menciones a estados/municipios en mañaneras → feature `atención_presidencial_por_entidad`. | ¿La atención discursiva a una entidad predice su voto? |
| W11 CRF/GAN | Secuencias/generativos | **N11. CRF para NER** (baseline interpretable vs. transformer) o data augmentation de texto escaso. | — |
| W12 Attention | Transformers | **N12. Fine-tune BETO/RoBERTa-es** para (a) sentimiento, (b) clasificación temática, (c) regresión directa texto→aprobación. Embeddings [CLS] → rama textual del modelo multimodal (A-MULTI). | Rama de texto del sistema final. |

**Datos que faltan y se requieren**: corpus de mañaneras indexado por fecha;
encuestas de aprobación (fuentes ya catalogadas en `VARIABLES_EXOGENAS.md`).
Sin corpus real → marcar notebooks N1–N12 como plantillas hasta integrarlo.

---

## 3. Análisis integradores (capstone — tocan varias semanas)

| # | Análisis | Temas cubiertos | Por qué es el proyecto estrella |
|---|---|---|---|
| **C1** | **GNN espacial**: grafo de municipios (adyacencia geográfica o similitud socioeconómica) → GCN/GraphSAGE prediciendo share por partido. | W04 (conv), W06 (latente), KB §6.2 efecto vecindario | La autocorrelación espacial del voto está documentada (Hernández 2025; Li et al. 2019 GCNN). `GEO_KEY_2` municipal da ~2,400 nodos recurrentes. |
| **C2** | **Modelo multimodal late-fusion**: MLP(tabular) ⊕ LSTM(series) ⊕ BETO(texto) ⊕ GNN(grafo) → softmax sobre partidos + incertidumbre (MC dropout/ensemble). | W01–W07, W09, N01–N12 | Es la arquitectura propuesta en `OPORTUNIDADES_MODELOS.md`; demuestra dominio de todas las ramas. |
| **C3** | **Forecasting de participación** (`PARTICIPACION`): problema más simple (escalar, no composicional) → comparar MLP vs LSTM vs Transformer vs SSM. | W01, W07, W09, W14 | Turnout tiene series más limpias que shares → buen primer experimento end-to-end. |
| **C4** | **Detección de anomalías electorales** (forense): autoencoder o isolation sobre features de casilla/municipio → outliers = posibles irregularidades (Cantú 2014 usó este enfoque con apellidos). | W06, W13 | Conecta con literatura forense (Cantú, AJPS 2014); útil para explicar `FLAG_CONSISTENCIA`. |
| **C5** | **Nowcasting con NLP**: usar solo texto/trends de los 90 días pre-elección → predecir resultado; contraste con modelo de fundamentales. | N02, N07, N12, W09 | Reproduce diseño de Brito et al. 2020 (posts → votos). |

---

## 4. Orden de ejecución sugerido

```
Sprint 1 (fundamentos):  C3 turnout → A1 MLP → A6 LSTM/GRU
Sprint 2 (estructura):   A4 VAE + clusters regionales → C1 GNN municipal
Sprint 3 (texto):        integrar corpus mañaneras → N7 LDA → N12 BETO
Sprint 4 (integración):  C2 multimodal + incertidumbre → C5 nowcasting
Sprint 5 (transversal):  A5 Transformer/SSM ablación → A7 RAG del KB
```

Cada sprint cierra con la misma tabla de comparación sobre test 2021/2024
para que el progreso sea medible y honesto (no "modelo X es mejor" sin
baseline de persistencia).

## 5. Riesgos metodológicos (del KB §7) aplicados al plan

1. `ID_DISTRITO` federal no comparable entre distritaciones → C1 usar
   municipios (unidad más estable) o nivel entidad.
2. Coaliciones desagregadas → en todos los targets de share, agregar
   combinaciones de la misma candidatura común primero.
3. 418 municipios Oaxaca usos y costumbres → excluir o modelar aparte en C1.
4. Solo ~6-8 ciclos presidenciales por unidad → regularización fuerte,
   pooling jerárquico (entidad↔municipio), nunca random-split temporal.
5. Texto: fecha de publicación, no de referencia — leakage real si se usan
   mañaneras post-elección.
