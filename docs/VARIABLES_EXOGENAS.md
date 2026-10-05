# Variables exógenas para explicar el voto en México

Este documento identifica variables candidatas a incorporar como covariables en
los modelos de predicción electoral. Se nutre del proyecto previo de análisis de
mañaneras (*What Mexico Hears vs. What Mexico Believes*, CSCI E-89b) y del
repositorio de investigación *Lo que México cree* (modelo de energía libre /
Friston aplicado a política mexicana).

## 1. Variables ya disponibles en el repo

El script `scripts/build_exogenous_features.py` genera dos tablas anuales a
partir de archivos del proyecto *Lo que México cree*:

### Nacional: `data/processed/features/features_anuales_nacional.csv`

| Variable | Descripción | Fuente |
|---|---|---|
| `enco_icc_mean` | Índice de confianza del consumidor (INEGI) | INEGI ENCO |
| `enco_p3_mean` | Situación económica personal hace 12 meses | INEGI ENCO |
| `enco_p4_mean` | Situación económica personal en 12 meses | INEGI ENCO |
| `enco_p5_mean` | Situación económica del país hace 12 meses | INEGI ENCO |
| `enco_p6_mean` | Situación económica del país en 12 meses | INEGI ENCO |
| `enco_p8_mean` | Intención de compra de bienes durables | INEGI ENCO |
| `panel_icc_mean` | ICC promediado del panel de estimación | INEGI/Banxico/CONASAMI |
| `panel_tlp_mean` | Tasa de interés interbancaria o referencia | Panel LQMC |
| `panel_ing_inpc_mean` | Ingreso corriente deflactado | Panel LQMC |
| `panel_ing_canasta_mean` | Ingreso vs. canasta básica | Panel LQMC |
| `panel_ing_urb_mean` | Ingreso urbano | Panel LQMC |
| `panel_ing_rur_mean` | Ingreso rural | Panel LQMC |
| `panel_covid_mean` | Indicador de pandemia (2020-2022) | Panel LQMC |
| `panel_post2019_mean` | Post-2019 estructural | Panel LQMC |
| `ocde_desempleo_pct_mean` | Tasa de desempleo | OECD/FRED |
| `ocde_inpc_indice_mean` | Índice de precios al consumidor | OECD/FRED |
| `fred_tipo_cambio_mxn_usd_mean` | Tipo de cambio MXN/USD | FRED |
| `remesas_total_mdd` | Remesas familiares anuales (miles de millones USD) | Banxico |

### Por entidad: `data/processed/features/features_anuales_entidad.csv`

| Variable | Descripción | Fuente |
|---|---|---|
| `remesas_total_mdd` | Remesas familiares por entidad (miles de millones USD) | Banxico |

## 2. Variables del proyecto de mañaneras a integrar

El corpus de 1,611 conferencias matutinas (AMLO + Sheinbaum) produce un panel
mensual de 75 × 54 features. Para usarlo como exógena electoral hay que
agregarlo al año de la elección (promedio del año, o promedio de los N meses
previos).

### Discurso presidencial

| Variable | Método de generación | Notas |
|---|---|---|
| `mananera_volumen_mensual` | Suma de palabras/intervenciones | Control de intensidad comunicativa |
| `mananera_speaker_share_pres` | % de palabras del presidente vs. otros | Agenda presidencial vs. técnica |
| `mananera_sentiment_bilstm_mean` | Sentimiento promedio vía BiLSTM entrenado desde cero | Dirección afectiva del discurso |
| `mananera_topic_k{i}_share` | Proporción del tópico i (STM K=12) | Salud, seguridad, economía, programas sociales, etc. |
| `mananera_policy_attention_pc1` | Componente principal de búsquedas Google Trends | Variable objetivo del paper previo; R² MLP = 0.54 |

### Opinión pública y atención

| Variable | Descripción | Disponibilidad |
|---|---|---|
| `approval_presidencial_mean` | Aprobación promedio anual del presidente | Encuestas (Reforma, El Financiero, Mitofsky) |
| `gt_economia_mean` | Búsquedas Google Trends de economía | Google Trends |
| `gt_seguridad_mean` | Búsquedas de seguridad/delincuencia | Google Trends |
| `gt_salud_mean` | Búsquedas de salud | Google Trends |
| `gt_programas_sociales_mean` | Búsquedas de programas sociales | Google Trends |

## 2.5 Encuestas de aprobación e intención de voto

La aprobación presidencial/esperanza de un partido o candidato suele ser una de
las variables con mayor poder predictivo sobre el voto. En México no existe una
única serie pública, estandarizada e ininterrumpida 1991-2024, pero se puede
construir una serie armohizada combinando encuestadoras y documentando cada
observación.

| Fuente | Cobertura | Nivel | Frecuencia | Observaciones |
|---|---|---|---|---|
| Consulta Mitofsky | Presidencial, gobernadores, presidentes municipales (desde 2000s) | Nacional, estatal, municipal selecto | Periódica/mensual | Serie institucional; datos históricos a veces como gráficas |
| Reforma | Presidencial y benchmark histórico (1990s-2020s) | Nacional | Periódica | Archivo parcialmente bajo muro de pago |
| El Financiero | Presidencial reciente (Peña, AMLO) | Nacional | Mensual/cuatrimestral | Histórico comparativo con Reforma |
| BGC / Parametría / OPR | Evaluaciones presidenciales | Nacional | Periódica | Referenciadas en trabajo académico comparativo |
| Latinobarómetro | Actitudes democráticas, confianza, evaluación presidencial | Nacional, comparativa latinoamericana | Anual | No es seguimiento mensual de aprobación |
| Pew Research Center | Confianza en el presidente, satisfacción con democracia | Nacional | Ondas esporádicas | Útil para validación cruzada |
| Demoscopia Digital | Gobernadores y presidentes municipales | Estatal, municipal selecto | Periódica | Muestras por internet/móvil; documentación variable |

**Campos obligatorios por observación:** encuestadora, fechas de campo, tamaño
de muestra, modo (teléfono, cara a cara, online), redacción exacta de la pregunta,
alcance geográfico y si la cifra es contemporánea o retrospectiva.

## 3. Variables socioeconómicas y territoriales por entidad

| Variable | Descripción | Fuente |
|---|---|---|
| `pobreza_multidimensional` | % de población en pobreza multidimensional | CONEVAL |
| `ingreso_promedio_entidad` | Ingreso laboral promedio | ENOE/INEGI |
| `homicidios_tasa` | Tasa de homicidios por 100k habitantes | SESNSP |
| `narcomenudeo_tasa` | Delitos reportados relacionados | SESNSP/INEGI |
| `crecimiento_pib_entidad` | PIB estatal anual | INEGI Cuentas Nacionales |
| `remesas_percapita_entidad` | Remesas / población | Banxico + CONAPO |
| `margina_coneval` | Índice de marginación | CONAPO |
| `analfabetismo` | Tasa de analfabetismo | INEGI Censo |
| `poblacion_mayor65_pct` | Envejecimiento | CONAPO |

## 4. Variables temporales / de contexto electoral

| Variable | Descripción |
|---|---|
| `anio` | Año electoral |
| `eleccion_presidencial` | 1 si hay elección presidencial ese año |
| `eleccion_intermedia` | 1 si es año de elecciones de medio término |
| `coalicion_ganadora_previa` | Partido o coalición ganador en proceso previo |
| `alternancia_ultima` | 1 si hubo cambio de partido ganador |
| `num_partidos_competitivos` | Número efectivo de partidos (Laakso-Taagepera) |
| `margen_victoria_previa` | Diferencia % 1°-2° en elección anterior |
| `participacion_previa` | Tasa de participación / lista nominal |

## 5. Marco teórico de canales

Siguiendo el marco del proyecto de mañaneras:

- **Canal atencional**: el discurso presidencial predice qué temas busca la
  gente (`policy_attention_pc1`, Google Trends), pero no necesariamente su
  aprobación o confianza.
- **Canal de creencia/económico**: confianza del consumidor, ingreso, empleo,
  inflación y remesas moldean la evaluación retrospectiva/prospectiva.
- **Canal de bienestar material**: pobreza, marginación, violencia y
  remesas capturan variación territorial del voto.
- **Canal político-institucional**: alternancia, competencia efectiva,
  identificación partidista y estructura de la oferta electoral.

## 6. Recomendación de uso en modelos

1. Para **predicción federal anual**: usar todas las nacionales + controles
   temporales.
2. Para **predicción estatal o municipal**: agregar las por entidad
   (remesas, violencia, pobreza) al año electoral.
3. Para **NLP**: generar embeddings o topic shares de noticias/mañaneras en
   la ventana previa a la elección y concatenar como features.
4. Validar siempre con **train/validation/test temporal** y no mezclar
   información posterior a la fecha de predicción (data leakage).

## 7. Fuentes y referencias

- INEGI ENCO: https://www.inegi.org.mx/programas/enco/
- Banxico: https://www.banxico.org.mx/
- OECD/FRED: https://fred.stlouisfed.org/
- CONEVAL: https://www.coneval.org.mx/
- SESNSP: https://www.gob.mx/sesnsp
- Consulta Mitofsky: https://www.mitofsky.mx/evaluacion-gobierno
- El Financiero encuestas: https://www.elfinanciero.com.mx/encuestas-ef/
- Reforma: https://www.reforma.com/
- Latinobarómetro: https://www.latinobarometro.org/
- Pew Research Center: https://www.pewresearch.org/global/
- Proyecto mañaneras (CSCI E-89b): `final-paper.html` y `master-document.html`
- Proyecto *Lo que México cree*: `MAESTRO_v14.html`
