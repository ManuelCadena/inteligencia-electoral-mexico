# Knowledge Base — Sistema y Geografía Electoral de México

> **Propósito.** Marco interpretativo para el dataset canónico
> (`data/processed/canonical/`) y para cualquier análisis/modelo construido
> sobre él. Cada afirmación está etiquetada con su fuente:
> **[OFICIAL]** INE/CPEUM/legislación · **[ACADÉMICO]** literatura revisada
> · **[DATASET]** inferido de los propios datos SICEE · **[aprox]** cifra que
> varía por proceso electoral.
>
> Generado: 2026-10-05 · Investigación: Consensus, Perplexity (sonar-pro),
> AION Brain. Fuentes al final del documento.

---

## 1. Mapa institucional

| Institución | Naturaleza | Función en los datos |
|---|---|---|
| **INE** (Instituto Nacional Electoral) | Organismo constitucional autónomo; sucede al **IFE** tras la reforma político-electoral de 2014. [OFICIAL] | Organiza elecciones **federales**; administra el **Padrón Electoral** y la **Lista Nominal**; define la **geografía electoral** (distritación federal **y local**); fiscaliza. Publica SICEE, PREP y el Atlas Electoral. |
| **OPLE** (Organismos Públicos Locales Electorales; 32) | Uno por entidad: IEE, IEPC, IECM, IEEPCO, etc. [OFICIAL] | Organizan elecciones de **gubernatura, diputaciones locales, ayuntamientos, alcaldías CDMX** y figuras submunicipales. Sus cómputos alimentan el SICEE. |
| **TEPJF** (Tribunal Electoral del Poder Judicial de la Federación) | Máxima autoridad jurisdiccional. [OFICIAL] | Califica la elección presidencial; resuelve impugnaciones que pueden **modificar resultados** (recuentos, anulaciones). |
| **Tribunales electorales locales (TEE/TJEE)** | Primera instancia local. [OFICIAL] | Resuelven controversias locales; explican parte de los grupos con `FLAG_CONSISTENCIA` (elecciones anuladas). |
| **Juntas (INE)** | Ejecutivas, distritales (300) y locales. [OFICIAL] | Escrutinio y cómputo por niveles: casilla → junta distrital → cómputos finales. |

**Implicación para el dataset:** los datos `LOCALES` provienen de 32 OPLEs con
leyes distintas; los `FEDERALES` de un solo organismo. La heterogeneidad
estructural de las series locales es inherente, no un error.

---

## 2. Jerarquía territorial y unidades de análisis

### 2.1 Cadena de anidación

```
Domicilio del elector
  └─ Sección electoral        (~68,000–71,000 en 2024) [aprox][OFICIAL]
       └─ Casilla             (~170,000–172,000 en 2024) [aprox][OFICIAL]
            ├─ Distrito local (uninominal)   699 en total (suma de MR de los 32 congresos) [OFICIAL]
            ├─ Distrito federal (uninominal) 300 [OFICIAL, art. 53 CPEUM]
            ├─ Municipio / demarcación       2,469 municipios + 16 alcaldías CDMX [aprox][OFICIAL]
            └─ Entidad federativa            32
                 ├─ Circunscripción plurinominal (dip. fed. RP)  5 × 40 curules
                 └─ Circunscripción nacional (sen. RP)          1 × 32 curules
```

### 2.2 Definiciones operativas

| Unidad | Qué es | Clave para el análisis |
|---|---|---|
| **Sección electoral** | Mínima unidad geográfica electoral; regla general 100–3,000 electores. [OFICIAL] | Es el denominador territorial básico; una sección puede estar en un distrito federal y otro local **distintos**. |
| **Casilla** | Mesa receptora de votos (básica, contigua, extraordinaria, especial). [OFICIAL] | Nivel de acta; el SICEE publica agregados por sección, no por casilla en nuestras series. |
| **Distrito federal uninominal** | 300; eligen 1 diputado federal MR cada uno. [OFICIAL] | Distritación **2005** (usada 2006–2015), **2017** (2018, 2021) y **2022** (2024→). Los `ID_DISTRITO` federales **no son comparables** entre distritaciones. |
| **Distrito local uninominal** | 699 en total (tabla §4.2); eligen diputados locales MR. [OFICIAL] | Número y trazo varían por entidad y redistritación local (INE también distritan localmente). |
| **Municipio** | División político-administrativa (2,469). [OFICIAL] | Unidad de `AYUNTAMIENTO`. La lista cambia: se crean municipios nuevos entre procesos (p. ej. nuevos municipios en Oaxaca, QRoo, Edomex). |
| **Alcaldía CDMX** | 16 demarcaciones; equivalente funcional de municipio. [OFICIAL] | En el dataset: `AYUNTAMIENTO`/`OTROS` en CDMX = alcaldías + concejalías. DF→CDMX en 2016: mismo `id_ine=09`. |
| **Entidad federativa** | 32. | Clave `ENTIDAD_CANONICA` + `ID_ENTIDAD` (01–32 INE/INEGI). |
| **Circunscripción plurinominal** | 5 regiones que agrupan entidades; cada una asigna 40 diputados federales RP. [OFICIAL] | La votación RP se emite en cada casilla pero se **agrupa por circunscripción**, no por distrito. En nuestros datos `DIPUTADO_FEDERAL_RP` está por entidad (voto emitido), no por circunscripción. |

### 2.3 Padrón vs. Lista Nominal (definición crítica)

- **Padrón Electoral** ⊇ **Lista Nominal**: el padrón registra a todas las
  personas inscritas; la lista nominal contiene solo quienes tienen
  **credencial vigente y recogida** — es el padrón "depurado" y es el
  denominador correcto de la participación. [OFICIAL]
- `LISTA_NOMINAL` en nuestros datos = lista nominal; `PARTICIPACION =
  TOTAL_VOTOS / LISTA_NOMINAL`.

---

## 3. Cargos y fórmulas de asignación

### 3.1 Nivel federal

| CARGO (dataset) | Curules | Fórmula | Territorio de decisión |
|---|---:|---|---|
| `PRESIDENTE` | 1 | Mayoría relativa nacional (un solo distrito electoral = la nación). [OFICIAL] | Nacional |
| `SENADOR_MR` | 64 | 2 fórmulas por entidad a la planilla más votada. [OFICIAL, art. 56] | Entidad |
| `SENADOR_RP` (en dataset: 1ª minoría + RP nacional) | 32 + 32 | 1 senaduría por entidad al partido en **segundo lugar** (primera minoría) + 32 por lista nacional RP. | Entidad (1ª min.) / Nacional (RP) |
| `DIPUTADO_FEDERAL_MR` | 300 | MR por distrito uninominal. | Distrito federal |
| `DIPUTADO_FEDERAL_RP` | 200 | Listas regionales en 5 circunscripciones; cociente natural + resto mayor, con topes de sobrerrepresentación (ningún partido >300 curules ni +8 pts de sobrerrepresentación nacional). [OFICIAL, art. 54] | Circunscripción |

> **Nota SICEE:** en los archivos el voto por senadurías y diputaciones RP se
> registra por entidad/sección (donde se emite), aunque el **escoño** se
> asigne en la circunscripción. Nuestro `CARGO` distingue MR vs RP por el
> archivo fuente (`_MR_`/`_RP_`).

### 3.2 Nivel local — congresos (composición vigente, ~2024)

MR = distritos locales uninominales; RP = plurinominales. [OFICIAL: INE
DECEyEC "congresos_locales"]

| Entidad | MR | RP | Total | | Entidad | MR | RP | Total |
|---|---:|---:|---:|---|---|---:|---:|---:|
| AGS | 18 | 9 | 27 | | MOR | 12 | 8 | 20 |
| BC | 17 | 8 | 25 | | NAY | 18 | 12 | 30 |
| BCS | 16 | 5 | 21 | | NL | 26 | 16 | 42 |
| CAMP | 21 | 14 | 35 | | OAX | 25 | 17 | 42 |
| COAH | 16 | 9 | 25 | | PUE | 26 | 15 | 41 |
| COL | 16 | 9 | 25 | | QRO | 15 | 10 | 25 |
| CHIS | 24 | 16 | 40 | | QROO | 15 | 10 | 25 |
| CHIH | 22 | 11 | 33 | | SLP | 15 | 12 | 27 |
| CDMX | 33 | 33 | 66 | | SIN | 24 | 16 | 40 |
| DGO | 15 | 10 | 25 | | SON | 21 | 12 | 33 |
| GTO | 22 | 14 | 36 | | TAB | 21 | 14 | 35 |
| GRO | 28 | 18 | 46 | | TAMPS | 22 | 14 | 36 |
| HGO | 18 | 12 | 30 | | TLAX | 15 | 10 | 25 |
| JAL | 20 | 18 | 38 | | VER | 30 | 20 | 50 |
| MEX | 45 | 30 | 75 | | YUC | 21 | 14 | 35 |
| MICH | 24 | 16 | 40 | | ZAC | 18 | 12 | 30 |
| **Total** | **699** | **459** | **1,158** | | | | | |

CDMX tiene Asamblea/Congreso de 66 integrantes (33 MR + 33 RP).

### 3.3 Cargos municipales

| Cargo | Fórmula típica | Notas |
|---|---|---|
| `AYUNTAMIENTO` | Planilla de mayoría (presidente + síndicos + regidores MR) + regidurías RP según ley estatal. | El número de regidores/síndicos depende de la **población del municipio** y de la ley orgánica municipal. [OFICIAL] |
| `OTROS_MUNICIPALES` | Juntas municipales, regidurías, sindicaturas, presidencias de comunidad, concejos (CDMX). | Figuras submunicipales; diseño heterogéneo por entidad. No comparar entre estados sin revisar la ley local. |
| `GOBERNADOR` | MR estatal, 6 años, sin reelección. | Jefatura de Gobierno CDMX es el equivalente. |

**Reelección consecutiva** (reforma 2014, efectiva ~2018): diputados locales
(hasta 4 periodos), síndicos y regidores pueden reelegirse; presidentes
municipales en algunas entidades. [OFICIAL] → introduce variable de
**incumbencia** real a partir de ~2021 en las series.

### 3.4 Regímenes especiales

- **Sistemas Normativos Indígenas ("usos y costumbres")** — Oaxaca: 418 de 570
  municipios eligen autoridad por asamblea comunitaria, sin partidos, con
  calendarios y periodos propios (1–3 años). IEEPCO califica. [OFICIAL/ACADÉMICO:
  García García 2024] → **esperable** que esos municipios tengan votos en 0 o
  no aparezcan en `AYUNTAMIENTO` (explica parte de `FLAG_CONSISTENCIA`).
- **Elecciones extraordinarias / anulaciones**: Puebla 2019 (gubernatura
  extraordinaria tras anulación), municipios de Chiapas 2015 con cómputos en
  cero (todos los partidos 0 → `FLAG_CONSISTENCIA`). [DATASET]

---

## 4. Sistema de partidos y coaliciones

### 4.1 Lectura de `PARTIDO_COALICION`

El campo contiene **siglas de boleta**, no necesariamente partidos:

| Patrón | Significado |
|---|---|
| `PAN`, `PRI`, `MORENA`… | Voto marcado solo en ese emblema. |
| `PAN_PRI_PRD`, `PT_MORENA_ES`, `PRI_PVEM_NVA_ALIANZA_PCU` | **Candidatura común/coalición**: voto marcado en todos los emblemas de la fórmula, o en la casilla de la coalición. |
| `CAND_IND1`…`CAND_INDn` | Candidaturas independientes (numeradas en la boleta). |
| `ES`, `MC`, `NA`, `PARM`, `DSPPN`, `PAZ`, `PP`, `MVC`, `PH`, `PCU`, `RSP`, `FXM`… | Partidos estatales o extintos (nacional o local según archivo). |

**Regla de oro** [OFICIAL]: cuando la boleta permite marcar cada partido de
la coalición por separado, el candidato común recibe la **suma** de
`PAN + PRI + PRD + PAN_PRI + PAN_PRD + …` (todas las combinaciones impresas).
Para el share del **candidato** se suman todas las combinaciones; para el
share del **partido** (efectos registro/RP/financiamiento) cada combinación
se redistribuye por reglas del OPLE/INE — **no usar la suma cruda**.

> En nuestro dataset, `PARTIDO_COALICION` conserva la sigla tal como vino;
> el análisis de "voto por candidato" requiere agregar todas las
> combinaciones de una misma coalición.

### 4.2 Cronología de siglas frecuentes [OFICIAL/ACADÉMICO]

| Sigla | Trayectoria |
|---|---|
| `PAN` | 1939–presente (nacional). |
| `PRI` | 1929/1946–presente (nacional). |
| `PRD` | 1989–~2024 (perdió registro nacional 2024). |
| `PT`, `PVEM` | Activos; `PT` perdió registro 2024 en algunos procesos. |
| `MC` (antes Convergencia) | 1997–presente. |
| `NVA_ALIANZA`/`NA`/`PANAL` | 2005–2018 (perdió registro). |
| `MORENA` | 2014–presente; partido hegemónico 2018–2024. |
| `ES` | Partido local Edomex (Encuentro Social), ~2014–2018. |
| `PES`, `RSP`, `FXM` | Registros 2018–2021; perdieron registro tras 2021. |
| `DSPPN`, `PARM`, `PCU`, `PAS`, `PDM`, `PSN` | Partidos extintos pre-2000. |

---

## 5. Cómo leer las cifras del dataset

| Campo | Definición | Precaución |
|---|---|---|
| `VOTOS` | Votos de la sigla de boleta en la unidad geográfica. | En coaliciones desagregadas, el total del candidato = suma de combinaciones. |
| `TOTAL_VOTOS` | Votos emitidos en la casilla/unidad (válidos + nulos + no registrados, según fuente). | En archivos con `TERRITORIO` (extranjero/nacional) ya consolidados por suma. |
| `LISTA_NOMINAL` | Electores habilitados. | Puede estar ausente en algunos archivos locales antiguos. |
| `NUM_VOTOS_NULOS` | Boletas sin opción válida. | Alto puede señalar protesta o problemas de cómputo. |
| `NUM_VOTOS_CAN_NREG` | Votos por candidatos no registrados. | Categoría separada de nulos. |
| `SHARE_TOTAL` / `SHARE_VALIDO` | VOTOS/TOTAL y VOTOS/válidos. | En coalición-desagregada puede sumar >1 si la fuente reporta también la coalición como fila — ver `FLAG_CONSISTENCIA`. |
| `PARTICIPACION` | TOTAL_VOTOS/LISTA_NOMINAL. | Comparar solo dentro del mismo cargo/año (denominadores distintos entre MR/RP). |
| `TERRITORIO` (raw) | "VOTO EN EL EXTRANJERO" vs "VOTO EN TERRITORIO NACIONAL". | Ya consolidado: ambos sumados por unidad. |

### Clasificación de casillas (contexto para datos más finos)

Básica (1ª por sección), contigua (misma sección, excede capacidad),
extraordinaria (acceso/ubicación), especial (voto fuera de sección, hasta
2018). [OFICIAL]

### PREP vs. SICEE vs. cómputos

- **PREP**: preliminar, no oficial, por acta recibida.
- **Cómputos distritales/municipales/estatales**: oficiales; pueden incluir
  recuentos y resoluciones.
- **SICEE**: consulta histórica de resultados **oficiales** (lo que tenemos).
- Diferencias PREP↔cómputo son normales; para entrenamiento usar solo SICEE.

---

## 6. Geografía electoral — patrones regionales [ACADÉMICO]

### 6.1 Periodización

| Periodo | Patrón | Regiones |
|---|---|---|
| 1988–1994 | PRI hegemónico, oposición en ciudades/norte | PRI nacional; PAN norte; cardenismo centro-sur/CDMX |
| 1997–2006 | Tripartidismo regionalizado | PAN norte/Bajío/occidente; PRD CDMX/sur (Mich., Gro., Oax., Tab.); PRI rural |
| 2000–2012 | Alternancia presidencial PAN | Fox 2000: norte + zonas metropolitanas; PRI conserva gobernaturas/municipios |
| 2012–2018 | PRI presidencial fragmentado | PRI gana 2012 sin recuperar ciudades |
| 2018–2024 | MORENA hegemónico creciente | MORENA sur/centro-sur/sectores populares urbanos + expansión norte; PAN resiste en Bajío/norte (Gto., Jal. parcial, NL parcial); PRI reducido |

### 6.2 Clivajes robustos documentados

1. **Norte–sur**: PAN históricamente fuerte en norte/Bajío; izquierda
   (PRD→MORENA) en centro-sur y CDMX. Persistente aun tras 2018
   [Skachkov 2020; Latinskaia Amerika 2020].
2. **Urbano–rural**: PRI y luego MORENA capturan voto rural/programas
   sociales ("green vote"); PAN histórico en zonas metropolitanas y clases
   medias [Gómez Díaz 2022; Baker et al. 2020].
3. **Nacionalización**: PRI era el partido más nacionalizado (~1994-2012);
   MORENA lo sustituyó como organización nacional tras 2018.
4. **Efecto vecindario**: autocorrelación espacial del voto por sección —
   base teórica para GNN (Hernández 2025, Cd. Juárez; Li et al. 2019).
5. **Coattails locales→congresales**: +1pp municipal ≈ +0.45–0.78pp en
   diputado local del mismo partido (Gomberg et al. 2019).
6. **Coordinación electoral**: los actores coordinan como en democracias
   consolidadas desde 2000; la fragmentación étnica reduce la coordinación
   (Gómez Díaz 2022).

### 6.3 Calendario de gubernaturas 2015–2024 [OFICIAL]

| Año | Entidades con gubernatura |
|---|---|
| 2015 | BCS, CAMP, COL, GRO, MICH, NL, QRO, SLP, SON |
| 2016 | AGS, CHIH, DGO, HGO, OAX, PUE, QROO, SIN, TAMPS, TLAX, VER*, ZAC |
| 2017 | COAH, MEX, NAY |
| 2018 | CHIS, GTO, JAL, MOR, PUE*, TAB, VER, YUC + Jefatura CDMX |
| 2019 | BC, PUE (extraordinaria) |
| 2021 | BC, BCS, CAMP, CHIH, COL, GRO, MICH, NAY, NL, QRO, SLP, SON, TLAX, ZAC |
| 2022 | AGS, DGO, HGO, OAX, QROO, TAMPS |
| 2023 | COAH, MEX |
| 2024 | CHIS, GTO, JAL, MOR, PUE, TAB, VER, YUC + Jefatura CDMX |

*VER 2016 eligió gubernatura de 2 años (transición a ciclo 2018).

> Regla: periodo ordinario 6 años, sin reelección; Puebla/VER tuvieron
> periodos excepcionales por anulación/transición. Las elecciones locales
> **no coinciden** con un único ciclo — la "cobertura" de cada año en las
> series locales refleja este calendario, no datos faltantes.

---

## 7. Advertencias para series temporales (checklist de leakage/comparabilidad)

1. **Redistritación federal**: 2005, 2017, 2022 — `ID_DISTRITO` federal no es
   una unidad estable entre distritaciones. Agregar a entidad o usar
   *crosswalk* municipal si se necesita panel distrital.
2. **Redistritación local**: cada OPLE/INE actualiza distritos locales;
   `GEO_KEY_2` en `DISTRITO` no es estable entre procesos.
3. **Municipios nuevos**: ~2,469 hoy pero la cuenta histórica cambia (nuevos
   municipios, reseccionamiento). Para panel municipal usar clave
   `ID_MUNICIPIO` INEGI y documentar altas.
4. **DF→CDMX (2016)**: misma `id_ine=09`; normalizado a `CIUDAD DE MEXICO`.
5. **Cambio de partidos**: extinciones y registros nuevos (MORENA 2014,
   PANAL -2018, PES/RSP/FXM 2018-2021); un análisis de partido fijo excluye
   historia. `NEP` y `SHARE_OTROS` absorben esto parcialmente.
6. **Coaliciones cambiantes**: la misma coalición puede no existir en dos
   elecciones; para series por partido usar desagregación a partido componente
   o trabajar a nivel "coalición-ganadora".
7. **Usos y costumbres**: ~418 municipios oaxaqueños sin elección partidista —
   aparecerán como ausentes o en ceros; no tratar como missing aleatorio.
8. **Reelección municipal** solo desde ~2018: la variable "incumbente gana"
   no existe antes para presidentes municipales.
9. **Voto extranjero**: consolidado por entidad en archivos con `TERRITORIO`.
10. **`FLAG_CONSISTENCIA`**: 318/11,739 grupos con suma de share fuera de
    [0.9,1.1] — anulaciones (Chiapas 2015) o inconsistencias de fuente;
    excluir o analizar aparte.

---

## 8. Glosario mínimo

| Término | Definición |
|---|---|
| MR | Mayoría relativa: gana quien obtiene más votos. |
| RP | Representación proporcional: curules por lista según votación partidista. |
| Primera minoría | Senaduría de la fórmula del partido en 2º lugar por entidad. |
| Fórmula | Par de candidaturas (propietario+suplente) en la boleta senatoria. |
| Cociente natural / resto mayor | Método de asignación RP (votación efectiva ÷ curules; remanentes por mayor residuo). |
| Sobrerrepresentación | Topes constitucionales: máx. 300 diputados y +8pts vs. % de voto nacional. |
| OPLE | Organismo Público Local Electoral. |
| SICEE | Sistema de Consulta de la Estadística de las Elecciones (INE). |
| PREP | Programa de Resultados Electorales Preliminares. |
| NEP | Número efectivo de partidos (Laakso–Taagepera): 1/Σsᵢ². |
| Lista nominal | Electores con credencial vigente habilitados para votar. |
| Candidatura común | Una candidatura postulada por ≥2 partidos sin coalición formal plena. |
| Usos y costumbres / SNI | Elección comunitaria sin partidos (Oaxaca y otros casos). |

---

## 9. Referencias

### Oficiales
- CPEUM arts. 52–56, 115–116 (composición Cámara, Senado, régimen municipal).
- LGIPE (Ley General de Instituciones y Procedimientos Electorales).
- INE — SICEE: https://sicee.ine.mx/ · Atlas Electoral · distritaciones
  2005/2017/2022 (acuerdo CG 14-dic-2022).
- INE DECEyEC — "Composición de los congresos locales"
  (`ine.mx/wp-content/uploads/2021/12/DECEyEC_congresos_locales.pdf`).
- IEEPCO — Catálogo de municipios por Sistemas Normativos Indígenas.

### Académicos (Consensus)
1. Weldon, J. (2003). *The Consequences of Mexico's Mixed-Member Electoral
   System, 1988–1997*. DOI: 10.1093/019925768x.003.0021.
2. Kerevel, Y. (2010). *The legislative consequences of Mexico's
   mixed-member electoral system, 2000–2009*. Electoral Studies.
   DOI: 10.1016/j.electstud.2010.07.004.
3. Molinar Horcasitas, J. & Weldon, J. (2003). *Reforming Electoral Systems
   in Mexico*. DOI: 10.1093/019925768x.003.0011.
4. Calvo, E. & Escobar, M. (2002). *Institutional gamblers…* Electoral
   Studies. DOI: 10.1016/s0261-3794(01)00011-7.
5. Gómez Díaz, A. (2022). *Still Learning to Make Votes Count?* Election
   Law Journal. DOI: 10.1089/elj.2021.0074.
6. Gomberg, A. et al. (2019). *Coattails and the forces that drive them:
   Evidence from Mexico*. EJPE. DOI: 10.1016/j.ejpoleco.2018.10.001.
7. Cantú, F. (2014). *Identifying Irregularities in Mexican Local
   Elections*. AJPS. DOI: 10.1111/ajps.12097.
8. Calderón-Hernández, B. et al. (2025). *Electoral precinct-level database
   for Mexican municipal elections*. Scientific Data.
   DOI: 10.1038/s41597-025-04918-9.
9. Hernández Hernández, V. (2025). *Autocorrelación espacial y distribución
   del voto en Ciudad Juárez (2018–2024)*. Noé sis. DOI: 10.20983/noesis.2025.2.2.
10. García García, E.P. (2024). *Geografía del voto en municipios de usos y
    costumbres en Oaxaca (1998–2022)*. Apuntes Electorales.
    DOI: 10.53985/ae.v23i70.889.
11. Skachkov, V. (2020). *Problems of study of electoral geography of
    Mexico in the XXI century*. DOI: 10.18384/2712-7621-2020-2-44-51.
12. Arredondo, V. et al. (2017). *Electoral Mathematics and Asymmetrical
    Treatment to Political Parties: The Mexican Case*.
    DOI: 10.5281/zenodo.1340054.
13. Baker, A. et al. (2020). *Discussion and the Regionalization of Voter
    Preferences*. Persuasive Peers. DOI: 10.2307/j.ctv10vkzkv.13.
