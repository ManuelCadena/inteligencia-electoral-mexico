# Diccionario de datos — SICEE (INE México)

Este documento describe las columnas más frecuentes en los archivos CSV del SICEE. Los nombres exactos varían ligeramente entre procesos electorales, por lo que los scripts normalizan a mayúsculas y sin espacios.

## 1. Identificación geográfica

| Variable (normalizada) | Descripción | Ejemplo |
|--------------------------|-------------|---------|
| `ID_ESTADO` | Clave numérica de la entidad federativa (INE). | 9 = Ciudad de México |
| `NOMBRE_ESTADO` / `NOMBRE_ENTIDAD` | Nombre de la entidad. | "CIUDAD DE MÉXICO" |
| `ID_DISTRITO` | Clave del distrito electoral federal/local. | 10 |
| `DISTRITO` | Nombre o número del distrito. | "10" |
| `ID_MUNICIPIO` | Clave del municipio. | 107 |
| `MUNICIPIO` / `NOMBRE_MUNICIPIO` | Nombre del municipio. | "BENITO JUÁREZ" |
| `SECCION` / `SECCIÓN` | Número de sección electoral. | 1234 |
| `CASILLA` / `TIPO_CASILLA` | Tipo o identificador de casilla. | "B", "C1", "M" |
| `CIRCUNSCRIPCION` | Circunscripción plurinominal (federal). | 4 |

## 2. Variables de votación

| Variable | Descripción |
|----------|-------------|
| `TOTAL_VOTOS` | Suma de votos válidos, nulos y no registrados. |
| `NUM_VOTOS_NULOS` | Votos anulados. |
| `NUM_VOTOS_CAN_NREG` | Votos para candidaturas no registradas. |
| `LISTA_NOMINAL` | Lista nominal de electores (denominador). |
| `PARTICIPACION` | `TOTAL_VOTOS / LISTA_NOMINAL` (calculada). |
| `ABSTENCION` | `1 - PARTICIPACION` (calculada). |

## 3. Variables de partido / candidatura

Los CSV pueden contener columnas con nombres de partidos o coaliciones, por ejemplo:

- `PAN`, `PRI`, `PRD`, `PT`, `PVEM`, `MC`, `MORENA`, `PES`, `RSP`, `FXM`, `NAZ`, etc.
- Coaliciones: `AC`, `PBT`, `JHH`, `FPXM`, etc.
- Candidaturas independientes: `CAND_IND_1`, `INDEPENDIENTE`, etc.

> **Importante:** las siglas de coaliciones no son estables a través del tiempo. Para análisis comparativo se recomienda usar los archivos `*_DISCAND.csv` (por candidato) o `_ENTCAND.csv`.

## 4. Archivos dentro de cada ZIP

Cada archivo ZIP suele contener varios CSV con el mismo prefijo y sufijo que indica el nivel de agregación:

| Sufijo | Significado |
|--------|-------------|
| `_ENT.csv` | Entidad federativa |
| `_DIS.csv` | Distrito federal/local |
| `_MUN.csv` | Municipio |
| `_SEC.csv` | Sección electoral |
| `_CAS.csv` | Casilla |
| `_CAND.csv` / `_DISCAND.csv` | Candidato |
| `_ENTCAND.csv` | Candidato por entidad |

## 5. Series generadas

En `data/processed/series/` se generan tablas en formato largo (tidy):

| Columna | Descripción |
|---------|-------------|
| `ANIO` | Año del proceso electoral. |
| `NIVEL` | Nivel de agregación (`ENT`, `MUN`, `SEC`, `CAS`). |
| `ENTIDAD` | Entidad federativa (o municipio/sección según nivel). |
| `PARTIDO_COALICION` | Sigla de partido o coalición. |
| `VOTOS` | Votos absolutos. |
| `TOTAL_VOTOS` | Total emitidos en la unidad. |
| `LISTA_NOMINAL` | Lista nominal en la unidad (cuando exista). |

## 6. Limitaciones

- Los nombres de columnas cambian entre procesos; los scripts aplican normalización pero pueden requerir ajustes para años especiales (ej. revocación de mandato, consulta popular).
- Algunos procesos no incluyen lista nominal a nivel casilla.
- La elección judicial de 2025 usa codificación diferente (cargo judicial, materia/especialidad); debe procesarse en script separado.
