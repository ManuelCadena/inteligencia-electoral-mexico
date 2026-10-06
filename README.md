# Inteligencia Electoral México

Dataset y modelos de **Deep Learning** y **NLP** para analizar resultados electorales históricos de México a partir de datos oficiales del **INE** (principalmente el SICEE).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ManuelCadena/inteligencia-electoral-mexico/blob/main/notebooks/01_exploracion_datos.ipynb)

## Objetivo del proyecto

Construir una base de datos reproducible, versionada y lista para modelos neuronales que permita:

1. **Predecir** resultados electorales por entidad, distrito o sección usando series temporales.
2. **Segmentar** territorios con comportamientos electorales similares (clustering espacial-temporal).
3. **Procesar lenguaje** de actas, cómputos y documentos electorales mediante embeddings y modelos de lenguaje en español.
4. **Cuantificar incertidumbre** mediante redes neuronales bayesianas y modelos jerárquicos.

## Fuentes de datos

| Fuente | Cobertura | URL |
|--------|-----------|-----|
| **SICEE** — Sistema de Consulta de la Estadística de las Elecciones | Federales 1991-2024; Locales 2015-2024 | https://sicee.ine.mx/home |
| **PREP** — Programa de Resultados Electorales Preliminares | Corte de la noche electoral (varios procesos) | https://www.ine.mx/voto-y-elecciones/resultados-electorales/ |
| **Cómputos Judiciales 2025** | Proceso Extraordinario PJF 2024-2025 | https://www.ine.mx/voto-y-elecciones/resultados-electorales/ |
| **Lista Nominal** | Padrón/lista nominal por entidad/sección | https://www.ine.mx/credencial/estadisticas-lista-nominal-padron-electoral/ |

## Estado actual de las descargas

En esta sesión se descargaron y verificaron **196 archivos ZIP** del SICEE (INE México):

- **Federal:** 50 archivos (1991-2024), ~1.1 GB.
- **Local:** 146 archivos (2015-2024), ~650 MB.
- **Total:** ~1.75 GB, 0 archivos corruptos.

Series procesadas y listas para modelos en `data/processed/series/`:

| Serie | Filas | Nivel |
|-------|-------|-------|
| `serie_presidencia_entidad.csv` | 3,936 | Entidad |
| `serie_senado_entidad.csv` | 4,992 | Entidad |
| `serie_diputados_fed_mr_entidad.csv` | 7,392 | Entidad |
| `serie_diputados_fed_rp_entidad.csv` | 1,984 | Entidad |
| `serie_estatal_gobernador.csv` | 1,622 | Entidad |
| `serie_estatal_diputados_local.csv` | 101,902 | Distrito |
| `serie_municipal_ayuntamiento.csv` | 362,900 | Municipio |
| `serie_municipal_otros.csv` | 15,936 | Municipio |

> **Nota:** los CSVs generados no se suben a GitHub por tamaño. Se generan
> localmente con `scripts/build_series.py`.

Variables exógenas ya integradas en `data/processed/features/`:

- `features_anuales_nacional.csv`: confianza del consumidor, desempleo,
  inflación, tipo de cambio y remesas.
- `features_anuales_entidad.csv`: remesas por entidad.
- Ver `docs/VARIABLES_EXOGENAS.md` para el catálogo completo de variables
candidatas (incluyendo mañaneras, Google Trends, pobreza, violencia, etc.).

## Estructura del repositorio

```
inteligencia-electoral-mexico/
├── data/
│   ├── raw/                 # ZIPs descargados del SICEE (no versionados)
│   ├── processed/series/    # Series largas tidy listas para modelos
│   └── external/            # PREP, cómputos judiciales, listado nominal
├── scripts/
│   ├── download_sicee.py    # Descarga masiva + catálogo
│   └── build_series.py      # Construye series temporales desde los ZIPs
├── notebooks/
│   ├── 01_exploracion_datos.ipynb
│   └── 02_modelo_lstm_series.ipynb
├── docs/
│   ├── ANTECEDENTES.md      # Antecedentes académicos
│   ├── METODOLOGIA.md       # Pipeline y riesgos metodológicos
│   └── DICCIONARIO_DATOS.md # Descripción de variables
├── references/papers/       # PDFs de papers relevantes
└── results/                 # Figuras, modelos entrenados, métricas
```

## Uso rápido

### 1. Clonar e instalar

```bash
git clone https://github.com/ManuelCadena/inteligencia-electoral-mexico.git
cd inteligencia-electoral-mexico
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Descargar datos federales

```bash
python scripts/download_sicee.py --ambito federal --anios 1991-2024 --workers 2
```

### 3. Descargar datos locales

```bash
python scripts/download_sicee.py --ambito local --anios 2015-2024 --workers 2
```

### 4. Construir series temporales

```bash
python scripts/build_series.py --dir data/raw --out data/processed/series
```

### 5. Abrir notebooks en Google Colab

Ve a `notebooks/` y abre el `.ipynb` con el botón "Open in Colab" o sube el archivo a https://colab.research.google.com.

## Modelos de referencia

- **LSTM / GRU** para series temporales de votación por entidad [5].
- **Transformers tabulares** para mezclar variables categóricas, numéricas y embeddings de texto [1].
- **MLP** como línea base para datos tabulares.
- **Redes neuronales bayesianas** para intervalos de predicción calibrados.
- **Graph Neural Networks (GNN)** para capturar autocorrelación espacial entre secciones/municipios.

Ver [`docs/ANTECEDENTES.md`](docs/ANTECEDENTES.md) para la bibliografía completa y [`docs/METODOLOGIA.md`](docs/METODOLOGIA.md) para el diseño experimental.

**Documento maestro (English):** [`docs/MASTER_DOCUMENTO.html`](docs/MASTER_DOCUMENTO.html) — complete pedagogical master document (self-contained HTML, Harvard-styled) presenting the Mexican electoral system framework, the canonical dataset (463,834 tidy rows; 355,176 canonical rows; 18.1M-word mañaneras corpus) and the 26 applied Deep Learning + NLP lectures with full theory and notebook specs. Generated by `docs/masterdoc/scripts/build_masterdoc.py`.

## Advertencias metodológicas

- **Coaliciones cambiantes:** una misma sigla no representa la misma alianza en diferentes procesos. Se conservan tres vistas: votos por partido, votos por coalición del proceso y votos por candidatura.
- **Reseccionamiento/redistritación:** una sección con el mismo número puede cambiar de territorio. Las comparaciones históricas deben usar agregaciones estables (entidad, municipio) o tablas de correspondencia.
- **Lista nominal variable:** cambios en el padrón afectan la participación. Preferir tasas sobre conteos absolutos.
- **PREP ≠ cómputo final:** el PREP es preliminar; para resultados jurídicos definitivos usar SICEE o cómputos distritales.

## Licencia

Los datos provienen del INE y se usan con fines de investigación académica. El código del repositorio se distribuye bajo MIT License (ver `LICENSE`).
