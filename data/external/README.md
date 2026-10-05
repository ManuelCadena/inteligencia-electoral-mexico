# Datos externos / exógenos

Esta carpeta contiene series auxiliares que pueden explicar o modular resultados
electorales. Los archivos CSV no se versionan en GitHub por provenir de otros
proyectos; se copian localmente desde las rutas indicadas.

## Fuentes incluidas (proyecto *Lo que México cree*)

| Archivo | Origen local | Frecuencia | Variables |
|---|---|---|---|
| `enco_5componentes_MENSUAL_2001_2026.csv` | `Lo que Mexico Cree/Datos/paquete_datos/` | Mensual | ICC, P3-P8 (confianza del consumidor INEGI) |
| `PANEL_ESTIMACION.csv` | `Lo que Mexico Cree/Datos/paquete_datos/` | Trimestral | ICC/P3-P8, TLP, ingreso, banderas COVID/post2019 |
| `desempleo_ocde_mensual_1987_2026.csv` | `Lo que Mexico Cree/Datos/paquete_datos/series_extendidas/` | Mensual | Tasa de desempleo (OECD/FRED) |
| `inpc_ocde_mensual_1969_2024.csv` | `Lo que Mexico Cree/Datos/paquete_datos/series_extendidas/` | Mensual | Índice de precios al consumidor |
| `tipo_cambio_diario_fred_1993_2026.csv` | `Lo que Mexico Cree/Datos/paquete_datos/series_extendidas/` | Diario | Tipo de cambio MXN/USD |
| `remesas_banxico_CE81_mensual_1995_2026.csv` | `Lo que Mexico Cree/Datos/paquete_datos/series_extendidas/` | Mensual | Remesas familiares (Banxico) |
| `remesas_por_entidad_banxico_trim_2003_2026.csv` | `Lo que Mexico Cree/Datos/paquete_datos/series_extendidas/` | Trimestral | Remesas por entidad |

## Variables propuestas para explicar el voto

Las variables se agrupan por canal teórico, siguiendo el marco del proyecto de
mañaneras (*What Mexico Hears vs. What Mexico Believes*):

1. **Canal económico / bienestar**
   - `enco_icc_mean`, `enco_p3_mean`..`p8_mean`: confianza del consumidor.
   - `panel_tlp_mean`, `panel_ing_*_mean`: salario mínimo real e ingresos.
   - `ocde_desempleo_pct_mean`: desempleo.
   - `ocde_inpc_indice_mean`: inflación.
   - `remesas_total_mdd`: remesas familiares (nacional y por entidad).
2. **Canal tipo de cambio / competitividad**
   - `fred_tipo_cambio_mxn_usd_mean`: peso/dólar.
3. **Canal discursivo (mañaneras)** — *requiere generación propia*
   - Volumen de palabras/intervenciones del presidente.
   - Proporciones de tópicos STM/LDA (`topic_*`).
   - Sentimiento promedio de la rueda de prensa (`sentiment_bilstm_mean`).
   - Atención: búsquedas Google Trends por programa/tema (`gt_*`).
   - Aprobación presidencial (`approval_mean`).
4. **Controles temporales**
   - Año, administración, elección presidencial/intermedia, pandemia COVID.

## Cómo reconstruir

```bash
python scripts/build_exogenous_features.py
python scripts/merge_electoral_features.py \
  --series data/processed/series/serie_presidencia_entidad.csv \
  --out data/processed/merged/presidencia_con_features.csv
```

## Nota metodológica

Todas las series exógenas se agregan a nivel anual (promedio para niveles,
suma para flujos como remesas). Para predecir elecciones locales o federales
específicas, lo ideal es usar el valor del mes/trimestre previo a la elección,
no el promedio anual. El script base sirve como punto de partida; el usuario
puede refinar ventanas temporales en los notebooks.
