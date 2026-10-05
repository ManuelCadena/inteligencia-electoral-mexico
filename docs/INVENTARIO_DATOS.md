# Inventario de datos — Inteligencia Electoral México

Este documento resume los datos disponibles en el repositorio. Los datos crudos
(ZIPs del SICEE) no se versionan en GitHub por tamaño; se reconstruyen con los
scripts de descarga. Las series procesadas se generan localmente con
`scripts/build_series.py`.

## Resumen ejecutivo

- **ZIPs descargados:** 196
- **Tamaño total crudo:** 1.63 GB
- **Archivos internos en ZIPs:** 6992
- **Series procesadas:** 9

## Datos crudos del SICEE

### Ámbito: federal

- **ZIPs:** 50
- **Bytes:** 1.08 GB (1,157,166,031 bytes)
- **Archivos internos:** 793
- **Años:** [1991, 1992, 1994, 1995, 1997, 2000, 2003, 2006, 2009, 2012, 2015, 2018, 2021, 2022, 2023, 2024]

| Año | ZIPs | Archivos internos | Tamaño |
|-----|------|-------------------|--------|
| 1991 | 3 | 41 | 45.88 MB |
| 1992 | 1 | 6 | 64.50 KB |
| 1994 | 4 | 54 | 64.58 MB |
| 1995 | 1 | 6 | 117.30 KB |
| 1997 | 3 | 41 | 48.59 MB |
| 2000 | 5 | 67 | 77.13 MB |
| 2003 | 3 | 34 | 38.47 MB |
| 2006 | 5 | 71 | 97.08 MB |
| 2009 | 2 | 36 | 49.52 MB |
| 2012 | 5 | 91 | 143.91 MB |
| 2015 | 2 | 36 | 58.58 MB |
| 2018 | 5 | 91 | 193.94 MB |
| 2021 | 4 | 63 | 72.28 MB |
| 2022 | 1 | 6 | 1.40 MB |
| 2023 | 1 | 21 | 1.02 MB |
| 2024 | 5 | 129 | 211.02 MB |

### Ámbito: local

- **ZIPs:** 146
- **Bytes:** 567.80 MB (595,384,527 bytes)
- **Archivos internos:** 6199
- **Años:** [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

| Año | ZIPs | Archivos internos | Tamaño |
|-----|------|-------------------|--------|
| 2015 | 17 | 702 | 54.50 MB |
| 2016 | 15 | 619 | 45.08 MB |
| 2017 | 4 | 172 | 15.02 MB |
| 2018 | 30 | 1223 | 118.85 MB |
| 2019 | 6 | 178 | 10.00 MB |
| 2020 | 2 | 42 | 2.57 MB |
| 2021 | 32 | 1556 | 134.81 MB |
| 2022 | 6 | 179 | 7.35 MB |
| 2023 | 2 | 78 | 7.64 MB |
| 2024 | 32 | 1450 | 171.99 MB |

#### Estados incluidos (local)

| Estado | ZIPs | Años distintos | Primero | Último |
|--------|------|----------------|---------|--------|
| Aguascalientes | 6 | 6 | 2016 | 2024 |
| Coahuila | 6 | 6 | 2017 | 2024 |
| Hidalgo | 6 | 6 | 2016 | 2024 |
| Tamaulipas | 6 | 6 | 2016 | 2024 |
| Quintana Roo | 6 | 6 | 2016 | 2024 |
| Durango | 6 | 6 | 2016 | 2024 |
| Oaxaca | 5 | 5 | 2016 | 2024 |
| Puebla | 5 | 5 | 2016 | 2024 |
| Colima | 5 | 5 | 2015 | 2024 |
| Veracruz | 5 | 5 | 2016 | 2024 |
| Morelos | 4 | 4 | 2015 | 2024 |
| Sinaloa | 4 | 4 | 2016 | 2024 |
| Sonora | 4 | 4 | 2015 | 2024 |
| San Luis Potosí | 4 | 4 | 2015 | 2024 |
| Tabasco | 4 | 4 | 2015 | 2024 |
| Querétaro | 4 | 4 | 2015 | 2024 |
| Tlaxcala | 4 | 4 | 2016 | 2024 |
| Yucatán | 4 | 4 | 2015 | 2024 |
| Nuevo León | 4 | 4 | 2015 | 2024 |
| Zacatecas | 4 | 4 | 2016 | 2024 |
| Baja California | 4 | 4 | 2016 | 2024 |
| Jalisco | 4 | 4 | 2015 | 2024 |
| Guerrero | 4 | 4 | 2015 | 2024 |
| Guanajuato | 4 | 4 | 2015 | 2024 |
| Chihuahua | 4 | 4 | 2016 | 2024 |
| Chiapas | 4 | 4 | 2015 | 2024 |
| Campeche | 4 | 4 | 2015 | 2024 |
| Baja California Sur | 4 | 4 | 2015 | 2024 |
| México | 3 | 3 | 2015 | 2024 |
| Nayarit | 3 | 3 | 2017 | 2024 |
| Estado de México | 3 | 3 | 2017 | 2023 |
| Michoacan | 2 | 2 | 2018 | 2024 |
| Distrito Federal | 2 | 2 | 2015 | 2016 |
| Ciudad de México | 2 | 2 | 2018 | 2024 |
| Michoacán | 2 | 2 | 2015 | 2021 |
| Ciudad De México | 1 | 1 | 2021 | 2021 |

## Series procesadas

| Serie | Filas | Ruta |
|-------|-------|------|
| presidencia | 3,936 | data/processed/series/serie_presidencia_entidad.csv |
| senado | 4,992 | data/processed/series/serie_senado_entidad.csv |
| diputados_fed_mr | 7,392 | data/processed/series/serie_diputados_fed_mr_entidad.csv |
| diputados_fed_rp | 1,984 | data/processed/series/serie_diputados_fed_rp_entidad.csv |
| estatal_gobernador | 1,622 | data/processed/series/serie_estatal_gobernador.csv |
| estatal_diputados_local | 101,902 | data/processed/series/serie_estatal_diputados_local.csv |
| municipal_ayuntamiento | 362,900 | data/processed/series/serie_municipal_ayuntamiento.csv |
| municipal_otros | 15,936 | data/processed/series/serie_municipal_otros.csv |
| catalogo_archivos | 5,879 | data/processed/series/catalogo_archivos.csv |

## Variables exógenas procesadas

- `features_anuales_nacional.csv`: 58 filas, 24 columnas.
- `features_anuales_entidad.csv`: 792 filas, 3 columnas.

## Datos externos copiados localmente

| Archivo | Tamaño |
|---------|--------|
| desempleo_ocde_mensual_1987_2026.csv | 9.29 KB |
| remesas_por_entidad_banxico_trim_2003_2026.csv | 118.80 KB |
| remesas_banxico_CE81_mensual_1995_2026.csv | 54.61 KB |
| inpc_ocde_mensual_1969_2024.csv | 21.38 KB |
| PANEL_ESTIMACION.csv | 7.36 KB |
| tipo_cambio_diario_fred_1993_2026.csv | 154.27 KB |
| enco_5componentes_MENSUAL_2001_2026.csv | 16.34 KB |

## Cobertura y omisiones conocidas

- **Federal:** se tienen todos los años que el SICEE expone para 1991-2024.
  No se listan años sin proceso federal (p. ej. 1993, 1996, 1998, 1999, 2001, 2002,
  2004, 2005, 2007, 2008, 2010, 2011, 2013, 2014, 2016, 2017, 2019, 2020, 2025).
- **Local/Estatal/Municipal:** se tienen todos los paquetes locales 2015-2024.
  Cada estado aparece solo en los años en que tuvo elecciones locales según el calendario.
- **PREP y Cómputos Judiciales:** aún no descargados; están documentados como fuentes
  complementarias en `docs/METODOLOGIA.md`.
- **Datos socioeconómicos por entidad:** se copiaron remesas por entidad; pobreza,
  violencia, PIB estatal y otras variables territoriales están catalogadas en
  `docs/VARIABLES_EXOGENAS.md` pero aún no se descargan automáticamente.
