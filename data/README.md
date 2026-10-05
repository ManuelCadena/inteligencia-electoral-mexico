# Datos

Esta carpeta contiene los datos del proyecto. **Los archivos grandes (ZIPs, CSVs de series) no se versionan en Git** por `../.gitignore`; se generan localmente con los scripts.

```
data/
├── raw/                 # ZIPs descargados del SICEE
│   ├── federal/<año>/
│   └── local/<año>/
├── processed/series/    # Series largas tidy listas para modelos
├── processed/tabular/   # Paneles y matrices de features
├── processed/text/      # Corpus de texto (actas, programas, comunicados)
└── external/            # PREP, cómputos judiciales 2025, listado nominal
```

Para regenerar los datos:

```bash
python scripts/download_sicee.py --ambito federal --anios 1991-2024 --workers 2
python scripts/download_sicee.py --ambito local   --anios 2015-2024 --workers 2
python scripts/build_series.py --dir data/raw --out data/processed/series
```
