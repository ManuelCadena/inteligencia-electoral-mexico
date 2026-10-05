#!/usr/bin/env python3
"""
merge_electoral_features.py
===========================
Toma una serie electoral (CSV tidy generado por build_series.py) y le agrega
variables exógenas nacionales y, cuando aplica, por entidad.

Ejemplos:
  python scripts/merge_electoral_features.py \
      --series data/processed/series/serie_presidencia_entidad.csv \
      --features data/processed/features/features_anuales_nacional.csv \
      --entidad-features data/processed/features/features_anuales_entidad.csv \
      --out data/processed/merged/presidencia_con_features.csv

  python scripts/merge_electoral_features.py \
      --series data/processed/series/serie_estatal_gobernador.csv \
      --features data/processed/features/features_anuales_nacional.csv \
      --entidad-features data/processed/features/features_anuales_entidad.csv \
      --out data/processed/merged/estatal_gobernador_con_features.csv

La columna de año se detecta automáticamente (ANIO o anio).
"""

import argparse
import pathlib
import sys

import pandas as pd


def detect_year_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.upper() == "ANIO":
            return c
    raise ValueError("No se encontró columna de año (ANIO/anio)")


def detect_entity_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.upper() in ("NOMBRE_ENTIDAD", "ENTIDAD", "NOMBRE_ESTADO", "ESTADO"):
            return c
    return None


def normalize_entity_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().str.replace(".", "", regex=False)


def merge_features(
    series_path: pathlib.Path,
    features_path: pathlib.Path,
    entidad_features_path: pathlib.Path | None,
    out_path: pathlib.Path,
) -> dict:
    series = pd.read_csv(series_path, encoding="utf-8")
    year_col = detect_year_col(series)
    features = pd.read_csv(features_path, encoding="utf-8")
    feat_year = detect_year_col(features)
    merged = series.merge(
        features,
        left_on=year_col,
        right_on=feat_year,
        how="left",
        suffixes=("", "_feat"),
    ).drop(columns=[feat_year])

    entidad_cols_added = []
    if entidad_features_path and entidad_features_path.exists():
        ent_feat = pd.read_csv(entidad_features_path, encoding="utf-8")
        ent_year = detect_year_col(ent_feat)
        ent_entity = detect_entity_col(ent_feat)
        series_entity = detect_entity_col(series)
        if ent_entity and series_entity:
            ent_feat[ent_entity] = normalize_entity_name(ent_feat[ent_entity])
            merged[series_entity + "_norm"] = normalize_entity_name(merged[series_entity])
            merged = merged.merge(
                ent_feat.rename(columns={ent_year: year_col}),
                left_on=[year_col, series_entity + "_norm"],
                right_on=[year_col, ent_entity],
                how="left",
                suffixes=("", "_ent"),
            )
            merged = merged.drop(columns=[series_entity + "_norm"])
            entidad_cols_added = [c for c in ent_feat.columns if c not in (ent_year, ent_entity)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8")
    return {
        "filas": len(merged),
        "columnas": list(merged.columns),
        "cols_exogenas_nacionales": [c for c in features.columns if c != feat_year],
        "cols_exogenas_entidad": entidad_cols_added,
        "path": str(out_path),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Une series electorales con variables exógenas anuales."
    )
    ap.add_argument("--series", required=True, help="CSV de serie electoral")
    ap.add_argument("--features", required=True, help="CSV de features nacionales anuales")
    ap.add_argument("--entidad-features", help="CSV de features por entidad anuales")
    ap.add_argument("--out", required=True, help="CSV de salida")
    args = ap.parse_args(argv)
    summary = merge_features(
        pathlib.Path(args.series),
        pathlib.Path(args.features),
        pathlib.Path(args.entidad_features) if args.entidad_features else None,
        pathlib.Path(args.out),
    )
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
