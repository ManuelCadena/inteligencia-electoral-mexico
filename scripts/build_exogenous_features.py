#!/usr/bin/env python3
"""
build_exogenous_features.py
===========================
Construye una tabla anual de variables exógenas potencialmente explicativas del
voto en México a partir de fuentes públicas y del proyecto previo
"Lo que México cree" / mañaneras.

Entradas (data/external/lo_que_mexico_cree/):
  - enco_5componentes_MENSUAL_2001_2026.csv
  - PANEL_ESTIMACION.csv
  - desempleo_ocde_mensual_1987_2026.csv
  - inpc_ocde_mensual_1969_2024.csv
  - tipo_cambio_diario_fred_1993_2026.csv
  - remesas_banxico_CE81_mensual_1995_2026.csv
  - remesas_por_entidad_banxico_trim_2003_2026.csv

Salidas (data/processed/features/):
  - features_anuales_nacional.csv  (año × variable)
  - features_anuales_entidad.csv   (año × entidad × remesas)
  - features_metadata.json         (descripción de cada serie)

Uso:
  python scripts/build_exogenous_features.py
"""

import argparse
import json
import logging
import pathlib
import re
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("exogenous")

EXTERNAL = pathlib.Path("data/external/lo_que_mexico_cree")
OUT = pathlib.Path("data/processed/features")


def read_csv(path: pathlib.Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        log.warning("No existe %s", path)
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        log.warning("Error leyendo %s: %s", path, e)
        return None


def enco_annual(df: pd.DataFrame) -> pd.DataFrame:
    df["fecha"] = pd.to_datetime(df["periodo"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["fecha"])
    df["anio"] = df["fecha"].dt.year
    numeric = [c for c in df.columns if c not in ("periodo", "fecha", "anio")]
    agg = df.groupby("anio")[numeric].mean().reset_index()
    agg.columns = ["anio"] + [f"{c}_mean" for c in numeric]
    return agg


def panel_annual(df: pd.DataFrame) -> pd.DataFrame:
    def to_date(x):
        m = re.match(r"(\d{4})-T(\d)", x)
        if m:
            y, q = int(m.group(1)), int(m.group(2))
            month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
            return pd.Timestamp(year=y, month=month, day=1)
        return pd.NaT

    df["fecha"] = df["periodo"].apply(to_date)
    df = df.dropna(subset=["fecha"])
    df["anio"] = df["fecha"].dt.year
    numeric = [c for c in df.columns if c not in ("periodo", "fecha", "anio")]
    agg = df.groupby("anio")[numeric].mean().reset_index()
    agg.columns = ["anio"] + [f"{c}_mean" for c in numeric]
    return agg


def monthly_annual(df: pd.DataFrame, date_col: str, value_col: str | None = None) -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["anio"] = df[date_col].dt.year
    numeric = [c for c in df.columns if c not in (date_col, "anio") and pd.api.types.is_numeric_dtype(df[c])]
    if value_col:
        numeric = [value_col]
    numeric = [c for c in numeric if c != "anio"]
    agg = df.groupby("anio")[numeric].mean().reset_index()
    agg.columns = ["anio"] + [f"{c}_mean" for c in numeric]
    return agg


def remesas_nacional_annual(path: pathlib.Path) -> pd.DataFrame | None:
    # Los archivos de Banxico traen metadata textual hasta la fila con datos.
    start = None
    date_re = re.compile(r"^\d{2}/\d{2}/\d{4}")
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        for i, line in enumerate(f):
            if date_re.match(line.strip().split(",")[0]):
                start = i
                break
    if start is None:
        log.warning("No se encontró inicio de datos en remesas nacional")
        return None
    # La primera fila de datos contiene las fechas; usamos el primer campo como fecha
    # y el segundo como el total de remesas en millones de dólares.
    data = pd.read_csv(path, encoding="latin-1", skiprows=start, header=None, usecols=[0, 1])
    data.columns = ["fecha", "remesas_total_mdd"]
    data["fecha"] = pd.to_datetime(data["fecha"], format="%d/%m/%Y", errors="coerce")
    data = data.dropna(subset=["fecha"])
    data["anio"] = data["fecha"].dt.year
    data["remesas_total_mdd"] = pd.to_numeric(data["remesas_total_mdd"], errors="coerce")
    agg = data.groupby("anio")["remesas_total_mdd"].sum().reset_index()
    agg["remesas_total_mdd"] = agg["remesas_total_mdd"] / 1e3  # miles de millones USD
    return agg


def remesas_entidad_annual(path: pathlib.Path) -> pd.DataFrame | None:
    df = read_csv(path, encoding="latin-1")
    if df is None:
        return None
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["fecha"])
    df["anio"] = df["fecha"].dt.year
    df["mdd"] = pd.to_numeric(df["mdd"], errors="coerce")
    agg = df.groupby(["anio", "entidad"])["mdd"].sum().reset_index()
    agg = agg.rename(columns={"mdd": "remesas_total_mdd"})
    return agg


def build_features(external_dir: pathlib.Path, out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    metadata = {}

    enco = read_csv(external_dir / "enco_5componentes_MENSUAL_2001_2026.csv")
    if enco is not None:
        enco_agg = enco_annual(enco)
        enco_agg = enco_agg.rename(columns={c: f"enco_{c}" for c in enco_agg.columns if c != "anio"})
        frames.append(enco_agg)
        metadata["enco_5componentes"] = {
            "source": "INEGI ENCO",
            "freq": "mensual -> anual",
            "variables": [c for c in enco_agg.columns if c != "anio"],
        }

    panel = read_csv(external_dir / "PANEL_ESTIMACION.csv")
    if panel is not None:
        panel_agg = panel_annual(panel)
        panel_agg = panel_agg.rename(columns={c: f"panel_{c}" for c in panel_agg.columns if c != "anio"})
        frames.append(panel_agg)
        metadata["panel_estimacion"] = {
            "source": "Proyecto Lo que México cree / INEGI/Banxico/CONASAMI",
            "freq": "trimestral -> anual",
            "variables": [c for c in panel_agg.columns if c != "anio"],
        }

    desempleo = read_csv(external_dir / "desempleo_ocde_mensual_1987_2026.csv")
    if desempleo is not None:
        de_agg = monthly_annual(
            desempleo.rename(columns={desempleo.columns[1]: "desempleo_pct"}),
            desempleo.columns[0],
        )
        de_agg = de_agg.rename(columns={c: f"ocde_{c}" for c in de_agg.columns if c != "anio"})
        frames.append(de_agg)
        metadata["desempleo_ocde"] = {
            "source": "OECD/FRED",
            "freq": "mensual -> anual",
            "variables": [c for c in de_agg.columns if c != "anio"],
        }

    inpc = read_csv(external_dir / "inpc_ocde_mensual_1969_2024.csv")
    if inpc is not None:
        inpc_agg = monthly_annual(
            inpc.rename(columns={inpc.columns[1]: "inpc_indice"}),
            inpc.columns[0],
        )
        inpc_agg = inpc_agg.rename(columns={c: f"ocde_{c}" for c in inpc_agg.columns if c != "anio"})
        frames.append(inpc_agg)
        metadata["inpc_ocde"] = {
            "source": "OECD/FRED",
            "freq": "mensual -> anual",
            "variables": [c for c in inpc_agg.columns if c != "anio"],
        }

    tc = read_csv(external_dir / "tipo_cambio_diario_fred_1993_2026.csv")
    if tc is not None:
        tc_agg = monthly_annual(
            tc.rename(columns={tc.columns[1]: "tipo_cambio_mxn_usd"}),
            tc.columns[0],
        )
        tc_agg = tc_agg.rename(columns={c: f"fred_{c}" for c in tc_agg.columns if c != "anio"})
        frames.append(tc_agg)
        metadata["tipo_cambio_fred"] = {
            "source": "FRED",
            "freq": "diario -> anual",
            "variables": [c for c in tc_agg.columns if c != "anio"],
        }

    rem_path = external_dir / "remesas_banxico_CE81_mensual_1995_2026.csv"
    if rem_path.exists():
        rem_agg = remesas_nacional_annual(rem_path)
        if rem_agg is not None:
            frames.append(rem_agg)
            metadata["remesas_banxico"] = {
                "source": "Banxico SE27803",
                "freq": "mensual -> anual (suma anual en miles de millones USD)",
                "variables": ["remesas_total_mdd"],
            }

    national = frames[0]
    for f in frames[1:]:
        national = national.merge(f, on="anio", how="outer")
    national = national.sort_values("anio").reset_index(drop=True)
    national_path = out_dir / "features_anuales_nacional.csv"
    national.to_csv(national_path, index=False, encoding="utf-8")
    log.info("features_anuales_nacional: %s filas -> %s", f"{len(national):,}", national_path)

    rem_ent = remesas_entidad_annual(external_dir / "remesas_por_entidad_banxico_trim_2003_2026.csv")
    if rem_ent is not None:
        rem_ent = rem_ent.sort_values(["anio", "entidad"]).reset_index(drop=True)
        ent_path = out_dir / "features_anuales_entidad.csv"
        rem_ent.to_csv(ent_path, index=False, encoding="utf-8")
        log.info("features_anuales_entidad: %s filas -> %s", f"{len(rem_ent):,}", ent_path)
        metadata["remesas_por_entidad"] = {
            "source": "Banxico",
            "freq": "trimestral -> anual (suma anual)",
            "variables": ["remesas_total_mdd"],
        }

    meta_path = out_dir / "features_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "nacional": {"filas": len(national), "columnas": list(national.columns), "path": str(national_path)},
        "entidad": {"filas": len(rem_ent) if rem_ent is not None else 0, "path": str(ent_path) if rem_ent is not None else None},
    }
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Construye variables exógenas anuales para modelos electorales.")
    ap.add_argument("--external", default=str(EXTERNAL), help="Carpeta con CSVs externos")
    ap.add_argument("--out", default=str(OUT), help="Carpeta de salida")
    args = ap.parse_args(argv)
    summary = build_features(pathlib.Path(args.external), pathlib.Path(args.out))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
