#!/usr/bin/env python3
"""
build_series.py
===============
Lee los ZIP descargados del SICEE y construye tablas largas (tidy) listas para
modelos de series temporales y NLP:

  - serie_presidencia_entidad.csv
  - serie_senado_entidad.csv
  - serie_diputados_entidad.csv
  - serie_local_entidad.csv (resumen por estado y año local)
  - catalogo_archivos.csv (contenido de cada zip)

Uso:
  python scripts/build_series.py --dir data/raw --out data/processed/series

Requiere: pandas (se instala con `pip install -r requirements.txt`).
"""

import argparse
import csv
import io
import json
import logging
import pathlib
import re
import sys
import zipfile

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("series")

ID_COLUMNS = {
    "ENT": {"CIRCUNSCRIPCION", "ID_ESTADO", "NOMBRE_ESTADO", "ID_ENTIDAD", "NOMBRE_ENTIDAD", "ENTIDAD",
            "TOTAL_VOTOS", "LISTA_NOMINAL", "NUM_VOTOS_NULOS", "NUM_VOTOS_CAN_NREG",
            "NUM_VOTOS_VALIDOS", "SECCIONES", "CASILLAS", "RUTA_ACTA", "OBSERVACIONES", "PARTICIPACION"},
    "DIS": {"CIRCUNSCRIPCION", "ID_ESTADO", "NOMBRE_ESTADO", "ID_DISTRITO", "DISTRITO",
            "ID_DISTRITO_LOCAL", "DISTRITO_LOCAL", "NOMBRE_DISTRITO",
            "CABECERA_DISTRITAL_LOCAL", "CABECERA_DISTRITAL",
            "TOTAL_VOTOS", "LISTA_NOMINAL", "NUM_VOTOS_NULOS", "NUM_VOTOS_CAN_NREG",
            "NUM_VOTOS_VALIDOS", "SECCIONES", "CASILLAS", "RUTA_ACTA", "OBSERVACIONES", "PARTICIPACION"},
    "MUN": {"CIRCUNSCRIPCION", "ID_ESTADO", "NOMBRE_ESTADO", "ID_MUNICIPIO", "MUNICIPIO",
            "NOMBRE_MUNICIPIO", "CABECERA_MUNICIPAL",
            "TOTAL_VOTOS", "LISTA_NOMINAL", "NUM_VOTOS_NULOS", "NUM_VOTOS_CAN_NREG",
            "NUM_VOTOS_VALIDOS", "SECCIONES", "CASILLAS", "RUTA_ACTA", "OBSERVACIONES", "PARTICIPACION"},
    "SEC": {"CIRCUNSCRIPCION", "ID_ESTADO", "NOMBRE_ESTADO", "ID_DISTRITO", "DISTRITO",
            "ID_MUNICIPIO", "MUNICIPIO", "SECCION", "TOTAL_VOTOS", "LISTA_NOMINAL",
            "NUM_VOTOS_NULOS", "NUM_VOTOS_CAN_NREG", "NUM_VOTOS_VALIDOS",
            "CASILLAS", "RUTA_ACTA", "OBSERVACIONES", "PARTICIPACION"},
}


def find_id_level(filename: str) -> str:
    fu = filename.upper()
    if "_MUN." in fu or "_MUNI." in fu:
        return "MUN"
    if "_DIS." in fu or "_DIST." in fu:
        return "DIS"
    if "_SEC." in fu or "_SECC." in fu:
        return "SEC"
    if "_CAS." in fu or "_CASILLA." in fu:
        return "CAS"
    return "ENT"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.replace("\ufeff", "").replace("ï»¿", "").strip().upper() for c in df.columns]
    return df


def numeric_party_columns(df: pd.DataFrame, level: str) -> list[str]:
    exclude = ID_COLUMNS.get(level, set())
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith(("ID_", "NUM_", "TOTAL_", "LISTA_", "CASILLAS", "SECCIONES")):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def entity_column(df: pd.DataFrame, level: str) -> tuple[str, list[str]]:
    """Detecta la(s) columna(s) de identificación geográfica y la columna de entidad."""
    candidates = {
        "ENT": ["NOMBRE_ENTIDAD", "NOMBRE_ESTADO", "ENTIDAD"],
        "DIS": ["ID_DISTRITO", "ID_DISTRITO_LOCAL", "DISTRITO", "DISTRITO_LOCAL",
               "NOMBRE_DISTRITO", "CABECERA_DISTRITAL_LOCAL", "CABECERA_DISTRITAL"],
        "MUN": ["MUNICIPIO", "NOMBRE_MUNICIPIO", "ID_MUNICIPIO", "CABECERA_MUNICIPAL"],
        "SEC": ["SECCION"],
        "CAS": ["CASILLA", "ID_CASILLA"],
    }
    present = [c for c in candidates.get(level, []) if c in df.columns]
    if present:
        # Conservar todas las columnas geográficas presentes (id + nombre).
        return present[0], present
    # Fallback: primera columna de texto que no sea ID_ESTADO
    for c in df.columns:
        if c not in {"CIRCUNSCRIPCION", "ID_ESTADO", "NOMBRE_ESTADO"} and df[c].dtype == object:
            return c, [c]
    return df.columns[0], [df.columns[0]]


def extract_zip_catalog(zf: zipfile.ZipFile, zip_path: pathlib.Path, anio: int) -> list[dict]:
    rows = []
    for name in zf.namelist():
        if name.lower().endswith((".csv", ".xlsx", ".xls")):
            rows.append({
                "anio": anio,
                "zip": zip_path.name,
                "zip_path": str(zip_path),
                "archivo": name,
                "size": zf.getinfo(name).file_size,
            })
    return rows


def read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """Prueba codificaciones comunes del INE."""
    data = zf.read(name)
    for enc in ("latin-1", "utf-8", "iso-8859-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, low_memory=False)
        except Exception:
            continue
    raise ValueError(f"No se pudo leer {name} con codificaciones probadas")


def build_long_for_zip(zip_path: pathlib.Path, pattern: str, folder_tokens: list[str] | None = None) -> pd.DataFrame | None:
    anio = int(re.search(r"(\d{4})", zip_path.name).group(1))
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.upper().endswith(pattern.upper() + ".CSV"):
                continue
            if folder_tokens:
                upper_name = name.upper().replace("_", " ")
                if not any(token.upper() in upper_name for token in folder_tokens):
                    continue
            try:
                df = read_csv_from_zip(zf, name)
            except Exception as e:
                log.warning("%s -> %s: no legible (%s)", zip_path.name, name, e)
                continue
            df = normalize_columns(df)
            # Eliminar filas completamente vacías (pies de página de algunos archivos SICEE).
            df = df.dropna(how="all")
            level = find_id_level(name)
            ent_col, id_cols = entity_column(df, level)
            partidos = numeric_party_columns(df, level)
            if not partidos or ent_col not in df.columns:
                continue
            keep = [c for c in id_cols + ["TOTAL_VOTOS", "LISTA_NOMINAL", "NUM_VOTOS_NULOS", "NUM_VOTOS_CAN_NREG"] if c in df.columns]
            long = df.melt(
                id_vars=keep,
                value_vars=partidos,
                var_name="PARTIDO_COALICION",
                value_name="VOTOS",
            )
            long.insert(0, "ARCHIVO", name.split("/")[-1])
            long.insert(0, "ANIO", anio)
            long.insert(0, "NIVEL", level)
            rows.append(long)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def build_series(raw_dir: pathlib.Path, out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    catalog_rows = []

    # Federal entidad por cargo
    federal_dir = raw_dir / "federal"
    if federal_dir.exists():
        zips = sorted(federal_dir.rglob("*.zip"))
        log.info("Procesando %d zips federales...", len(zips))
        for z in zips:
            anio = int(re.search(r"(\d{4})", z.name).group(1))
            try:
                with zipfile.ZipFile(z) as zf:
                    catalog_rows.extend(extract_zip_catalog(zf, z, anio))
            except Exception as e:
                log.warning("No se pudo abrir %s: %s", z, e)

        for cargo_token, label in [
            ("PRESIDENCIA", "presidencia"),
            ("SENADUR", "senado"),
            ("DIPUTACIONES_FED_MR", "diputados_fed_mr"),
            ("DIPUTACIONES_FED_RP", "diputados_fed_rp"),
        ]:
            parts = []
            for z in zips:
                if cargo_token in z.name.upper():
                    df = build_long_for_zip(z, "_ENT")
                    if df is not None:
                        parts.append(df)
            if parts:
                out = pd.concat(parts, ignore_index=True)
                path = out_dir / f"serie_{label}_entidad.csv"
                out.to_csv(path, index=False, encoding="utf-8")
                summary[label] = {"filas": len(out), "path": str(path)}
                log.info("%s: %s filas -> %s", label, f"{len(out):,}", path)

    # Local: separar elecciones estatales (gobernador/diputados locales) de municipales
    local_dir = raw_dir / "local"
    if local_dir.exists():
        zips = sorted(local_dir.rglob("*.zip"))
        log.info("Procesando %d zips locales...", len(zips))
        for z in zips:
            anio = int(re.search(r"(\d{4})", z.name).group(1))
            try:
                with zipfile.ZipFile(z) as zf:
                    catalog_rows.extend(extract_zip_catalog(zf, z, anio))
            except Exception as e:
                log.warning("No se pudo abrir %s: %s", z, e)

        scopes = [
            ("_ENT", ["GUBERNATURA", "JEFATURA GOBIERNO", "JEFATURA DE GOBIERNO", "ASAMB CONST"],
             "estatal_gobernador", "entidad"),
            ("_DIS", ["DIPUTACIONES LOC"], "estatal_diputados_local", "distrito"),
            ("_MUN", ["AYUNTAMIENTOS", "ALCALDIAS"], "municipal_ayuntamiento", "municipio"),
            ("_MUN", ["JUNTAS MUNICIPALES", "REGIDURIAS", "SINDICATURAS", "PRESIDENCIA DE COMUNIDAD",
                      "PRESIDENCIAS DE COMUNIDAD"], "municipal_otros", "municipio"),
        ]
        for pattern, tokens, label, level_name in scopes:
            parts = []
            for z in zips:
                df = build_long_for_zip(z, pattern, folder_tokens=tokens)
                if df is not None:
                    parts.append(df)
            if parts:
                out = pd.concat(parts, ignore_index=True)
                path = out_dir / f"serie_{label}.csv"
                out.to_csv(path, index=False, encoding="utf-8")
                summary[label] = {"filas": len(out), "path": str(path)}
                log.info("%s: %s filas -> %s", label, f"{len(out):,}", path)

    # Catálogo de archivos
    if catalog_rows:
        cat_df = pd.DataFrame(catalog_rows)
        cat_path = out_dir / "catalogo_archivos.csv"
        cat_df.to_csv(cat_path, index=False, encoding="utf-8")
        summary["catalogo_archivos"] = {"filas": len(cat_df), "path": str(cat_path)}

    # Resumen JSON
    json_path = out_dir / "resumen_series.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Resumen guardado en %s", json_path)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Construye series temporales a partir de zips del SICEE.")
    ap.add_argument("--dir", default="data/raw", help="Carpeta raíz con zips descargados")
    ap.add_argument("--out", default="data/processed/series", help="Carpeta de salida")
    args = ap.parse_args(argv)
    summary = build_series(pathlib.Path(args.dir), pathlib.Path(args.out))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
