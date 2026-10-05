#!/usr/bin/env python3
"""
build_inventory.py
==================
Genera el inventario final de datos descargados y procesados:
  - docs/INVENTARIO_DATOS.md
  - data/processed/inventario_datos.json

Uso:
  python scripts/build_inventory.py
"""

import json
import pathlib
import sys

import pandas as pd

RAW = pathlib.Path("data/raw")
PROCESSED = pathlib.Path("data/processed")
DOCS = pathlib.Path("docs")


def size_human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def build_inventory() -> dict:
    inv: dict = {"raw": {}, "series": {}, "features": {}, "external": {}, "meta": {}}

    # RAW catalogs
    for scope in ["federal", "local"]:
        cat_path = RAW / f"catalog_{scope}.csv"
        if not cat_path.exists():
            continue
        df = pd.read_csv(cat_path)
        df["year"] = df["anio"].astype(int)
        inv["raw"][scope] = {
            "total_zips": int(len(df)),
            "total_bytes": int(df["bytes"].sum()),
            "total_bytes_human": size_human(int(df["bytes"].sum())),
            "total_internal_files": int(df["archivos_zip"].sum()),
            "years": sorted(df["year"].unique().tolist()),
            "by_year": df.groupby("year")
            .agg(zips=("year", "size"), bytes=("bytes", "sum"), files=("archivos_zip", "sum"))
            .astype({"zips": int, "bytes": int, "files": int})
            .to_dict(orient="index"),
        }
        if scope == "local":
            by_state = (
                df.groupby("nombre")
                .agg(zips=("year", "size"), years=("year", "nunique"), first_year=("year", "min"), last_year=("year", "max"))
                .astype({"zips": int, "years": int, "first_year": int, "last_year": int})
                .sort_values("zips", ascending=False)
                .to_dict(orient="index")
            )
            inv["raw"][scope]["by_state"] = by_state

    # Series
    res_path = PROCESSED / "series/resumen_series.json"
    if res_path.exists():
        inv["series"] = json.loads(res_path.read_text(encoding="utf-8"))

    # Features
    feat_path = PROCESSED / "features/features_metadata.json"
    if feat_path.exists():
        inv["features"]["metadata"] = json.loads(feat_path.read_text(encoding="utf-8"))
    for csv_name in ["features_anuales_nacional.csv", "features_anuales_entidad.csv"]:
        p = PROCESSED / "features" / csv_name
        if p.exists():
            df = pd.read_csv(p)
            inv["features"][csv_name] = {
                "filas": len(df),
                "columnas": list(df.columns),
                "path": str(p),
            }

    # External copies
    ext_dir = pathlib.Path("data/external/lo_que_mexico_cree")
    if ext_dir.exists():
        files = []
        for f in ext_dir.iterdir():
            if f.is_file():
                files.append({"name": f.name, "bytes": f.stat().st_size, "size": size_human(f.stat().st_size)})
        inv["external"]["lo_que_mexico_cree"] = files

    inv["meta"]["total_raw_zips"] = sum(v["total_zips"] for v in inv["raw"].values())
    inv["meta"]["total_raw_bytes"] = sum(v["total_bytes"] for v in inv["raw"].values())
    inv["meta"]["total_raw_bytes_human"] = size_human(inv["meta"]["total_raw_bytes"])
    return inv


def render_markdown(inv: dict) -> str:
    lines = [
        "# Inventario de datos — Inteligencia Electoral México",
        "",
        "Este documento resume los datos disponibles en el repositorio. Los datos crudos",
        "(ZIPs del SICEE) no se versionan en GitHub por tamaño; se reconstruyen con los",
        "scripts de descarga. Las series procesadas se generan localmente con",
        "`scripts/build_series.py`.",
        "",
    ]

    meta = inv.get("meta", {})
    lines += [
        "## Resumen ejecutivo",
        "",
        f"- **ZIPs descargados:** {meta.get('total_raw_zips', 0)}",
        f"- **Tamaño total crudo:** {meta.get('total_raw_bytes_human', 'N/A')}",
        f"- **Archivos internos en ZIPs:** {sum(v.get('total_internal_files', 0) for v in inv.get('raw', {}).values())}",
        f"- **Series procesadas:** {len(inv.get('series', {}))}",
        "",
    ]

    # Raw
    lines += ["## Datos crudos del SICEE", ""]
    for scope in ["federal", "local"]:
        info = inv.get("raw", {}).get(scope)
        if not info:
            continue
        lines += [
            f"### Ámbito: {scope}",
            "",
            f"- **ZIPs:** {info['total_zips']}",
            f"- **Bytes:** {info['total_bytes_human']} ({info['total_bytes']:,} bytes)",
            f"- **Archivos internos:** {info['total_internal_files']}",
            f"- **Años:** {info['years']}",
            "",
            "| Año | ZIPs | Archivos internos | Tamaño |",
            "|-----|------|-------------------|--------|",
        ]
        for y in sorted(info["by_year"].keys()):
            d = info["by_year"][y]
            lines.append(f"| {y} | {d['zips']} | {d['files']} | {size_human(d['bytes'])} |")
        lines.append("")
        if scope == "local" and info.get("by_state"):
            lines += [
                "#### Estados incluidos (local)",
                "",
                "| Estado | ZIPs | Años distintos | Primero | Último |",
                "|--------|------|----------------|---------|--------|",
            ]
            for state, d in info["by_state"].items():
                lines.append(
                    f"| {state} | {d['zips']} | {d['years']} | {d['first_year']} | {d['last_year']} |"
                )
            lines.append("")

    # Series
    lines += ["## Series procesadas", "", "| Serie | Filas | Ruta |", "|-------|-------|------|"]
    for name, data in inv.get("series", {}).items():
        rows = data.get("filas", "N/A")
        path = data.get("path", "")
        lines.append(f"| {name} | {rows:,} | {path} |")
    lines.append("")

    # Features
    feat = inv.get("features", {})
    if feat:
        lines += ["## Variables exógenas procesadas", ""]
        for csv_name, meta_csv in feat.items():
            if csv_name == "metadata":
                continue
            lines.append(f"- `{csv_name}`: {meta_csv.get('filas')} filas, {len(meta_csv.get('columnas', []))} columnas.")
        lines.append("")

    # External
    ext = inv.get("external", {}).get("lo_que_mexico_cree", [])
    if ext:
        lines += ["## Datos externos copiados localmente", "", "| Archivo | Tamaño |", "|---------|--------|"]
        for f in ext:
            lines.append(f"| {f['name']} | {f['size']} |")
        lines.append("")

    # Gaps / notes
    lines += [
        "## Cobertura y omisiones conocidas",
        "",
        "- **Federal:** se tienen todos los años que el SICEE expone para 1991-2024.",
        "  No se listan años sin proceso federal (p. ej. 1993, 1996, 1998, 1999, 2001, 2002,",
        "  2004, 2005, 2007, 2008, 2010, 2011, 2013, 2014, 2016, 2017, 2019, 2020, 2025).",
        "- **Local/Estatal/Municipal:** se tienen todos los paquetes locales 2015-2024.",
        "  Cada estado aparece solo en los años en que tuvo elecciones locales según el calendario.",
        "- **PREP y Cómputos Judiciales:** aún no descargados; están documentados como fuentes",
        "  complementarias en `docs/METODOLOGIA.md`.",
        "- **Datos socioeconómicos por entidad:** se copiaron remesas por entidad; pobreza,",
        "  violencia, PIB estatal y otras variables territoriales están catalogadas en",
        "  `docs/VARIABLES_EXOGENAS.md` pero aún no se descargan automáticamente.",
        "",
    ]

    return "\n".join(lines)


def main(argv=None):
    inv = build_inventory()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    json_path = PROCESSED / "inventario_datos.json"
    json_path.write_text(json.dumps(inv, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = DOCS / "INVENTARIO_DATOS.md"
    md_path.write_text(render_markdown(inv), encoding="utf-8")
    print(f"Inventario JSON: {json_path}")
    print(f"Inventario Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
