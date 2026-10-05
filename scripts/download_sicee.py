#!/usr/bin/env python3
"""
download_sicee.py
=================
Descarga masiva de las bases de datos del SICEE (INE México) y genera un
manifiesto/catalog.csv con metadatos de cada archivo.

Uso (ejemplos):
  python scripts/download_sicee.py --ambito federal --anios 1991-2024
  python scripts/download_sicee.py --ambito federal --cargo PRESIDENCIA --dir data/raw
  python scripts/download_sicee.py --ambito local   --anios 2015-2024 --workers 4
  python scripts/download_sicee.py --ambito federal --list-only

La estructura de salida es:
  <dir>/federal/<año>/<archivo>.zip
  <dir>/local/<año>/<archivo>.zip

Compatible con Google Colab: ejecuta desde la raíz del repo clonado.
"""

import argparse
import csv
import io
import json
import logging
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
API = "https://sicee-api.ine.mx/api/v1/local/downloads/getCargosElecciones"
BASE = "https://sicee.ine.mx/ACTASSICEEN2024/download/"

AMBITOS = {"federal": "FEDERALES_", "local": "LOCALES"}

# Años con elecciones federales conocidas según el SICEE (1991-2024).
# El script acepta cualquier rango; esta lista solo sirve como default.
DEFAULT_FEDERAL_YEARS = "1991-2024"
DEFAULT_LOCAL_YEARS = "2015-2024"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sicee")


# ---------------------------------------------------------------------------
# Utilidades HTTP
# ---------------------------------------------------------------------------
def post(url: str, body: dict, timeout: int = 60) -> list | dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; INE-SICEE-dataset; PhD-research)",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def http_get_bytes(url: str, timeout: int = 600, retries: int = 3) -> bytes:
    """Descarga un recurso con reintentos exponenciales."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; INE-SICEE-dataset; PhD-research)"
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, Exception) as e:
            last_err = e
            wait = min(2 ** attempt * 2, 60)
            log.warning("Reintentando %s en %ds (error: %s)", url, wait, e)
            time.sleep(wait)
    raise last_err


# ---------------------------------------------------------------------------
# Listado de archivos disponibles
# ---------------------------------------------------------------------------
def listar(ambito: str, anio: int) -> list[dict]:
    tipo = AMBITOS[ambito]
    data = post(API, {"tipo": tipo, "anio": anio})
    if not isinstance(data, list):
        raise ValueError(f"Respuesta inesperada para {ambito} {anio}: {data}")
    return data


def url_zip(item: dict) -> str:
    nombre = urllib.parse.quote(item["ruta_zip"])
    if item["tipo_ambito"] == "FEDERALES_":
        return f"{BASE}FEDERALES_{item['anio']}/{nombre}"
    return f"{BASE}{item['tipo_ambito']}/{item['anio']}/{nombre}"


# ---------------------------------------------------------------------------
# Descarga de un archivo individual
# ---------------------------------------------------------------------------
def download_one(item: dict, dest: pathlib.Path, min_size: int = 10_000) -> dict:
    """Descarga un zip si no existe o está incompleto. Retorna metadatos."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = url_zip(item)
    record = {
        "anio": item["anio"],
        "ambito": item.get("tipo_ambito", ""),
        "nombre": item.get("nombre", ""),
        "ruta_zip": item["ruta_zip"],
        "url": url,
        "destino": str(dest),
        "estado": "pendiente",
        "bytes": 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if dest.exists():
        size = dest.stat().st_size
        if size >= min_size:
            record["estado"] = "ya_existe"
            record["bytes"] = size
            return record
        else:
            log.warning("%s existe pero es demasiado pequeño (%d bytes); re-descargando", dest, size)

    try:
        raw = http_get_bytes(url, timeout=600)
        dest.write_bytes(raw)
        # Validación rápida de que sea ZIP
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            namelist = zf.namelist()
        record["estado"] = "descargado"
        record["bytes"] = len(raw)
        record["archivos_zip"] = len(namelist)
    except zipfile.BadZipFile:
        record["estado"] = "error_zip_invalido"
    except Exception as e:
        record["estado"] = f"error_{type(e).__name__}: {e}"
    return record


# ---------------------------------------------------------------------------
# Rango de años
# ---------------------------------------------------------------------------
def parse_rango(txt: str) -> list[int]:
    if "-" not in txt:
        return [int(txt)]
    a, _, b = txt.partition("-")
    start, end = int(a), int(b)
    return list(range(start, end + 1))


# ---------------------------------------------------------------------------
# Manifiesto / catalog
# ---------------------------------------------------------------------------
def write_catalog(records: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("[]", encoding="utf-8")
        return
    keys = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    log.info("Catálogo guardado en %s (%d registros)", path, len(records))


def load_catalog(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# CLI principal
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Descarga bases de datos electorales del SICEE (INE México).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/download_sicee.py --ambito federal --anios 1991-2024
  python scripts/download_sicee.py --ambito federal --cargo PRESIDENCIA --dir data/raw
  python scripts/download_sicee.py --ambito local --anios 2015-2024 --workers 4
  python scripts/download_sicee.py --ambito federal --list-only
        """,
    )
    ap.add_argument("--ambito", choices=["federal", "local"], required=True)
    ap.add_argument(
        "--anios",
        help="Rango de años, ej. 1991-2024. Default: federal 1991-2024, local 2015-2024.",
    )
    ap.add_argument(
        "--cargo",
        default="",
        help="Filtro por texto del nombre del zip, ej. PRESIDENCIA, SENADUR, DIPUTACIONES_FED_MR, AGS.",
    )
    ap.add_argument("--dir", default="data/raw", help="Carpeta raíz de descarga")
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Descargas concurrentes (default: 2; usar <=4 para no saturar al servidor).",
    )
    ap.add_argument(
        "--list-only",
        action="store_true",
        help="Solo lista los archivos disponibles; no descarga.",
    )
    ap.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = ap.parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.anios:
        anios = parse_rango(args.anios)
    else:
        anios = parse_rango(DEFAULT_FEDERAL_YEARS if args.ambito == "federal" else DEFAULT_LOCAL_YEARS)

    base_dir = pathlib.Path(args.dir)
    catalog_path = base_dir / f"catalog_{args.ambito}.csv"

    # ------------------------------------------------------------------
    # Fase 1: listar archivos por año
    # ------------------------------------------------------------------
    items = []
    for anio in anios:
        try:
            year_items = listar(args.ambito, anio)
        except urllib.error.HTTPError as e:
            log.warning("%d: no hay datos o error HTTP (%s)", anio, e)
            continue
        except Exception as e:
            log.warning("%d: error al listar (%s)", anio, e)
            continue
        if not year_items:
            log.info("%d: sin archivos disponibles", anio)
            continue
        for it in year_items:
            it.setdefault("anio", anio)
            it.setdefault("tipo_ambito", AMBITOS[args.ambito])
        items.extend(year_items)
        log.info("%d: %d archivos listados", anio, len(year_items))

    if args.cargo:
        items = [it for it in items if args.cargo.upper() in it["ruta_zip"].upper()]
        log.info("Filtrado por cargo '%s': %d archivos", args.cargo, len(items))

    if args.list_only:
        for it in items:
            print(f"{it['anio']:4d}  {it['nombre']:<45}  {url_zip(it)}")
        return 0

    # ------------------------------------------------------------------
    # Fase 2: descargar con reanudación y concurrencia controlada
    # ------------------------------------------------------------------
    prior = {r["ruta_zip"]: r for r in load_catalog(catalog_path) if r.get("estado") != "error"}
    tasks = []
    for it in items:
        dest = base_dir / args.ambito / str(it["anio"]) / it["ruta_zip"]
        # Si ya está en catálogo y existe, download_one lo detectará
        tasks.append((it, dest))

    records = []
    total = len(tasks)
    workers = max(1, min(args.workers, 6))
    log.info("Iniciando descarga de %d archivos con %d workers...", total, workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_meta = {
            pool.submit(download_one, it, dest): (it, dest) for it, dest in tasks
        }
        for i, future in enumerate(as_completed(future_to_meta), start=1):
            rec = future.result()
            records.append(rec)
            status = rec["estado"]
            mb = rec.get("bytes", 0) / 1e6
            log.info("[%3d/%3d] %s -> %s (%.1f MB)", i, total, rec["ruta_zip"], status, mb)

    # ------------------------------------------------------------------
    # Fase 3: guardar catálogo
    # ------------------------------------------------------------------
    write_catalog(records, catalog_path)
    ok = sum(1 for r in records if r["estado"] in ("descargado", "ya_existe"))
    log.info("Resumen: %d/%d descargados o ya existentes", ok, len(records))
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
