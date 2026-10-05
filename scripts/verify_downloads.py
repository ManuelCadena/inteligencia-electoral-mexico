#!/usr/bin/env python3
"""
verify_downloads.py
===================
Valida que todos los ZIPs descargados sean archivos ZIP legibles y muestre
estadísticas de tamaño. Útil después de correr download_sicee.py.

Uso:
  python scripts/verify_downloads.py --dir data/raw
"""

import argparse
import csv
import logging
import pathlib
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("verify")


def verify(base_dir: pathlib.Path):
    zips = sorted(base_dir.rglob("*.zip"))
    ok, bad = 0, []
    total_bytes = 0
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zf:
                n = len(zf.namelist())
            ok += 1
            total_bytes += z.stat().st_size
            log.debug("%s OK (%d archivos internos)", z, n)
        except Exception as e:
            bad.append((z, e))
            log.warning("%s ERROR: %s", z, e)

    print(f"\nZIPs verificados: {ok}/{len(zips)}")
    print(f"ZIPs corruptos: {len(bad)}")
    print(f"Tamaño total: {total_bytes/1e9:.2f} GB\n")

    if bad:
        print("Archivos con error:")
        for z, e in bad:
            print(f"  {z}: {e}")

    # Resumen por año
    resumen = {}
    for z in zips:
        anio_match = pathlib.Path(z).stem[-4:]
        if anio_match.isdigit():
            resumen.setdefault(anio_match, {"n": 0, "bytes": 0})
            resumen[anio_match]["n"] += 1
            resumen[anio_match]["bytes"] += z.stat().st_size

    print("Resumen por año:")
    print("  Año  | Archivos | MB")
    for anio in sorted(resumen):
        info = resumen[anio]
        print(f"  {anio} | {info['n']:>8} | {info['bytes']/1e6:>7.1f}")

    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verifica integridad de ZIPs del SICEE.")
    ap.add_argument("--dir", default="data/raw", help="Carpeta raíz con zips")
    args = ap.parse_args(argv)
    return verify(pathlib.Path(args.dir))


if __name__ == "__main__":
    import sys
    sys.exit(main())
