#!/usr/bin/env python3
"""Construye el grafo de adyacencia municipal para el capstone GNN (C1).

Opción A (preferida): shapefile INEGI → adyacencia por contigüidad real (touches).
Opción B (fallback):  centroides → k-NN por distancia haversine.

Salida: data/external/geo/adyacencia_municipios.csv  (cvegeo_a, cvegeo_b, metodo)
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

GEO_DIR = Path("data/external/geo")


def build_from_shapefile(shp_path: Path) -> pd.DataFrame:
    """Adyacencia por contigüidad de frontera via geopandas."""
    import geopandas as gpd  # noqa: import perezoso — dependencia pesada

    gdf = gpd.read_file(shp_path)
    cve_col = next((c for c in gdf.columns if c.upper() in ("CVEGEO", "CVE_GEO", "CVE_MUN")), None)
    if cve_col is None:
        raise ValueError(f"Sin columna CVEGEO en {shp_path}; cols={gdf.columns.tolist()}")
    if cve_col.upper() == "CVE_MUN":
        ent_col = next(c for c in gdf.columns if c.upper() in ("CVE_ENT", "CVE_ENTIDAD"))
        gdf["CVEGEO"] = gdf[ent_col].astype(str).str.zfill(2) + gdf[cve_col].astype(str).str.zfill(3)
        cve_col = "CVEGEO"
    pairs = []
    geoms = gdf.set_index(cve_col).geometry
    for i, (a, ga) in enumerate(geoms.items()):
        for b, gb in geoms.items():
            if a < b and ga.touches(gb):
                pairs.append({"cvegeo_a": a, "cvegeo_b": b, "metodo": "contiguidad_inegi"})
        if i % 500 == 0:
            log.info("fila %d/%d", i, len(geoms))
    return pd.DataFrame(pairs)


def haversine_km(lon1, lat1, lon2, lat2):
    import numpy as np
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_from_centroids(csv_path: Path, k: int = 6) -> pd.DataFrame:
    """k-NN geográfico como fallback documentado."""
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    assert {"cvegeo", "lon", "lat"} <= set(df.columns), "se requieren cvegeo, lon, lat"
    pairs = set()
    for _, r in df.iterrows():
        d = haversine_km(r.lon, r.lat, df.lon.values, df.lat.values)
        idx = d.argsort()[1 : k + 1]
        for j in idx:
            a, b = sorted((str(r.cvegeo), str(df.iloc[j].cvegeo)))
            pairs.add((a, b))
    return pd.DataFrame(sorted(pairs), columns=["cvegeo_a", "cvegeo_b"]).assign(
        metodo=f"knn_centroide_k{k}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shp", type=Path, default=GEO_DIR / "inegi" / "municipios.shp")
    ap.add_argument("--centroids", type=Path, default=GEO_DIR / "municipios_centroides.csv")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    if args.shp.exists():
        out = build_from_shapefile(args.shp)
    elif args.centroids.exists():
        out = build_from_centroids(args.centroids, args.k)
    else:
        raise SystemExit(
            f"Ni {args.shp} ni {args.centroids} existen. Ver data/external/geo/README.md."
        )
    dst = GEO_DIR / "adyacencia_municipios.csv"
    out.to_csv(dst, index=False)
    log.info("aristas: %d -> %s", len(out), dst)


if __name__ == "__main__":
    main()
