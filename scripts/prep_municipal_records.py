#!/usr/bin/env python3
"""T2.6 prep — municipal election records to parquet (pandas-only process).

Separated from train_gnn_municipal.py because pandas' C CSV parser +
torch in the same process triggers an OMP pthread_mutex crash on this
macOS env. Run this FIRST, then the trainer reads the parquet.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "data/processed/canonical/canonical_long.csv"
DST = ROOT / "data/processed/tabular/municipal_records.parquet"
PARTIES = ["PAN", "PRI", "PRD_IZQ", "MORENA", "PVEM", "MC_CONV", "OTROS"]

PARTY_MAP = {
    "PAN": "PAN", "PRI": "PRI", "MORENA": "MORENA",
    "PVEM": "PVEM", "PES": "PVEM",
    "MC": "MC_CONV", "CONV": "MC_CONV", "NVA_ALIANZA": "MC_CONV", "PANAL": "MC_CONV",
    "PRD": "PRD_IZQ", "PT": "PRD_IZQ", "FXM": "PRD_IZQ", "PSD": "PRD_IZQ",
    "PARM": "OTROS", "PDM": "OTROS", "PAS": "OTROS", "PPS": "OTROS",
}


def canonical_party(name):
    if not isinstance(name, str) or not name:
        return "OTROS"
    for tok in name.replace("-", "_").split("_"):
        if tok in PARTY_MAP:
            return PARTY_MAP[tok]
    return "OTROS"


def main():
    df = pd.read_csv(CANON, low_memory=False)
    m = df[(df.NIVEL_GEO == "MUNICIPIO") & (df.CARGO == "AYUNTAMIENTO")].copy()
    m = m.dropna(subset=["ID_ENTIDAD", "ID_MUNICIPIO"])
    m["cvegeo"] = (
        m.ID_ENTIDAD.astype(int).astype(str).str.zfill(2)
        + m.ID_MUNICIPIO.astype(int).astype(str).str.zfill(3)
    )
    m["party"] = m.PARTIDO_COALICION.map(canonical_party)
    m["share"] = m.SHARE_VALIDO.clip(0, 1)

    pvt = m.groupby(["cvegeo", "ANIO", "party"])["share"].sum().unstack(fill_value=0.0)
    for p in PARTIES:
        if p not in pvt.columns:
            pvt[p] = 0.0
    pvt = pvt[PARTIES]
    pvt = pvt.div(pvt.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    meta = m.groupby(["cvegeo", "ANIO"])[["PARTICIPACION", "MARGEN_VICTORIA"]].first()
    rec = pvt.join(meta).reset_index().sort_values(["cvegeo", "ANIO"])

    DST.parent.mkdir(parents=True, exist_ok=True)
    rec.to_parquet(DST, index=False)
    log.info("records %d -> %s", len(rec), DST)

    # adjacency as parquet too (read_csv C-engine + torch = OMP crash)
    adj = pd.read_csv(ROOT / "data/external/geo/adyacencia_municipios.csv", dtype=str)
    adj_dst = ROOT / "data/processed/tabular/adyacencia_municipios.parquet"
    adj.to_parquet(adj_dst, index=False)
    log.info("adjacency %d -> %s", len(adj), adj_dst)


if __name__ == "__main__":
    main()
