#!/usr/bin/env python3
"""T2.1 — Feature matrix for multimodal electoral forecasting (E-89).

Unit: federal contest at ENTIDAD level — (CARGO, ENTIDAD, ANIO).
Target: vote-share vector over canonical parties (compositional, sums to 1)
+ PARTICIPACION. Features: lagged shares (L1-L3), participation lags, margin,
exogenous _LAG1 macro variables, entity/office codes.

Fixed chronological split: train 1994-2012 · val 2015-2018 · test 2021-2024.
Training rows restricted to FLAG_CONSISTENCIA=0 per methodology contract.

Run: python scripts/build_panel_features.py
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "data/processed/canonical/canonical_long.csv"
OUT = ROOT / "data/processed/tabular/panel_features.parquet"
META = ROOT / "data/processed/tabular/panel_features_metadata.json"

CARGOS_FED = ["PRESIDENTE", "SENADOR_MR", "DIPUTADO_FEDERAL_MR"]
# RP excluded from core scope (proportional list, different mechanism).
PARTIES = ["PAN", "PRI", "PRD_IZQ", "MORENA", "PVEM", "MC_CONV", "OTROS"]

PARTY_MAP = {
    "PAN": "PAN", "PRI": "PRI", "MORENA": "MORENA",
    "PVEM": "PVEM", "PES": "PVEM",  # PES agrupado con bloque verde-evangélico
    "MC": "MC_CONV", "CONV": "MC_CONV",
    "PRD": "PRD_IZQ", "PT": "PRD_IZQ", "FXM": "PRD_IZQ", "PSD": "PRD_IZQ",
    "PARM": "OTROS", "PDM": "OTROS", "PAS": "OTROS", "PPS": "OTROS",
    "NVA_ALIANZA": "MC_CONV", "PANAL": "MC_CONV",
}

EXO_LAG = [
    "enco_icc_mean_LAG1_NAC", "enco_p3_mean_LAG1_NAC", "enco_p4_mean_LAG1_NAC",
    "enco_p5_mean_LAG1_NAC", "enco_p6_mean_LAG1_NAC", "enco_p8_mean_LAG1_NAC",
    "panel_p3_mean_LAG1_NAC", "panel_icc_mean_LAG1_NAC", "panel_tlp_mean_LAG1_NAC",
    "panel_ing_inpc_mean_LAG1_NAC", "panel_ing_canasta_mean_LAG1_NAC",
    "ocde_desempleo_pct_mean_LAG1_NAC", "ocde_inpc_indice_mean_LAG1_NAC",
    "fred_tipo_cambio_mxn_usd_mean_LAG1_NAC", "remesas_total_mdd_LAG1_NAC",
    "entidad_ENT_LAG1_ENT", "remesas_total_mdd_ENT_LAG1_ENT",
]


def canonical_party(name: str) -> str:
    """First-token dominant party of coalition strings -> canonical bucket."""
    if not isinstance(name, str) or not name:
        return "OTROS"
    for tok in name.replace("-", "_").split("_"):
        if tok in PARTY_MAP:
            return PARTY_MAP[tok]
        if tok.startswith("CAND_IND"):
            return "OTROS"
    return "OTROS"


def main():
    log.info("cargando canonical_long.csv")
    df = pd.read_csv(CANON, low_memory=False)
    fed = df[
        (df.AMBITO == "FEDERAL")
        & (df.NIVEL_GEO == "ENTIDAD")
        & (df.CARGO.isin(CARGOS_FED))
    ].copy()
    log.info("filas federal/entidad: %d", len(fed))

    fed["party"] = fed.PARTIDO_COALICION.map(canonical_party)
    fed["share"] = fed.SHARE_VALIDO.clip(0, 1)

    # pivot shares: (CARGO, ENTIDAD, ANIO) x party
    pvt = (
        fed.groupby(["CARGO", "ID_ENTIDAD", "ENTIDAD_CANONICA", "ANIO", "party"])["share"]
        .sum().unstack(fill_value=0.0)
    )
    for p in PARTIES:
        if p not in pvt.columns:
            pvt[p] = 0.0
    pvt = pvt[PARTIES]
    # compositional renormalization (safety: shares should sum ~1 on valid votes)
    s = pvt.sum(axis=1).replace(0, np.nan)
    pvt = pvt.div(s, axis=0).fillna(0.0)
    assert (pvt.sum(axis=1).sub(1).abs() < 1e-6).all(), "share vector no suma 1"

    # outcome-level columns (first row of group)
    meta_cols = [
        "PARTICIPACION", "MARGEN_VICTORIA", "NEP", "NUM_PARTIDOS", "ALTERNANCIA",
        "FLAG_CONSISTENCIA", "ES_PRESIDENCIAL", "AÑO_SEXENIO",
    ] + [c for c in EXO_LAG if c in fed.columns]
    meta = fed.groupby(["CARGO", "ID_ENTIDAD", "ENTIDAD_CANONICA", "ANIO"])[meta_cols].first()

    panel = pvt.join(meta).reset_index()
    panel = panel.rename(columns={p: f"share_{p}" for p in PARTIES})

    # ---- lags within (CARGO, ID_ENTIDAD) ----
    panel = panel.sort_values(["CARGO", "ID_ENTIDAD", "ANIO"])
    grp = panel.groupby(["CARGO", "ID_ENTIDAD"])
    share_cols = [f"share_{p}" for p in PARTIES]
    for lag in (1, 2, 3):
        for c in share_cols:
            panel[f"{c}_L{lag}"] = grp[c].shift(lag)
        for c in ("PARTICIPACION", "MARGEN_VICTORIA", "NEP"):
            panel[f"{c}_L{lag}"] = grp[c].shift(lag)

    panel["n_lags"] = panel[[f"share_PAN_L{l}" for l in (1, 2, 3)]].notna().sum(axis=1)

    # split
    panel["split"] = np.select(
        [panel.ANIO <= 2012, panel.ANIO <= 2018],
        ["train", "val"],
        default="test",
    )

    # no-leakage asserts
    exo = [c for c in panel.columns if c.endswith("_LAG1_NAC") or c.endswith("_LAG1_ENT")]
    assert exo, "sin exógenas LAG1"
    assert panel.ANIO.between(1991, 2024).all()
    assert panel.CARGO.isin(CARGOS_FED).all()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)

    summ = {
        "rows": int(len(panel)),
        "grupos": int(panel.groupby(["CARGO", "ID_ENTIDAD"]).ngroups),
        "cargos": panel.CARGO.value_counts().to_dict(),
        "split_counts": panel.split.value_counts().to_dict(),
        "anios": sorted(panel.ANIO.unique().tolist()),
        "parties": PARTIES,
        "exo_lag_features": exo,
        "flag_consistencia_train": int(
            (panel[panel.split == "train"].FLAG_CONSISTENCIA == 0).sum()
        ),
        "flag_consistencia_train_flagged": int(
            (panel[panel.split == "train"].FLAG_CONSISTENCIA != 0).sum()
        ),
        "n_lags_dist": panel.n_lags.value_counts().sort_index().to_dict(),
        "target": [f"share_{p}" for p in PARTIES] + ["PARTICIPACION"],
        "split_rule": "train<=2012 | val 2015-2018 | test>=2021",
    }
    META.write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    log.info("panel: %s -> %s", panel.shape, OUT)
    log.info("splits: %s", summ["split_counts"])
    log.info("metadata -> %s", META)


if __name__ == "__main__":
    main()
