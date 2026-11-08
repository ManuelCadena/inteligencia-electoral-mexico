#!/usr/bin/env python3
"""T2.4 — Exogenous ablation: does macro/structural signal beat persistence?

Core question of the E-89 paper. On the entity-level panel:
  A. persistence               -> ŷ = share_L1
  B. persistence + exo residual -> ŷ = share_L1 + Ridge(exo_features)
  C. exo only                   -> ŷ = Ridge(exo_features)
  D. lags + exo                 -> ŷ = Ridge(lags + exo)

If B < A on val/test MAE, exogenous features carry signal persistence lacks.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/tabular/panel_features.parquet"
OUT = ROOT / "models/baselines/exo_ablation.json"
PARTIES = ["PAN", "PRI", "PRD_IZQ", "MORENA", "PVEM", "MC_CONV", "OTROS"]
SHARE = [f"share_{p}" for p in PARTIES]


def mae(yt, yp):
    return float(np.abs(yt - yp).mean())


def main():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(PANEL)
    df = df[df.n_lags >= 1].copy()

    exo = [c for c in df.columns if (c.endswith("_LAG1_NAC") or c.endswith("_LAG1_ENT"))
           and pd.api.types.is_numeric_dtype(df[c])]
    lags = [f"{c}_L{l}" for c in SHARE for l in (1, 2, 3)] + [
        "PARTICIPACION_L1", "MARGEN_VICTORIA_L1", "NEP_L1"]
    log.info("exo %d · lags %d", len(exo), len(lags))

    tr = df[df.split == "train"]
    results = {}
    for split in ("val", "test"):
        te = df[df.split == split]
        yt = te[SHARE].to_numpy()
        l1 = te[[f"{c}_L1" for c in SHARE]].to_numpy()
        res = {}

        def fit_pred(feats, target="shares"):
            imp = SimpleImputer(strategy="mean").fit(tr[feats])
            sc = StandardScaler().fit(imp.transform(tr[feats]))
            Xtr = sc.transform(imp.transform(tr[feats]))
            Xte = sc.transform(imp.transform(te[feats]))
            if target == "shares":
                m = Ridge(alpha=1.0).fit(Xtr, tr[SHARE].to_numpy())
                return np.clip(m.predict(Xte), 0, 1)
            m = Ridge(alpha=1.0).fit(Xtr, tr[SHARE].to_numpy() - tr[[f"{c}_L1" for c in SHARE]].to_numpy())
            return np.clip(l1 + m.predict(Xte), 0, 1)

        res["persistence"] = mae(yt, l1)
        res["exo_residual"] = mae(yt, fit_pred(exo, "residual"))
        res["exo_only"] = mae(yt, fit_pred(exo))
        res["lags_plus_exo"] = mae(yt, fit_pred(lags + exo))
        res["lags_only"] = mae(yt, fit_pred(lags))
        results[split] = res
        log.info("%s | " + " | ".join(f"{k} {v:.4f}" for k, v in res.items()), split)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    log.info("-> %s", OUT)


if __name__ == "__main__":
    main()
