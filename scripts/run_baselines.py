#!/usr/bin/env python3
"""T2.2 — Persistence / AR baselines for E-89 electoral forecasting.

Every neural model must beat these numbers to justify itself.
Baselines on panel_features.parquet (federal/entidad):
  1. persistence      : ŷ = share_L1 vector (last election, same cargo/entity)
  2. national_mean    : mean share per (CARGO, party) over train
  3. ridge AR-lite    : Ridge on lag + exogenous features (sklearn, no NN)
Metrics: MAE/RMSE over the 7-party share vector, RMSE on PARTICIPACION,
winner accuracy (argmax match). Val + test reported separately.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/tabular/panel_features.parquet"
OUT_DIR = ROOT / "models/baselines"
PARTIES = ["PAN", "PRI", "PRD_IZQ", "MORENA", "PVEM", "MC_CONV", "OTROS"]
SHARE = [f"share_{p}" for p in PARTIES]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.abs(y_true - y_pred)
    return {
        "mae_share_vec": float(err.mean()),
        "rmse_share_vec": float(np.sqrt(((y_true - y_pred) ** 2).mean())),
        "mae_per_party": {p: float(err[:, i].mean()) for i, p in enumerate(PARTIES)},
        "winner_acc": float((y_true.argmax(1) == y_pred.argmax(1)).mean()),
    }


def main():
    df = pd.read_parquet(PANEL)
    n0 = len(df)
    df = df[df.n_lags >= 1].copy()
    log.info("panel %d -> %d filas con >=1 lag", n0, len(df))

    y_true = df[SHARE].to_numpy()
    out = {}

    for split in ("val", "test"):
        m = df.split == split
        yt, idx = y_true[m], df[m]
        res = {}

        # persistence
        yp = idx[[f"{c}_L1" for c in SHARE]].to_numpy()
        res["persistence"] = metrics(yt, yp)
        # participation persistence
        res["persistence"]["rmse_participacion"] = float(
            np.sqrt(((idx.PARTICIPACION - idx.PARTICIPACION_L1) ** 2).mean())
        )

        # national mean per cargo (fit on train only)
        tr = df[df.split == "train"]
        means = tr.groupby("CARGO")[SHARE].mean()
        yp2 = np.vstack([means.loc[c].to_numpy() for c in idx.CARGO])
        res["national_mean"] = metrics(yt, yp2)

        # ridge AR-lite on lags + exogenous
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        feat = [f"{c}_L{l}" for c in SHARE for l in (1, 2, 3)] + [
            "PARTICIPACION_L1", "MARGEN_VICTORIA_L1", "NEP_L1",
        ] + [
            c for c in idx.columns
            if (c.endswith("_LAG1_NAC") or c.endswith("_LAG1_ENT"))
            and pd.api.types.is_numeric_dtype(idx[c])
        ]
        Xtr, Xv = tr[feat].to_numpy(), idx[feat].to_numpy()
        imp = SimpleImputer(strategy="mean").fit(Xtr)
        rd = Ridge(alpha=1.0).fit(imp.transform(Xtr), tr[SHARE].to_numpy())
        yp3 = np.clip(rd.predict(imp.transform(Xv)), 0, 1)
        res["ridge_ar_lite"] = metrics(yt, yp3)

        out[split] = res
        log.info(
            "%s | persistence MAE %.4f | ridge MAE %.4f | mean MAE %.4f | winner_acc pers %.3f",
            split, res["persistence"]["mae_share_vec"],
            res["ridge_ar_lite"]["mae_share_vec"],
            res["national_mean"]["mae_share_vec"],
            res["persistence"]["winner_acc"],
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # predictions of the strongest baseline for inspection
    df.to_csv(OUT_DIR / "baseline_panel_rows.csv", index=False)
    (OUT_DIR / "baseline_metrics.json").write_text(json.dumps(out, indent=2))
    log.info("-> %s", OUT_DIR / "baseline_metrics.json")


if __name__ == "__main__":
    main()
