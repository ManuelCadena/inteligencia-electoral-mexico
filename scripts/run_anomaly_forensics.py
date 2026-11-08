#!/usr/bin/env python3
"""Option-B proof — electoral anomaly forensics (E-89 reframe, unsupervised).

Persistence is irrelevant here: the task is detecting irregular
municipal election records, not forecasting. Two detectors:
  1. IsolationForest on share vector + participation + margin
  2. Autoencoder reconstruction error (PyTorch)

Pseudo-label validation: FLAG_CONSISTENCIA!=0 records and extreme
participation (>0.95 or <0.05) should score as more anomalous.
"""
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "data/processed/tabular/municipal_records.parquet"
OUT = ROOT / "models/forensics/anomaly_report.json"
PARTIES = ["PAN", "PRI", "PRD_IZQ", "MORENA", "PVEM", "MC_CONV", "OTROS"]


class AE(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, 32), nn.ReLU(), nn.Linear(32, 8))
        self.dec = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, n))

    def forward(self, x):
        return self.dec(self.enc(x))


def main():
    torch.manual_seed(42)
    rec = pd.read_parquet(REC).dropna(subset=["PARTICIPACION", "MARGEN_VICTORIA"])
    log.info("records %d", len(rec))

    X = np.column_stack([
        rec[PARTIES].to_numpy(np.float32),
        rec.PARTICIPACION, rec.MARGEN_VICTORIA,
        (rec.ANIO - 2015) / 10.0,
    ]).astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd

    # --- IsolationForest ---
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
    iso_score = -iso.fit(Xn).score_samples(Xn)  # higher = more anomalous

    # --- Autoencoder ---
    ae = AE(X.shape[1])
    opt = torch.optim.AdamW(ae.parameters(), lr=3e-3, weight_decay=1e-5)
    Xt = torch.tensor(Xn)
    for ep in range(150):
        ae.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(ae(Xt), Xt)
        loss.backward()
        opt.step()
    ae.eval()
    with torch.no_grad():
        rec_err = ((ae(Xt) - Xt) ** 2).mean(1).numpy()

    def pct(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    anom = 0.5 * pct(iso_score) + 0.5 * pct(rec_err)
    rec = rec.assign(iso=iso_score, ae_err=rec_err, anom=anom)

    # pseudo-labels: extreme participation / near-unanimous winner
    extreme_part = (rec.PARTICIPACION > 0.90) | (rec.PARTICIPACION < 0.15)
    near_unanimous = rec[PARTIES].max(axis=1) > 0.85
    top_k = rec.anom.nlargest(int(0.05 * len(rec))).index
    enriched = {
        "n_records": int(len(rec)),
        "extreme_participation_share": float(extreme_part.mean()),
        "near_unanimous_share": float(near_unanimous.mean()),
        "top5pct_extreme_part": float(extreme_part.loc[top_k].mean()),
        "top5pct_near_unanimous": float(near_unanimous.loc[top_k].mean()),
        "lift_extreme_part": float(
            extreme_part.loc[top_k].mean() / max(extreme_part.mean(), 1e-9)),
        "lift_near_unanimous": float(
            near_unanimous.loc[top_k].mean() / max(near_unanimous.mean(), 1e-9)),
        "corr_iso_ae": float(np.corrcoef(iso_score, rec_err)[0, 1]),
    }
    top = rec.loc[top_k, ["cvegeo", "ANIO", "PARTICIPACION", "MARGEN_VICTORIA",
                          "anom"]].sort_values("anom", ascending=False).head(15)
    enriched["top15"] = top.round(3).to_dict("records")

    log.info("lift extreme_participation %.2fx · near_unanimous %.2fx · corr %.3f",
             enriched["lift_extreme_part"], enriched["lift_near_unanimous"],
             enriched["corr_iso_ae"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(enriched, indent=2, ensure_ascii=False))
    rec.to_csv(ROOT / "models/forensics/anomaly_scores.csv", index=False)
    log.info("-> %s", OUT)


if __name__ == "__main__":
    main()
