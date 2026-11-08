#!/usr/bin/env python3
"""T2.3 — Tabular MLP baseline (PyTorch) for E-89 electoral forecasting.

Input: lag features (L1-L3 shares, participation, margin, NEP) + exogenous
_LAG1 macro + one-hot CARGO + ID_ENTIDAD embedding.
Output: 7-party vote-share vector via softmax + PARTICIPACION head.
Loss: KL on shares (compositional) + MSE on participation.
Chronological split, early stopping on val; the number to beat is
persistence MAE (test 0.0496 / val 0.1065, winner_acc 0.852).

Run: python scripts/train_mlp_panel.py [--epochs 300]
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

log = logging.getLogger(__name__)
logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/tabular/panel_features.parquet"
OUT_DIR = ROOT / "models/mlp"
PARTIES = ["PAN", "PRI", "PRD_IZQ", "MORENA", "PVEM", "MC_CONV", "OTROS"]
SHARE = [f"share_{p}" for p in PARTIES]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class PanelMLP(nn.Module):
    """Residual MLP: predicts delta over last-election shares (L1 skip)."""

    def __init__(self, n_feat, n_ent, emb=8, hid=64, drop=0.15):
        super().__init__()
        self.emb = nn.Embedding(n_ent, emb)
        self.trunk = nn.Sequential(
            nn.Linear(n_feat + emb + 3, hid), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(drop),
        )
        self.head_share = nn.Linear(hid, len(PARTIES))
        self.head_part = nn.Linear(hid, 1)
        nn.init.zeros_(self.head_share.weight)
        nn.init.zeros_(self.head_share.bias)

    def forward(self, x, ent, cargo_oh, share_l1):
        h = self.trunk(torch.cat([x, self.emb(ent), cargo_oh], dim=1))
        logit = self.head_share(h) + torch.log(share_l1.clamp_min(1e-4))
        return torch.softmax(logit, dim=1), torch.sigmoid(self.head_part(h)).squeeze(-1)


def load_split(df, split, feat, ent2idx, cargo2idx, scaler):
    d = df[df.split == split]
    X = scaler.transform(d[feat].to_numpy(np.float32))
    ent = torch.tensor([ent2idx[e] for e in d.ID_ENTIDAD], dtype=torch.long)
    cargo = torch.zeros(len(d), 3)
    for i, c in enumerate(d.CARGO):
        cargo[i, cargo2idx[c]] = 1.0
    y = torch.tensor(d[SHARE].to_numpy(np.float32))
    yp = torch.tensor(d.PARTICIPACION.to_numpy(np.float32))
    l1 = torch.tensor(d[[f"{c}_L1" for c in SHARE]].to_numpy(np.float32))
    return (torch.tensor(X), ent, cargo, l1, y, yp)


def evaluate(model, tensors):
    model.eval()
    x, ent, cargo, l1, y, yp = [t.to(DEVICE) for t in tensors]
    with torch.no_grad():
        sh, part = model(x, ent, cargo, l1)
    sh, part = sh.cpu().numpy(), part.cpu().numpy()
    y, yp = y.cpu().numpy(), yp.cpu().numpy()
    return {
        "mae_share_vec": float(np.abs(y - sh).mean()),
        "rmse_share_vec": float(np.sqrt(((y - sh) ** 2).mean())),
        "mae_per_party": {p: float(np.abs(y - sh)[:, i].mean()) for i, p in enumerate(PARTIES)},
        "winner_acc": float((y.argmax(1) == sh.argmax(1)).mean()),
        "rmse_participacion": float(np.sqrt(((yp - part) ** 2).mean())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_parquet(PANEL)
    df = df[df.n_lags >= 1].copy()

    exo = [c for c in df.columns if (c.endswith("_LAG1_NAC") or c.endswith("_LAG1_ENT"))
           and pd.api.types.is_numeric_dtype(df[c])]
    feat = [f"{c}_L{l}" for c in SHARE for l in (1, 2, 3)] + [
        "PARTICIPACION_L1", "PARTICIPACION_L2", "PARTICIPACION_L3",
        "MARGEN_VICTORIA_L1", "NEP_L1", "AÑO_SEXENIO",
    ] + exo
    log.info("features: %d (exo %d)", len(feat), len(exo))

    ent2idx = {e: i for i, e in enumerate(sorted(df.ID_ENTIDAD.unique()))}
    cargo2idx = {"PRESIDENTE": 0, "SENADOR_MR": 1, "DIPUTADO_FEDERAL_MR": 2}

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    tr = df[df.split == "train"]
    imp = SimpleImputer(strategy="mean").fit(tr[feat])

    class Scaler:
        def transform(self, X):
            return np.nan_to_num(imp.transform(X), nan=0.0).astype(np.float32)

    scaler = Scaler()
    tr_t = load_split(df, "train", feat, ent2idx, cargo2idx, scaler)
    va_t = load_split(df, "val", feat, ent2idx, cargo2idx, scaler)
    te_t = load_split(df, "test", feat, ent2idx, cargo2idx, scaler)

    # standardize features (fit on train)
    mean = tr_t[0].mean(0, keepdim=True)
    std = tr_t[0].std(0, keepdim=True).clamp_min(1e-6)
    for t in (tr_t, va_t, te_t):
        t = (t[0] - mean) / std

    tr_x = (tr_t[0] - mean) / std
    va_x = (va_t[0] - mean) / std
    te_x = (te_t[0] - mean) / std
    tr_t = (tr_x, *tr_t[1:])
    va_t = (va_x, *va_t[1:])
    te_t = (te_x, *te_t[1:])

    model = PanelMLP(len(feat), len(ent2idx), hid=48, drop=0.25).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    xtr, etr, ctr, l1tr, ytr, yptr = [t.to(DEVICE) for t in tr_t]

    best_val, best_state, patience = np.inf, None, 0
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        sh, part = model(xtr, etr, ctr, l1tr)
        loss = nn.functional.mse_loss(sh, ytr) + \
            0.2 * nn.functional.mse_loss(part, yptr)
        loss.backward()
        opt.step()
        sched.step()
        m = evaluate(model, va_t)
        if m["mae_share_vec"] < best_val - 1e-5:
            best_val, best_state, patience = m["mae_share_vec"], model.state_dict(), 0
        else:
            patience += 1
            if patience >= 40:
                break
        if ep % 50 == 0:
            log.info("ep %d loss %.4f val MAE %.4f", ep, loss.item(), m["mae_share_vec"])

    model.load_state_dict(best_state)
    results = {
        "val": evaluate(model, va_t),
        "test": evaluate(model, te_t),
        "epochs_run": ep + 1,
        "features": len(feat),
    }
    log.info("MLP test: MAE %.4f winner_acc %.3f | baseline persistence MAE 0.0496 acc 0.852",
             results["test"]["mae_share_vec"], results["test"]["winner_acc"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "mlp_panel.pt")
    (OUT_DIR / "mlp_metrics.json").write_text(json.dumps(results, indent=2))
    log.info("-> %s", OUT_DIR)


if __name__ == "__main__":
    main()
