#!/usr/bin/env python3
"""T2.6 — Municipal GCN over real INEGI contiguity graph (E-89).

Hypothesis: entity-level tabular models can't beat persistence (n=32 × 7
elections); municipal granularity (2,053 nodes × ~2-3 elections each) +
spatial pooling via contiguity can.

Model: 2-layer GCN with residual L1 skip — predicts delta over the
municipality's own last-election shares, neighbour features propagated
through Â = D^-1/2 (A+I) D^-1/2.

Node-year feature (per election record): [share_L1(7), part_L1,
margin_L1, gap_years, year]. Target: 7-party share vector + PARTICIPACION.

Split (chronological): train ≤2018 · val 2019-2021 · test 2022-2024.
Metric vs municipal persistence (own last election).

Run: python scripts/train_gnn_municipal.py [--epochs 400]
"""
import argparse
import json
import logging
import os
from pathlib import Path

# OMP pthread_mutex segfault workaround on this macOS/conda env —
# must be set BEFORE torch/numpy threading libs initialize
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

log = logging.getLogger(__name__)
logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "data/processed/canonical/canonical_long.csv"
ADJ = ROOT / "data/external/geo/adyacencia_municipios.csv"
OUT_DIR = ROOT / "models/gnn"
PARTIES = ["PAN", "PRI", "PRD_IZQ", "MORENA", "PVEM", "MC_CONV", "OTROS"]
DEVICE = "cpu"  # dense Â (2056²) matmul segfaults on MPS; CPU is fast enough here

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


def build_records():
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
    rec = pvt.join(meta).reset_index()
    return rec.sort_values(["cvegeo", "ANIO"])


def build_graph(nodes):
    adj_pq = ROOT / "data/processed/tabular/adyacencia_municipios.parquet"
    adj = pd.read_parquet(adj_pq) if adj_pq.exists() else pd.read_csv(ADJ, dtype=str)
    n2i = {c: i for i, c in enumerate(nodes)}
    edges = [(n2i[a], n2i[b]) for a, b in zip(adj.cvegeo_a, adj.cvegeo_b)
             if a in n2i and b in n2i]
    n = len(nodes)
    A = np.zeros((n, n), dtype=np.float32)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    A += np.eye(n, dtype=np.float32)
    d = A.sum(1) ** -0.5
    d[np.isinf(d)] = 0.0
    Ahat = (d[:, None] * A) * d[None, :]
    return torch.from_numpy(Ahat)  # Â symmetric norm (built in numpy: torch alloc crash workaround)


class GCN(nn.Module):
    def __init__(self, n_feat, hid=64, drop=0.2):
        super().__init__()
        self.g1 = nn.Linear(n_feat, hid)
        self.g2 = nn.Linear(hid, hid)
        self.head_share = nn.Linear(hid, len(PARTIES))
        self.head_part = nn.Linear(hid, 1)
        self.drop = nn.Dropout(drop)
        nn.init.zeros_(self.head_share.weight)
        nn.init.zeros_(self.head_share.bias)

    def forward(self, X, Ahat, share_l1):
        H = self.drop(torch.relu(Ahat @ self.g1(X)))
        H = self.drop(torch.relu(Ahat @ self.g2(H)))
        logit = self.head_share(H) + torch.log(share_l1.clamp_min(1e-4))
        return torch.softmax(logit, dim=1), torch.sigmoid(self.head_part(H)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prep = ROOT / "data/processed/tabular/municipal_records.parquet"
    if not prep.exists():
        raise SystemExit(
            "Ejecuta primero scripts/prep_municipal_records.py "
            "(pandas-CSV + torch en un proceso crashea por OMP en este env)"
        )
    rec = pd.read_parquet(prep)
    nodes = sorted(rec.cvegeo.unique())
    n2i = {c: i for i, c in enumerate(nodes)}
    Ahat = build_graph(nodes).to(DEVICE)
    log.info("nodos %d · aristas %.0f", len(nodes), int((Ahat > 0).sum() - len(nodes)))

    # per-node election timeline: L1 = previous election of same node
    for p in PARTIES:
        rec[f"{p}_L1"] = rec.groupby("cvegeo")[p].shift(1)
    rec["part_L1"] = rec.groupby("cvegeo").PARTICIPACION.shift(1)
    rec["marg_L1"] = rec.groupby("cvegeo").MARGEN_VICTORIA.shift(1)
    rec["year_L1"] = rec.groupby("cvegeo").ANIO.shift(1)
    rec["gap"] = rec.ANIO - rec.year_L1
    rec = rec.dropna(subset=["part_L1", "gap"]).copy()
    log.info("registros con lag: %d", len(rec))

    share_L1 = rec[[f"{p}_L1" for p in PARTIES]].to_numpy(np.float32)
    y_share = rec[PARTIES].to_numpy(np.float32)
    feats = np.column_stack([
        share_L1, rec.part_L1, rec.marg_L1.fillna(0), rec.gap.fillna(4) / 10.0,
        (rec.ANIO - 2015) / 10.0,
    ]).astype(np.float32)
    node_idx = rec.cvegeo.map(n2i).to_numpy()
    years = rec.ANIO.to_numpy()
    yp = rec.PARTICIPACION.to_numpy(np.float32)

    # global standardization of features (fit on train rows)
    tr_m = years <= 2018
    mu, sd = feats[tr_m].mean(0), feats[tr_m].std(0) + 1e-6
    F = torch.tensor((feats - mu) / sd)

    # snapshot node features per election year = each node's own feature row
    # (nodes without election that year are masked out of the loss)
    masks = {"train": tr_m, "val": (years >= 2019) & (years <= 2021), "test": years >= 2022}
    X_all = torch.zeros(len(nodes), F.shape[1])
    for i, ni in enumerate(node_idx):
        X_all[ni] = F[i]  # last write wins within same year; nodes have ≤1 election/yr
    L1_all = torch.zeros(len(nodes), len(PARTIES))
    for i, ni in enumerate(node_idx):
        L1_all[ni] = torch.tensor(share_L1[i])
    Y_all = torch.zeros(len(nodes), len(PARTIES))
    YP_all = torch.zeros(len(nodes))
    LP_all = torch.zeros(len(nodes))
    for i, ni in enumerate(node_idx):
        Y_all[ni] = torch.tensor(y_share[i])
        YP_all[ni] = float(yp[i])
        LP_all[ni] = float(rec.part_L1.iloc[i])

    X_all, L1_all, Y_all = X_all.to(DEVICE), L1_all.to(DEVICE), Y_all.to(DEVICE)
    YP_all, LP_all = YP_all.to(DEVICE), LP_all.to(DEVICE)

    # training node indices per split
    tr_nodes = torch.tensor(sorted({node_idx[i] for i in np.where(masks["train"])[0]}))
    va_nodes = torch.tensor(sorted({node_idx[i] for i in np.where(masks["val"])[0]}))
    te_nodes = torch.tensor(sorted({node_idx[i] for i in np.where(masks["test"])[0]}))
    # node-level targets = the node's election in that split (dedup already per year)
    log.info("split nodos: train %d val %d test %d", len(tr_nodes), len(va_nodes), len(te_nodes))

    model = GCN(F.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    def eval_nodes(idx):
        model.eval()
        with torch.no_grad():
            sh, part = model(X_all, Ahat, L1_all)
        idx_d = idx.to(DEVICE)
        yt, ys = Y_all[idx_d], sh[idx_d]
        return {
            "mae_share_vec": float(torch.abs(yt - ys).mean()),
            "winner_acc": float((yt.argmax(1) == ys.argmax(1)).float().mean()),
            "rmse_participacion": float(torch.sqrt(((YP_all[idx_d] - part[idx_d]) ** 2).mean())),
        }

    best_val, best_state, patience = np.inf, None, 0
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        sh, part = model(X_all, Ahat, L1_all)
        idx_d = tr_nodes.to(DEVICE)
        loss = nn.functional.mse_loss(sh[idx_d], Y_all[idx_d]) + \
            0.2 * nn.functional.mse_loss(part[idx_d], YP_all[idx_d])
        loss.backward()
        opt.step()
        sched.step()
        m = eval_nodes(va_nodes)
        if m["mae_share_vec"] < best_val - 1e-5:
            best_val, best_state, patience = m["mae_share_vec"], model.state_dict(), 0
        else:
            patience += 1
            if patience >= 50:
                break
        if ep % 60 == 0:
            log.info("ep %d loss %.4f val MAE %.4f", ep, loss.item(), m["mae_share_vec"])

    model.load_state_dict(best_state)
    results = {"val": eval_nodes(va_nodes), "test": eval_nodes(te_nodes), "epochs_run": ep + 1}

    # municipal persistence baseline on same nodes
    pers = {
        s: {
            "mae_share_vec": float(torch.abs(Y_all[idx.to(DEVICE)] - L1_all[idx.to(DEVICE)]).mean()),
            "winner_acc": float((Y_all[idx.to(DEVICE)].argmax(1) == L1_all[idx.to(DEVICE)].argmax(1)).float().mean()),
            "rmse_participacion": float(torch.sqrt(((YP_all[idx.to(DEVICE)] - LP_all[idx.to(DEVICE)]) ** 2).mean())),
        }
        for s, idx in (("val", va_nodes), ("test", te_nodes))
    }
    results["persistence_baseline"] = pers
    log.info("GCN test MAE %.4f acc %.3f | persistence %.4f acc %.3f",
             results["test"]["mae_share_vec"], results["test"]["winner_acc"],
             pers["test"]["mae_share_vec"], pers["test"]["winner_acc"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "gcn_municipal.pt")
    (OUT_DIR / "gnn_metrics.json").write_text(json.dumps(results, indent=2))
    log.info("-> %s", OUT_DIR)


if __name__ == "__main__":
    main()
