#!/usr/bin/env python3
"""E-89 slot 17 — Time-Series Foundation Model Benchmarking.

Panel A (macro monthly): ENCO ICC, OECD unemployment, OECD CPI,
Banxico remittances. Rolling-origin evaluation, horizon h=6.
Panel B (electoral entity): presidential vote share by state,
1994-2024 (~7 elections), horizon h=1 — the persistence-wall test.

Models: persistence, seasonal-naive, Ridge-AR(12), Chronos-bolt-small
(zero-shot), TimesFM-2.5 (zero-shot), pooled LSTM (trained baselines).
Outputs: models/benchmark/benchmark_metrics.json + predictions.parquet.
"""
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON_MACRO = 6
N_ORIGINS = 12          # rolling cutoffs, last 12+ months
CONTEXT_MAX = 512       # cap context length for foundation models
SEED = 42


# ---------- data loading ----------

def load_macro_series():
    """Monthly macro series -> {name: (dates, values)}."""
    series = {}
    d = pd.read_csv(
        ROOT / "data/external/mananeras/enco/"
        "enco_5componentes_MENSUAL_2001_2026.csv")
    series["enco_icc"] = (
        pd.to_datetime(d.periodo), d.icc.astype(float).values)

    d = pd.read_csv(
        ROOT / "data/external/lo_que_mexico_cree/"
        "desempleo_ocde_mensual_1987_2026.csv")
    series["desempleo_ocde"] = (
        pd.to_datetime(d.observation_date),
        d.LRHUTTTTMXM156S.astype(float).values)

    d = pd.read_csv(
        ROOT / "data/external/lo_que_mexico_cree/"
        "inpc_ocde_mensual_1969_2024.csv")
    series["inpc_ocde"] = (
        pd.to_datetime(d.observation_date),
        d.MEXCPIALLMINMEI.astype(float).values)

    # Banxico remittances: file has a Spanish header block — find the row
    # whose first col parses as a date.
    raw = pd.read_csv(
        ROOT / "data/external/lo_que_mexico_cree/"
        "remesas_banxico_CE81_mensual_1995_2026.csv",
        header=None, encoding="latin-1", skiprows=17,
        usecols=[0, 1], names=["fecha", "total"])
    raw["fecha"] = pd.to_datetime(raw.fecha, format="%d/%m/%Y",
                                  errors="coerce")
    raw = raw.dropna()
    if len(raw):
        series["remesas_banxico"] = (
            raw.fecha.reset_index(drop=True),
            pd.to_numeric(raw.total, errors="coerce")
              .astype(np.float64).values)
    return {k: (d, v) for k, (d, v) in series.items() if len(v) > 60}


def _norm_ent(name):
    import unicodedata
    n = unicodedata.normalize("NFKD", str(name))
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.upper().strip()
    if n in ("DISTRITO FEDERAL", "CIUDAD DE MEXICO"):
        return "CDMX"
    if n == "MEXICO":
        return "EDO MEXICO"
    return n


def load_electoral_series():
    """Federal-deputy MR vote share of the leading party per entity.

    Elections every 3 years (1991-2024, ~12 obs) — the low-frequency
    persistence-wall test. Share = votes of last-election winner."""
    d = pd.read_csv(
        ROOT / "data/processed/series/serie_diputados_fed_mr_entidad.csv")
    d["ENT"] = d.NOMBRE_ESTADO.map(_norm_ent)
    d["share"] = d.VOTOS / d.TOTAL_VOTOS
    pivot = d.pivot_table(
        index=["ENT", "ANIO"], columns="PARTIDO_COALICION",
        values="share", aggfunc="sum").fillna(0.0)
    series = {}
    for ent, g in pivot.groupby(level=0):
        g2 = g.droplevel(0).sort_index()
        if len(g2) < 8:
            continue
        top = g2.iloc[-1].idxmax()  # party winning last election
        series[f"dipfed_share_{ent}"] = (
            pd.Series(g2.index.astype(int)), g2[top].values)
    return series


# ---------- baselines ----------

def predict_persistence(ctx, h):
    return np.repeat(ctx[-1], h)


def predict_seasonal(ctx, h, season=12):
    if len(ctx) < season:
        return predict_persistence(ctx, h)
    reps = ctx[-season:]
    return np.tile(reps, int(np.ceil(h / season)))[:h]


def predict_ridge(ctx, h, n_lags=12):
    from sklearn.linear_model import Ridge
    if len(ctx) < n_lags + 5:
        return predict_persistence(ctx, h)
    X = np.lib.stride_tricks.sliding_window_view(ctx, n_lags)
    Xtr, ytr = X[:-1], ctx[n_lags:]
    m = Ridge(alpha=1.0).fit(Xtr, ytr)
    hist = list(ctx[-n_lags:])
    preds = []
    for _ in range(h):
        x = np.array(hist[-n_lags:]).reshape(1, -1)
        p = float(m.predict(x)[0])
        preds.append(p)
        hist.append(p)
    return np.array(preds)


def _pooled_windows(all_ctx, n_lags):
    """Standardized lag windows pooled across series."""
    Xs, ys = [], []
    for v in all_ctx:
        v = np.asarray(v, dtype=np.float64)
        if len(v) <= n_lags + 5:
            continue
        z = (v - v.mean()) / (v.std() + 1e-8)
        X = np.lib.stride_tricks.sliding_window_view(z, n_lags)
        Xs.append(X[:-1])
        ys.append(z[n_lags:])
    X = np.concatenate(Xs).astype(np.float32)[..., None]
    return X, np.concatenate(ys).astype(np.float32)


def _train_net(net, X, y, epochs=40, lr=1e-3):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt, yt = torch.tensor(X), torch.tensor(y)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xt), yt)
        loss.backward()
        opt.step()
    return net


def _rollout(net, ctx, h, n_lags, seq=True):
    import torch
    mu, sd = ctx.mean(), ctx.std() + 1e-8
    hist = list((ctx - mu) / sd)
    preds = []
    with torch.no_grad():
        for _ in range(h):
            w = np.array(hist[-n_lags:], dtype=np.float32)
            x = torch.tensor(
                w[None, :, None] if seq else w[None, :])
            p = float(net(x).item())
            preds.append(p * sd + mu)
            hist.append(p)
    return np.array(preds)


def predict_lstm_pooled(all_ctx, ctx, h, n_lags=12, epochs=40):
    """Small LSTM trained across all series' windows (shared weights)."""
    import torch.nn as nn
    X, y = _pooled_windows(all_ctx, n_lags)

    class L(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(1, 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.head(o[:, -1]).squeeze(-1)

    net = _train_net(L(), X, y, epochs)
    return _rollout(net, ctx, h, n_lags)


def predict_mlp_pooled(all_ctx, ctx, h, n_lags=12, epochs=40):
    """Flat MLP on the lag window — simplest trained NN baseline."""
    import torch.nn as nn
    X, y = _pooled_windows(all_ctx, n_lags)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(), nn.Linear(n_lags, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    net = _train_net(M(), X, y, epochs)
    return _rollout(net, ctx, h, n_lags, seq=False)


def predict_transformer_pooled(all_ctx, ctx, h, n_lags=12, epochs=40):
    """Tiny Transformer encoder (2 layers, 2 heads) + linear head."""
    import torch
    import torch.nn as nn
    X, y = _pooled_windows(all_ctx, n_lags)

    class T(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 32)
            self.pos = nn.Parameter(torch.randn(1, n_lags, 32) * 0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=32, nhead=2, dim_feedforward=64,
                batch_first=True, dropout=0.0)
            self.enc = nn.TransformerEncoder(enc, num_layers=2)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            z = self.enc(self.proj(x) + self.pos)
            return self.head(z[:, -1]).squeeze(-1)

    net = _train_net(T(), X, y, epochs, lr=5e-4)
    return _rollout(net, ctx, h, n_lags)


# ---------- foundation models ----------

_CHRONOS = None
_TIMESFM = None


def get_chronos():
    global _CHRONOS
    if _CHRONOS is None:
        import torch
        from chronos import BaseChronosPipeline
        _CHRONOS = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small", device_map="cpu")
        torch.set_num_threads(1)
    return _CHRONOS


def predict_chronos(ctx, h):
    import torch
    pipe = get_chronos()
    t = torch.tensor(np.asarray(ctx, dtype=np.float32))
    ctx_in = t[-CONTEXT_MAX:]
    fc = pipe.predict(ctx_in, prediction_length=h)
    return fc[0].median(dim=0).values.numpy()


def get_timesfm():
    global _TIMESFM
    if _TIMESFM is None:
        import timesfm
        _TIMESFM = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch")
        _TIMESFM.compile(
            timesfm.ForecastConfig(max_context=CONTEXT_MAX,
                                   max_horizon=64))
    return _TIMESFM


def predict_timesfm(ctx, h):
    m = get_timesfm()
    point, _ = m.forecast(horizon=h,
                          inputs=[np.asarray(ctx[-CONTEXT_MAX:], dtype=np.float32)])
    return np.asarray(point[0])[:h]


# ---------- evaluation ----------

def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def mase(pred, true, ctx, season=12):
    """MASE vs in-sample naive-1 (random walk) denominator."""
    denom = (np.mean(np.abs(np.diff(ctx)))
             if len(ctx) > 1 else np.nan)
    # floor at 1e-3: near-constant series make MASE undefined, not infinite
    if np.isnan(denom) or denom < 1e-3:
        denom = 1e-3
    return mae(pred, true) / float(denom)


def run_panel(name, series, horizon, n_origins, models):
    rows = []
    for sname, (dates, vals) in series.items():
        v = np.asarray(vals, dtype=np.float64)
        if len(v) < horizon + n_origins + 5:
            log.warning("skip %s: too short (%d)", sname, len(v))
            continue
        all_ctx = [x[1] for x in series.values()]
        for k in range(n_origins):
            cut = len(v) - n_origins - horizon + k + 1
            ctx, true = v[:cut], v[cut:cut + horizon]
            for mname, fn in models.items():
                t0 = time.time()
                try:
                    pred = fn(ctx, horizon)
                except Exception as e:
                    log.error("%s/%s/%s failed: %s",
                              name, sname, mname, e)
                    continue
                rows.append({
                    "panel": name, "series": sname, "origin": k,
                    "model": mname,
                    "mae": mae(pred, true),
                    "mase": mase(pred, true, ctx,
                                 season=12 if name == "macro" else 1),
                    "secs": round(time.time() - t0, 3),
                })
        log.info("%s %s done (%d origins)", name, sname, n_origins)
    return rows


def main():
    macro = load_macro_series()
    electoral = load_electoral_series()
    log.info("macro series: %s | electoral: %d",
             {k: len(v) for k, (_, v) in macro.items()}, len(electoral))

    models = {
        "persistence": predict_persistence,
        "seasonal_naive": predict_seasonal,
        "ridge_ar12": predict_ridge,
        "chronos_bolt_small": predict_chronos,
        "timesfm_2p5": predict_timesfm,
    }
    # pooled trained baselines need all series contexts
    macro_all = [v for _, v in macro.values()]
    models["lstm_pooled"] = lambda c, h: predict_lstm_pooled(
        macro_all, c, h)
    models["mlp_pooled"] = lambda c, h: predict_mlp_pooled(
        macro_all, c, h)
    models["transformer_pooled"] = lambda c, h: predict_transformer_pooled(
        macro_all, c, h)

    rows = []
    rows += run_panel("macro", macro, HORIZON_MACRO, N_ORIGINS, models)
    e_models = dict(models)
    e_models["seasonal_naive"] = lambda ctx, h: predict_seasonal(
        ctx, h, season=1)
    e_models["lstm_pooled"] = lambda ctx, h: predict_lstm_pooled(
        [v for _, v in electoral.values()], ctx, h, n_lags=3, epochs=40)
    e_models["mlp_pooled"] = lambda ctx, h: predict_mlp_pooled(
        [v for _, v in electoral.values()], ctx, h, n_lags=3, epochs=40)
    e_models["transformer_pooled"] = lambda ctx, h: (
        predict_transformer_pooled(
            [v for _, v in electoral.values()], ctx, h,
            n_lags=3, epochs=40))
    e_models["ridge_ar12"] = lambda ctx, h: predict_ridge(ctx, h, n_lags=3)
    rows += run_panel("electoral", electoral, 1, 3, e_models)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "benchmark_predictions.parquet", index=False)

    summ = (df.groupby(["panel", "model"])
              .agg(mae_mean=("mae", "mean"),
                   mae_median=("mae", "median"),
                   mase_mean=("mase", "mean"),
                   mase_median=("mase", "median"),
                   n=("mae", "size"),
                   secs_mean=("secs", "mean"))
              .reset_index())
    res = {
        "horizon_macro": HORIZON_MACRO,
        "n_origins_macro": N_ORIGINS,
        "electoral_horizon": 1,
        "electoral_origins": 3,
        "summary": summ.to_dict("records"),
    }
    (OUT / "benchmark_metrics.json").write_text(json.dumps(res, indent=2))
    log.info("=== summary ===\n%s", summ.to_string())


if __name__ == "__main__":
    main()
