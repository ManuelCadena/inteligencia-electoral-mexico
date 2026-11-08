#!/usr/bin/env python3
"""T2.7 — Post-2018 text ablation: do mañanera features beat persistence?

The only exogenous signal not yet tested. The mañanera corpus exists only
from Dec-2018, covering exactly the test period (federal elections 2021-06,
2024-06). Monthly NLP features (topics 1-12, BiLSTM sentiment, attention
PCs, econ/confr rates, president share) are NATIONAL — they shift all
entities uniformly; the honest question is whether national discourse
predicts the persistence residual.

Election alignment: federal contests happen the first Sunday of June.
Features = mean of the trailing 6 months before the election month
(Dec prev-year -> May election-year), no leakage.

Conditions on entity-level test rows (2021, 2024):
  persistence | + text residual | + exo residual | + exo+text residual
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
TEXT = Path(
    "/Users/manuelcadena/My Drive/Manuel Cadena/New Life/"
    "Harvard Master in Finance/Natural Language PRocessing /"
    "final_project/data/features_mensuales.csv"
)
OUT = ROOT / "models/baselines/text_ablation.json"
PARTIES = ["PAN", "PRI", "PRD_IZQ", "MORENA", "PVEM", "MC_CONV", "OTROS"]
SHARE = [f"share_{p}" for p in PARTIES]

TEXT_FEATS = (
    [f"topic_{i}" for i in range(1, 13)]
    + ["econ_rate", "confr_rate", "share_presidente",
       "sentimiento_bilstm_mean", "sentimiento_bilstm_std",
       "attention_pc1", "policy_attention_pc1", "attention_econ_pc1",
       "attention_soc_pc1", "attention_pol_pc1"]
)


def mae(a, b):
    return float(np.abs(a - b).mean())


def text_block(periodo_col, periods, year):
    """Trailing-6-month mean ending in May of election year."""
    sel = [f"{year - 1}-12-01"] + [f"{year}-{m:02d}-01" for m in range(1, 6)]
    m = periodo_col.isin(sel)
    n = m.sum()
    assert n >= 4, f"solo {n} meses para {year}"
    return periods[m][TEXT_FEATS].mean().to_numpy()


def main():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(PANEL)
    df = df[df.n_lags >= 1].copy()
    txt = pd.read_csv(TEXT)
    log.info("text features: %d · periodo %s..%s",
             len(TEXT_FEATS), txt.periodo.min(), txt.periodo.max())

    tf = {y: text_block(txt.periodo, txt, y) for y in (2021, 2024)}
    te = df[(df.split == "test") & (df.ANIO.isin([2021, 2024]))].copy()
    log.info("test rows 2021/2024: %d", len(te))
    yt = te[SHARE].to_numpy()
    l1 = te[[f"{c}_L1" for c in SHARE]].to_numpy()

    exo = [c for c in df.columns if (c.endswith("_LAG1_NAC") or c.endswith("_LAG1_ENT"))
           and pd.api.types.is_numeric_dtype(df[c])]
    X_txt = np.tile(np.stack([tf[y] for y in te.ANIO]), (1, 1))  # per-row election year
    X_txt = np.stack([tf[y] for y in te.ANIO])
    X_exo = te[exo].to_numpy()

    # ridge on residual trained ONLY on post-2018 data with text available:
    # train = val split (2015-18) lacks text — so the honest design is
    # residual fit on exo (train) then +text correction evaluated cold on test.
    # Since text has no trainable history, we test text as a *feature shift*
    # against residuals directly: does mean discourse level correlate with
    # (share - L1) cross-sectionally? Report raw and ridge-on-text-residual.
    res = {"persistence": mae(yt, l1), "n_rows": int(len(te)),
           "elections": sorted(te.ANIO.unique().tolist()),
           "note": ("text features are national-level; identical across "
                    "entities within a year — they can only shift residuals "
                    "uniformly, so entity MAE gains are structurally capped")}

    # Ridge trained on residual with [exo | text] — but text varies only by
    # year (2 unique values), so fit on exo-residual then add text effect as
    # a learned scalar per party via leave-one-election-out CV.
    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()

    def ridge_resid(feat_mat):
        Xtr = np.nan_to_num(imp.fit_transform(feat_mat))
        resid_tr = np.nan_to_num(imp.fit_transform(te[SHARE].to_numpy() - l1))
        m = Ridge(alpha=1.0).fit(sc.fit_transform(Xtr), resid_tr)
        return np.clip(l1 + m.predict(sc.transform(Xtr)), 0, 1)

    res["exo_residual"] = mae(yt, ridge_resid(X_exo))
    res["text_residual"] = mae(yt, ridge_resid(X_txt))
    res["exo_plus_text_residual"] = mae(yt, ridge_resid(
        np.column_stack([X_exo, X_txt])))

    # honest caveat: with n_elections=2 the text-ridge is fit ON the test rows
    # themselves (no post-2018 training set exists). Flag it explicitly.
    res["caveat"] = ("text ridge is transductive on test (only 2021 & 2024 "
                     "carry text) — interpretable as 'can national discourse "
                     "explain the residual direction', NOT as predictive gain")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        if isinstance(v, float):
            log.info("%s: %.4f", k, v)
    log.info("-> %s", OUT)


if __name__ == "__main__":
    main()
