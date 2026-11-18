"""
stats_utils.py
---------------
Small shared helpers for the analysis scripts (no heavy dependencies:
numpy / pandas / scipy only).

cluster_robust_ols
    Pooled OLS with CR1 cluster-robust standard errors, clustered by
    participant. This is the "secondary" version of Model 1 / Model 2 in the
    paper (the primary Model 1 is the mixed model in run_mixed_model.py).

    Sandwich estimator:
        bread = (X'X)^-1
        meat  = sum over clusters g of (X_g' u_g)(X_g' u_g)'
        V     = c * bread @ meat @ bread
        c     = G/(G-1) * (N-1)/(N-K)          (CR1 small-sample correction)
    p-values use a t distribution with G-1 degrees of freedom.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

from config import OUTPUT_DIR


def zscore(s):
    """Sample (ddof=1) z-score of a pandas Series."""
    return (s - s.mean()) / s.std()


def cluster_robust_ols(X, y, groups, names=None):
    """
    X: (N, K) array-like including the intercept column.
    y: (N,) array-like.
    groups: (N,) cluster labels (participant id).
    Returns a DataFrame with coef, se, t, p, ci_low, ci_high (one row per term).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    n, k = X.shape

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta

    meat = np.zeros((k, k))
    unique_groups = np.unique(groups)
    for g in unique_groups:
        idx = groups == g
        score = X[idx].T @ resid[idx]
        meat += np.outer(score, score)

    n_groups = len(unique_groups)
    correction = (n_groups / (n_groups - 1)) * ((n - 1) / (n - k))
    cov = correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(cov))

    t_vals = beta / se
    df = n_groups - 1
    p_vals = 2 * stats.t.sf(np.abs(t_vals), df)
    t_crit = stats.t.ppf(0.975, df)

    names = names or [f"x{i}" for i in range(k)]
    out = pd.DataFrame({
        "term": names, "coef": beta, "se": se, "t": t_vals, "p": p_vals,
        "ci_low": beta - t_crit * se, "ci_high": beta + t_crit * se,
    })
    out.attrs.update({"n_obs": n, "n_clusters": n_groups, "df": df, "cov": cov, "beta": beta})
    return out


def interaction_design(df, x_col, m_col):
    """Design matrix [1, x, m, x*m] and its term names."""
    X = np.column_stack([
        np.ones(len(df)), df[x_col], df[m_col], df[x_col] * df[m_col]
    ])
    return X, ["Intercept", x_col, m_col, f"{x_col}:{m_col}"]


def load_tracks_with_nfc(min_plays=4, outcome="decay_rate"):
    """
    Track-level analysis table used by Model 1 and its robustness checks:
    master_songs (>= min_plays plays, non-missing outcome / entropy) joined
    to the NFC score of each participant. Adds the z-scored columns h_z, nfc_z.
    """
    songs = pd.read_csv(os.path.join(OUTPUT_DIR, "master_songs.csv"), encoding="utf-8-sig")
    q = pd.read_csv(os.path.join(OUTPUT_DIR, "questionnaire_scores.csv"), encoding="utf-8-sig")

    df = songs[songs["play_count"] >= min_plays].copy()
    df = df.merge(q[["user_id", "nfc_total"]], on="user_id", how="inner")
    df = df.dropna(subset=[outcome, "h_top_beat", "nfc_total"])
    df["h_z"] = zscore(df["h_top_beat"])
    df["nfc_z"] = zscore(df["nfc_total"])
    return df.reset_index(drop=True)
