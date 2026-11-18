"""
make_figures.py
----------------
Figure 1: standardized decay rate as a function of topological entropy, fitted
          separately for participants below / above the median NFC score
          (lines = pooled OLS fits, bands = 95% confidence intervals from
          participant-clustered standard errors).
Figure 2: morningness-eveningness (MEQ) vs. circular mean listening hour, one
          dot per participant, with the OLS regression line.

Run after build_master_dataset.py, score_questionnaires.py and
listening_time_analysis.py. Writes figure1_model1.png and figure2_chronotype.png
to OUTPUT_DIR.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config import OUTPUT_DIR
from stats_utils import cluster_robust_ols, load_tracks_with_nfc, zscore

LOW_COLOR, HIGH_COLOR = "#0072B2", "#D55E00"  # colour-blind-safe pair


def figure1():
    df = load_tracks_with_nfc(min_plays=4, outcome="decay_rate")
    df["decay_z"] = zscore(df["decay_rate"])
    median_nfc = df.drop_duplicates("user_id")["nfc_total"].median()
    df["nfc_group"] = np.where(df["nfc_total"] > median_nfc, "High NFC", "Low NFC")
    # participants exactly at the median are kept in "Low NFC"

    grid = np.linspace(df["h_z"].quantile(0.02), df["h_z"].quantile(0.98), 100)
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=200)

    for label, color in [("Low NFC", LOW_COLOR), ("High NFC", HIGH_COLOR)]:
        g = df[df["nfc_group"] == label]
        X = np.column_stack([np.ones(len(g)), g["h_z"]])
        res = cluster_robust_ols(X, g["decay_z"], g["user_id"], ["Intercept", "h_z"])
        beta, cov = res.attrs["beta"], res.attrs["cov"]
        G = np.column_stack([np.ones_like(grid), grid])
        fit = G @ beta
        se = np.sqrt(np.einsum("ij,jk,ik->i", G, cov, G))
        tcrit = stats.t.ppf(0.975, res.attrs["df"])
        n_users = g["user_id"].nunique()
        ax.plot(grid, fit, color=color, lw=2,
                label=f"{label} (n = {n_users} participants, slope = {beta[1]:.3f})")
        ax.fill_between(grid, fit - tcrit * se, fit + tcrit * se, color=color, alpha=0.18, lw=0)

    ax.axhline(0, color="#888", lw=0.8, ls=":")
    ax.set_xlabel("Topological entropy of the track (z-score)")
    ax.set_ylabel("Decay rate (z-score)")
    ax.set_title("Track complexity and decay rate, by NFC group")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure1_model1.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def figure2():
    q = pd.read_csv(os.path.join(OUTPUT_DIR, "questionnaire_scores.csv"), encoding="utf-8-sig")
    t = pd.read_csv(os.path.join(OUTPUT_DIR, "listening_time_summary.csv"), encoding="utf-8-sig")
    d = q.merge(t, on="user_id").dropna(subset=["meq_total", "circular_mean_hour"])

    fit = stats.linregress(d["meq_total"], d["circular_mean_hour"])
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    ax.scatter(d["meq_total"], d["circular_mean_hour"], s=38, color=LOW_COLOR, zorder=3)
    xs = np.array([d["meq_total"].min(), d["meq_total"].max()])
    ax.plot(xs, fit.intercept + fit.slope * xs, color="#444", lw=1.6)
    ax.set_xlabel("MEQ score (higher = more morning-oriented)")
    ax.set_ylabel("Circular mean listening hour")
    ax.set_title(f"Chronotype and mean listening hour (r = {fit.rvalue:.2f}, p = {fit.pvalue:.3f}, n = {len(d)})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure2_chronotype.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    figure1()
    figure2()
