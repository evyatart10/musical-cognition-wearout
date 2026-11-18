"""
analysis_model1_ols.py
-----------------------
Model 1 (secondary / robustness version): pooled OLS with participant-clustered
(CR1) standard errors.

    decay_rate ~ h_z * nfc_z          (tracks with >= 4 plays)

The *primary* Model 1 in the paper is the linear mixed-effects model with a
random intercept and random slope (run_mixed_model.py, "Model B"). This script
reproduces the cluster-robust OLS reported alongside it, in two variants:

  (a) all tracks
  (b) excluding tracks flagged low_beat_confidence (beat tracking unreliable)

Run after build_master_dataset.py and score_questionnaires.py.
"""

import pandas as pd

from stats_utils import cluster_robust_ols, interaction_design, load_tracks_with_nfc


def fit_and_print(df, label):
    X, names = interaction_design(df, "h_z", "nfc_z")
    res = cluster_robust_ols(X, df["decay_rate"], df["user_id"], names)
    print("=" * 70)
    print(f"{label}")
    print(f"N tracks = {res.attrs['n_obs']}, participants = {res.attrs['n_clusters']} "
          f"(t df = {res.attrs['df']})")
    print("=" * 70)
    print(res.round(4).to_string(index=False))
    print()
    return res


def main():
    df = load_tracks_with_nfc(min_plays=4, outcome="decay_rate")
    fit_and_print(df, "Model 1 (cluster-robust OLS), all tracks with >= 4 plays")

    if "low_beat_confidence" in df.columns:
        # NaN / False = reliable beat tracking
        reliable = df[~df["low_beat_confidence"].fillna(False).astype(bool)].copy()
        # re-standardize inside the subsample so coefficients are per-SD of that sample
        reliable["h_z"] = (reliable["h_top_beat"] - reliable["h_top_beat"].mean()) / reliable["h_top_beat"].std()
        fit_and_print(reliable, "Model 1 (cluster-robust OLS), excluding low_beat_confidence tracks")
    else:
        print("Column 'low_beat_confidence' not found in master_songs.csv; "
              "skipping the exclusion variant (re-run process_user.py to create it).")


if __name__ == "__main__":
    main()
