"""
analysis_secondary.py
----------------------
Secondary analyses reported in Section 4.4 of the paper:

  1. Per-participant slopes: for every participant with >= MIN_TRACKS tracks
     (>= 4 plays, valid decay_rate), regress decay_rate on h_top_beat; then
     correlate the 23 (or fewer) individual slopes with NFC.
  2. Confound check: do track popularity / listening volume (total_ms,
     play_count) relate to decay_rate or to h_top_beat? Includes the partial
     correlation of h_top_beat and decay_rate controlling for each.
  3. Artist entropy per participant (Shannon entropy of the distribution of
     plays over artists, base 2, plus a version normalized by log2(#artists))
     and its correlation with NFC. Needs the raw Spotify JSON files.

Run after build_master_dataset.py and score_questionnaires.py.
Writes: artist_diversity_by_user.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr, entropy

from config import OUTPUT_DIR, DATA_ROOT
from stats_utils import load_tracks_with_nfc

MIN_TRACKS = 30


# ---------------------------------------------------------------- 1. slopes
def per_user_slopes(df):
    rows = []
    for uid, g in df.groupby("user_id"):
        if len(g) < MIN_TRACKS:
            continue
        fit = linregress(g["h_top_beat"], g["decay_rate"])
        rows.append({"user_id": uid, "n_tracks": len(g), "slope": fit.slope,
                     "nfc_total": g["nfc_total"].iloc[0]})
    out = pd.DataFrame(rows)
    print("=" * 70)
    print(f"1. Per-participant slope of decay_rate on h_top_beat (users with >= {MIN_TRACKS} tracks)")
    print("=" * 70)
    if len(out) >= 4:
        r, p = pearsonr(out["slope"], out["nfc_total"])
        rs, ps = spearmanr(out["slope"], out["nfc_total"])
        print(f"n = {len(out)} participants")
        print(f"Pearson  r(slope, NFC) = {r:.3f}, p = {p:.4f}")
        print(f"Spearman r(slope, NFC) = {rs:.3f}, p = {ps:.4f}")
    else:
        print("Not enough participants with enough tracks.")
    print()
    return out


# ------------------------------------------------------------ 2. confounds
def partial_corr(x, y, z):
    """Correlation of x and y after regressing z out of both."""
    def resid(a, b):
        slope, intercept, *_ = linregress(b, a)
        return a - (intercept + slope * b)
    return pearsonr(resid(np.asarray(x, float), np.asarray(z, float)),
                    resid(np.asarray(y, float), np.asarray(z, float)))


def confound_check(df):
    print("=" * 70)
    print("2. Confound check: popularity / volume vs. decay_rate and h_top_beat")
    print(f"   (n = {len(df)} tracks, {df['user_id'].nunique()} participants)")
    print("=" * 70)
    for conf in ["total_ms", "play_count"]:
        for target in ["decay_rate", "h_top_beat"]:
            r, p = pearsonr(df[conf], df[target])
            print(f"corr({conf}, {target}) = {r:.3f} (p = {p:.4g})")
    print()
    for conf in ["total_ms", "play_count"]:
        r, p = partial_corr(df["h_top_beat"], df["decay_rate"], df[conf])
        print(f"partial corr(h_top_beat, decay_rate | {conf}) = {r:.3f} (p = {p:.4g})")
    r, p = pearsonr(df["h_top_beat"], df["decay_rate"])
    print(f"zero-order  corr(h_top_beat, decay_rate)      = {r:.3f} (p = {p:.4g})")
    print()


# ---------------------------------------------------------- 3. artist entropy
def artist_entropy_by_user(nfc_by_user):
    from process_user import load_and_normalize_json  # needs librosa installed

    rows = []
    for uid in sorted(os.listdir(DATA_ROOT)):
        folder = os.path.join(DATA_ROOT, uid)
        if not os.path.isdir(folder):
            continue
        records = []
        for dirpath, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(".json"):
                    records.extend(load_and_normalize_json(os.path.join(dirpath, f)))
        if not records:
            continue
        artists = pd.Series([str(r.get("artistName", "")).strip().lower() for r in records])
        counts = artists.value_counts().to_numpy()
        h = entropy(counts / counts.sum(), base=2)
        rows.append({
            "user_id": uid, "n_plays": int(counts.sum()),
            "n_distinct_artists": int(len(counts)),
            "artist_entropy": h,
            "artist_entropy_norm": h / np.log2(len(counts)) if len(counts) > 1 else np.nan,
        })

    out = pd.DataFrame(rows).merge(nfc_by_user, on="user_id", how="left")
    path = os.path.join(OUTPUT_DIR, "artist_diversity_by_user.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("3. Artist entropy vs. NFC")
    print("=" * 70)
    for col in ["artist_entropy", "artist_entropy_norm"]:
        pair = out[[col, "nfc_total"]].dropna()
        r, p = pearsonr(pair[col], pair["nfc_total"])
        rs, ps = spearmanr(pair[col], pair["nfc_total"])
        print(f"{col} vs NFC (n={len(pair)}): Pearson r={r:.3f} (p={p:.4f}), "
              f"Spearman r={rs:.3f} (p={ps:.4f})")
    print(f"Saved: {path}\n")
    return out


def main():
    df = load_tracks_with_nfc(min_plays=4, outcome="decay_rate")
    per_user_slopes(df)
    confound_check(df)

    q = pd.read_csv(os.path.join(OUTPUT_DIR, "questionnaire_scores.csv"), encoding="utf-8-sig")
    if os.path.isdir(DATA_ROOT):
        artist_entropy_by_user(q[["user_id", "nfc_total"]])
    else:
        print(f"DATA_ROOT not found ({DATA_ROOT}); skipping artist entropy (needs raw JSON).")


if __name__ == "__main__":
    main()
