"""
analysis_intermediate_variability.py
-------------------------------------
Supplementary analysis (Section 4.3): Model 2 had to exclude Intermediate
chronotypes because they have no defined peak. This script asks a different
question that *can* include them: do Intermediate types vary less in the
complexity of what they play across the day than morning/evening types?

For every participant:
  * split the day into six 4-hour blocks (00-03, 04-07, ..., 20-23);
  * keep blocks with >= MIN_EVENTS_PER_BIN plays that have a complexity score;
  * require >= MIN_BINS qualifying blocks;
  * time-of-day range = max - min of the block means of h_top_beat.
Then compare the range of Intermediate vs. non-Intermediate participants
(Welch's t-test and Mann-Whitney U). The one-sided hypothesis is that the
Intermediate group has the SMALLER range; two-sided results are printed too.

Input: event_level_complexity_by_hour.csv, questionnaire_scores.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

from config import OUTPUT_DIR

MIN_EVENTS_PER_BIN = 10
MIN_BINS = 3
BLOCK_HOURS = 4


def time_of_day_range(g):
    g = g.assign(block=g["hour"] // BLOCK_HOURS)
    stats_by_block = g.groupby("block")["h_top_beat"].agg(["mean", "count"])
    stats_by_block = stats_by_block[stats_by_block["count"] >= MIN_EVENTS_PER_BIN]
    if len(stats_by_block) < MIN_BINS:
        return np.nan
    return stats_by_block["mean"].max() - stats_by_block["mean"].min()


def main():
    ev = pd.read_csv(os.path.join(OUTPUT_DIR, "event_level_complexity_by_hour.csv"),
                     encoding="utf-8-sig").dropna(subset=["h_top_beat"])
    q = pd.read_csv(os.path.join(OUTPUT_DIR, "questionnaire_scores.csv"), encoding="utf-8-sig")
    q = q.dropna(subset=["meq_category"])

    ranges = (ev.groupby("user_id").apply(time_of_day_range)
                .rename("tod_range").reset_index())
    d = ranges.merge(q[["user_id", "meq_category"]], on="user_id", how="inner").dropna(subset=["tod_range"])
    d["group"] = np.where(d["meq_category"] == "intermediate", "Intermediate", "Morning/Evening")

    print(d.sort_values("tod_range").to_string(index=False))
    a = d.loc[d["group"] == "Intermediate", "tod_range"]
    b = d.loc[d["group"] == "Morning/Evening", "tod_range"]
    print(f"\nIntermediate: n={len(a)}, mean range={a.mean():.4f} (SD {a.std():.4f})")
    print(f"Morning/Evening: n={len(b)}, mean range={b.mean():.4f} (SD {b.std():.4f})")

    if len(a) >= 2 and len(b) >= 2:
        for alt in ["two-sided", "less"]:
            t, p = stats.ttest_ind(a, b, equal_var=False, alternative=alt)
            u, pu = stats.mannwhitneyu(a, b, alternative=alt)
            print(f"[{alt}] Welch t = {t:.3f}, p = {p:.4f} | Mann-Whitney U = {u:.1f}, p = {pu:.4f}")
    else:
        print("Not enough participants in one of the groups for a test.")


if __name__ == "__main__":
    main()
