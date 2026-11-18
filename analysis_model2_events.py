"""
analysis_model2_events.py
--------------------------
Model 2: does the complexity of what people play depend on whether they are
listening at their chronotype's peak time of day, and does NFC moderate that?

    h_z ~ nfc_z * on_peak          (event level, clustered by participant)

Definitions (fixed in advance):
  * Morning window = hours 8-12 (08:00-12:59); evening window = hours 17-21 (17:00-21:59).
  * Only participants whose MEQ category is morning-type or evening-type are
    included (Intermediate types have no defined peak, so they are excluded
    here; see analysis_intermediate_variability.py for them).
  * on_peak = 1 when a play falls inside the window that matches the
    participant's chronotype (morning type -> morning window, evening type ->
    evening window); every other hour of the day is on_peak = 0.
  * Only plays that could be matched to a track with a complexity score are used.
  (Validated: this definition reproduces the event counts and on-peak shares of
  the first version of event_level_model2.csv exactly.)

Input : event_level_complexity_by_hour.csv (from listening_time_analysis.py,
        which needs master_songs.csv from build_master_dataset.py first),
        questionnaire_scores.csv (from score_questionnaires.py)
Output: event_level_model2.csv  (the exact data the model is fitted on)
"""

import os
import pandas as pd

from config import OUTPUT_DIR
from stats_utils import cluster_robust_ols, interaction_design, zscore

MORNING_HOURS = range(8, 13)   # 8..12
EVENING_HOURS = range(17, 22)  # 17..21


def build_event_table():
    ev = pd.read_csv(os.path.join(OUTPUT_DIR, "event_level_complexity_by_hour.csv"),
                     encoding="utf-8-sig")
    q = pd.read_csv(os.path.join(OUTPUT_DIR, "questionnaire_scores.csv"), encoding="utf-8-sig")

    ev = ev.dropna(subset=["h_top_beat"]).merge(
        q[["user_id", "nfc_total", "meq_category"]], on="user_id", how="inner")
    ev = ev.dropna(subset=["meq_category"])

    is_morning_type = ev["meq_category"].str.contains("morning")
    is_evening_type = ev["meq_category"].str.contains("evening")
    ev = ev[is_morning_type | is_evening_type].copy()  # drops Intermediate

    ev["on_peak"] = (
        (ev["meq_category"].str.contains("morning") & ev["hour"].isin(MORNING_HOURS)) |
        (ev["meq_category"].str.contains("evening") & ev["hour"].isin(EVENING_HOURS))
    )
    return ev[["user_id", "nfc_total", "meq_category", "on_peak", "h_top_beat"]].reset_index(drop=True)


def main():
    ev = build_event_table()
    out_path = os.path.join(OUTPUT_DIR, "event_level_model2.csv")
    ev.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Events: {len(ev)} | participants: {ev['user_id'].nunique()}")
    print(ev.groupby("meq_category")["user_id"].nunique().to_string())
    print(f"Saved: {out_path}\n")

    ev["h_z"] = zscore(ev["h_top_beat"])
    ev["nfc_z"] = zscore(ev["nfc_total"])
    ev["on_peak_num"] = ev["on_peak"].astype(float)

    X, names = interaction_design(ev, "nfc_z", "on_peak_num")
    res = cluster_robust_ols(X, ev["h_z"], ev["user_id"], names)
    print("=" * 70)
    print("Model 2: h_z ~ nfc_z * on_peak (cluster-robust OLS, clustered by participant)")
    print(f"N events = {res.attrs['n_obs']}, participants = {res.attrs['n_clusters']} "
          f"(t df = {res.attrs['df']})")
    print("=" * 70)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
