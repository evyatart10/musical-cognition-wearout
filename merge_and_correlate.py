"""
merge_and_correlate.py
------------------------
ממזג את שלוש הטבלאות (שאלונים, שעות האזנה, מורכבות מוזיקלית) לפי
user_id, ומריץ את הקורלציות הראשוניות שהוגדרו מראש בתוכנית הניתוח.

הרץ שוב בדיוק כמו שהוא בכל פעם שמתעדכן נתון (נבדק נוסף, שאלון חסר
שהושלם) - אין צורך לשנות שום דבר בקוד עצמו.
"""

import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from config import OUTPUT_DIR

QUESTIONNAIRE_PATH = os.path.join(OUTPUT_DIR, "questionnaire_scores.csv")
LISTENING_TIME_PATH = os.path.join(OUTPUT_DIR, "listening_time_summary.csv")
MASTER_USERS_PATH = os.path.join(OUTPUT_DIR, "master_users_mci.csv")  # מ-build_master_dataset.py

# תוכנית הניתוח הראשונית - נקבעה מראש, לא לשנות אחרי ראיית התוצאות
PLANNED_CORRELATIONS = [
    ("nfc_total", "mci_score", "NFC מול מורכבות מוזיקלית ממוצעת"),
    ("nfc_total", "hour_entropy", "NFC מול פיזור שעות ההאזנה"),
    ("meq_total", "circular_mean_hour", "כרונוטיפ מול שעת האזנה ממוצעת"),
    ("meq_total", "pct_late_night", "כרונוטיפ מול אחוז האזנה בלילה מאוחר"),
    ("meq_total", "hour_entropy", "כרונוטיפ מול פיזור שעות ההאזנה"),
]


def main():
    q = pd.read_csv(QUESTIONNAIRE_PATH, encoding='utf-8-sig')
    t = pd.read_csv(LISTENING_TIME_PATH, encoding='utf-8-sig')

    merged = q.merge(t, on='user_id', how='inner')
    print(f"Questionnaires: {len(q)} users | Listening-time: {len(t)} users | "
          f"Merged (inner): {len(merged)} users")

    if os.path.exists(MASTER_USERS_PATH):
        m = pd.read_csv(MASTER_USERS_PATH, encoding='utf-8-sig')
        merged = merged.merge(m, on='user_id', how='left')
        print(f"Merged with complexity data: {merged['mci_score'].notna().sum()} users have mci_score.")
    else:
        print(f"Note: {MASTER_USERS_PATH} not found yet - skipping complexity merge "
              f"(run build_master_dataset.py once run_all_users.py finishes).")

    save_path = os.path.join(OUTPUT_DIR, "merged_analysis_table.csv")
    merged.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Saved merged table: {save_path}\n")

    print("=" * 60)
    print(f"PRELIMINARY CORRELATIONS (n={len(merged)})")
    print("=" * 60)

    for var_x, var_y, label in PLANNED_CORRELATIONS:
        if var_x not in merged.columns or var_y not in merged.columns:
            print(f"{label}: SKIPPED ({var_x} or {var_y} not available yet)")
            continue
        pair = merged[[var_x, var_y]].dropna()
        n = len(pair)
        if n < 4:
            print(f"{label}: SKIPPED (only {n} valid pairs)")
            continue
        r_p, p_p = pearsonr(pair[var_x], pair[var_y])
        r_s, p_s = spearmanr(pair[var_x], pair[var_y])
        print(f"\n{label} (n={n})")
        print(f"  Pearson  r={r_p:.3f}, p={p_p:.4f}")
        print(f"  Spearman r={r_s:.3f}, p={p_s:.4f}")

    print("\n" + "=" * 60)
    print("NOTE: this is a PRELIMINARY run. Re-run this exact script "
          "(unchanged) once all subjects are in - do not alter the "
          "analysis plan based on interim results.")


if __name__ == "__main__":
    main()
