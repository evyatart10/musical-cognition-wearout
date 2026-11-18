"""
analyze_window_sensitivity.py
--------------------------------
מנתח את תוצאות multiscale_sensitivity_pipeline.py: האם מדדי המורכבות
יציבים ביחס לאורך-החלון, או שיש רגישות משמעותית?

לכל שיר שנדגם, יש 4 שורות (5/10/20/30 שניות). הסקריפט בודק:
1. קורלציה בין הערך בכל אורך-חלון קצר לבין הערך ב-30 שניות (החלון
   שבו כל שאר העבודה חושבה) - קורלציה גבוהה = יציבות טובה.
2. שינוי ממוצע (%) בין 5 שניות ל-30 שניות, כדי לתת תחושה של גודל-האפקט
   לא רק מובהקות.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import os

from config import OUTPUT_DIR


def main():
    path = os.path.join(OUTPUT_DIR, 'multiscale_entropy_sample.csv')
    df = pd.read_csv(path, encoding='utf-8-sig')
    print(f"Loaded {len(df)} rows ({df.groupby(['artistname','trackname','user_id']).ngroups} unique tracks)")

    # לכל שיר, נשווה כל אורך-חלון קצר מול אורך 30 (full)
    wide = df.pivot_table(
        index=['artistname', 'trackname', 'user_id'],
        columns='window_sec_requested',
        values=['mean_entropy', 'h_top_beat', 'negentropy'],
    )

    print("\n" + "=" * 70)
    print("STABILITY CHECK: correlation of each short window vs. the 30-second window")
    print("=" * 70)

    for measure in ['mean_entropy', 'h_top_beat', 'negentropy']:
        print(f"\n--- {measure} ---")
        if 30 not in wide[measure].columns:
            print("  30-second column not found, skipping.")
            continue
        ref = wide[measure][30]
        for w in [5, 10, 20]:
            if w not in wide[measure].columns:
                continue
            pair = pd.concat([wide[measure][w], ref], axis=1, keys=['short', 'full']).dropna()
            if len(pair) < 5:
                print(f"  {w}s: not enough paired data (n={len(pair)})")
                continue
            r_p, p_p = pearsonr(pair['short'], pair['full'])
            r_s, p_s = spearmanr(pair['short'], pair['full'])
            pct_change = ((pair['short'] - pair['full']) / pair['full'].abs()).mean() * 100
            print(f"  {w}s vs 30s (n={len(pair)}): Pearson r={r_p:.3f} (p={p_p:.4f}), "
                  f"Spearman r={r_s:.3f}, mean % change={pct_change:+.1f}%")

    print("\nInterpretation guide: r > .7-.8 generally indicates the measure is reasonably")
    print("stable across window length; lower values suggest the 30-second window choice")
    print("materially affects the complexity estimate, which should be reported as a")
    print("concrete limitation (or motivate re-running the main analysis with a longer window).")

    # ---- בדיקה נוספת: האם הרגישות לאורך-חלון שונה בין שירים פשוטים/מורכבים? ----
    # (מנצל את זה שהדגימה נבנתה מדורגת לפי טרציל מורכבות - ראו pipeline script)
    print("\n" + "=" * 70)
    print("Does window-length sensitivity differ by complexity level?")
    print("=" * 70)
    for measure in ['h_top_beat']:
        ref = wide[measure].get(30)
        w5 = wide[measure].get(5)
        if ref is None or w5 is None:
            continue
        diff = (w5 - ref).abs()
        combined = pd.concat([diff.rename('abs_diff_5v30'), ref.rename('full_value')], axis=1).dropna()
        combined['complexity_tercile'] = pd.qcut(combined['full_value'], 3,
                                                   labels=['Low', 'Medium', 'High'], duplicates='drop')
        print(f"\n--- {measure}: mean |5s - 30s| difference by complexity tercile ---")
        print(combined.groupby('complexity_tercile', observed=True)['abs_diff_5v30'].agg(['mean', 'std', 'count']))


if __name__ == "__main__":
    main()
