"""
run_mixed_model.py
---------------------
מודל 1 האמיתי (#2א): Linear Mixed-Effects Model עם statsmodels,
במקום קירוב ה-OLS+cluster-robust ששימש עד כה כי לא הייתה גישה
ל-statsmodels בסביבת הניתוח הקודמת.

לפני ההרצה: pip install statsmodels --break-system-packages

מריץ שני מודלים ומשווה:
1. Random intercept בלבד (הכי יציב, כמעט תמיד מתכנס)
2. Random intercept + random slope ל-h_top_beat_z לפי משתתף (עשיר יותר,
   אבל עלול לא להתכנס עם מדגם קטן - אם זה קורה, הסקריפט ינפול חזרה
   למודל 1 ויגיד לך במפורש)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

from config import OUTPUT_DIR


def main():
    songs = pd.read_csv(os.path.join(OUTPUT_DIR, 'master_songs.csv'), encoding='utf-8-sig')
    q = pd.read_csv(os.path.join(OUTPUT_DIR, 'questionnaire_scores.csv'), encoding='utf-8-sig')

    df = songs[songs['play_count'] >= 4].copy()
    df = df.merge(q[['user_id', 'nfc_total']], on='user_id', how='inner')
    df = df.dropna(subset=['decay_rate', 'h_top_beat', 'nfc_total'])

    df['h_z'] = (df['h_top_beat'] - df['h_top_beat'].mean()) / df['h_top_beat'].std()
    df['nfc_z'] = (df['nfc_total'] - df['nfc_total'].mean()) / df['nfc_total'].std()

    # אבחון: decay_rate בקנה-מידה גדול/עם חריגים עלול למנוע התכנסות.
    print("decay_rate distribution:")
    print(df['decay_rate'].describe())
    print(f"decay_rate | mean={df['decay_rate'].mean():.3f}, std={df['decay_rate'].std():.3f}, "
          f"min={df['decay_rate'].min():.3f}, max={df['decay_rate'].max():.3f}\n")

    # מנרמלים גם את המשתנה התלוי - עוזר מאוד ליציבות ההתכנסות של REML/ML
    # ב-statsmodels, ולא משנה שום דבר בפרשנות (p-values, כיוון האפקט)
    df['decay_z'] = (df['decay_rate'] - df['decay_rate'].mean()) / df['decay_rate'].std()

    print(f"Model dataset: {len(df)} tracks across {df['user_id'].nunique()} participants\n")

    # ===== מודל A: random intercept בלבד =====
    print("=" * 70)
    print("MODEL A: Random intercept for participant (decay_rate standardized)")
    print("=" * 70)
    model_a = smf.mixedlm(
        "decay_z ~ h_z * nfc_z", df, groups=df["user_id"]
    )
    result_a = None
    for method in ["lbfgs", "cg", "bfgs", "powell"]:
        try:
            result_a = model_a.fit(reml=True, method=method, maxiter=200)
            if result_a.converged:
                print(f"(converged using method='{method}')")
                break
        except Exception:
            continue
    print(result_a.summary())
    print(f"\nConverged: {result_a.converged}")

    # ===== מודל B: random intercept + random slope ל-h_z =====
    print("\n" + "=" * 70)
    print("MODEL B: Random intercept + random slope for Topological Entropy")
    print("=" * 70)
    try:
        model_b = smf.mixedlm(
            "decay_z ~ h_z * nfc_z", df, groups=df["user_id"],
            re_formula="~h_z",
        )
        result_b = None
        for method in ["lbfgs", "cg", "bfgs", "powell"]:
            try:
                result_b = model_b.fit(reml=True, method=method, maxiter=200)
                if result_b.converged:
                    print(f"(converged using method='{method}')")
                    break
            except Exception:
                continue
        print(result_b.summary())
        print(f"\nConverged: {result_b.converged}")
        if not result_b.converged:
            print("WARNING: Model B did not converge cleanly even after trying multiple")
            print("optimizers. Prefer Model A's results if Model A converged.")
    except Exception as e:
        print(f"Model B failed to fit: {type(e).__name__}: {e}")
        print("Falling back to Model A as the primary result.")

    print("\n" + "=" * 70)
    print("Report whichever model converged cleanly as the primary Model 1 result")
    print("in Section 4.2, replacing the cluster-robust OLS approximation used earlier.")
    print("If both converge, prefer Model B (random slopes) but note both in a footnote")
    print("if their conclusions differ meaningfully.")


if __name__ == "__main__":
    main()
