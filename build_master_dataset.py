"""
build_master_dataset.py
------------------------
מריצים פעם אחת בלבד, אחרי שכל 22 קבצי המשתמשים הגולמיים (raw) קיימים
בתיקיית OUTPUT_DIR (תוצרי process_user.py).

מה הסקריפט עושה:
1. מאחד את כל קבצי המשתמשים הגולמיים לטבלה אחת (master_songs).
2. מבצע נרמול Z-score גלובלי ל-mean_entropy ול-h_top_beat על פני *כל*
   השירים של *כל* הנבדקים ביחד (לא בנפרד לכל משתמש כמו שהיה בקוד
   המקורי) - זו הנקודה הקריטית שתוקנה.
3. מחשב song_complexity_index לכל שיר, ו-mci_score לכל נבדק (ממוצע
   ה-song_complexity_index שלו) - עכשיו יש שונות אמיתית בין הנבדקים.
4. מריץ בדיקת קונפאונדים בסיסית: קורלציה בין המדדים הגולמיים
   (mean_entropy, h_top_beat) לבין total_ms (פרוקסי לאורך/פופולריות
   השיר) ואחוז השירים המסומנים low_beat_confidence לכל נבדק - כדי
   לחשוף הטיות שיטתיות אפשריות (בעיקר רלוונטי לנבדקים עם הרבה קלאסי).
5. שומר שני קבצים: master_songs.csv (רמת-שיר) ו-
   master_users_mci.csv (רמת-נבדק, מוכן לצירוף עם NFC/כרונוטיפ).

אופציונלי: אם קיים קובץ מטא-דאטה עם עמודת genre (למשל ה-CSV המקורי
שכלל audio-features + genre), אפשר לצרף אותו לפני הרצת בדיקת
הקונפאונדים לפי ז'אנר - ראו את החלק המסומן "OPTIONAL GENRE JOIN" למטה.
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr

from config import OUTPUT_DIR

MASTER_SONGS_PATH = os.path.join(OUTPUT_DIR, "master_songs.csv")
MASTER_USERS_PATH = os.path.join(OUTPUT_DIR, "master_users_mci.csv")

# OPTIONAL: אם יש קובץ המכיל עמודת genre פר שיר-פר-משתמש, ציינו כאן
GENRE_METADATA_PATH = None  # לדוגמה: os.path.join(OUTPUT_DIR, "Master_Users_Metadata_with_genre.csv")


def load_all_raw_user_files(directory):
    frames = []
    for filename in os.listdir(directory):
        if not filename.endswith(".csv"):
            continue
        if "master" in filename.lower() or "metadata" in filename.lower():
            continue
        path = os.path.join(directory, filename)
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
            required = ['artistname', 'trackname', 'mean_entropy', 'h_top_beat', 'negentropy', 'user_id']
            if all(c in df.columns for c in required):
                frames.append(df)
            else:
                print(f"Skipping {filename}: missing required raw columns.")
        except Exception as e:
            print(f"Warning: could not read {filename}: {e}")

    if not frames:
        raise RuntimeError("No valid raw user CSVs found in OUTPUT_DIR.")

    return pd.concat(frames, ignore_index=True)


def main():
    master = load_all_raw_user_files(OUTPUT_DIR)
    n_users = master['user_id'].nunique()
    print(f"Loaded {len(master)} songs across {n_users} users.")

    before = len(master)
    master = master.dropna(subset=['mean_entropy', 'h_top_beat'])
    print(f"Dropped {before - len(master)} songs with missing raw audio metrics.")

    # --- GLOBAL NORMALIZATION (התיקון המרכזי) ---
    shannon_mean, shannon_std = master['mean_entropy'].mean(), master['mean_entropy'].std()
    topo_mean, topo_std = master['h_top_beat'].mean(), master['h_top_beat'].std()

    master['norm_shannon'] = (master['mean_entropy'] - shannon_mean) / shannon_std
    master['norm_topo'] = (master['h_top_beat'] - topo_mean) / topo_std
    master['song_complexity_index'] = (master['norm_shannon'] + master['norm_topo']) / 2

    print(f"\nGlobal normalization stats (n={len(master)} songs, {n_users} users):")
    print(f"  mean_entropy: mean={shannon_mean:.3f}, std={shannon_std:.3f}")
    print(f"  h_top_beat:   mean={topo_mean:.3f}, std={topo_std:.3f}")

    # --- PER-USER MCI (רמת נבדק, לצירוף עם NFC/כרונוטיפ) ---
    user_agg = master.groupby('user_id').agg(
        mci_score=('song_complexity_index', 'mean'),
        mci_std=('song_complexity_index', 'std'),
        n_songs=('song_complexity_index', 'count'),
        total_listening_hours=('total_ms', lambda x: x.sum() / 3600000),
        pct_low_beat_confidence=('low_beat_confidence', 'mean') if 'low_beat_confidence' in master.columns else ('song_complexity_index', lambda x: np.nan),
    ).reset_index()

    print("\nPer-user MCI summary:")
    print(user_agg[['user_id', 'mci_score', 'n_songs']].to_string(index=False))

    # --- CONFOUND CHECKS ---
    print("\n--- Confound diagnostics ---")
    valid = master.dropna(subset=['total_ms'])
    if len(valid) > 10:
        r_shannon, p_shannon = pearsonr(valid['total_ms'], valid['mean_entropy'])
        r_topo, p_topo = pearsonr(valid['total_ms'], valid['h_top_beat'])
        print(f"corr(total_ms, mean_entropy) = {r_shannon:.3f} (p={p_shannon:.4f})")
        print(f"corr(total_ms, h_top_beat)   = {r_topo:.3f} (p={p_topo:.4f})")
        print("* קורלציה חזקה (|r|>0.3) כאן מעידה על confound אפשרי לפי אורך/פופולריות השיר.")

    if 'low_beat_confidence' in master.columns:
        pct_flagged = master['low_beat_confidence'].mean() * 100
        print(f"\n{pct_flagged:.1f}% מכלל השירים מסומנים low_beat_confidence.")
        print("מומלץ לבדוק אם האחוז הזה גבוה במיוחד אצל נבדקים ספציפיים (למשל מאזיני קלאסי כבדים),")
        print("ולשקול לדווח על תת-קבוצה זו בנפרד או להחריג אותה מניתוח h_top_beat/negentropy.")

    # --- OPTIONAL GENRE JOIN ---
    if GENRE_METADATA_PATH and os.path.exists(GENRE_METADATA_PATH):
        genre_df = pd.read_csv(GENRE_METADATA_PATH, encoding='utf-8-sig')
        if {'artistname', 'trackname', 'user_id', 'genre'}.issubset(genre_df.columns):
            merged = master.merge(
                genre_df[['artistname', 'trackname', 'user_id', 'genre']],
                on=['artistname', 'trackname', 'user_id'], how='left'
            )
            genre_summary = merged.groupby('genre')['song_complexity_index'].agg(['mean', 'std', 'count'])
            print("\n--- Complexity index by genre ---")
            print(genre_summary.sort_values('mean', ascending=False).to_string())
        else:
            print("GENRE_METADATA_PATH provided but missing required columns; skipping genre check.")

    # --- SAVE ---
    master.to_csv(MASTER_SONGS_PATH, index=False, encoding='utf-8-sig')
    user_agg.to_csv(MASTER_USERS_PATH, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {MASTER_SONGS_PATH}")
    print(f"Saved: {MASTER_USERS_PATH}  <- זה הקובץ לצירוף עם NFC/כרונוטיפ")


if __name__ == "__main__":
    main()
