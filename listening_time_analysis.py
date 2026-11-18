"""
listening_time_analysis.py
----------------------------
מנתח את שעות ההאזנה של כל הנבדקים, ללא תלות בהורדת אודיו - רץ מיידית
על קבצי ה-JSON הגולמיים + קבצי ה-CSV שכבר חושבו ע"י process_user.py.

חשוב: משתמש בהיסטוריית ההאזנה *המלאה* (לא רק ב-500 השירים
הפופולריים), כי סינון-לפי-פופולריות היה רלוונטי רק לצמצום עומס
ההורדות, ולא רלוונטי/רצוי לניתוח שעות - הוא היה מטה את הדגימה כלפי
האזנות חוזרות ומפספס האזנות חד-פעמיות/גילוי.

מייצר שני קבצי פלט:
1. listening_time_summary.csv - רמת-משתמש: אנטרופיית שעות ההאזנה,
   שעת-שיא, וכו'. מוכן לצירוף עם כרונוטיפ/NFC לניתוח בין-נבדקים.
2. event_level_complexity_by_hour.csv - רמת-אירוע (כל השמעה בודדת),
   עם שעת ההשמעה וציון המורכבות של השיר *במקרים שהשיר נמצא בקבוצת
   ה-500 שכבר עובדו* (עמודת has_complexity_score מציינת אם יש התאמה).
   מיועד לניתוח mixed-effects (נבדק כ-random intercept) של הקשר בין
   שעת האזנה למורכבות המוזיקה, ואיך זה נבדל לפי כרונוטיפ - בעל עוצמה
   סטטיסטית גבוהה בהרבה מקורלציה פשוטה על 22 נקודות (ממוצע לכל נבדק).
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import entropy

from config import DATA_ROOT, OUTPUT_DIR
from process_user import load_and_normalize_json

ROOT_DATA_DIR = DATA_ROOT  # אותה תיקייה כמו ב-run_all_users.py
MASTER_SONGS_PATH = os.path.join(OUTPUT_DIR, "master_songs.csv")  # תוצר build_master_dataset.py


def compute_hour_entropy(hours):
    """אנטרופיית שאנון של התפלגות שעות ההאזנה (0-23)."""
    counts = np.bincount(hours, minlength=24)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return entropy(probs, base=2)


NIGHT_HOURS = {23, 0, 1, 2, 3, 4}  # חלון "לילה מאוחר": 23:00-04:59


def pct_late_night(hours):
    """אחוז ההשמעות שנופלות בחלון הלילה המאוחר - מדד ישיר וקל-פרשנות לכרונוטיפ."""
    return np.isin(hours, list(NIGHT_HOURS)).mean()


def circular_mean_hour(hours):
    """
    שעה 'ממוצעת' בצורה נכונה למשתנה מעגלי (23:00 ו-01:00 קרובות,
    לא רחוקות, כמו שממוצע רגיל היה מחשב).
    """
    radians = hours / 24 * 2 * np.pi
    mean_angle = np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    mean_hour = (mean_angle / (2 * np.pi)) * 24
    return mean_hour % 24


def find_user_folders(root_dir):
    users = []
    for entry in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, entry)
        if not os.path.isdir(full_path):
            continue
        json_files = glob.glob(os.path.join(full_path, "**", "*.json"), recursive=True)
        if json_files:
            users.append((entry, json_files))
    return users


def main():
    if not os.path.exists(MASTER_SONGS_PATH):
        print(f"Warning: {MASTER_SONGS_PATH} not found. "
              f"Run build_master_dataset.py first if you want complexity scores joined in. "
              f"Continuing with listening-time-only analysis.")
        master_songs = None
    else:
        master_songs = pd.read_csv(MASTER_SONGS_PATH, encoding='utf-8-sig')
        master_songs['match_key'] = (
            master_songs['artistname'].str.strip().str.lower() + "|||" +
            master_songs['trackname'].str.strip().str.lower()
        )

    users = find_user_folders(ROOT_DATA_DIR)
    print(f"Found {len(users)} user folders.\n")

    summary_rows = []
    event_rows = []

    for user_id, json_files in users:
        print(f"Processing listening times for {user_id}...")
        all_records = []
        for path in json_files:
            all_records.extend(load_and_normalize_json(path))

        if not all_records:
            print(f"  No music records found for {user_id}, skipping.")
            continue

        df = pd.DataFrame(all_records)
        df['endTime'] = pd.to_datetime(df['endTime'])
        df = df.dropna(subset=['endTime'])
        df['hour'] = df['endTime'].dt.hour
        df['weekday'] = df['endTime'].dt.dayofweek  # 0=שני ... 6=ראשון
        df['is_weekend'] = df['weekday'].isin([4, 5])  # שישי-שבת בישראל

        hours = df['hour'].to_numpy()

        summary_rows.append({
            'user_id': user_id,
            'n_plays_total': len(df),
            'hour_entropy': compute_hour_entropy(hours),
            'circular_mean_hour': circular_mean_hour(hours),
            'peak_hour': int(df['hour'].mode().iloc[0]),
            'pct_late_night': pct_late_night(hours),
            'pct_weekend_listening': df['is_weekend'].mean(),
            'hour_entropy_weekday': compute_hour_entropy(df.loc[~df['is_weekend'], 'hour'].to_numpy()),
            'hour_entropy_weekend': compute_hour_entropy(df.loc[df['is_weekend'], 'hour'].to_numpy())
            if df['is_weekend'].sum() > 5 else np.nan,
        })

        # --- רמת-אירוע: חיבור עם ציון מורכבות (אם קיים) ---
        if master_songs is not None:
            df['match_key'] = (
                df['artistName'].astype(str).str.strip().str.lower() + "|||" +
                df['trackName'].astype(str).str.strip().str.lower()
            )
            user_songs = master_songs[master_songs['user_id'] == user_id][
                ['match_key', 'song_complexity_index', 'mean_entropy', 'h_top_beat']
            ]
            merged = df.merge(user_songs, on='match_key', how='left')
            merged['has_complexity_score'] = merged['song_complexity_index'].notna()
            merged['user_id'] = user_id

            event_rows.append(
                merged[['user_id', 'hour', 'weekday', 'is_weekend',
                        'has_complexity_score', 'song_complexity_index',
                        'mean_entropy', 'h_top_beat']]
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "listening_time_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {summary_path}  <- לצירוף עם כרונוטיפ/NFC (ניתוח בין-נבדקים)")

    if event_rows:
        event_df = pd.concat(event_rows, ignore_index=True)
        event_path = os.path.join(OUTPUT_DIR, "event_level_complexity_by_hour.csv")
        event_df.to_csv(event_path, index=False, encoding='utf-8-sig')
        pct_matched = event_df['has_complexity_score'].mean() * 100
        print(f"Saved: {event_path}  <- לניתוח mixed-effects (נבדק כ-random intercept)")
        print(f"({pct_matched:.1f}% מהאירועים הותאמו לשיר עם ציון מורכבות ידוע)")


if __name__ == "__main__":
    main()
