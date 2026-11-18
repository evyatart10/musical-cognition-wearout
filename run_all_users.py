"""
run_all_users.py
------------------
אורקסטרציה: מריץ את process_one_user (מ-process_user.py) על כל
הנבדקים בזה אחר זה, אוטומטית, בלי צורך לערוך ולהריץ ידנית כל פעם.

איך להשתמש:
1. ארגן תיקייה אחת (ROOT_DATA_DIR) שבתוכה תת-תיקייה אחת לכל נבדק.
   שם תת-התיקייה = ה-USER_ID הרצוי (למשל "USER_01_A").
   בתוך כל תת-תיקייה שימו את כל קבצי ה-JSON של אותו נבדק (כל מבנה
   תיקיות פנימי מתחתיה בסדר - הסקריפט מחפש רקורסיבית).

   דוגמה למבנה:
   USERS SPOTIFY DATA/
     USER_01_A/
       StreamingHistory_music_0.json
       StreamingHistory_music_1.json
     USER_02_B/
       Spotify Extended Streaming History/
         Streaming_History_Audio_2025.json

2. הרץ את הסקריפט הזה (לא process_user.py הבודד). הוא ירוץ ברקע על
   כל הנבדקים ברצף - אפשר להשאיר את זה רץ כל הלילה.

3. אם נבדק כבר יש לו CSV קיים ב-OUTPUT_DIR עם שיעור השלמה גבוה
   (SKIP_COMPLETION_THRESHOLD ומעלה) - הוא ידולג אוטומטית. אם יש לו
   CSV אבל עם שיעור השלמה נמוך (כמו המקרה של 20/500) - הוא יעובד
   מחדש מאפס (ה-lookup הגלובלי עדיין יחסוך הורדות של שירים שכבר
   הצליחו בעבר עבור נבדקים אחרים/ריצות קודמות).

4. אם נבדק ספציפי נכשל לגמרי (חריגה/exception) - הסקריפט ידווח על
   זה בסיכום בסוף, אך ימשיך לנבדק הבא ולא יעצור את כל הריצה.
"""

import os
import glob
import traceback
import pandas as pd

from config import DATA_ROOT, OUTPUT_DIR
from process_user import (
    get_existing_songs_lookup,
    process_one_user,
)

ROOT_DATA_DIR = DATA_ROOT
SKIP_COMPLETION_THRESHOLD = 0.5  # נבדק עם פחות מ-50% שירים תקינים יעובד מחדש


def get_existing_completion_rate(user_id):
    """
    בודק אם כבר קיים CSV לנבדק הזה, ומחזיר את שיעור השורות התקינות
    שלו (0.0 אם אין קובץ בכלל).
    """
    path = os.path.join(OUTPUT_DIR, f"{user_id}.csv")
    if not os.path.exists(path):
        return 0.0
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
        if len(df) == 0 or 'mean_entropy' not in df.columns:
            return 0.0
        return df['mean_entropy'].notna().mean()
    except Exception:
        return 0.0


def find_user_folders(root_dir):
    """
    מחזיר list של (user_id, [json_files]) עבור כל תת-תיקייה תחת
    root_dir שמכילה לפחות קובץ JSON אחד (רקורסיבית).
    """
    users = []
    if not os.path.exists(root_dir):
        print(f"ERROR: ROOT_DATA_DIR not found: {root_dir}")
        return users

    for entry in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, entry)
        if not os.path.isdir(full_path):
            continue
        json_files = glob.glob(os.path.join(full_path, "**", "*.json"), recursive=True)
        if json_files:
            users.append((entry, json_files))
        else:
            print(f"Warning: no JSON files found under {full_path}, skipping.")
    return users


def main():
    users = find_user_folders(ROOT_DATA_DIR)
    print(f"Found {len(users)} user folders under {ROOT_DATA_DIR}\n")

    lookup = get_existing_songs_lookup(OUTPUT_DIR)

    summary = []
    for user_id, json_files in users:
        completion = get_existing_completion_rate(user_id)
        if completion >= SKIP_COMPLETION_THRESHOLD:
            print(f"=== {user_id}: SKIPPED (already {completion:.0%} complete) ===\n")
            summary.append({'user_id': user_id, 'status': 'skipped', 'completion': completion})
            continue

        print(f"=== Processing {user_id} ({len(json_files)} json file(s)) ===")
        try:
            result = process_one_user(json_files, user_id, lookup)
            result['status'] = 'done'
            summary.append(result)
        except Exception as e:
            print(f"!!! FAILED processing {user_id}: {e}")
            traceback.print_exc()
            summary.append({'user_id': user_id, 'status': 'error', 'error': str(e)})
        print()

    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    for row in summary:
        if row['status'] == 'done':
            print(f"{row['user_id']}: DONE - {row['n_found']}/{row['n_total']} found, "
                  f"{row['n_low_conf']} low-beat-confidence")
        elif row['status'] == 'skipped':
            print(f"{row['user_id']}: SKIPPED (already {row['completion']:.0%} complete)")
        else:
            print(f"{row['user_id']}: ERROR - {row.get('error')}")


if __name__ == "__main__":
    main()
