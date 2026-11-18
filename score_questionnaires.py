"""
score_questionnaires.py
-------------------------
מנקד את שני השאלונים (NFC-18, MEQ-19) מתוך קובץ תגובות גוגל פורמס,
ומייצא טבלה נקייה: user_id, nfc_total, meq_total, meq_category.

הערות מתודולוגיות חשובות:
--------------------------
1. NFC-18 (Cacioppo, Petty & Kao 1984): משתמש בסולם הבי-פולרי המקורי
   (4- עד 4+ דרך 0, 9 דרגות). 9 מתוך 18 הפריטים מנוסחים הפוך (reverse-
   scored) ולכן ההיגד "מסכים" שלהם מבטא NFC *נמוך* - הניקוד שלהם
   הופך (מוכפל ב--1) לפני הסכימה. הסיווג ל-9 הפריטים ההפוכים נעשה
   ידנית לפי תוכן הפריט (סמנטיקה), לא לפי ניחוש - וזה תואם בדיוק את
   העובדה הידועה שבגרסה זו 9/18 פריטים הפוכים.
   טווח ציון סופי (אחרי טרנספורמציה ל-1..9 לכל פריט): 18-162.

2. MEQ-19 (Horne & Östberg 1976, גרסת Terman et al. 2001/2008
   Self-Assessment): לכל פריט מפתח ניקוד ייחודי משלו (לא סולם אחיד).
   מפתח הניקוד כאן נבנה מהתאמה ישירה בין אפשרויות התשובה שהוצעו
   בטופס לבין הטווחים/הניסוחים במפתח הרשמי - אומת פריט-פריט מול
   המקור. שני פריטים (33, 37) דורשים אימות נוסף ע"י בעל השאלון כי
   הניסוח שלהם בטופס שונה מעט מהמקור (ראו הערות inline).
   טווח ציון סופי: 16-86, עם 5 קטגוריות סטנדרטיות.
"""

import re
import pandas as pd

import os
from config import OUTPUT_DIR, QUESTIONNAIRE_XLSX

INPUT_PATH = QUESTIONNAIRE_XLSX
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "questionnaire_scores.csv")

USER_ID_COL = "Column 42"  # העמודה שהוספת עם ה-USER_ID

# ============================================================
# NFC-18
# ============================================================

# אינדקסים 1-18 (מיקום בתוך 18 הפריטים, לא אינדקס העמודה בקובץ)
# של הפריטים המנוסחים הפוך - מזוהים לפי תוכן הפריט:
# "חשיבה היא לא ההנאה המועדפת", "אעדיף מעט מחשבה", "מנסה להימנע
# מחשיבה מעמיקה", "חושב רק במידה הנדרשת", "מעדיף פרויקטים קטנים",
# "אוהב משימות שדורשות מעט מחשבה אחרי שלמדתי", "למידה חדשה לא
# מרגשת", "מרגיש הקלה ולא סיפוק", "מספיק לי שזה עובד, לא אכפת לי
# איך/למה" - כולם ניסוחים בכיוון NFC נמוך.
NFC_REVERSE_POSITIONS = {3, 4, 5, 7, 8, 9, 12, 16, 17}


def parse_nfc_value(raw):
    """
    ממיר ערך גולמי מהטופס (למשל '4+ (מסכים מאוד)', '2-', 0) לערך
    מספרי בטווח -4..+4.
    """
    if pd.isna(raw):
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    s = str(raw).strip()
    if s == "0":
        return 0
    m = re.match(r"^(\d)([+-])", s)
    if not m:
        raise ValueError(f"Unrecognized NFC response format: {raw!r}")
    digit, sign = m.groups()
    val = int(digit)
    return val if sign == "+" else -val


def score_nfc_row(row, nfc_columns):
    total = 0
    for i, col in enumerate(nfc_columns, start=1):  # i = 1..18
        raw_val = parse_nfc_value(row[col])
        if raw_val is None:
            return None  # פריט חסר - לא ניתן לחשב ציון מלא לשורה זו
        if i in NFC_REVERSE_POSITIONS:
            raw_val = -raw_val
        # טרנספורמציה מ--4..4 ל-1..9 (מוסכמה נפוצה, לא משפיעה על
        # קורלציות - רק הופכת את הטווח לחיובי/קריא יותר)
        total += raw_val + 5
    return total  # טווח: 18-162


# ============================================================
# MEQ-19
# ============================================================
# כל dict ממפה את הטקסט המדויק שהופיע בטופס לניקוד הרשמי המתאים,
# מאומת מול המקור (Terman et al. Self-Assessment Version).

MEQ_SCORING = [
    # Q1 - שעת קימה חופשית
    {"07:45 - 09:45": 3, "09:45 - 11:00": 2, "06:30 - 07:45": 4, "11:00 - 12:00": 1},
    # Q2 - שעת שינה חופשית
    {"22:15 - 00:30": 3, "00:30 - 01:45": 2, "01:45 - 03:00": 1, "21:00 - 22:15": 4},
    # Q3 - תלות בשעון מעורר
    {"תלוי/ה במידה מסוימת": 2, "מעט תלוי/ה": 3, "תלוי/ה מאוד": 1},
    # Q4 - קלות קימה בבוקר
    {"קל למדי": 3, "קשה למדי": 2, "קשה מאוד": 1, "קל מאוד": 4},
    # Q5 - עירנות חצי שעה אחרי יקיצה
    {"מעט עירני/ת": 2, "בכלל לא עירני/ת": 1, "עירני/ת למדי": 3, "עירני/ת מאוד": 4},
    # Q6 - תיאבון חצי שעה אחרי יקיצה
    {"גרוע למדי": 2, "גרוע מאוד": 1, "טוב למדי": 3, "טוב מאוד": 4},
    # Q7 - תחושה חצי שעה אחרי יקיצה
    {"עייף/ה למדי": 2, "רענן/ה למדי": 3, "עייף/ה מאוד": 1, "רענן/ה מאוד": 4},
    # Q8 - דחיית שינה אם אין התחייבות למחרת
    {"1-2 שעות מאוחר יותר": 2, "פחות משעה מאוחר יותר": 3,
     "אף פעם לא מאוחר יותר": 4, "יותר מ-2 שעות מאוחר יותר": 1},
    # Q9 - תפקוד בפעילות גופנית 07:00
    {"אהיה במצב טוב": 4, "יהיה לי קשה מאוד": 1, "אתפקד בקושי": 2},
    # Q10 - שעת עייפות בערב
    {"22:15 - 00:30": 3, "01:45 - 03:00": 1, "00:30 - 01:45": 2, "21:00 - 22:15": 4},
    # Q11 - שעת שיא למבחן שעתיים
    {"11:00 - 13:00": 4, "15:00 - 17:00": 2, "08:00 - 10:00": 6},
    # Q12 - עייפות אם שוכב לישון ב-23:00
    {"מעט עייף/ה": 2, "בכלל לא עייף/ה": 0, "עייף/ה למדי": 3},
    # Q13 - התעוררות אחרי שינה מאוחרת - *** לאמת מול הכוונה המקורית ***
    # מיפוי: שתי הגרסאות של "אתעורר מאוחר" מקבלות את הניקוד הנמוך
    # ביותר (1) בהיעדר אפשרות ביניים מדויקת יותר במקור.
    {"אתעורר מאוחר מהרגיל": 1, "אתעורר בזמן הרגיל ואנמנם": 3,
     "אתעורר הרבה יותר מאוחר מהרגיל": 1, "אתעורר בזמן הרגיל ולא ארדם שוב": 4},
    # Q14 - משמרת לילה 04:00-06:00
    {"אשן טוב לפני ואנמנם אחרי": 3, "אשן רק לפני המשימה": 4,
     "אנמנם לפני ואשן אחרי": 2, "לא אלך לישון עד שהמשימה תסתיים": 1},
    # Q15 - שעתיים עבודה פיזית קשה
    {"11:00 - 13:00": 3, "15:00 - 17:00": 2, "19:00 - 21:00": 1, "08:00 - 10:00": 4},
    # Q16 - פעילות גופנית 23:00
    {"אהיה במצב טוב": 1, "אתפקד בקושי": 3, "יהיה לי קשה מאוד": 4},
    # Q17 - בחירת 5 שעות עבודה - *** לאמת מול הכוונה המקורית ***
    # הטווחים בטופס גסים יותר מהמקור; מופו לפי שעת ההתחלה המשוערת.
    {"08:00 - 13:00": 4, "13:00 - 18:00": 3, "17:00 - 22:00": 1},
    # Q18 - שעת שיא ביום
    {"10:00 - 17:00": 3, "17:00 - 22:00": 2, "08:00 - 10:00": 4},
    # Q19 - סיווג עצמי בוקר/ערב
    {"יותר ערב מבוקר": 2, "טיפוס ערב מובהק": 1,
     "יותר בוקר מערב": 4, "טיפוס בוקר מובהק": 6},
]


def meq_category(score):
    if score <= 30:
        return "definite_evening"
    elif score <= 41:
        return "moderate_evening"
    elif score <= 58:
        return "intermediate"
    elif score <= 69:
        return "moderate_morning"
    else:
        return "definite_morning"


def score_meq_row(row, meq_columns):
    total = 0
    for col, mapping in zip(meq_columns, MEQ_SCORING):
        raw_val = row[col]
        if pd.isna(raw_val):
            return None
        raw_val = str(raw_val).strip()
        if raw_val not in mapping:
            raise ValueError(f"Unrecognized MEQ response '{raw_val}' for column '{col}'")
        total += mapping[raw_val]
    return total  # טווח: 16-86


def main():
    df = pd.read_excel(INPUT_PATH)

    nfc_columns = df.columns[3:21]   # 18 פריטים
    meq_columns = df.columns[21:40]  # 19 פריטים
    assert len(nfc_columns) == 18, f"Expected 18 NFC columns, got {len(nfc_columns)}"
    assert len(meq_columns) == 19, f"Expected 19 MEQ columns, got {len(meq_columns)}"

    df = df.dropna(subset=[USER_ID_COL]).copy()

    # Optional ID corrections (e.g. a participant whose ID was mistyped in the form).
    # Put a JSON object {"WRONG_ID": "CORRECT_ID"} in data_raw/id_corrections.json.
    # The file is private (data_raw/ is git-ignored) and is simply skipped if absent.
    corrections_path = os.path.join(os.path.dirname(QUESTIONNAIRE_XLSX), "id_corrections.json")
    if os.path.exists(corrections_path):
        import json
        with open(corrections_path, encoding="utf-8") as fh:
            df[USER_ID_COL] = df[USER_ID_COL].replace(json.load(fh))

    print(f"{len(df)} rows with a valid USER_ID.")

    df["nfc_total"] = df.apply(lambda r: score_nfc_row(r, nfc_columns), axis=1)
    df["meq_total"] = df.apply(lambda r: score_meq_row(r, meq_columns), axis=1)
    df["meq_category"] = df["meq_total"].apply(lambda s: meq_category(s) if pd.notna(s) else None)

    result = df[[USER_ID_COL, "nfc_total", "meq_total", "meq_category"]].rename(
        columns={USER_ID_COL: "user_id"}
    )

    n_incomplete = result["nfc_total"].isna().sum() + result["meq_total"].isna().sum()
    if n_incomplete:
        print(f"Warning: {n_incomplete} missing scores due to incomplete questionnaire responses.")

    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_PATH}")
    print(result)


if __name__ == "__main__":
    main()
