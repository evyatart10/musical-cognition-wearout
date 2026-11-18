"""
process_user.py  (v2 - iTunes Preview API)
--------------------------------------------
מעבד היסטוריית האזנה של משתמש בודד: מזהה את 500 השירים הפופולריים
ביותר שלו, מוריד עבור כל שיר קטע תצוגה מקדימה (preview, ~30 שניות)
מ-iTunes Search API (חינמי, ללא צורך במפתח/הרשמה), ומחשב עליו מדדי
מורכבות אודיו גולמיים (Shannon entropy ספקטרלי, topological entropy
מבוסס-כרומה, ו-negentropy הרמוני). שומר קובץ CSV גולמי.

שינוי מרכזי מול הגרסה הקודמת (מבוססת yt-dlp/YouTube):
--------------------------------------------------------
הוחלף מקור האודיו מהורדת יוטיוב (yt-dlp) ל-iTunes Search API.
הסיבה: יוטיוב אוכפת בחודשים האחרונים מנגנון סטרימינג (SABR) שדורש
PO Tokens ומחסום קליינטים, מה שהופך הורדה אוטומטית להמונית לבלתי-
יציבה (ראו https://github.com/yt-dlp/yt-dlp/issues/12482). ל-iTunes
Search API אין הגבלה כזו - זו קריאת HTTP פשוטה שמיועדת רשמית לשימוש
תכנותי, ומחזירה קובץ audio/m4a ישיר בן ~30 שניות לכל שיר (כשקיים
בקטלוג של Apple).

מגבלה שכדאי לציין בפרק השיטה של העבודה:
------------------------------------------
1. אורך הקטע מוגבל ל-~30 שניות (preview) במקום השיר המלא. זה מספיק
   לחישוב אנטרופיה ספקטרלית/הרמונית (מחושבת ממילא על פני חלונות STFT
   קצרים בתוך השיר), אך פחות מתאים למדידת תופעות מבניות ארוכות-טווח
   (למשל שינויי סקשן/פזמון-בית).
2. לא כל שיר נמצא בקטלוג iTunes (בעיקר שירים ישראליים נישתיים/עצמאיים
   מאוד) - שירים כאלה יידלגו ויירשמו עם ערכים ריקים, ותוכל לבדוק כמה
   מתוך ה-500 חסרים בסוף הריצה.
3. ה-preview לא תמיד מתחיל מתחילת השיר (לרוב Apple בוחרת קטע מייצג/
   פזמון) - ייתכן הבדל שיטתי קל מול "השיר כולו" שכדאי להזכיר כמגבלה.
"""

import pandas as pd
import numpy as np
import json
import os
import time
import requests
import librosa
from scipy.stats import entropy, linregress
from math import log

from config import DATA_ROOT, OUTPUT_DIR

# --- CONFIGURATION (EXACT PATHS) ---
# יש לעדכן את שני השדות הבאים עבור כל נבדק בנפרד.
INPUT_JSON_FILES = [
    os.path.join(DATA_ROOT, "USER_XX_NAME", "StreamingHistory_music_0.json"),
    os.path.join(DATA_ROOT, "USER_XX_NAME", "StreamingHistory_music_1.json"),
    # הוסיפו כאן עוד שורות לפי הצורך, או מחקו את השורה השנייה אם יש רק קובץ אחד
]
USER_ID = "USER_XX_NAME"

TEMP_FILE = "processing_temp"
L = 4  # Sequence length for Topological Entropy
MIN_BEATS_PER_SEC = 0.5  # סף גס לזיהוי ניתוח פעימה לא אמין (כוונון אפשרי)
ITUNES_TIMEOUT_SEC = 10
ITUNES_MIN_INTERVAL_SEC = 3.5  # ~17 בקשות/דקה - מרווח בטוח מתחת למגבלת ה-~20/דקה של iTunes
ITUNES_RATE_LIMIT_BACKOFF_SEC = 65  # המתנה אחרי 429, מעל חלון הדקה שעליו מבוססת המגבלה
_last_itunes_call_time = 0.0


def get_existing_songs_lookup(directory):
    """
    Scans the output directory for all existing raw CSVs to reuse
    already-computed audio complexity data (avoids re-downloading).
    """
    lookup = {}
    if not os.path.exists(directory):
        os.makedirs(directory)
        return lookup

    print(f"Indexing existing data in {directory}...")
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and "Metadata" not in filename and "master" not in filename.lower():
            try:
                temp_df = pd.read_csv(os.path.join(directory, filename), encoding='utf-8-sig')
                required = ['artistname', 'trackname', 'mean_entropy', 'h_top_beat', 'negentropy']
                if all(c in temp_df.columns for c in required):
                    valid = temp_df.dropna(subset=['mean_entropy', 'h_top_beat', 'negentropy'])
                    for _, row in valid.iterrows():
                        key = (str(row['artistname']).strip().lower(), str(row['trackname']).strip().lower())
                        if key not in lookup:
                            lookup[key] = {
                                'mean_entropy': row['mean_entropy'],
                                'h_top_beat': row['h_top_beat'],
                                'negentropy': row['negentropy'],
                                'low_beat_confidence': row.get('low_beat_confidence', np.nan),
                            }
            except Exception as e:
                print(f"Warning: Skipping {filename}: {e}")
    return lookup


def get_itunes_preview_url(artist, track):
    """
    מחפש שיר ב-iTunes Search API ומחזיר URL להורדה ישירה של קטע
    התצוגה המקדימה (audio/m4a, בדרך כלל ~30 שניות). מחזיר None אם
    לא נמצאה התאמה. כולל rate limiting פנימי כדי לא לחצות את מגבלת
    ה-~20 בקשות/דקה שמוטלת ע"י iTunes Search API (חצייתה גורמת ל-429
    Too Many Requests, ולעיתים אף לתוצאות ריקות בשקט - 200 OK עם
    results=[] - מה שנראה בטעות כ"השיר לא נמצא בקטלוג").
    """
    global _last_itunes_call_time

    for attempt in range(2):
        # --- Rate limiting: מוודא מרווח מינימלי בין קריאות ---
        elapsed = time.time() - _last_itunes_call_time
        if elapsed < ITUNES_MIN_INTERVAL_SEC:
            time.sleep(ITUNES_MIN_INTERVAL_SEC - elapsed)

        try:
            resp = requests.get(
                "https://itunes.apple.com/search",
                params={"term": f"{artist} {track}", "media": "music", "limit": 1},
                timeout=ITUNES_TIMEOUT_SEC,
            )
            _last_itunes_call_time = time.time()

            if resp.status_code == 429:
                print(f"iTunes rate-limit (429) hit on '{artist} - {track}', "
                      f"waiting {ITUNES_RATE_LIMIT_BACKOFF_SEC}s and retrying...")
                time.sleep(ITUNES_RATE_LIMIT_BACKOFF_SEC)
                continue  # לא סופר כ"לא נמצא" - זו חסימת קצב, לא היעדר בקטלוג

            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                return results[0].get("previewUrl")
            return None  # 200 OK אבל אין תוצאות - כנראה באמת לא בקטלוג

        except Exception as e:
            print(f"iTunes search error for '{artist} - {track}': {e}")
            _last_itunes_call_time = time.time()
            return None

    print(f"Giving up on '{artist} - {track}' after repeated rate-limit backoffs.")
    return None


def download_and_process(artist, track):
    """
    מוצא preview דרך iTunes, מוריד אותו, ומחשב מדדי מורכבות אודיו.
    Returns: shannon, topo, negentropy, low_beat_confidence (bool)
    """
    audio_path = f"{TEMP_FILE}.m4a"

    for attempt in range(2):  # ניסיון ראשון + ניסיון חוזר אחד
        try:
            preview_url = get_itunes_preview_url(artist, track)
            if not preview_url:
                return None, None, None, None  # לא נמצא בקטלוג - אין טעם לנסות שוב

            if os.path.exists(audio_path):
                os.remove(audio_path)

            r = requests.get(preview_url, timeout=ITUNES_TIMEOUT_SEC)
            r.raise_for_status()
            with open(audio_path, "wb") as f:
                f.write(r.content)

            y, sr = librosa.load(audio_path, sr=22050)
            duration_sec = len(y) / sr

            stft = np.abs(librosa.stft(y))
            psd = np.mean(stft ** 2, axis=1)
            shannon = entropy(psd / np.sum(psd), base=2)

            _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            n_beats = len(beat_frames)
            low_conf = (n_beats / max(duration_sec, 1)) < MIN_BEATS_PER_SEC

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            beat_chroma = librosa.util.sync(chroma, beat_frames)
            notes = np.argmax(beat_chroma, axis=0)
            ngrams = [tuple(notes[i:i + L]) for i in range(len(notes) - L + 1)]
            topo = log(len(set(ngrams))) / L if len(ngrams) > 0 else 0

            dist = np.bincount(notes, minlength=12) / len(notes)
            negentropy = log(12) - entropy(dist)

            os.remove(audio_path)
            return shannon, topo, negentropy, low_conf

        except Exception as e:
            print(f"Error processing {artist} - {track} (attempt {attempt + 1}/2): "
                  f"[{type(e).__name__}] {e!r}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if attempt == 0:
                time.sleep(1)
                continue
            return None, None, None, None


def load_and_normalize_json(path):
    """
    טוען קובץ streaming history של ספוטיפיי ומנרמל אותו לסכימה האחידה
    (endTime, artistName, trackName, msPlayed), בלי קשר לאיזה משני
    הפורמטים המקובלים של ספוטיפיי הוא:

    1. "Account data" (הפורמט הרגיל): שמות קבצים כמו
       StreamingHistory_music_0.json, שדות: endTime, artistName,
       trackName, msPlayed.

    2. "Extended Streaming History": שמות קבצים כמו
       Streaming_History_Audio_2025.json / Streaming_History_Video_2025.json,
       שדות: ts, ms_played, master_metadata_track_name,
       master_metadata_album_artist_name, episode_name (לפודקאסטים).
       מהפורמט הזה מסננים אוטומטית כל רשומה שהיא פודקאסט (episode_name
       לא ריק) או שאין לה שם שיר (master_metadata_track_name ריק).

    מחזיר: list של dict בסכימה האחידה.
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    if not raw:
        return []

    sample = raw[0]

    if 'endTime' in sample and 'artistName' in sample:
        return raw  # פורמט Account data - אין צורך בהמרה

    if 'ts' in sample and 'master_metadata_track_name' in sample:
        normalized = []
        for rec in raw:
            track = rec.get('master_metadata_track_name')
            episode = rec.get('episode_name')
            if not track or episode:
                continue  # פודקאסט / וידאו / רשומה בלי שיר - לא רלוונטי
            normalized.append({
                'endTime': rec.get('ts'),
                'artistName': rec.get('master_metadata_album_artist_name'),
                'trackName': track,
                'msPlayed': rec.get('ms_played'),
            })
        return normalized

    raise ValueError(
        f"Unrecognized JSON schema in {path}. "
        f"Sample keys found: {list(sample.keys())}"
    )


def process_one_user(json_files, user_id, lookup):
    """
    מעבד נבדק בודד: טוען את קבצי ה-JSON שלו, מזהה את 500 השירים
    הפופולריים ביותר, מביא/מחשב עבורם מדדי מורכבות, ושומר CSV.
    מחזיר dict עם סטטיסטיקת סיכום (להצגה/ללוג).
    """
    all_records = []
    for path in json_files:
        records = load_and_normalize_json(path)
        print(f"Loaded {len(records)} music records from {os.path.basename(path)}")
        all_records.extend(records)
    df_h = pd.DataFrame(all_records)

    before_dedup = len(df_h)
    df_h = df_h.drop_duplicates()
    if before_dedup != len(df_h):
        print(f"Removed {before_dedup - len(df_h)} exact-duplicate rows across input files.")

    df_h['endTime'] = pd.to_datetime(df_h['endTime'])
    max_date = df_h['endTime'].max()

    stats = df_h.groupby(['artistName', 'trackName']).agg(
        play_count=('msPlayed', 'count'),
        total_ms=('msPlayed', 'sum'),
        est_duration=('msPlayed', 'max'),
        first_listen=('endTime', 'min')
    ).reset_index()

    stats['pop_score'] = np.log1p(stats['play_count']) * (stats['total_ms'] / stats['est_duration'])
    top_500 = stats.sort_values('pop_score', ascending=False).head(500)

    final_results = []
    not_found_count = 0
    for idx, row in top_500.iterrows():
        art, trk = row['artistName'], row['trackName']
        key = (str(art).strip().lower(), str(trk).strip().lower())

        print(f"[{len(final_results) + 1}/500] {art} - {trk}...", end="")

        if key in lookup:
            print(" [DATA REUSED]")
            s = lookup[key]['mean_entropy']
            t = lookup[key]['h_top_beat']
            n = lookup[key]['negentropy']
            low_conf = lookup[key]['low_beat_confidence']
        else:
            print(" [FETCHING FROM ITUNES]")
            s, t, n, low_conf = download_and_process(art, trk)
            if s is None:
                not_found_count += 1
            # שים לב: ה-rate limiting האמיתי מול iTunes כבר קורה בתוך
            # get_itunes_preview_url עצמה - אין צורך בהמתנה נוספת כאן.
            # ניתן להוסיף לזיכרון-לוקאלי כדי לחסוך הורדות עתידיות
            if s is not None:
                lookup[key] = {
                    'mean_entropy': s, 'h_top_beat': t, 'negentropy': n,
                    'low_beat_confidence': low_conf,
                }

        song_hist = df_h[(df_h['artistName'] == art) & (df_h['trackName'] == trk)].sort_values('endTime')
        gaps = song_hist['endTime'].diff().dt.total_seconds() / 86400
        decay = linregress(np.arange(len(gaps.dropna())), gaps.dropna()).slope if len(gaps.dropna()) >= 3 else 0

        final_results.append({
            'artistname': art, 'trackname': trk, 'play_count': row['play_count'],
            'total_ms': row['total_ms'],
            'ppm': (row['play_count'] / max((max_date - row['first_listen']).days, 1)) * 30,
            'decay_rate': decay,
            'mean_entropy': s, 'h_top_beat': t, 'negentropy': n,
            'low_beat_confidence': low_conf,
            'user_id': user_id
        })

    user_df = pd.DataFrame(final_results)

    save_path = os.path.join(OUTPUT_DIR, f"{user_id}.csv")
    user_df.to_csv(save_path, index=False, encoding='utf-8-sig')

    n_low_conf = user_df['low_beat_confidence'].sum()
    n_total = len(user_df)
    print(f"\nSUCCESS: Raw data saved for {user_id} -> {save_path}")
    print(f"Note: {not_found_count}/{n_total} songs not found on iTunes (missing values).")
    print(f"Note: {n_low_conf}/{n_total} songs flagged with low beat-tracking confidence.")

    return {
        'user_id': user_id,
        'n_total': n_total,
        'n_found': n_total - not_found_count,
        'not_found_count': not_found_count,
        'n_low_conf': int(n_low_conf),
        'save_path': save_path,
    }


def main():
    lookup = get_existing_songs_lookup(OUTPUT_DIR)
    process_one_user(INPUT_JSON_FILES, USER_ID, lookup)


if __name__ == "__main__":
    main()
