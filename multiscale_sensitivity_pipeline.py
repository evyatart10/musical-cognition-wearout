"""
multiscale_sensitivity_pipeline.py
------------------------------------
בדיקת חוסן (#3ב): האם מדדי המורכבות (mean_entropy, h_top_beat,
negentropy) יציבים ביחס לאורך חלון-הניתוח, או שהם תלויים באופן
משמעותי באורך ה-30-שניות שנבחר?

במקום לעבד מחדש את כל 11,055 השירים (איטי מדי בגלל rate limiting של
iTunes), נדגם מדגם מייצג (ברירת מחדל: 15 שירים לכל נבדק, כ-345 סה"כ),
נוריד אותם מחדש, ונחשב את כל שלושת המדדים בארבעה אורכי-חלון: 5, 10,
20, ו-30 שניות (=החלון המלא).

הפלט: multiscale_entropy_sample.csv - שורה לכל (שיר, אורך-חלון).
"""

import pandas as pd
import numpy as np
import json
import os
import time
import requests
import librosa
from scipy.stats import entropy
from math import log

from config import OUTPUT_DIR
from process_user import get_itunes_preview_url, MIN_BEATS_PER_SEC

N_PER_USER = 15  # כמה שירים לדגום מכל נבדק בסה"כ (יחולק ~שווה בשווה בין 3 רמות המורכבות)
WINDOW_LENGTHS_SEC = [5, 10, 20, 30]  # 30 = כל האודיו שיש (מקדימי iTunes)
L = 4  # אורך רצף לטופולוגית אנטרופיה (כמו בפייפליין המקורי)
TEMP_FILE = "multiscale_temp"
RANDOM_SEED = 42


def compute_measures(y, sr):
    """מחשב mean_entropy, h_top_beat, negentropy, low_beat_confidence על מקטע אודיו נתון."""
    duration_sec = len(y) / sr
    if duration_sec < 2:
        return None  # קצר מדי לניתוח משמעותי

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

    dist = np.bincount(notes, minlength=12) / max(len(notes), 1)
    negentropy = log(12) - entropy(dist[dist > 0]) if notes.size > 0 else np.nan

    return {
        'mean_entropy': shannon, 'h_top_beat': topo,
        'negentropy': negentropy, 'low_beat_confidence': low_conf,
        'n_beats': n_beats,
    }


def main():
    master_path = os.path.join(OUTPUT_DIR, 'master_songs.csv')
    master = pd.read_csv(master_path, encoding='utf-8-sig')
    master = master.dropna(subset=['mean_entropy', 'h_top_beat'])

    # דגימה מדורגת: לכל נבדק, מחלקים את השירים שלו ל-3 טרצילים לפי
    # h_top_beat (פשוט/בינוני/מורכב) ודוגמים בערך N_PER_USER/3 מכל
    # טרציל - כך שהמדגם מייצג את כל טווח-המורכבות, לא רק את
    # השירים הפופולריים ביותר.
    def stratified_sample(g):
        n_strata = 3
        try:
            g = g.copy()
            g['_stratum'] = pd.qcut(g['h_top_beat'], n_strata, labels=False, duplicates='drop')
        except ValueError:
            g['_stratum'] = 0  # אין מספיק ערכים שונים לחלוקה - דוגמים מהכל יחד
        n_per_stratum = max(1, N_PER_USER // n_strata)
        return (
            g.groupby('_stratum', group_keys=False)
            .apply(lambda s: s.sample(n=min(n_per_stratum, len(s)), random_state=RANDOM_SEED))
            .drop(columns='_stratum')
        )

    sample = master.groupby('user_id', group_keys=False).apply(stratified_sample)
    print(f"Sampled {len(sample)} tracks across {sample['user_id'].nunique()} users "
          f"(stratified by complexity tercile within each participant).")

    results = []
    for idx, row in sample.reset_index(drop=True).iterrows():
        artist, track, user_id = row['artistname'], row['trackname'], row['user_id']
        print(f"[{idx + 1}/{len(sample)}] {artist} - {track} ({user_id})...", end=" ")

        preview_url = get_itunes_preview_url(artist, track)
        if not preview_url:
            print("NOT FOUND")
            continue

        audio_path = f"{TEMP_FILE}.m4a"
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            r = requests.get(preview_url, timeout=10)
            r.raise_for_status()
            with open(audio_path, "wb") as f:
                f.write(r.content)

            y_full, sr = librosa.load(audio_path, sr=22050)
            full_duration = len(y_full) / sr

            for window_sec in WINDOW_LENGTHS_SEC:
                if window_sec >= full_duration:
                    y_window = y_full  # החלון המלא שקיים (בד"כ ~30 שניות)
                    actual_window = full_duration
                else:
                    y_window = y_full[: int(window_sec * sr)]
                    actual_window = window_sec

                measures = compute_measures(y_window, sr)
                if measures is None:
                    continue
                measures.update({
                    'artistname': artist, 'trackname': track, 'user_id': user_id,
                    'window_sec_requested': window_sec, 'window_sec_actual': actual_window,
                })
                results.append(measures)

            os.remove(audio_path)
            print("OK")
        except Exception as e:
            print(f"ERROR: {type(e).__name__} {e!r}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
        time.sleep(0.3)  # קצב עדין; ה-rate limiting האמיתי כבר בתוך get_itunes_preview_url

    out_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, 'multiscale_entropy_sample.csv')
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {out_path} ({len(out_df)} rows = tracks x window-lengths)")


if __name__ == "__main__":
    main()
