# Musical cognition and listener wear-out

Analysis code for the seminar paper *"Musical Cognition and Listener Wear-Out: Need for Cognition and
Chronotype as Moderators of Structural Engagement with Music"* (Hebrew University, seminar of Prof. Roni Granot).

The code turns raw Spotify streaming histories and two questionnaires (NFC-18, MEQ-19) into the
measures and statistics reported in the paper.

> **Privacy.** The Spotify exports, questionnaire responses and every derived per-participant table
> are personal data. They are **not** part of this repository, and `.gitignore` blocks all
> `.json / .csv / .xlsx / .docx / .zip` files. Check `git status` before the first push.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`librosa` needs an audio backend that can decode `.m4a` (install `ffmpeg` and put it on the PATH).

## Where the data goes

All paths live in `src/config.py`; nothing else contains a path.

```
data_raw/
  questionnaires.xlsx          Google-Forms export of the responses
  USER_01_NAME/*.json          one folder per participant (any sub-folder depth)
  USER_02_NAME/*.json
output/                        every generated file is written here
```

Use other locations through environment variables: `SEMINAR_DATA_ROOT`, `SEMINAR_OUTPUT_DIR`,
`SEMINAR_QUESTIONNAIRE`. Both Spotify formats are accepted (Account data `StreamingHistory_music_*.json`
and Extended `Streaming_History_Audio_*.json`).

## Running order

Run from inside `src/` (`cd src`).

| # | Script | What it does | Main output (in `output/`) |
|---|--------|--------------|----------------------------|
| 1 | `score_questionnaires.py` | Scores NFC-18 (9 reverse items, 18-162) and MEQ-19 (+ category) | `questionnaire_scores.csv` |
| 2 | `run_all_users.py` | For every participant: top-500 tracks, 30 s iTunes preview, spectral entropy, topological entropy, negentropy, decay rate. Resumable (skips finished users) | `<USER_ID>.csv` |
| 3 | `build_master_dataset.py` | Merges users, **global** z-normalization, complexity index | `master_songs.csv`, `master_users_mci.csv` |
| 4 | `listening_time_analysis.py` | Hour entropy, circular mean hour, % late night; event-level table | `listening_time_summary.csv`, `event_level_complexity_by_hour.csv` |
| 5 | `merge_and_correlate.py` | Merges questionnaires + listening + complexity; the five pre-planned correlations | `merged_analysis_table.csv` |
| 6 | `run_mixed_model.py` | **Primary Model 1**: mixed model, random intercept (A) and random intercept + slope (B) | console |
| 7 | `analysis_model1_ols.py` | Model 1, cluster-robust OLS (secondary), with/without low-beat-confidence tracks | console |
| 8 | `analysis_model2_events.py` | Model 2: complexity at on-peak vs. off-peak hours x NFC | `event_level_model2.csv` |
| 9 | `analysis_intermediate_variability.py` | Supplementary: time-of-day variability, Intermediate vs. morning/evening types | console |
| 10 | `analysis_secondary.py` | Per-participant slopes vs. NFC, confound checks, artist entropy | `artist_diversity_by_user.csv` |
| 11 | `multiscale_sensitivity_pipeline.py` | Audio-window sensitivity: 15 stratified tracks per participant at 5/10/20/30 s | `multiscale_entropy_sample.csv` |
| 12 | `analyze_window_sensitivity.py` | Correlations of each short window with the 30 s window | console |
| 13 | `make_figures.py` | Figure 1 (decay vs. entropy by NFC group) and Figure 2 (MEQ vs. mean listening hour) | `figure1_model1.png`, `figure2_chronotype.png` |

`process_user.py` is the per-participant engine used by step 2 (it can also be run on its own for one user);
`stats_utils.py` and `config.py` are shared helpers.

## Measures in one paragraph

* **Spectral entropy** (`mean_entropy`): Shannon entropy of the mean STFT power spectrum.
* **Topological entropy** (`h_top_beat`): `log(#distinct 4-grams) / 4` over beat-synchronous dominant-chroma symbols;
  tracks with too few detected beats get `low_beat_confidence = True`.
* **Decay rate**: OLS slope of the gap (days) between consecutive plays over the play sequence; needs >= 4 plays.
* **Listening time**: Shannon entropy of the hour-of-day distribution, circular mean hour, share of plays 23:00-04:59.
  Hours are taken exactly as stored in the Spotify export (no time-zone conversion).
* z-scores are **global** (across all tracks / participants), not within participant.

## Statistics

* Primary Model 1: `decay_z ~ h_z * nfc_z` with a participant random intercept and a random slope for `h_z` (REML).
  Model A (intercept only) can converge to a boundary solution; report Model B.
* Secondary: pooled OLS with CR1 participant-clustered standard errors (t, G-1 df).
* Bonferroni alpha = .0125 for the four planned tests.

## Reproducibility notes

* iTunes preview lookups are rate limited (about 3.5 s per request); a full run over 23 participants takes hours.
  Already-computed tracks are re-used through the lookup in `process_user.get_existing_songs_lookup`.
* The audio is a 30-second preview, not the full track (see Limitations in the paper, and
  `multiscale_sensitivity_pipeline.py` for the window-length check).
* If a participant's ID was mistyped in the questionnaire form, list the fix in `data_raw/id_corrections.json`
  (`{"WRONG_ID": "CORRECT_ID"}`); `score_questionnaires.py` applies it if the file exists. The file stays private (`data_raw/` is git-ignored).
* `analysis_model1_ols.py`, `analysis_model2_events.py`, `analysis_intermediate_variability.py`,
  `analysis_secondary.py` and `make_figures.py` were consolidated from interactive analysis into scripts
  at the end of the project. Re-run them on the final data and compare with the tables in the paper.
