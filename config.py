"""
config.py
---------
Central place for every path used by the project, so no script contains a
personal/absolute path. Override any of them with an environment variable,
or just keep the defaults and place the files inside this repository:

    data_raw/<USER_ID>/*.json     one folder per participant (Spotify exports)
    data_raw/questionnaires.xlsx  the raw questionnaire responses (Google Form export)
    output/                       every generated CSV / figure is written here

Environment variables:
    SEMINAR_DATA_ROOT        folder that holds one sub-folder per participant
    SEMINAR_OUTPUT_DIR       folder for all generated files
    SEMINAR_QUESTIONNAIRE    path of the questionnaire .xlsx
"""

import os

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.environ.get("SEMINAR_DATA_ROOT", os.path.join(REPO_ROOT, "data_raw"))
OUTPUT_DIR = os.environ.get("SEMINAR_OUTPUT_DIR", os.path.join(REPO_ROOT, "output"))
QUESTIONNAIRE_XLSX = os.environ.get(
    "SEMINAR_QUESTIONNAIRE", os.path.join(DATA_ROOT, "questionnaires.xlsx")
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
