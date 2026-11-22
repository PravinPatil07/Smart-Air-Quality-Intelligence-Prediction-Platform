
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    SECRET_KEY = os.getenv(
        "SECRET_KEY",
        "airsense-local-development-key"
    )

    SQLALCHEMY_DATABASE_URI = (
        f"sqlite:///{BASE_DIR / 'database' / 'airsense.db'}"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    MODEL_DIR = BASE_DIR / "models"