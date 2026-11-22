
import json
import sqlite3
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# AIRSENSE INDIA - DATA PREPROCESSING PIPELINE
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "city_day.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
DB_FILE = ROOT / "database" / "airsense.db"
REPORT_FILE = PROCESSED_DIR / "data_quality_report.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("AirSense")

def load_dataset():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}\n"
            "Download city_day.csv from Kaggle and place it "
            "inside data/raw/"
        )

    logger.info("Loading original dataset...")
    return pd.read_csv(RAW_FILE)

def clean_dataset(df):

    logger.info("Starting data cleaning...")

    df.columns = [
        str(col).strip().replace(" ", "_")
        for col in df.columns
    ]

    required = {"City", "Date"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing required columns: {required - set(df.columns)}"
        )

    original_rows = len(df)

    # Normalize city names
    df["City"] = (
        df["City"]
        .astype("string")
        .str.strip()
    )

    # Convert dates
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Convert measurements to numeric
    numeric_columns = [
        "PM2.5", "PM10", "NO", "NO2", "NOx",
        "NH3", "CO", "SO2", "O3", "Benzene",
        "Toluene", "Xylene", "AQI"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Remove invalid dates and city names
    df = df.dropna(subset=["City", "Date"])

    df = df[df["City"] != ""]

    # Remove duplicate city-date observations
    df = df.drop_duplicates(
        subset=["City", "Date"],
        keep="last"
    )

    # Negative measurements are invalid
    for column in numeric_columns:
        if column in df.columns:
            df.loc[df[column] < 0, column] = np.nan

    # Retain original AQI classification
    if "AQI_Bucket" in df.columns:
        df["AQI_Bucket"] = (
            df["AQI_Bucket"]
            .astype("string")
            .str.strip()
        )

    df = df.sort_values(
        ["City", "Date"]
    ).reset_index(drop=True)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    logger.info(
        "Rows before cleaning: %s", original_rows
    )
    logger.info(
        "Rows after cleaning: %s", len(df)
    )

    return df

def generate_quality_report(df, original_rows):

    report = {
        "project": "AirSense India",
        "dataset": RAW_FILE.name,
        "original_rows": int(original_rows),
        "processed_rows": int(len(df)),
        "removed_rows": int(original_rows - len(df)),
        "unique_cities": int(df["City"].nunique()),
        "date_range": {
            "start": str(df["Date"].min().date()),
            "end": str(df["Date"].max().date())
        },
        "missing_values": {
            key: int(value)
            for key, value in df.isna().sum().items()
        },
        "city_coverage": sorted(
            df["City"].dropna().unique().tolist()
        ),
        "maharashtra_city_check": {}
    }

    # Do not assume a city exists in the source dataset.
    maharashtra_cities = [
        "Mumbai", "Pune", "Nashik", "Nagpur",
        "Chhatrapati Sambhaji Nagar",
        "Aurangabad", "Thane", "Navi Mumbai"
    ]

    available = set(report["city_coverage"])

    for city in maharashtra_cities:
        report["maharashtra_city_check"][city] = (
            city in available
        )

    if "AQI" in df.columns:
        report["aqi_summary"] = {
            "valid_records": int(df["AQI"].notna().sum()),
            "mean": (
                round(float(df["AQI"].mean()), 2)
                if df["AQI"].notna().any() else None
            ),
            "minimum": (
                float(df["AQI"].min())
                if df["AQI"].notna().any() else None
            ),
            "maximum": (
                float(df["AQI"].max())
                if df["AQI"].notna().any() else None
            )
        }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    logger.info("Quality report generated: %s", REPORT_FILE)

    return report

def save_to_csv(df):

    output = PROCESSED_DIR / "city_day_cleaned.csv"

    df.to_csv(output, index=False)

    logger.info("Processed CSV saved: %s", output)

def save_to_database(df):

    logger.info("Saving dataset to SQLite...")

    with sqlite3.connect(DB_FILE) as connection:

        df.to_sql(
            "city_day",
            connection,
            if_exists="replace",
            index=False
        )

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_city_date
            ON city_day (City, Date)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_city_aqi
            ON city_day (City, AQI)
        """)

        connection.commit()

    logger.info("SQLite database updated: %s", DB_FILE)

def verify_database():

    with sqlite3.connect(DB_FILE) as connection:

        total = pd.read_sql_query(
            "SELECT COUNT(*) AS total FROM city_day",
            connection
        )

        cities = pd.read_sql_query("""
            SELECT City, COUNT(*) AS records
            FROM city_day
            GROUP BY City
            ORDER BY records DESC
        """, connection)

    logger.info("Database record count:\n%s", total)
    logger.info("City coverage:\n%s", cities.to_string(index=False))

def main():

    logger.info("========== AIRSENSE INDIA ==========")

    raw_df = load_dataset()
    original_rows = len(raw_df)

    cleaned_df = clean_dataset(raw_df)

    if cleaned_df.empty:
        raise ValueError("No valid records remain after cleaning.")

    generate_quality_report(
        cleaned_df,
        original_rows
    )

    save_to_csv(cleaned_df)
    save_to_database(cleaned_df)
    verify_database()

    logger.info("PHASE 1 DATA PROCESSING COMPLETED!")

if __name__ == "__main__":
    main()