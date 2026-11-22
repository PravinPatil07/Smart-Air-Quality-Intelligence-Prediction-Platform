
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "database" / "airsense.db"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "statistics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_NAME = "city_day"

TARGET = "AQI"

POLLUTANTS = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene"
]

LAG_VALUES = [1, 2, 3, 7, 14]
ROLLING_WINDOWS = [3, 7, 14]


# =====================================================
# LOAD DATA
# =====================================================

def load_data():

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    connection = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        f"SELECT * FROM {TABLE_NAME}",
        connection
    )

    connection.close()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    ).dt.normalize()

    df["City"] = df["City"].astype(str).str.strip()

    numeric_columns = [
        col for col in POLLUTANTS + [TARGET]
        if col in df.columns
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        df.loc[df[col] < 0, col] = np.nan

    df = df.dropna(subset=["City", "Date"])

    df = df.sort_values(
        ["City", "Date"]
    )

    if df.duplicated(["City", "Date"]).any():
        raise ValueError(
            "Duplicate City-Date records found. "
            "Resolve duplicates before feature generation."
        )

    print("Loaded records:", len(df))
    print("Cities:", df["City"].nunique())

    return df


# =====================================================
# CALENDAR FEATURES
# =====================================================

def add_calendar_features(df):

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfYear"] = df["Date"].dt.dayofyear

    # Cyclical encoding
    df["Month_Sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )

    df["Month_Cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )

    df["DayOfWeek_Sin"] = np.sin(
        2 * np.pi * df["DayOfWeek"] / 7
    )

    df["DayOfWeek_Cos"] = np.cos(
        2 * np.pi * df["DayOfWeek"] / 7
    )

    def get_season(month):

        if month in [12, 1, 2]:
            return "Winter"

        elif month in [3, 4, 5]:
            return "Summer"

        elif month in [6, 7, 8, 9]:
            return "Monsoon"

        return "Post-Monsoon"

    df["Season"] = df["Month"].apply(get_season)

    return df


# =====================================================
# DAILY FEATURES FOR EACH CITY
# =====================================================

def create_city_features(city_df):

    city_df = city_df.sort_values("Date").copy()

    city_name = city_df["City"].iloc[0]

    city_df = city_df.set_index("Date")

    # Create a complete daily calendar.
    # Missing dates remain missing; no measurements
    # are fabricated.
    daily_index = pd.date_range(
        start=city_df.index.min(),
        end=city_df.index.max(),
        freq="D"
    )

    city_df = city_df.reindex(daily_index)

    city_df.index.name = "Date"

    city_df["City"] = city_name

    city_df = city_df.reset_index()

    city_df = add_calendar_features(city_df)

    source_columns = [
        col for col in POLLUTANTS + [TARGET]
        if col in city_df.columns
    ]

    # Historical lag features
    for col in source_columns:

        for lag in LAG_VALUES:

            city_df[f"{col}_lag_{lag}"] = (
                city_df[col].shift(lag)
            )

    # Rolling statistics, including today's
    # observation, which is available at forecast time.
    for col in source_columns:

        for window in ROLLING_WINDOWS:

            city_df[f"{col}_rolling_mean_{window}"] = (
                city_df[col]
                .rolling(window=window, min_periods=window)
                .mean()
            )

            city_df[f"{col}_rolling_std_{window}"] = (
                city_df[col]
                .rolling(window=window, min_periods=window)
                .std()
            )

    # Change from previous calendar day
    for col in [TARGET, "PM2.5", "PM10"]:

        if col in city_df.columns:

            city_df[f"{col}_change_1d"] = (
                city_df[col] - city_df[col].shift(1)
            )

    # Tomorrow's AQI is the target.
    city_df["target_date"] = (
        city_df["Date"] + pd.Timedelta(days=1)
    )

    city_df["target_aqi_next_day"] = (
        city_df[TARGET].shift(-1)
    )

    return city_df


# =====================================================
# BUILD COMPLETE DATASET
# =====================================================

def build_features(df):

    city_frames = []

    for city, city_df in df.groupby("City"):

        print("Processing:", city)

        features = create_city_features(city_df)

        city_frames.append(features)

    result = pd.concat(
        city_frames,
        ignore_index=True
    )

    # Only rows with a known next-day AQI
    # can be used as supervised training examples.
    result = result.dropna(
        subset=["target_aqi_next_day"]
    )

    result = result.sort_values(
        ["target_date", "City"]
    ).reset_index(drop=True)

    return result


# =====================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# =====================================================

def chronological_split(df):

    unique_dates = np.sort(
        df["target_date"].dropna().unique()
    )

    if len(unique_dates) < 10:
        raise ValueError(
            "Not enough distinct target dates for a "
            "chronological split."
        )

    split_position = int(len(unique_dates) * 0.80)

    split_position = min(
        max(split_position, 1),
        len(unique_dates) - 1
    )

    split_date = unique_dates[split_position]

    train_df = df[
        df["target_date"] < split_date
    ].copy()

    test_df = df[
        df["target_date"] >= split_date
    ].copy()

    return train_df, test_df, pd.Timestamp(split_date)


# =====================================================
# SAVE DATA AND REPORT
# =====================================================

def main():

    print("\nAirSense India - Phase 3")
    print("=" * 50)

    df = load_data()

    feature_df = build_features(df)

    train_df, test_df, split_date = (
        chronological_split(feature_df)
    )

    # Save complete feature dataset
    feature_path = (
        OUTPUT_DIR / "airsense_features.csv"
    )

    feature_df.to_csv(
        feature_path,
        index=False
    )

    # Save chronological partitions
    train_df.to_csv(
        OUTPUT_DIR / "train_features.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test_features.csv",
        index=False
    )

    # Feature columns for model training.
    # Exclude target and future-date information.
    excluded = {
        "target_aqi_next_day",
        "target_date",
        "AQI_Bucket"
    }

    feature_columns = [
        col for col in feature_df.columns
        if col not in excluded
        and col != "Date"
    ]

    report = {

        "project": "AirSense India",

        "task": "Next-day city-level AQI prediction",

        "target": "target_aqi_next_day",

        "total_feature_rows": int(len(feature_df)),

        "number_of_cities": int(
            feature_df["City"].nunique()
        ),

        "date_start": str(
            feature_df["Date"].min().date()
        ),

        "date_end": str(
            feature_df["Date"].max().date()
        ),

        "train_rows": int(len(train_df)),

        "test_rows": int(len(test_df)),

        "split_target_date": str(
            split_date.date()
        ),

        "feature_count": len(feature_columns),

        "feature_columns": feature_columns,

        "target_missing": int(
            feature_df["target_aqi_next_day"].isna().sum()
        ),

        "training_strategy": (
            "Chronological 80/20 split by target date"
        ),

        "missing_value_strategy": (
            "Preserved missing feature values. "
            "Imputation will be fitted only on training data."
        )
    }

    report_path = (
        REPORT_DIR / "phase3_feature_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\nPHASE 3 COMPLETED")
    print("=" * 50)

    print("Feature rows:", len(feature_df))
    print("Cities:", feature_df["City"].nunique())
    print("Features:", len(feature_columns))

    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))

    print("Split date:", split_date.date())

    print("\nSaved files:")

    print(feature_path)
    print(OUTPUT_DIR / "train_features.csv")
    print(OUTPUT_DIR / "test_features.csv")
    print(report_path)


if __name__ == "__main__":
    main()
