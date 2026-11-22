
import json
import sqlite3
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# AIRSENSE INDIA - EXPLORATORY DATA ANALYSIS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "database" / "airsense.db"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
STAT_DIR = REPORT_DIR / "statistics"

for folder in [FIGURE_DIR, STAT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("AirSense_EDA")

sns.set_theme(style="darkgrid")


def load_data():

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "SQLite database not found. Run Phase 1 first."
        )

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM city_day",
            conn,
            parse_dates=["Date"]
        )

    if df.empty:
        raise ValueError("Database contains no records.")

    logger.info("Loaded %s records", len(df))

    return df


def basic_information(df):

    logger.info("Generating basic dataset information...")

    info = {
        "total_records": int(len(df)),
        "total_columns": int(len(df.columns)),
        "unique_cities": int(df["City"].nunique()),
        "start_date": str(df["Date"].min().date()),
        "end_date": str(df["Date"].max().date()),
        "columns": list(df.columns),
        "data_types": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
        "missing_values": {
            col: int(value)
            for col, value in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum())
    }

    with open(
        STAT_DIR / "basic_information.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(info, file, indent=4)

    print("\n========== DATASET INFORMATION ==========")
    print(df.info())
    print("\nDataset shape:", df.shape)
    print("\nUnique cities:", df["City"].nunique())
    print("\nMissing values:\n", df.isna().sum())
    print("\nDescriptive statistics:\n", df.describe())

    return info


def missing_value_analysis(df):

    missing = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing.columns = ["Column", "Missing_Count"]

    missing["Missing_Percentage"] = (
        missing["Missing_Count"] / len(df) * 100
    )

    missing = missing.sort_values(
        "Missing_Percentage",
        ascending=False
    )

    missing.to_csv(
        STAT_DIR / "missing_value_analysis.csv",
        index=False
    )

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=missing.head(15),
        x="Missing_Percentage",
        y="Column"
    )

    plt.title("Top 15 Columns - Missing Value Percentage")
    plt.xlabel("Missing Values (%)")
    plt.ylabel("Column")

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "missing_values.png",
        dpi=300
    )
    plt.close()


def pollutant_analysis(df):

    pollutants = [
        col for col in [
            "PM2.5", "PM10", "NO", "NO2", "NOx",
            "NH3", "CO", "SO2", "O3",
            "Benzene", "Toluene", "Xylene"
        ]
        if col in df.columns
    ]

    summary = df[pollutants].describe().T

    summary.to_csv(
        STAT_DIR / "pollutant_statistics.csv"
    )

    # Pollutant distributions
    for pollutant in pollutants:

        values = df[pollutant].dropna()

        if values.empty:
            continue

        plt.figure(figsize=(10, 5))

        sns.histplot(
            values,
            bins=50,
            kde=True
        )

        plt.title(f"{pollutant} Distribution")
        plt.xlabel(f"{pollutant} concentration")
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / f"{pollutant}_distribution.png",
            dpi=250
        )

        plt.close()

    # Correlation matrix
    corr_columns = pollutants.copy()

    if "AQI" in df.columns:
        corr_columns.append("AQI")

    correlation = df[corr_columns].corr(
        method="pearson"
    )

    correlation.to_csv(
        STAT_DIR / "pollutant_correlation.csv"
    )

    plt.figure(figsize=(13, 10))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Pollutant Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "correlation_matrix.png",
        dpi=300
    )

    plt.close()


def city_analysis(df):

    if "AQI" not in df.columns:
        logger.warning("AQI column unavailable.")
        return

    city_stats = (
        df.groupby("City")
        .agg(
            total_records=("AQI", "count"),
            average_aqi=("AQI", "mean"),
            maximum_aqi=("AQI", "max"),
            minimum_aqi=("AQI", "min")
        )
        .reset_index()
    )

    city_stats = city_stats[
        city_stats["total_records"] >= 30
    ]

    city_stats = city_stats.sort_values(
        "average_aqi",
        ascending=False
    )

    city_stats.to_csv(
        STAT_DIR / "city_aqi_ranking.csv",
        index=False
    )

    top_cities = city_stats.head(10)

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=top_cities,
        x="average_aqi",
        y="City"
    )

    plt.title("Top 10 Cities by Average AQI")
    plt.xlabel("Average AQI")
    plt.ylabel("City")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "top_10_polluted_cities.png",
        dpi=300
    )

    plt.close()

    # Lowest average AQI cities
    cleanest = city_stats.sort_values(
        "average_aqi"
    ).head(10)

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=cleanest,
        x="average_aqi",
        y="City"
    )

    plt.title("10 Cities with Lowest Average AQI")
    plt.xlabel("Average AQI")
    plt.ylabel("City")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "cleanest_cities.png",
        dpi=300
    )

    plt.close()


def time_analysis(df):

    data = df.dropna(
        subset=["AQI"]
    ).copy()

    data["YearMonth"] = (
        data["Date"].dt.to_period("M").astype(str)
    )

    monthly = (
        data.groupby("YearMonth")["AQI"]
        .mean()
        .reset_index()
    )

    monthly.to_csv(
        STAT_DIR / "monthly_aqi_trend.csv",
        index=False
    )

    plt.figure(figsize=(16, 6))

    sns.lineplot(
        data=monthly,
        x="YearMonth",
        y="AQI"
    )

    plt.title("Monthly Average AQI Trend")
    plt.xlabel("Month")
    plt.ylabel("Average AQI")

    plt.xticks(rotation=60)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "monthly_aqi_trend.png",
        dpi=300
    )

    plt.close()

    # Seasonal analysis
    month_to_season = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Summer",
        4: "Summer",
        5: "Summer",
        6: "Monsoon",
        7: "Monsoon",
        8: "Monsoon",
        9: "Monsoon",
        10: "Post-Monsoon",
        11: "Post-Monsoon"
    }

    data["Season"] = (
        data["Date"].dt.month.map(month_to_season)
    )

    seasonal = (
        data.groupby("Season")["AQI"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    seasonal.to_csv(
        STAT_DIR / "seasonal_aqi_analysis.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=seasonal,
        x="Season",
        y="mean",
        order=[
            "Winter",
            "Summer",
            "Monsoon",
            "Post-Monsoon"
        ]
    )

    plt.title("Average AQI by Season")
    plt.xlabel("Season")
    plt.ylabel("Average AQI")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "seasonal_aqi.png",
        dpi=300
    )

    plt.close()


def maharashtra_analysis(df):

    # City aliases must be reviewed against actual source names.
    aliases = {
        "Mumbai": ["Mumbai"],
        "Pune": ["Pune"],
        "Nashik": ["Nashik"],
        "Nagpur": ["Nagpur"],
        "Thane": ["Thane"],
        "Navi Mumbai": ["Navi Mumbai"],
        "Chhatrapati Sambhaji Nagar": [
            "Chhatrapati Sambhaji Nagar",
            "Aurangabad"
        ]
    }

    available = set(
        df["City"].dropna().unique()
    )

    rows = []

    for canonical, variants in aliases.items():

        matched = [
            name for name in variants
            if name in available
        ]

        for city in matched:

            subset = df[
                df["City"] == city
            ]

            rows.append({
                "City": canonical,
                "Source_City": city,
                "Records": int(
                    subset["AQI"].count()
                ),
                "Average_AQI": (
                    round(float(subset["AQI"].mean()), 2)
                    if subset["AQI"].notna().any()
                    else None
                ),
                "Maximum_AQI": (
                    float(subset["AQI"].max())
                    if subset["AQI"].notna().any()
                    else None
                )
            })

    result = pd.DataFrame(rows)

    result.to_csv(
        STAT_DIR / "maharashtra_city_analysis.csv",
        index=False
    )

    logger.info(
        "Maharashtra cities found: %s",
        result["City"].nunique() if not result.empty else 0
    )


def main():

    logger.info("Starting AirSense EDA...")

    df = load_data()

    basic_information(df)
    missing_value_analysis(df)
    pollutant_analysis(df)
    city_analysis(df)
    time_analysis(df)
    maharashtra_analysis(df)

    logger.info("EDA COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
