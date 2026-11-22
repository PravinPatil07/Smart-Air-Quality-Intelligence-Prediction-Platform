
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "database" / "airsense.db"
OUTPUT = ROOT / "reports" / "statistics"

OUTPUT.mkdir(parents=True, exist_ok=True)


def load_data():

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM city_day",
            conn,
            parse_dates=["Date"]
        )


def distribution_analysis(df):

    results = {}

    pollutants = [
        "PM2.5", "PM10", "NO2",
        "SO2", "CO", "O3", "AQI"
    ]

    for col in pollutants:

        if col not in df.columns:
            continue

        values = df[col].dropna()

        if len(values) == 0:
            continue

        results[col] = {
            "count": int(values.count()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "skewness": float(values.skew()),
            "kurtosis": float(values.kurtosis()),
            "p25": float(values.quantile(0.25)),
            "p75": float(values.quantile(0.75)),
            "p90": float(values.quantile(0.90)),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99))
        }

    return results


def outlier_analysis(df):

    results = []

    pollutants = [
        "PM2.5", "PM10", "NO2",
        "SO2", "CO", "O3", "AQI"
    ]

    for col in pollutants:

        if col not in df.columns:
            continue

        values = df[col].dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            ((values < lower) | (values > upper)).sum()
        )

        results.append({
            "Pollutant": col,
            "Q1": float(q1),
            "Q3": float(q3),
            "IQR": float(iqr),
            "Lower_Bound": float(lower),
            "Upper_Bound": float(upper),
            "Outlier_Count": count,
            "Outlier_Percentage": float(
                count / len(values) * 100
            )
        })

    return results


def correlation_analysis(df):

    pollutants = [
        col for col in [
            "PM2.5", "PM10", "NO2",
            "SO2", "CO", "O3", "AQI"
        ]
        if col in df.columns
    ]

    results = {}

    for col in pollutants:

        if col == "AQI":
            continue

        pair = df[[col, "AQI"]].dropna()

        if len(pair) < 3:
            continue

        pearson_r, pearson_p = stats.pearsonr(
            pair[col], pair["AQI"]
        )

        spearman_r, spearman_p = stats.spearmanr(
            pair[col], pair["AQI"]
        )

        results[col] = {
            "observations": int(len(pair)),
            "pearson_correlation": float(pearson_r),
            "pearson_p_value": float(pearson_p),
            "spearman_correlation": float(spearman_r),
            "spearman_p_value": float(spearman_p)
        }

    return results


def seasonal_hypothesis_test(df):

    data = df.dropna(
        subset=["AQI", "Date"]
    ).copy()

    data["Month"] = data["Date"].dt.month

    seasons = {
        "Winter": [12, 1, 2],
        "Summer": [3, 4, 5],
        "Monsoon": [6, 7, 8, 9],
        "Post-Monsoon": [10, 11]
    }

    groups = {}

    for season, months in seasons.items():

        values = data.loc[
            data["Month"].isin(months),
            "AQI"
        ].dropna()

        if len(values) > 0:
            groups[season] = values

    if len(groups) < 2:
        return {
            "test": "Kruskal-Wallis",
            "status": "Insufficient seasonal groups"
        }

    statistic, p_value = stats.kruskal(
        *groups.values()
    )

    return {
        "test": "Kruskal-Wallis",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "groups": {
            name: int(len(values))
            for name, values in groups.items()
        },
        "significance_level": 0.05,
        "interpretation": (
            "Evidence of a seasonal distribution difference"
            if p_value < 0.05
            else "No statistically significant seasonal difference detected"
        ),
        "note": (
            "This is a preliminary unadjusted test. "
            "Repeated observations and city-level differences "
            "should be considered before causal conclusions."
        )
    }


def city_comparison_test(df):

    data = df.dropna(
        subset=["AQI"]
    )

    city_counts = data["City"].value_counts()

    eligible = city_counts[
        city_counts >= 30
    ].index[:2]

    if len(eligible) < 2:
        return {
            "status": "Not enough cities with 30 valid AQI records"
        }

    first = data.loc[
        data["City"] == eligible[0], "AQI"
    ]

    second = data.loc[
        data["City"] == eligible[1], "AQI"
    ]

    statistic, p_value = stats.mannwhitneyu(
        first,
        second,
        alternative="two-sided"
    )

    return {
        "test": "Mann-Whitney U",
        "city_1": str(eligible[0]),
        "city_2": str(eligible[1]),
        "u_statistic": float(statistic),
        "p_value": float(p_value),
        "interpretation": (
            "Evidence of a difference in distributions"
            if p_value < 0.05
            else "No statistically significant difference detected"
        )
    }


def main():

    df = load_data()

    report = {
        "distribution": distribution_analysis(df),
        "outliers": outlier_analysis(df),
        "correlations": correlation_analysis(df),
        "seasonal_test": seasonal_hypothesis_test(df),
        "city_test": city_comparison_test(df)
    }

    with open(
        OUTPUT / "statistical_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\nSTATISTICAL ANALYSIS COMPLETED")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()
