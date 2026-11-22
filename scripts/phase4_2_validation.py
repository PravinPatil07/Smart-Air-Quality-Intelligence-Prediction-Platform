
# ============================================================
# AIRSENSE INDIA - PHASE 4.2
# MODEL VALIDATION, EXPLAINABILITY & RELIABILITY
#
# Hardware: Intel Core i3 | 8 GB RAM | CPU only
# Champion model: Ridge Regression
# ============================================================

import os

# CPU and memory settings - set before NumPy / sklearn imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

REPORT_DIR = ROOT_DIR / "reports" / "statistics"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"

TRAIN_FILE = DATA_DIR / "train_features.csv"
TEST_FILE = DATA_DIR / "test_features.csv"

MODEL_FILE = MODEL_DIR / "best_aqi_model_optimized.joblib"

TARGET = "target_aqi_next_day"
DATE_COLUMN = "target_date"

VALIDATION_FRACTION = 0.15
CV_FOLDS = 3

RANDOM_STATE = 42

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(
        f"[{time.strftime('%H:%M:%S')}] {message}",
        flush=True
    )


def save_json(data, filepath):

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),
        "RMSE": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "R2": float(
            r2_score(actual, predicted)
        )
    }


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

def load_resources():

    log("Loading datasets and trained model...")

    required_files = [
        TRAIN_FILE,
        TEST_FILE,
        MODEL_FILE
    ]

    for filepath in required_files:

        if not filepath.exists():
            raise FileNotFoundError(
                f"Required file not found: {filepath}"
            )

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    train[TARGET] = pd.to_numeric(
        train[TARGET], errors="coerce"
    )

    test[TARGET] = pd.to_numeric(
        test[TARGET], errors="coerce"
    )

    train = train.dropna(
        subset=[TARGET]
    ).reset_index(drop=True)

    test = test.dropna(
        subset=[TARGET]
    ).reset_index(drop=True)

    for df in [train, test]:

        if DATE_COLUMN not in df.columns:
            raise ValueError(
                f"Missing required column: {DATE_COLUMN}"
            )

        df[DATE_COLUMN] = pd.to_datetime(
            df[DATE_COLUMN],
            errors="coerce"
        )

    train = train.dropna(
        subset=[DATE_COLUMN]
    ).reset_index(drop=True)

    test = test.dropna(
        subset=[DATE_COLUMN]
    ).reset_index(drop=True)

    model = joblib.load(MODEL_FILE)

    log(f"Training records: {len(train):,}")
    log(f"Testing records: {len(test):,}")

    return train, test, model


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(train, test, model):

    log("Preparing features...")

    # Use the exact original feature columns saved in the model.
    # This avoids differences between Phase 4.1 and Phase 4.2.

    preprocessor = model.named_steps["preprocessor"]

    if not hasattr(preprocessor, "feature_names_in_"):
        raise ValueError(
            "The saved model does not contain feature_names_in_. "
            "Please check the Phase 4.1 model."
        )

    feature_columns = list(
        preprocessor.feature_names_in_
    )

    missing_train = [
        col for col in feature_columns
        if col not in train.columns
    ]

    missing_test = [
        col for col in feature_columns
        if col not in test.columns
    ]

    if missing_train or missing_test:

        raise ValueError(
            "Model feature mismatch.\n"
            f"Missing from training: {missing_train}\n"
            f"Missing from testing: {missing_test}"
        )

    X_train = train[feature_columns].copy()
    X_test = test[feature_columns].copy()

    log(f"Features loaded: {len(feature_columns)}")

    return X_train, X_test, feature_columns


# ============================================================
# CHRONOLOGICAL HOLDOUT VALIDATION
# ============================================================

def create_validation_split(train):

    log("Creating chronological validation split...")

    unique_dates = np.sort(
        train[DATE_COLUMN].unique()
    )

    if len(unique_dates) < 10:
        raise ValueError(
            "Not enough unique dates for validation."
        )

    split_index = int(
        len(unique_dates) *
        (1 - VALIDATION_FRACTION)
    )

    split_index = max(
        1, min(split_index, len(unique_dates) - 1)
    )

    validation_start = unique_dates[split_index]

    fit_mask = (
        train[DATE_COLUMN] < validation_start
    )

    validation_mask = (
        train[DATE_COLUMN] >= validation_start
    )

    fit_indices = train.index[fit_mask]
    validation_indices = train.index[validation_mask]

    log(f"Validation begins: {validation_start}")
    log(f"Fit records: {len(fit_indices):,}")
    log(f"Validation records: {len(validation_indices):,}")

    return fit_indices, validation_indices


# ============================================================
# TIME-SERIES CROSS-VALIDATION
# ============================================================

def run_time_series_validation(
    train,
    X_train,
    original_model
):

    log("Starting expanding-window time-series validation...")

    dates = np.sort(
        train[DATE_COLUMN].unique()
    )

    if len(dates) < 12:
        raise ValueError(
            "Insufficient dates for three time-series folds."
        )

    # Start after the initial training history.
    initial_position = int(len(dates) * 0.50)

    remaining_dates = dates[initial_position:]

    date_chunks = np.array_split(
        remaining_dates, CV_FOLDS
    )

    results = []

    for fold_number, validation_dates in enumerate(
        date_chunks, start=1
    ):

        if len(validation_dates) == 0:
            continue

        validation_start = validation_dates[0]
        validation_end = validation_dates[-1]

        fit_mask = (
            train[DATE_COLUMN] < validation_start
        )

        validation_mask = (
            train[DATE_COLUMN].isin(validation_dates)
        )

        fit_indices = train.index[fit_mask]
        validation_indices = train.index[validation_mask]

        if len(fit_indices) == 0 or len(validation_indices) == 0:
            continue

        X_fit = X_train.loc[fit_indices]
        X_valid = X_train.loc[validation_indices]

        y_fit = train.loc[fit_indices, TARGET]
        y_valid = train.loc[validation_indices, TARGET]

        log(
            f"Fold {fold_number}: "
            f"Training {len(fit_indices):,} | "
            f"Validation {len(validation_indices):,}"
        )

        fold_model = clone(original_model)

        start = time.time()

        fold_model.fit(X_fit, y_fit)

        predictions = fold_model.predict(X_valid)

        elapsed = time.time() - start

        metrics = calculate_metrics(
            y_valid,
            predictions
        )

        results.append({
            "Fold": fold_number,
            "Training_Records": len(fit_indices),
            "Validation_Records": len(validation_indices),
            "Validation_Start": str(validation_start),
            "Validation_End": str(validation_end),
            **metrics,
            "Training_Time_Seconds": round(elapsed, 2)
        })

        log(
            f"Fold {fold_number} completed | "
            f"RMSE: {metrics['RMSE']:.3f} | "
            f"R2: {metrics['R2']:.4f}"
        )

        del fold_model
        gc.collect()

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "phase4_2_time_series_cv.csv",
        index=False
    )

    return results_df


# ============================================================
# VALIDATION PREDICTIONS AND RESIDUALS
# ============================================================

def run_residual_analysis(
    train,
    X_train,
    original_model
):

    log("Running residual analysis...")

    fit_indices, validation_indices = (
        create_validation_split(train)
    )

    validation_model = clone(original_model)

    X_fit = X_train.loc[fit_indices]
    X_valid = X_train.loc[validation_indices]

    y_fit = train.loc[fit_indices, TARGET]
    y_valid = train.loc[validation_indices, TARGET]

    validation_model.fit(
        X_fit,
        y_fit
    )

    predicted = validation_model.predict(X_valid)

    result = train.loc[
        validation_indices
    ].copy()

    result["Actual_AQI"] = y_valid.values
    result["Predicted_AQI"] = predicted

    result["Residual"] = (
        result["Actual_AQI"]
        - result["Predicted_AQI"]
    )

    result["Absolute_Error"] = np.abs(
        result["Residual"]
    )

    metrics = calculate_metrics(
        result["Actual_AQI"],
        result["Predicted_AQI"]
    )

    result.to_csv(
        REPORT_DIR / "phase4_2_validation_residuals.csv",
        index=False
    )

    log(
        f"Holdout MAE: {metrics['MAE']:.3f} | "
        f"RMSE: {metrics['RMSE']:.3f} | "
        f"R2: {metrics['R2']:.4f}"
    )

    return (
        result,
        validation_model,
        metrics
    )


# ============================================================
# RESIDUAL PLOTS
# ============================================================

def create_residual_plots(residuals):

    log("Creating residual analysis charts...")

    # 1. Actual vs Predicted
    plt.figure(figsize=(9, 6))

    plt.scatter(
        residuals["Actual_AQI"],
        residuals["Predicted_AQI"],
        alpha=0.35
    )

    minimum = min(
        residuals["Actual_AQI"].min(),
        residuals["Predicted_AQI"].min()
    )

    maximum = max(
        residuals["Actual_AQI"].max(),
        residuals["Predicted_AQI"].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title("Actual vs Predicted AQI")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "phase4_2_actual_vs_predicted.png",
        dpi=200
    )

    plt.close()

    # 2. Residual distribution
    plt.figure(figsize=(9, 6))

    plt.hist(
        residuals["Residual"],
        bins=40,
        edgecolor="black"
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Prediction Residual Distribution")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "phase4_2_residual_distribution.png",
        dpi=200
    )

    plt.close()

    # 3. Residual vs predicted
    plt.figure(figsize=(9, 6))

    plt.scatter(
        residuals["Predicted_AQI"],
        residuals["Residual"],
        alpha=0.35
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Predicted AQI")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted AQI")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "phase4_2_residual_vs_predicted.png",
        dpi=200
    )

    plt.close()

    # 4. Error over time
    daily_error = (
        residuals
        .groupby("target_date")["Absolute_Error"]
        .mean()
        .sort_index()
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        daily_error.index,
        daily_error.values
    )

    plt.xlabel("Date")
    plt.ylabel("Mean Absolute Error")
    plt.title("Prediction Error Over Time")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "phase4_2_error_over_time.png",
        dpi=200
    )

    plt.close()


# ============================================================
# CITY-WISE VALIDATION
# ============================================================

def city_wise_analysis(residuals):

    log("Generating city-wise validation report...")

    if "City" not in residuals.columns:
        log("City column missing. Skipping city analysis.")
        return pd.DataFrame()

    results = []

    for city, group in residuals.groupby("City"):

        if len(group) < 10:
            continue

        metrics = calculate_metrics(
            group["Actual_AQI"],
            group["Predicted_AQI"]
        )

        results.append({
            "City": city,
            "Records": len(group),
            **metrics,
            "Mean_Bias": float(
                group["Residual"].mean()
            )
        })

    city_df = pd.DataFrame(results)

    if not city_df.empty:

        city_df = city_df.sort_values(
            "MAE"
        ).reset_index(drop=True)

    city_df.to_csv(
        REPORT_DIR / "phase4_2_city_validation.csv",
        index=False
    )

    return city_df


# ============================================================
# PERMUTATION FEATURE IMPORTANCE
# ============================================================

def explain_features(
    validation_model,
    X_train,
    train,
    feature_columns
):

    log("Calculating permutation feature importance...")

    fit_indices, validation_indices = (
        create_validation_split(train)
    )

    X_valid = X_train.loc[validation_indices]
    y_valid = train.loc[validation_indices, TARGET]

    # Permutation importance works on original input columns,
    # before the pipeline transforms them.
    result = permutation_importance(
        validation_model,
        X_valid,
        y_valid,
        scoring="neg_root_mean_squared_error",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std
    })

    importance_df = importance_df.sort_values(
        "Importance_Mean",
        ascending=False
    ).reset_index(drop=True)

    importance_df.to_csv(
        REPORT_DIR / "phase4_2_permutation_importance.csv",
        index=False
    )

    top_features = importance_df.head(20).sort_values(
        "Importance_Mean"
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["Feature"],
        top_features["Importance_Mean"],
        xerr=top_features["Importance_Std"]
    )

    plt.xlabel("Increase in RMSE after permutation")
    plt.title("Top 20 Permutation Feature Importances")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "phase4_2_feature_importance.png",
        dpi=200
    )

    plt.close()

    return importance_df


# ============================================================
# PREDICTION UNCERTAINTY
# ============================================================

def prediction_uncertainty(
    residuals,
    test,
    X_test,
    final_model
):

    log("Calculating approximate prediction intervals...")

    # Use the absolute errors from the chronological
    # validation period to calibrate the interval.

    calibration_errors = residuals[
        "Absolute_Error"
    ].dropna().values

    if len(calibration_errors) == 0:
        raise ValueError(
            "No calibration residuals available."
        )

    # Finite-sample corrected 90% absolute-error quantile.
    alpha = 0.10

    n = len(calibration_errors)

    quantile_level = min(
        1.0,
        np.ceil((n + 1) * (1 - alpha)) / n
    )

    error_margin = float(
        np.quantile(
            calibration_errors,
            quantile_level,
            method="higher"
        )
    )

    predictions = final_model.predict(X_test)

    lower = predictions - error_margin
    upper = predictions + error_margin

    # AQI is non-negative; this clipping is for display.
    lower_display = np.maximum(lower, 0)

    interval_result = test[
        ["City", DATE_COLUMN]
    ].copy() if "City" in test.columns else test[
        [DATE_COLUMN]
    ].copy()

    interval_result["Actual_AQI"] = test[TARGET].values
    interval_result["Predicted_AQI"] = predictions
    interval_result["Lower_90"] = lower_display
    interval_result["Upper_90"] = upper
    interval_result["Interval_Width"] = upper - lower_display

    interval_result["Covered"] = (
        (interval_result["Actual_AQI"] >= lower_display)
        &
        (interval_result["Actual_AQI"] <= upper)
    )

    coverage = float(
        interval_result["Covered"].mean()
    )

    interval_result.to_csv(
        REPORT_DIR / "phase4_2_prediction_intervals.csv",
        index=False
    )

    log(f"Calibrated error margin: +/- {error_margin:.2f}")
    log(f"Test interval coverage: {coverage * 100:.2f}%")

    return {
        "nominal_coverage": 0.90,
        "calibration_records": int(n),
        "absolute_error_margin": error_margin,
        "test_empirical_coverage": coverage,
        "test_records": len(test),
        "interpretation": (
            "Approximate historical error-based interval. "
            "Temporal distribution changes may affect coverage; "
            "this is not a guaranteed forecast range."
        )
    }


# ============================================================
# FINAL TEST AUDIT
# ============================================================

def test_set_audit(test, final_model, X_test):

    log("Evaluating final saved model on test data...")

    predicted = final_model.predict(X_test)

    metrics = calculate_metrics(
        test[TARGET],
        predicted
    )

    log(
        f"Final test MAE: {metrics['MAE']:.4f} | "
        f"RMSE: {metrics['RMSE']:.4f} | "
        f"R2: {metrics['R2']:.4f}"
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 65)
    print(" AIRSENSE INDIA - PHASE 4.2")
    print(" MODEL VALIDATION & EXPLAINABILITY")
    print("=" * 65)

    train, test, final_model = load_resources()

    X_train, X_test, feature_columns = prepare_features(
        train,
        test,
        final_model
    )

    # 1. Expanding-window validation
    cv_results = run_time_series_validation(
        train,
        X_train,
        final_model
    )

    # 2. Out-of-sample residual analysis
    residuals, validation_model, validation_metrics = (
        run_residual_analysis(
            train,
            X_train,
            final_model
        )
    )

    # 3. Residual charts
    create_residual_plots(residuals)

    # 4. City-level diagnostics
    city_results = city_wise_analysis(residuals)

    # 5. Explainability
    importance_df = explain_features(
        validation_model,
        X_train,
        train,
        feature_columns
    )

    # 6. Prediction uncertainty
    uncertainty = prediction_uncertainty(
        residuals,
        test,
        X_test,
        final_model
    )

    # 7. Final test audit
    test_metrics = test_set_audit(
        test,
        final_model,
        X_test
    )

    # 8. Summary report
    report = {
        "project": "AirSense India",
        "phase": "Phase 4.2 - Model Validation and Explainability",
        "model": "Ridge Regression",
        "training_records": len(train),
        "testing_records": len(test),
        "feature_count": len(feature_columns),
        "city_count": (
            int(train["City"].nunique())
            if "City" in train.columns else None
        ),
        "chronological_validation_metrics": validation_metrics,
        "time_series_cv": (
            cv_results.to_dict(orient="records")
        ),
        "test_metrics": test_metrics,
        "prediction_uncertainty": uncertainty,
        "top_10_features": (
            importance_df.head(10).to_dict(orient="records")
        ),
        "methodology_notes": [
            "Validation uses chronological holdout data.",
            "Time-series folds use expanding training windows.",
            "Permutation importance uses validation data.",
            "The final saved model is not retrained or overwritten.",
            "Test metrics are an audit, not a model-selection criterion.",
            "Prediction intervals are approximate and may not retain "
            "nominal coverage under temporal distribution changes."
        ]
    }

    save_json(
        report,
        REPORT_DIR / "phase4_2_validation_report.json"
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print(" PHASE 4.2 COMPLETED")
    print("=" * 65)

    print(f"\nValidation MAE: {validation_metrics['MAE']:.3f}")
    print(f"Validation RMSE: {validation_metrics['RMSE']:.3f}")
    print(f"Validation R2: {validation_metrics['R2']:.4f}")

    print(f"\nTest MAE: {test_metrics['MAE']:.3f}")
    print(f"Test RMSE: {test_metrics['RMSE']:.3f}")
    print(f"Test R2: {test_metrics['R2']:.4f}")

    print(
        "\nApproximate 90% interval margin: "
        f"+/- {uncertainty['absolute_error_margin']:.2f} AQI"
    )

    print(
        "\nEmpirical test interval coverage: "
        f"{uncertainty['test_empirical_coverage'] * 100:.2f}%"
    )

    print(
        f"\nTotal execution time: {elapsed / 60:.2f} minutes"
    )

    print(f"\nReports: {REPORT_DIR}")
    print(f"Charts: {FIGURE_DIR}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
