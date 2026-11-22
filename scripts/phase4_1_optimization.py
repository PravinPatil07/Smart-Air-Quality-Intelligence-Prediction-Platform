
# ============================================================
# AIRSENSE INDIA - PHASE 4.1
# CPU-OPTIMIZED MODEL TRAINING AND EVALUATION
# Hardware: Intel Core i3 | 8 GB RAM | CPU only
# ============================================================

import os

# Set thread limits BEFORE importing NumPy / sklearn
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

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.base import clone

warnings.filterwarnings("ignore")


# ============================================================
# OPTIONAL XGBOOST
# ============================================================

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports" / "statistics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_features.csv"
TEST_FILE = DATA_DIR / "test_features.csv"

TARGET = "target_aqi_next_day"

RANDOM_STATE = 42

# Use chronological validation
VALIDATION_FRACTION = 0.15

# Keep CPU usage low
CPU_JOBS = 1


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def log(message):
    print(f"\n[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted))
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    log("Loading training and testing datasets...")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TRAIN_FILE}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TEST_FILE}")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    if TARGET not in train.columns or TARGET not in test.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found. "
            "Check your Phase 3 feature files."
        )

    # Remove invalid target rows
    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
    test[TARGET] = pd.to_numeric(test[TARGET], errors="coerce")

    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    test = test.dropna(subset=[TARGET]).reset_index(drop=True)

    # Parse dates
    for df in (train, test):
        for col in ["target_date", "Date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(
                    df[col], errors="coerce"
                )

    log(f"Training records: {len(train):,}")
    log(f"Testing records: {len(test):,}")

    if "City" in train.columns:
        log(f"Cities: {train['City'].nunique()}")

    return train, test


# ============================================================
# FEATURE SELECTION
# ============================================================

def prepare_features(train, test):

    log("Preparing model features...")

    excluded_columns = {
        TARGET,
        "target_date",
        "Date",
        "AQI_Bucket",
        "target"
    }

    # Use only columns present in both datasets
    feature_columns = [
        col for col in train.columns
        if col in test.columns
        and col not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError("No usable feature columns found.")

    X_train = train[feature_columns].copy()
    X_test = test[feature_columns].copy()

    # Remove columns that are completely empty in training.
    # This avoids imputer failures and unnecessary features.
    empty_columns = [
        col for col in X_train.columns
        if X_train[col].isna().all()
    ]

    if empty_columns:
        log(f"Removing {len(empty_columns)} all-empty features.")
        X_train.drop(columns=empty_columns, inplace=True)
        X_test.drop(columns=empty_columns, inplace=True)

    # Detect categorical columns
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_columns = [
        col for col in X_train.columns
        if col not in categorical_columns
    ]

    # Convert numeric features safely
    for col in numeric_columns:
        X_train[col] = pd.to_numeric(
            X_train[col], errors="coerce"
        )
        X_test[col] = pd.to_numeric(
            X_test[col], errors="coerce"
        )

    log(f"Total features: {len(X_train.columns)}")
    log(f"Numeric features: {len(numeric_columns)}")
    log(f"Categorical features: {len(categorical_columns)}")

    return (
        X_train,
        X_test,
        numeric_columns,
        categorical_columns
    )


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor(
    numeric_columns,
    categorical_columns,
    scale_numeric=False
):

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ])

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# CHRONOLOGICAL VALIDATION SPLIT
# ============================================================

def chronological_split(train):

    log("Creating chronological validation split...")

    if "target_date" not in train.columns:
        raise ValueError(
            "target_date is missing. "
            "Chronological validation cannot be performed."
        )

    dates = pd.to_datetime(
        train["target_date"], errors="coerce"
    )

    valid_mask = dates.notna()

    train = train.loc[valid_mask].copy()
    dates = dates.loc[valid_mask]

    unique_dates = np.sort(dates.unique())

    if len(unique_dates) < 5:
        raise ValueError(
            "Not enough unique dates for chronological validation."
        )

    split_position = int(
        len(unique_dates) * (1 - VALIDATION_FRACTION)
    )

    split_position = max(
        1, min(split_position, len(unique_dates) - 1)
    )

    split_date = unique_dates[split_position]

    fit_mask = dates < split_date
    validation_mask = dates >= split_date

    fit_indices = train.index[fit_mask]
    validation_indices = train.index[validation_mask]

    log(f"Validation start date: {split_date}")
    log(f"Fit records: {len(fit_indices):,}")
    log(f"Validation records: {len(validation_indices):,}")

    return fit_indices, validation_indices


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_models(
    numeric_columns,
    categorical_columns
):

    models = {}

    # 1. Ridge Regression
    ridge_pipeline = Pipeline([
        (
            "preprocessor",
            create_preprocessor(
                numeric_columns,
                categorical_columns,
                scale_numeric=True
            )
        ),
        (
            "model",
            Ridge(
                alpha=10.0,
                solver="lsqr"
            )
        )
    ])

    models["Ridge"] = ridge_pipeline

    # 2. Random Forest
    rf_pipeline = Pipeline([
        (
            "preprocessor",
            create_preprocessor(
                numeric_columns,
                categorical_columns
            )
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=60,
                max_depth=12,
                min_samples_leaf=2,
                max_features=0.8,
                n_jobs=CPU_JOBS,
                random_state=RANDOM_STATE
            )
        )
    ])

    models["Random Forest"] = rf_pipeline

    # 3. XGBoost
    if XGBOOST_AVAILABLE:

        xgb_pipeline = Pipeline([
            (
                "preprocessor",
                create_preprocessor(
                    numeric_columns,
                    categorical_columns
                )
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    reg_lambda=2,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    tree_method="hist",
                    n_jobs=CPU_JOBS,
                    random_state=RANDOM_STATE,
                    verbosity=0
                )
            )
        ])

        models["XGBoost"] = xgb_pipeline

    return models


# ============================================================
# MODEL TRAINING AND VALIDATION
# ============================================================

def train_and_compare(
    train,
    X_train,
    numeric_columns,
    categorical_columns
):

    log("Starting CPU-optimized model comparison...")

    fit_indices, validation_indices = chronological_split(train)

    X_fit = X_train.loc[fit_indices]
    X_validation = X_train.loc[validation_indices]

    y_fit = train.loc[fit_indices, TARGET]
    y_validation = train.loc[validation_indices, TARGET]

    models = build_models(
        numeric_columns,
        categorical_columns
    )

    results = []
    fitted_models = {}

    # Dummy baseline
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_fit, y_fit)

    dummy_predictions = dummy.predict(X_validation)

    dummy_metrics = calculate_metrics(
        y_validation,
        dummy_predictions
    )

    results.append({
        "Model": "Dummy Baseline",
        **dummy_metrics
    })

    log(
        f"Dummy baseline RMSE: "
        f"{dummy_metrics['RMSE']:.4f}"
    )

    for model_name, pipeline in models.items():

        log(f"Training {model_name}...")

        start_time = time.time()

        try:

            pipeline.fit(X_fit, y_fit)

            predictions = pipeline.predict(X_validation)

            metrics = calculate_metrics(
                y_validation,
                predictions
            )

            elapsed = time.time() - start_time

            results.append({
                "Model": model_name,
                **metrics,
                "Training_Time_Seconds": round(elapsed, 2)
            })

            fitted_models[model_name] = pipeline

            log(
                f"{model_name} completed | "
                f"MAE: {metrics['MAE']:.3f} | "
                f"RMSE: {metrics['RMSE']:.3f} | "
                f"R2: {metrics['R2']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

        except Exception as error:

            log(f"{model_name} failed: {error}")

        finally:
            gc.collect()

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        "RMSE"
    ).reset_index(drop=True)

    comparison.to_csv(
        REPORT_DIR / "phase4_1_validation_comparison.csv",
        index=False
    )

    if not fitted_models:
        raise RuntimeError(
            "All candidate models failed during training."
        )

    # Select model using validation RMSE only.
    best_name = min(
        fitted_models,
        key=lambda name: comparison.loc[
            comparison["Model"] == name, "RMSE"
        ].iloc[0]
    )

    log(f"Selected model: {best_name}")

    return best_name, comparison


# ============================================================
# FINAL MODEL TRAINING
# ============================================================

def train_final_model(
    best_name,
    train,
    X_train,
    numeric_columns,
    categorical_columns
):

    log("Retraining selected model on complete training data...")

    models = build_models(
        numeric_columns,
        categorical_columns
    )

    final_model = models[best_name]

    start_time = time.time()

    final_model.fit(
        X_train,
        train[TARGET]
    )

    elapsed = time.time() - start_time

    log(f"Final training completed in {elapsed:.1f} seconds.")

    return final_model


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def evaluate_final_model(
    model,
    test,
    X_test
):

    log("Evaluating final model on untouched test data...")

    actual = test[TARGET].values
    predicted = model.predict(X_test)

    metrics = calculate_metrics(actual, predicted)

    log(f"Test MAE:  {metrics['MAE']:.4f}")
    log(f"Test RMSE: {metrics['RMSE']:.4f}")
    log(f"Test R2:   {metrics['R2']:.4f}")

    predictions = test.copy()

    predictions["Actual_AQI"] = actual
    predictions["Predicted_AQI"] = predicted
    predictions["Absolute_Error"] = np.abs(
        actual - predicted
    )

    predictions.to_csv(
        REPORT_DIR / "phase4_1_predictions.csv",
        index=False
    )

    return metrics, predictions


# ============================================================
# CITY-WISE EVALUATION
# ============================================================

def evaluate_cities(predictions):

    log("Generating city-wise evaluation...")

    if "City" not in predictions.columns:
        log("City column unavailable. Skipping city evaluation.")
        return pd.DataFrame()

    records = []

    for city, group in predictions.groupby("City"):

        if len(group) < 2:
            continue

        metrics = calculate_metrics(
            group["Actual_AQI"],
            group["Predicted_AQI"]
        )

        records.append({
            "City": city,
            "Records": len(group),
            **metrics
        })

    result = pd.DataFrame(records)

    if not result.empty:
        result = result.sort_values(
            "MAE"
        ).reset_index(drop=True)

    result.to_csv(
        REPORT_DIR / "phase4_1_city_evaluation.csv",
        index=False
    )

    return result


# ============================================================
# AQI SEVERITY EVALUATION
# ============================================================

def get_aqi_category(value):

    if value <= 50:
        return "Good"

    elif value <= 100:
        return "Satisfactory"

    elif value <= 200:
        return "Moderate"

    elif value <= 300:
        return "Poor"

    elif value <= 400:
        return "Very Poor"

    elif value <= 500:
        return "Severe"

    return "Above 500 - Outside standard AQI scale"


def evaluate_severity(predictions):

    log("Generating AQI severity evaluation...")

    result = predictions.copy()

    result["Actual_Category"] = result[
        "Actual_AQI"
    ].apply(get_aqi_category)

    result["Predicted_Category"] = result[
        "Predicted_AQI"
    ].apply(get_aqi_category)

    result["Category_Match"] = (
        result["Actual_Category"]
        == result["Predicted_Category"]
    )

    severity = result.groupby(
        "Actual_Category"
    ).agg(
        Records=("Actual_AQI", "count"),
        Mean_Absolute_Error=("Absolute_Error", "mean"),
        Category_Match_Percentage=("Category_Match", "mean")
    ).reset_index()

    severity["Category_Match_Percentage"] *= 100

    severity.to_csv(
        REPORT_DIR / "phase4_1_severity_evaluation.csv",
        index=False
    )

    return severity


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(model):

    log("Generating feature importance...")

    try:

        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()

        if hasattr(estimator, "feature_importances_"):

            importance = estimator.feature_importances_

        elif hasattr(estimator, "coef_"):

            importance = np.abs(
                np.asarray(estimator.coef_).ravel()
            )

        else:
            log("Feature importance unavailable for selected model.")
            return

        if len(feature_names) != len(importance):
            log("Feature importance dimensions do not match.")
            return

        result = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        result = result.sort_values(
            "Importance",
            ascending=False
        ).reset_index(drop=True)

        result.to_csv(
            REPORT_DIR / "phase4_1_feature_importance.csv",
            index=False
        )

    except Exception as error:
        log(f"Feature importance error: {error}")


# ============================================================
# SAVE MODEL AND REPORT
# ============================================================

def save_final_outputs(
    model,
    best_name,
    metrics,
    comparison,
    train,
    test
):

    model_path = MODEL_DIR / "best_aqi_model_optimized.joblib"

    joblib.dump(
        model,
        model_path,
        compress=1
    )

    report = {
        "project": "AirSense India",
        "phase": "Phase 4.1 - CPU Optimized",
        "selected_model": best_name,
        "hardware_mode": "CPU-only / Single-thread",
        "training_records": int(len(train)),
        "testing_records": int(len(test)),
        "city_count": int(
            train["City"].nunique()
            if "City" in train.columns else 0
        ),
        "test_metrics": metrics,
        "models_compared": comparison.to_dict(
            orient="records"
        ),
        "model_path": str(model_path),
        "test_data_used_for_model_selection": False
    }

    save_json(
        report,
        REPORT_DIR / "phase4_1_optimization_report.json"
    )

    log(f"Saved optimized model: {model_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    print("\n" + "=" * 65)
    print(" AIRSENSE INDIA - PHASE 4.1 CPU OPTIMIZATION")
    print("=" * 65)

    log("Hardware configuration: CPU only, single-threaded")
    log("Loading datasets...")

    train, test = load_data()

    (
        X_train,
        X_test,
        numeric_columns,
        categorical_columns
    ) = prepare_features(train, test)

    # Model comparison
    best_name, comparison = train_and_compare(
        train,
        X_train,
        numeric_columns,
        categorical_columns
    )

    # Train selected model on complete training data
    final_model = train_final_model(
        best_name,
        train,
        X_train,
        numeric_columns,
        categorical_columns
    )

    # Test only after model selection and final training
    metrics, predictions = evaluate_final_model(
        final_model,
        test,
        X_test
    )

    # Additional reports
    city_results = evaluate_cities(predictions)

    severity_results = evaluate_severity(predictions)

    save_feature_importance(final_model)

    save_final_outputs(
        final_model,
        best_name,
        metrics,
        comparison,
        train,
        test
    )

    total_time = time.time() - total_start

    print("\n" + "=" * 65)
    print(" PHASE 4.1 COMPLETED")
    print("=" * 65)

    print(f"\nSelected Model: {best_name}")

    print("\nFinal Test Metrics:")
    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"R2   : {metrics['R2']:.4f}")

    print(f"\nTotal training time: {total_time / 60:.2f} minutes")

    print("\nGenerated Reports:")
    print(REPORT_DIR)

    print("\nGenerated Model:")
    print(MODEL_DIR / "best_aqi_model_optimized.joblib")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
