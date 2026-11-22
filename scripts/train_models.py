
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =====================================================
# PATHS AND CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports" / "statistics"
MODEL_DIR = BASE_DIR / "models"

TRAIN_PATH = DATA_DIR / "train_features.csv"
TEST_PATH = DATA_DIR / "test_features.csv"
PHASE3_REPORT = REPORT_DIR / "phase3_feature_report.json"

TARGET = "target_aqi_next_day"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================

def load_data():

    for path in [TRAIN_PATH, TEST_PATH, PHASE3_REPORT]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    train_df = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["Date", "target_date"]
    )

    test_df = pd.read_csv(
        TEST_PATH,
        parse_dates=["Date", "target_date"]
    )

    with open(PHASE3_REPORT, "r", encoding="utf-8") as file:
        phase3_report = json.load(file)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column missing: {TARGET}")

    if TARGET not in test_df.columns:
        raise ValueError(f"Target column missing: {TARGET}")

    train_df = train_df.dropna(subset=[TARGET]).copy()
    test_df = test_df.dropna(subset=[TARGET]).copy()

    # Confirm the test period is later than training.
    latest_train_date = train_df["target_date"].max()
    earliest_test_date = test_df["target_date"].min()

    if latest_train_date >= earliest_test_date:
        raise ValueError(
            "Temporal leakage risk: training and test "
            "target-date ranges overlap."
        )

    print("\nDATASET INFORMATION")
    print("-" * 50)
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Training target-date end:", latest_train_date)
    print("Testing target-date start:", earliest_test_date)

    return train_df, test_df, phase3_report


# =====================================================
# PREPARE FEATURES
# =====================================================

def prepare_features(train_df, test_df, phase3_report):

    excluded_columns = {
        TARGET,
        "target_date",
        "Date",
        "AQI_Bucket"
    }

    # Use the feature list generated in Phase 3.
    feature_columns = [
        col for col in phase3_report["feature_columns"]
        if col not in excluded_columns
        and col in train_df.columns
        and col in test_df.columns
    ]

    if not feature_columns:
        raise ValueError("No usable model features found.")

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    y_train = pd.to_numeric(
        train_df[TARGET],
        errors="coerce"
    )

    y_test = pd.to_numeric(
        test_df[TARGET],
        errors="coerce"
    )

    # Remove any rows with invalid target values.
    valid_train = y_train.notna()
    valid_test = y_test.notna()

    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train]

    X_test = X_test.loc[valid_test]
    y_test = y_test.loc[valid_test]

    # Identify categorical and numeric columns.
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_columns = [
        col for col in feature_columns
        if col not in categorical_columns
    ]

    print("\nFEATURE INFORMATION")
    print("-" * 50)
    print("Total features:", len(feature_columns))
    print("Numeric features:", len(numeric_columns))
    print("Categorical features:", len(categorical_columns))

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
        numeric_columns,
        categorical_columns
    )


# =====================================================
# PREPROCESSING PIPELINES
# =====================================================

def build_preprocessor(
    numeric_columns,
    categorical_columns,
    scale_numeric=False
):

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        )
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
                keep_empty_features=True
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop"
    )


# =====================================================
# DEFINE MODELS
# =====================================================

def build_models(
    numeric_columns,
    categorical_columns
):

    models = {

        "Dummy Baseline": Pipeline([
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns,
                    categorical_columns
                )
            ),
            (
                "model",
                DummyRegressor(strategy="median")
            )
        ]),

        "Ridge Regression": Pipeline([
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns,
                    categorical_columns,
                    scale_numeric=True
                )
            ),
            (
                "model",
                Ridge(alpha=10.0)
            )
        ]),

        "Random Forest": Pipeline([
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns,
                    categorical_columns
                )
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=18,
                    min_samples_leaf=2,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=42
                )
            )
        ]),

        "XGBoost": Pipeline([
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns,
                    categorical_columns
                )
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=600,
                    learning_rate=0.04,
                    max_depth=6,
                    min_child_weight=3,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.1,
                    reg_lambda=2.0,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=42
                )
            )
        ])
    }

    return models


# =====================================================
# EVALUATION
# =====================================================

def calculate_metrics(y_true, predictions):

    rmse = np.sqrt(
        mean_squared_error(y_true, predictions)
    )

    return {
        "MAE": float(
            mean_absolute_error(y_true, predictions)
        ),
        "RMSE": float(rmse),
        "R2": float(
            r2_score(y_true, predictions)
        )
    }


# =====================================================
# CITY-WISE EVALUATION
# =====================================================

def evaluate_by_city(test_df, predictions, model_name):

    results = []

    evaluation_df = test_df[
        ["City", "Date", "target_date", TARGET]
    ].copy()

    evaluation_df["Predicted_AQI"] = predictions

    evaluation_df["Absolute_Error"] = (
        evaluation_df[TARGET]
        - evaluation_df["Predicted_AQI"]
    ).abs()

    for city, group in evaluation_df.groupby("City"):

        if len(group) < 30:
            continue

        results.append({

            "City": city,

            "Records": int(len(group)),

            "MAE": float(
                mean_absolute_error(
                    group[TARGET],
                    group["Predicted_AQI"]
                )
            ),

            "RMSE": float(
                np.sqrt(
                    mean_squared_error(
                        group[TARGET],
                        group["Predicted_AQI"]
                    )
                )
            ),

            "R2": float(
                r2_score(
                    group[TARGET],
                    group["Predicted_AQI"]
                )
            )
        })

    city_report = pd.DataFrame(results)

    city_report = city_report.sort_values(
        "MAE"
    )

    city_report.to_csv(
        REPORT_DIR / "phase4_city_evaluation.csv",
        index=False
    )

    return city_report


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

def save_feature_importance(
    best_pipeline,
    model_name
):

    model = best_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        print(
            "\nSelected model does not expose "
            "tree-based feature importance."
        )
        return

    preprocessor = best_pipeline.named_steps[
        "preprocessor"
    ]

    transformed_names = (
        preprocessor.get_feature_names_out()
    )

    importance = pd.DataFrame({

        "Feature": transformed_names,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    importance.to_csv(
        REPORT_DIR / "phase4_feature_importance.csv",
        index=False
    )

    print("\nTOP 15 FEATURES")
    print(importance.head(15).to_string(index=False))


# =====================================================
# MAIN TRAINING FUNCTION
# =====================================================


def main():

    print("\nAIR SENSE INDIA")
    print("PHASE 4 - MACHINE LEARNING")
    print("=" * 60)

    (
        train_df,
        test_df,
        phase3_report
    ) = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
        numeric_columns,
        categorical_columns
    ) = prepare_features(
        train_df,
        test_df,
        phase3_report
    )

    # -------------------------------------------------
    # CHRONOLOGICAL VALIDATION SPLIT INSIDE TRAINING
    # -------------------------------------------------

    training_dates = np.sort(
        train_df.loc[X_train.index, "target_date"]
        .dropna()
        .unique()
    )

    if len(training_dates) < 10:
        raise ValueError(
            "Insufficient training dates for validation."
        )

    validation_position = int(
        len(training_dates) * 0.85
    )

    validation_position = min(
        max(validation_position, 1),
        len(training_dates) - 1
    )

    validation_start = training_dates[
        validation_position
    ]

    fit_mask = (
        train_df.loc[X_train.index, "target_date"]
        < validation_start
    )

    validation_mask = ~fit_mask

    X_fit = X_train.loc[fit_mask]
    y_fit = y_train.loc[fit_mask]

    X_validation = X_train.loc[validation_mask]
    y_validation = y_train.loc[validation_mask]

    print("\nVALIDATION SPLIT")
    print("-" * 50)
    print("Fit rows:", len(X_fit))
    print("Validation rows:", len(X_validation))
    print("Validation starts:", validation_start)

    if len(X_fit) == 0 or len(X_validation) == 0:
        raise ValueError(
            "Invalid chronological validation split."
        )

    # -------------------------------------------------
    # TRAIN AND COMPARE ON VALIDATION DATA
    # -------------------------------------------------

    models = build_models(
        numeric_columns,
        categorical_columns
    )

    validation_results = {}
    trained_models = {}

    for name, pipeline in models.items():

        print("\nTraining:", name)

        pipeline.fit(X_fit, y_fit)

        validation_predictions = pipeline.predict(
            X_validation
        )

        metrics = calculate_metrics(
            y_validation,
            validation_predictions
        )

        validation_results[name] = metrics
        trained_models[name] = pipeline

        print("Validation MAE:", round(metrics["MAE"], 4))
        print("Validation RMSE:", round(metrics["RMSE"], 4))
        print("Validation R2:", round(metrics["R2"], 4))

    comparison = pd.DataFrame([

        {
            "Model": name,
            "Validation_MAE": values["MAE"],
            "Validation_RMSE": values["RMSE"],
            "Validation_R2": values["R2"]
        }

        for name, values in validation_results.items()

    ])

    comparison = comparison.sort_values(
        "Validation_RMSE"
    ).reset_index(drop=True)

    comparison.to_csv(
        REPORT_DIR / "phase4_model_comparison.csv",
        index=False
    )

    print("\nVALIDATION MODEL COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False))

    # Select winner using validation RMSE only.
    best_name = comparison.iloc[0]["Model"]

    best_model = trained_models[best_name]

    print("\nSELECTED MODEL:", best_name)

    # -------------------------------------------------
    # REFIT SELECTED MODEL ON FULL TRAINING DATA
    # -------------------------------------------------

    print("\nRetraining selected model on full training data...")

    best_model.fit(X_train, y_train)

    # -------------------------------------------------
    # FINAL TEST EVALUATION
    # -------------------------------------------------

    print("\nEvaluating on untouched test data...")

    test_predictions = best_model.predict(X_test)

    final_metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    print("\nFINAL TEST RESULTS")
    print("=" * 60)
    print("Model:", best_name)
    print("MAE:", round(final_metrics["MAE"], 4))
    print("RMSE:", round(final_metrics["RMSE"], 4))
    print("R2:", round(final_metrics["R2"], 4))

    # -------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------

    model_bundle = {

        "model": best_model,
        "model_name": best_name,
        "target": TARGET,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "training_end_date": str(
            train_df["target_date"].max().date()
        ),
        "testing_start_date": str(
            test_df["target_date"].min().date()
        )
    }

    model_path = MODEL_DIR / "best_aqi_model.joblib"

    joblib.dump(
        model_bundle,
        model_path
    )

    # -------------------------------------------------
    # SAVE TEST PREDICTIONS
    # -------------------------------------------------

    prediction_df = test_df[
        ["City", "Date", "target_date", TARGET]
    ].copy()

    prediction_df["Predicted_AQI"] = test_predictions

    prediction_df["Absolute_Error"] = (
        prediction_df[TARGET]
        - prediction_df["Predicted_AQI"]
    ).abs()

    prediction_df.to_csv(
        REPORT_DIR / "phase4_predictions.csv",
        index=False
    )

    # -------------------------------------------------
    # CITY-WISE EVALUATION
    # -------------------------------------------------

    city_report = evaluate_by_city(
        test_df,
        test_predictions,
        best_name
    )

    # -------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------

    save_feature_importance(
        best_model,
        best_name
    )

    # -------------------------------------------------
    # FINAL REPORT
    # -------------------------------------------------

    report = {

        "project": "AirSense India",

        "task": "Next-day AQI prediction",

        "best_model": best_name,

        "validation_metrics": validation_results[best_name],

        "test_metrics": final_metrics,

        "validation_model_comparison": comparison.to_dict(
            orient="records"
        ),

        "training_rows": len(X_train),

        "testing_rows": len(X_test),

        "feature_count": len(feature_columns),

        "city_count": int(
            test_df["City"].nunique()
        ),

        "validation_start_date": str(
            pd.Timestamp(validation_start).date()
        ),

        "model_file": str(model_path),

        "evaluation_strategy": (
            "Chronological validation for model selection; "
            "untouched chronological test set for final evaluation."
        )
    }

    with open(
        REPORT_DIR / "phase4_model_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\nPHASE 4 COMPLETED")
    print("=" * 60)

    print("Best model:", best_name)
    print("Final test MAE:", round(final_metrics["MAE"], 4))
    print("Final test RMSE:", round(final_metrics["RMSE"], 4))
    print("Final test R2:", round(final_metrics["R2"], 4))

    print("\nSaved model:", model_path)

    print("\nReports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()


