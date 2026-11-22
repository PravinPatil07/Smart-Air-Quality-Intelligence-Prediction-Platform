
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATABASE_FILE = ROOT_DIR / "database" / "airsense.db"

FEATURE_FILE = (
    ROOT_DIR / "data" / "processed" / "airsense_features.csv"
)

TRAIN_FILE = (
    ROOT_DIR / "data" / "processed" / "train_features.csv"
)

MODEL_FILE = (
    ROOT_DIR / "models" / "best_aqi_model_optimized.joblib"
)

PHASE41_REPORT = (
    ROOT_DIR / "reports" / "statistics"
    / "phase4_1_optimization_report.json"
)

PHASE42_REPORT = (
    ROOT_DIR / "reports" / "statistics"
    / "phase4_2_validation_report.json"
)

FEATURE_IMPORTANCE_FILE = (
    ROOT_DIR / "reports" / "statistics"
    / "phase4_2_permutation_importance.csv"
)


# ============================================================
# CITY COORDINATES
# Approximate city-centre coordinates, not station locations.
# ============================================================

CITY_COORDINATES = {
    "Ahmedabad": [23.0225, 72.5714],
    "Aizawl": [23.7271, 92.7176],
    "Amaravati": [16.5062, 80.6480],
    "Amritsar": [31.6340, 74.8723],
    "Bengaluru": [12.9716, 77.5946],
    "Bhopal": [23.2599, 77.4126],
    "Brajrajnagar": [21.8167, 83.9167],
    "Chandigarh": [30.7333, 76.7794],
    "Chennai": [13.0827, 80.2707],
    "Coimbatore": [11.0168, 76.9558],
    "Delhi": [28.6139, 77.2090],
    "Ernakulam": [9.9816, 76.2999],
    "Gurugram": [28.4595, 77.0266],
    "Guwahati": [26.1445, 91.7362],
    "Hyderabad": [17.3850, 78.4867],
    "Jaipur": [26.9124, 75.7873],
    "Jorapokhar": [23.7565, 86.4140],
    "Kochi": [9.9312, 76.2673],
    "Kolkata": [22.5726, 88.3639],
    "Lucknow": [26.8467, 80.9462],
    "Mumbai": [19.0760, 72.8777],
    "Patna": [25.5941, 85.1376],
    "Shillong": [25.5788, 91.8933],
    "Talcher": [20.9517, 85.2167],
    "Thiruvananthapuram": [8.5241, 76.9366],
    "Visakhapatnam": [17.6868, 83.2185],
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_columns(dataframe):

    aliases = {
        "city": "City",
        "date": "Date",
        "target_date": "target_date",
        "aqi": "AQI",
        "aqi_bucket": "AQI_Bucket",
        "pm2.5": "PM2.5",
        "pm10": "PM10",
        "no": "NO",
        "no2": "NO2",
        "nox": "NOx",
        "nh3": "NH3",
        "so2": "SO2",
        "co": "CO",
        "o3": "O3",
        "benzene": "Benzene",
        "toluene": "Toluene",
        "xylene": "Xylene",
    }

    rename_map = {}

    for column in dataframe.columns:

        key = str(column).strip().lower()

        if key in aliases:
            rename_map[column] = aliases[key]

    return dataframe.rename(columns=rename_map)


def get_aqi_category(aqi):

    if aqi is None or pd.isna(aqi):
        return "Unavailable"

    aqi = float(aqi)

    if aqi < 0:
        return "Invalid"
    elif aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    elif aqi <= 500:
        return "Severe"

    return "Above 500"


def get_aqi_color(category):

    colors = {
        "Good": "#22c55e",
        "Satisfactory": "#84cc16",
        "Moderate": "#eab308",
        "Poor": "#f97316",
        "Very Poor": "#ef4444",
        "Severe": "#991b1b",
        "Above 500": "#7f1d1d",
        "Unavailable": "#64748b",
        "Invalid": "#64748b",
    }

    return colors.get(category, "#64748b")


def find_nested_value(data, possible_keys):

    if isinstance(data, dict):

        for key, value in data.items():

            if str(key).lower() in possible_keys:
                return value

        for value in data.values():

            result = find_nested_value(
                value,
                possible_keys
            )

            if result is not None:
                return result

    elif isinstance(data, list):

        for item in data:

            result = find_nested_value(
                item,
                possible_keys
            )

            if result is not None:
                return result

    return None


def find_metrics_dictionary(data):

    if isinstance(data, dict):

        normalized_keys = {
            str(key).lower() for key in data.keys()
        }

        has_mae = "mae" in normalized_keys
        has_rmse = "rmse" in normalized_keys

        has_r2 = any(
            key in normalized_keys
            for key in ["r2", "r2_score", "r_squared"]
        )

        if has_mae and has_rmse and has_r2:
            return data

        for value in data.values():

            result = find_metrics_dictionary(value)

            if result is not None:
                return result

    elif isinstance(data, list):

        for item in data:

            result = find_metrics_dictionary(item)

            if result is not None:
                return result

    return None


def normalize_metrics(metrics):

    if not isinstance(metrics, dict):
        return {}

    lookup = {
        str(key).lower(): value
        for key, value in metrics.items()
    }

    return {
        "MAE": lookup.get("mae"),
        "RMSE": lookup.get("rmse"),
        "R2": lookup.get(
            "r2",
            lookup.get(
                "r2_score",
                lookup.get("r_squared")
            )
        )
    }


# ============================================================
# AIRSENSE SERVICE
# ============================================================

class AirSenseService:

    def __init__(self):

        self.model = None

        self.features = pd.DataFrame()
        self.history = pd.DataFrame()

        self.feature_columns = []

        # Internal set for fast prediction validation.
        self.trained_cities = set()

        # Public sorted list used by the API and dashboard.
        self.training_cities = []

        self.interval_margin = None

        self.phase41_report = {}
        self.phase42_report = {}

        self.test_metrics = {}
        self.validation_metrics = {}

        self.top_features = []

        self.load_resources()

    # ========================================================
    # LOAD MODEL AND DATA
    # ========================================================

    def load_resources(self):

        print("[AirSense] Loading resources...")

        required_files = [
            MODEL_FILE,
            FEATURE_FILE,
            TRAIN_FILE,
        ]

        for file_path in required_files:

            if not file_path.exists():

                raise FileNotFoundError(
                    f"Required AirSense file not found: {file_path}"
                )

        # Load saved Ridge Regression pipeline.
        self.model = joblib.load(MODEL_FILE)

        if not hasattr(self.model, "named_steps"):

            raise ValueError(
                "Expected a scikit-learn Pipeline model."
            )

        if "preprocessor" not in self.model.named_steps:

            raise ValueError(
                "Saved model does not contain "
                "a 'preprocessor' pipeline step."
            )

        preprocessor = self.model.named_steps["preprocessor"]

        if not hasattr(preprocessor, "feature_names_in_"):

            raise ValueError(
                "Model preprocessing step does not contain "
                "feature_names_in_."
            )

        self.feature_columns = list(
            preprocessor.feature_names_in_
        )

        # Load engineered features.
        self.features = pd.read_csv(FEATURE_FILE)

        self.features = normalize_columns(self.features)

        if "City" not in self.features.columns:

            raise ValueError(
                "City column is missing from airsense_features.csv."
            )

        self.features["City"] = (
            self.features["City"]
            .astype("string")
            .str.strip()
        )

        # Validate exact model input columns.
        missing_columns = [
            column
            for column in self.feature_columns
            if column not in self.features.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing model input columns in feature dataset: "
                + ", ".join(missing_columns)
            )

        # ----------------------------------------------------
        # LOAD TRAINING CITIES
        # ----------------------------------------------------

        training_data = pd.read_csv(
            TRAIN_FILE,
            usecols=["City"]
        )

        training_data["City"] = (
            training_data["City"]
            .astype("string")
            .str.strip()
        )

        self.trained_cities = set(
            training_data["City"]
            .dropna()
            .tolist()
        )

        # FIX: expose trained cities as a sorted list.
        self.training_cities = sorted(
            self.trained_cities,
            key=str.lower
        )

        del training_data

        # Parse dates.
        for column in ["Date", "target_date"]:

            if column in self.features.columns:

                self.features[column] = pd.to_datetime(
                    self.features[column],
                    errors="coerce"
                )

        # Load remaining resources.
        self.load_history()
        self.load_reports()
        self.load_feature_importance()

        print("[AirSense] Model loaded successfully.")

        print(
            f"[AirSense] Model input features: "
            f"{len(self.feature_columns)}"
        )

        print(
            f"[AirSense] Historical cities: "
            f"{self.features['City'].nunique()}"
        )

        print(
            f"[AirSense] Trained cities: "
            f"{len(self.training_cities)}"
        )

        print(
            "[AirSense] Prediction cities:",
            self.training_cities
        )

    # ========================================================
    # PREDICTION CITY ACCESSOR
    # ========================================================

    def get_prediction_cities(self):

        return self.training_cities.copy()

    # ========================================================
    # LOAD SQLITE HISTORY
    # ========================================================

    def load_history(self):

        self.history = pd.DataFrame()

        if DATABASE_FILE.exists():

            try:

                with sqlite3.connect(DATABASE_FILE) as connection:

                    tables = pd.read_sql_query(
                        """
                        SELECT name
                        FROM sqlite_master
                        WHERE type = 'table'
                        """,
                        connection
                    )

                    table_names = tables["name"].tolist()

                    if "city_day" in table_names:

                        self.history = pd.read_sql_query(
                            'SELECT * FROM "city_day"',
                            connection
                        )

            except Exception as error:

                print(
                    "[AirSense] Database loading warning:",
                    error
                )

        # Fallback to engineered features.
        if self.history.empty:

            self.history = self.features.copy()

        self.history = normalize_columns(self.history)

        if "City" not in self.history.columns:

            raise ValueError(
                "City column missing from historical data."
            )

        self.history["City"] = (
            self.history["City"]
            .astype(str)
            .str.strip()
        )

        if "Date" in self.history.columns:

            self.history["Date"] = pd.to_datetime(
                self.history["Date"],
                errors="coerce"
            )

            self.history = self.history.dropna(
                subset=["Date"]
            )

        if "AQI" in self.history.columns:

            self.history["AQI"] = pd.to_numeric(
                self.history["AQI"],
                errors="coerce"
            )

        self.history = self.history.reset_index(drop=True)

    # ========================================================
    # LOAD REPORTS
    # ========================================================

    def load_reports(self):

        if PHASE41_REPORT.exists():

            try:

                with open(
                    PHASE41_REPORT,
                    "r",
                    encoding="utf-8"
                ) as file:

                    self.phase41_report = json.load(file)

            except (OSError, json.JSONDecodeError) as error:

                print(
                    "[AirSense] Phase 4.1 report warning:",
                    error
                )

        if PHASE42_REPORT.exists():

            try:

                with open(
                    PHASE42_REPORT,
                    "r",
                    encoding="utf-8"
                ) as file:

                    self.phase42_report = json.load(file)

            except (OSError, json.JSONDecodeError) as error:

                print(
                    "[AirSense] Phase 4.2 report warning:",
                    error
                )

        metrics = find_metrics_dictionary(
            self.phase42_report
        )

        if metrics is None:

            metrics = find_metrics_dictionary(
                self.phase41_report
            )

        self.test_metrics = normalize_metrics(metrics)

        validation = find_nested_value(
            self.phase42_report,
            {
                "chronological_validation_metrics",
                "validation_metrics",
                "holdout_metrics"
            }
        )

        self.validation_metrics = normalize_metrics(
            validation
        )

        uncertainty = find_nested_value(
            self.phase42_report,
            {"prediction_uncertainty"}
        )

        if isinstance(uncertainty, dict):

            margin = uncertainty.get(
                "absolute_error_margin"
            )

            if margin is not None:

                try:

                    self.interval_margin = float(margin)

                except (TypeError, ValueError):

                    self.interval_margin = None

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    def load_feature_importance(self):

        self.top_features = []

        if not FEATURE_IMPORTANCE_FILE.exists():
            return

        try:

            dataframe = pd.read_csv(
                FEATURE_IMPORTANCE_FILE
            )

            dataframe.columns = [
                str(column).strip()
                for column in dataframe.columns
            ]

            lower_columns = {
                column.lower(): column
                for column in dataframe.columns
            }

            feature_column = None

            for name in ["feature", "features", "variable"]:

                if name in lower_columns:

                    feature_column = lower_columns[name]
                    break

            importance_column = None

            for name in [
                "importance_mean",
                "mean_importance",
                "importance",
                "permutation_importance",
                "mean"
            ]:

                if name in lower_columns:

                    importance_column = lower_columns[name]
                    break

            if (
                feature_column is None
                or importance_column is None
            ):

                print(
                    "[AirSense] Could not identify feature "
                    "importance columns."
                )

                return

            dataframe[importance_column] = pd.to_numeric(
                dataframe[importance_column],
                errors="coerce"
            )

            dataframe = dataframe.dropna(
                subset=[feature_column, importance_column]
            )

            dataframe = dataframe.sort_values(
                importance_column,
                ascending=False
            ).head(10)

            for _, row in dataframe.iterrows():

                self.top_features.append({
                    "feature": str(row[feature_column]),
                    "importance": round(
                        float(row[importance_column]),
                        4
                    )
                })

        except Exception as error:

            print(
                "[AirSense] Feature importance warning:",
                error
            )

    # ========================================================
    # CITY INFORMATION
    # ========================================================

    def get_cities(self):

        return sorted(
            self.features["City"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

    def get_latest_observations(self):

        required_columns = {"City", "Date", "AQI"}

        if not required_columns.issubset(
            self.history.columns
        ):

            return pd.DataFrame()

        valid = self.history.dropna(
            subset=["City", "Date", "AQI"]
        ).copy()

        if valid.empty:

            return pd.DataFrame()

        latest = (
            valid.sort_values("Date")
            .groupby("City", as_index=False)
            .tail(1)
            .copy()
        )

        latest["Category"] = latest["AQI"].apply(
            get_aqi_category
        )

        latest["Color"] = latest["Category"].apply(
            get_aqi_color
        )

        latest["Coordinates"] = latest["City"].apply(
            lambda city: CITY_COORDINATES.get(city)
        )

        return latest

    # ========================================================
    # DASHBOARD OVERVIEW
    # ========================================================

    def get_overview(self):

        latest = self.get_latest_observations()

        if latest.empty:

            return {
                "city_count": 0,
                "record_count": len(self.history),
                "average_aqi": None,
                "latest_date": None,
                "cities": []
            }

        average_aqi = latest["AQI"].mean()

        latest_date = latest["Date"].max()

        cities = []

        for _, row in latest.iterrows():

            coordinates = row["Coordinates"]

            cities.append({
                "city": str(row["City"]),
                "aqi": round(float(row["AQI"]), 2),
                "category": row["Category"],
                "color": row["Color"],
                "date": row["Date"].strftime("%Y-%m-%d"),
                "coordinates": coordinates
            })

        return {
            "city_count": len(cities),
            "record_count": len(self.history),
            "average_aqi": (
                round(float(average_aqi), 2)
                if pd.notna(average_aqi) else None
            ),
            "latest_date": (
                latest_date.strftime("%Y-%m-%d")
                if pd.notna(latest_date) else None
            ),
            "cities": cities
        }

    # ========================================================
    # CITY HISTORICAL ANALYTICS
    # ========================================================

    def get_city_history(self, city, limit=180):

        city_data = self.history[
            self.history["City"].str.lower()
            == city.strip().lower()
        ].copy()

        if city_data.empty:
            return None

        if "Date" not in city_data.columns:
            return None

        city_data = city_data.dropna(
            subset=["Date"]
        ).sort_values("Date")

        if "AQI" in city_data.columns:

            city_data["AQI"] = pd.to_numeric(
                city_data["AQI"],
                errors="coerce"
            )

        city_data = city_data.tail(limit)

        history = []

        if "AQI" in city_data.columns:

            valid_aqi = city_data.dropna(
                subset=["AQI"]
            )

            for _, row in valid_aqi.iterrows():

                history.append({
                    "date": row["Date"].strftime("%Y-%m-%d"),
                    "aqi": round(float(row["AQI"]), 2)
                })

        pollutant_names = [
            "PM2.5",
            "PM10",
            "NO2",
            "SO2",
            "CO",
            "O3",
            "NH3"
        ]

        pollutant_averages = {}

        for pollutant in pollutant_names:

            if pollutant not in city_data.columns:
                continue

            values = pd.to_numeric(
                city_data[pollutant],
                errors="coerce"
            ).dropna()

            if not values.empty:

                pollutant_averages[pollutant] = round(
                    float(values.mean()),
                    3
                )

        latest_row = city_data.iloc[-1]

        latest_aqi = latest_row.get("AQI")

        return {
            "city": city,
            "record_count": len(city_data),
            "latest_date": latest_row["Date"].strftime(
                "%Y-%m-%d"
            ),
            "latest_aqi": (
                round(float(latest_aqi), 2)
                if pd.notna(latest_aqi) else None
            ),
            "category": get_aqi_category(latest_aqi),
            "history": history,
            "pollutants": pollutant_averages
        }

    # ========================================================
    # NEXT-DAY AQI PREDICTION
    # ========================================================

    def predict_city(self, city):

        city = str(city).strip()

        # Validate city against the model's training cities.
        trained_city_lookup = {
            item.lower(): item
            for item in self.trained_cities
        }

        if city.lower() not in trained_city_lookup:
            return None

        actual_city = trained_city_lookup[city.lower()]

        city_data = self.features[
            self.features["City"].str.lower()
            == actual_city.lower()
        ].copy()

        if city_data.empty:
            return None

        # Prefer the next-day target date.
        sort_column = (
            "target_date"
            if "target_date" in city_data.columns
            else "Date"
        )

        if sort_column in city_data.columns:

            city_data = city_data.dropna(
                subset=[sort_column]
            )

            city_data = city_data.sort_values(
                sort_column
            )

        if city_data.empty:
            return None

        # Select the most recent usable feature row.
        city_data = city_data.dropna(
            subset=self.feature_columns,
            how="all"
        )

        if city_data.empty:
            return None

        latest = city_data.iloc[[-1]]

        X = latest[self.feature_columns].copy()

        # Saved pipeline applies fitted preprocessing.
        prediction = float(
            self.model.predict(X)[0]
        )

        prediction = max(0.0, prediction)

        observed_aqi = None

        if "AQI" in latest.columns:

            value = latest["AQI"].iloc[0]

            if pd.notna(value):
                observed_aqi = float(value)

        observation_date = None
        forecast_date = None

        if "Date" in latest.columns:

            value = latest["Date"].iloc[0]

            if pd.notna(value):

                observation_date = value.strftime(
                    "%Y-%m-%d"
                )

        if "target_date" in latest.columns:

            value = latest["target_date"].iloc[0]

            if pd.notna(value):

                forecast_date = value.strftime(
                    "%Y-%m-%d"
                )

        if forecast_date is None and observation_date:

            forecast_date = (
                pd.Timestamp(observation_date)
                + pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d")

        lower = None
        upper = None

        if self.interval_margin is not None:

            lower = max(
                0.0,
                prediction - self.interval_margin
            )

            upper = prediction + self.interval_margin

        category = get_aqi_category(prediction)

        return {
            "city": actual_city,

            "observation_date": observation_date,

            "forecast_date": forecast_date,

            "latest_observed_aqi": (
                round(observed_aqi, 2)
                if observed_aqi is not None else None
            ),

            "predicted_aqi": round(prediction, 2),

            "category": category,

            "color": get_aqi_color(category),

            "lower_90": (
                round(lower, 2)
                if lower is not None else None
            ),

            "upper_90": (
                round(upper, 2)
                if upper is not None else None
            ),

            "interval_note": (
                "Approximate historical error-based interval. "
                "It is not a guaranteed forecast range."
            )
        }

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    def get_model_information(self):

        return {
            "model": "Ridge Regression",

            "test_metrics": self.test_metrics,

            "validation_metrics": self.validation_metrics,

            "feature_count": len(self.feature_columns),

            "trained_city_count": len(self.trained_cities),

            "top_features": self.top_features,

            "interval_margin": self.interval_margin,

            "historical_data_only": True
        }


# ============================================================
# SHARED SERVICE INSTANCE
# ============================================================

airsense_service = AirSenseService()
