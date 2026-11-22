
from flask import Blueprint, jsonify, current_app


# ============================================================
# API BLUEPRINT
# ============================================================

api_bp = Blueprint(
    "api",
    __name__,
    url_prefix="/api"
)


# ============================================================
# SERVICE ACCESS
# ============================================================

def get_service():
    """Get the AirSense service registered with Flask."""

    service = current_app.extensions.get("airsense")

    if service is None:
        raise RuntimeError(
            "AirSense service is not registered in Flask."
        )

    return service


# ============================================================
# 1. DASHBOARD OVERVIEW
# ============================================================

@api_bp.route("/overview", methods=["GET"])
def overview():

    try:
        service = get_service()

        result = service.get_overview()

        return jsonify(result), 200

    except Exception as error:

        current_app.logger.exception(
            "Overview API error"
        )

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ============================================================
# 2. HISTORICAL AND PREDICTION CITY LISTS
# ============================================================

@api_bp.route("/cities", methods=["GET"])
def cities():

    try:
        service = get_service()

        # All cities in historical dataset.
        all_cities = service.get_cities()

        # Only cities represented in model training.
        prediction_cities = service.get_prediction_cities()

        return jsonify({
            "success": True,
            "cities": all_cities,
            "prediction_cities": prediction_cities,
            "historical_city_count": len(all_cities),
            "prediction_city_count": len(prediction_cities)
        }), 200

    except Exception as error:

        current_app.logger.exception(
            "City list API error"
        )

        return jsonify({
            "success": False,
            "error": str(error),
            "cities": [],
            "prediction_cities": [],
            "historical_city_count": 0,
            "prediction_city_count": 0
        }), 500


# ============================================================
# 3. CITY HISTORICAL ANALYTICS
# ============================================================

@api_bp.route("/city/<city>", methods=["GET"])
def city_details(city):

    try:
        service = get_service()

        result = service.get_city_history(city)

        if result is None:

            return jsonify({
                "success": False,
                "error": (
                    f"{city} was not found in the "
                    "historical dataset."
                )
            }), 404

        return jsonify(result), 200

    except Exception as error:

        current_app.logger.exception(
            "City history API error: %s",
            city
        )

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ============================================================
# 4. NEXT-DAY AQI PREDICTION
# ============================================================

@api_bp.route("/predict/<city>", methods=["GET"])
def predict(city):

    try:
        service = get_service()

        # Get the model-supported cities.
        supported_cities = service.get_prediction_cities()

        # Case-insensitive city matching.
        canonical_city = next(
            (
                item
                for item in supported_cities
                if item.strip().casefold()
                == city.strip().casefold()
            ),
            None
        )

        if canonical_city is None:

            return jsonify({
                "success": False,
                "error": (
                    f"{city} is not supported by the "
                    "trained AQI prediction model."
                ),
                "supported_cities": supported_cities
            }), 400

        # Generate prediction using saved Ridge pipeline.
        result = service.predict_city(canonical_city)

        if result is None:

            return jsonify({
                "success": False,
                "error": (
                    "No usable engineered features found "
                    f"for {canonical_city}."
                )
            }), 404

        return jsonify({
            "success": True,
            "model": "Ridge Regression",
            "historical_data_only": True,
            **result
        }), 200

    except Exception as error:

        current_app.logger.exception(
            "AQI prediction API error: %s",
            city
        )

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ============================================================
# 5. MODEL PERFORMANCE AND EXPLAINABILITY
# ============================================================

@api_bp.route("/model", methods=["GET"])
def model_information():

    try:
        service = get_service()

        result = service.get_model_information()

        return jsonify(result), 200

    except Exception as error:

        current_app.logger.exception(
            "Model information API error"
        )

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ============================================================
# 6. API HEALTH CHECK
# ============================================================

@api_bp.route("/health", methods=["GET"])
def api_health():

    try:
        service = get_service()

        return jsonify({
            "success": True,
            "status": "healthy",
            "service": "AirSense India API",
            "model_loaded": service.model is not None,
            "historical_cities": len(service.get_cities()),
            "prediction_cities": len(
                service.get_prediction_cities()
            )
        }), 200

    except Exception as error:

        current_app.logger.exception(
            "API health check error"
        )

        return jsonify({
            "success": False,
            "status": "unhealthy",
            "service": "AirSense India API",
            "error": str(error)
        }), 500
