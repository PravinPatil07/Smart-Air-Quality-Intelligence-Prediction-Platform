
from flask import Flask


def create_app():

    app = Flask(__name__)

    app.config["SECRET_KEY"] = "airsense-local-development-key"

    # Load the trained model and datasets once.
    from app.services.airsense_service import airsense_service

    app.extensions["airsense"] = airsense_service

    from app.routes.main import main_bp
    from app.routes.api import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)

    return app
