from flask import Flask
from app.routes.health import health_bp
from app.routes.api import api_bp
from app.routes.ui import ui_bp


def create_app() -> Flask:
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_mapping(
        SECRET_KEY="change-me",
        JSON_SORT_KEYS=False,
    )
    app.register_blueprint(health_bp)
    app.register_blueprint(api_bp, url_prefix="/api/v1")
    app.register_blueprint(ui_bp)
    return app
