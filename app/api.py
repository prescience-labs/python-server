from flask import Flask

from .modules import *
from .settings import Settings

class Api():
    app = Flask(__name__)
    settings = Settings()

    def __init__(self):
        """API config
        This contains all settings for the Flask API which serves as the single
        external connection point to this container.
        """

        # Add new blueprints here
        self.app.register_blueprint(mod_core)
        self.app.register_blueprint(mod_status)
