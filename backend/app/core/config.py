import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


class Settings(BaseSettings):
    PROJECT_NAME: str = "DeepQuant: DRL Trading Dashboard API"
    API_V1_STR: str = "/api"
    DASHBOARD_RESULTS_DIR: str = "outputs/dashboard"

    BACKEND_CORS_ORIGINS: list = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    @property
    def dashboard_results_path(self) -> Path:
        return Path(BASE_DIR) / self.DASHBOARD_RESULTS_DIR


settings = Settings()