from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "dev"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "substitution-ranking"
    model_name: str = "totto-substitution-ranker"
    model_stage: str = "Production"

    top_k_default: int = 3
    target_season: str = "252"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    models_dir: str = "artifacts/models"
    reports_dir: str = "artifacts/reports"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def debug(self) -> bool:
        return self.app_debug

    @property
    def host(self) -> str:
        return self.app_host

    @property
    def port(self) -> int:
        return self.app_port


settings = Settings()
BASE_DIR = Path(__file__).resolve().parents[2]
