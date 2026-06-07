from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://quant:quant@localhost:5432/quant"

    # Auth
    auth_enabled: bool = True
    supabase_url: str = ""
    supabase_jwt_secret: str = ""

    # Paths
    quant_data_dir: Path = Path("website/data")

    # API
    cors_origins: list[str] = ["*"]
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Scheduler
    scheduler_enabled: bool = True
    scheduler_cron: str = "0 11 * * *"   # 06:00 ET = 11:00 UTC


settings = Settings()
