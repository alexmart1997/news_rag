from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/news_rag",
    )
    rbc_request_delay: float = float(os.getenv("RBC_REQUEST_DELAY", "0.5"))
    rbc_request_timeout: int = int(os.getenv("RBC_REQUEST_TIMEOUT", "20"))


settings = Settings()
