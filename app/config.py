from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE_URL = f"sqlite:///{(BASE_DIR / 'news_rag.db').as_posix()}"

@dataclass(slots=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    database_url: str = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)
    rbc_request_delay: float = float(os.getenv("RBC_REQUEST_DELAY", "0.5"))
    rbc_request_timeout: int = int(os.getenv("RBC_REQUEST_TIMEOUT", "20"))

settings = Settings()