# backend/app/config.py
from pydantic import BaseSettings
from typing import List

class Settings(BaseSettings):
    BYBIT_REST_BASE: str = "https://api.bybit.com"
    BYBIT_WS_PUBLIC: str = "wss://stream.bybit.com"
    SUPPORTED_PAIRS: List[str] = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT"]
    SEQUENCE_LENGTH: int = 60
    MODEL_PATH: str = "backend/app/models/lstm_model.h5"
    SCALER_PATH: str = "backend/app/models/scalers.joblib"
    EMBEDDING_SIZE: int = 8
    MIN_CONFIDENCE: float = 0.55
    EMA_FILTER_PERIOD: int = 50
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()