# backend/app/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class PredictResponse(BaseModel):
    signal: str
    confidence: float
    pair: str
    last_updated: str
    reason: Optional[str] = None

class TrainRequest(BaseModel):
    pairs: Optional[List[str]] = None
    epochs: Optional[int] = 5
    batch_size: Optional[int] = 64