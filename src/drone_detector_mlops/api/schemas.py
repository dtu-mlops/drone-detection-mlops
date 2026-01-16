from pydantic import BaseModel
from datetime import datetime


class PredictionScores(BaseModel):
    """Class probability scores."""

    drone: float
    bird: float


class Prediction(BaseModel):
    """Prediction result."""

    class_name: str  # "drone" or "bird"
    confidence: float
    scores: PredictionScores


class PredictionMetadata(BaseModel):
    """Metadata about the prediction."""

    model_version: str
    inference_time_ms: float
    timestamp: datetime


class PredictionResponse(BaseModel):
    """Complete prediction response."""

    prediction: Prediction
    metadata: PredictionMetadata


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    """API info response."""

    model_version: str
    uptime_seconds: float
