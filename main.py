from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Detect API", version="1.0.0")


class DetectRequest(BaseModel):
    data: Optional[str] = None
    image_url: Optional[str] = None


class DetectResponse(BaseModel):
    detected: bool
    confidence: Optional[float] = None
    message: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "detect-api"}


@app.post("/detect")
async def detect(request: DetectRequest):
    """Detection endpoint"""
    # Placeholder detection logic
    # Replace this with your actual detection implementation
    detected = request.data is not None or request.image_url is not None
    
    return DetectResponse(
        detected=detected,
        confidence=0.85 if detected else 0.0,
        message="Detection completed" if detected else "No data provided"
    )

