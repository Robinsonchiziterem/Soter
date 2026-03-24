"""
Soter AI Service - FastAPI Application
Main entry point for the AI service layer
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from contextlib import asynccontextmanager
import logging
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("Starting up Soter AI Service...")
    
    # Validate API keys on startup
    if not settings.validate_api_keys():
        logger.warning("No API keys configured. AI features will be unavailable.")
    else:
        provider = settings.get_active_provider()
        logger.info(f"AI provider configured: {provider}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Soter AI Service...")


app = FastAPI(
    title="Soter AI Service",
    description="AI service layer for Soter platform using FastAPI",
    version="0.1.0",
    lifespan=lifespan
)


class Claim(BaseModel):
    id: int
    amount: float
    ipAddress: str = None
    evidenceRef: str = None


class BatchRequest(BaseModel):
    claims: List[Claim]


def encode_ip(ip: str) -> int:
    if not ip:
        return 0
    # Simple deterministic hash to int
    return sum([int(part) for part in ip.split('.') if part.isdigit()])


def encode_evidence(evidence: str) -> int:
    if not evidence:
        return 0
    # Use built‑in hash, modulo to keep range reasonable
    return abs(hash(evidence)) % 10000


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service availability
    """
    return {
        "status": "healthy",
        "service": "soter-ai-service",
        "version": "0.1.0"
    }


@app.get("/")
async def root():
    """
    Root endpoint with service information
    """
    return {
        "service": "Soter AI Service",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/analyze-batch")
async def analyze_batch(request: BatchRequest):
    if not request.claims:
        raise HTTPException(status_code=400, detail="No claims provided")

    # Build DataFrame
    df = pd.DataFrame([
        {
            "id": c.id,
            "amount": c.amount,
            "ip": encode_ip(c.ipAddress),
            "evidence": encode_evidence(c.evidenceRef)
        } for c in request.claims
    ])

    # Scale numeric features
    scaler = StandardScaler()
    features = scaler.fit_transform(df[["amount", "ip", "evidence"]])

    # DBSCAN clustering – eps and min_samples tuned for demo purposes
    clustering = DBSCAN(eps=0.8, min_samples=3).fit(features)
    df["cluster"] = clustering.labels_

    # Compute risk score: outliers (label -1) get high risk, tight clusters get lower risk
    def risk_for_label(label: int) -> float:
        if label == -1:
            return 0.95
        # Count members in cluster
        size = (df["cluster"] == label).sum()
        # Smaller clusters are riskier
        return max(0.2, 1.0 - (size / len(df)))

    df["fraudRiskScore"] = df["cluster"].apply(risk_for_label)

    # Return list of results
    results = df[["id", "fraudRiskScore"]].to_dict(orient="records")
    return results


# Global error handler for HTTP exceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """
    Global error handler for HTTP exceptions
    """
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "service": "soter-ai-service"
        }
    )


# Global error handler for general exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """
    Global error handler for unhandled exceptions
    """
    logger.error(f"Unhandled Exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": "Internal server error",
            "service": "soter-ai-service"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
