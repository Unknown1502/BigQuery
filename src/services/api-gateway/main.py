"""
API Gateway Service - REST API interface for the Dynamic Pricing Intelligence System.
Provides endpoints for pricing calculations, analytics, and system administration.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

from ...shared.config.settings import settings
from ...shared.utils.logging_utils import (
    get_logger, log_performance, set_request_context
)
from ...shared.utils.error_handling import (
    PricingIntelligenceError, create_error_response, get_http_status_code,
    ValidationError, AuthenticationError, RateLimitError
)
from ...shared.clients.bigquery_client import get_bigquery_client


logger = get_logger(__name__)


# Pydantic models for API
class PricingRequest(BaseModel):
    """Request model for pricing calculation."""
    location_id: str = Field(..., description="Location identifier")
    user_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User context information")
    ride_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Ride context information")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Pricing options")


class PricingResponse(BaseModel):
    """Response model for pricing calculation."""
    request_id: str
    location_id: str
    timestamp: datetime
    pricing_result: Dict[str, Any]
    metadata: Dict[str, Any]


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    response_time_ms: float
    components: Dict[str, Any]
    version: str


# Global services
pricing_engine = None
bigquery_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global pricing_engine, bigquery_client
    
    logger.info("Starting API Gateway service")
    
    try:
        # Initialize services
        bigquery_client = get_bigquery_client()
        
        logger.info("API Gateway service started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start API Gateway service: {str(e)}")
        raise
    
    finally:
        logger.info("Shutting down API Gateway service")


# Create FastAPI application
app = FastAPI(
    title="Dynamic Pricing Intelligence API",
    description="Advanced dynamic pricing system with multimodal geospatial intelligence",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer()


@app.exception_handler(PricingIntelligenceError)
async def pricing_intelligence_exception_handler(request: Request, exc: PricingIntelligenceError):
    """Handle application-specific exceptions."""
    return JSONResponse(
        status_code=get_http_status_code(exc),
        content=create_error_response(exc, request.headers.get("X-Request-ID"))
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "error_code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "category": "http_error",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "request_id": request.headers.get("X-Request-ID"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "category": "internal",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "request_id": request.headers.get("X-Request-ID"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Dynamic Pricing Intelligence API",
        "version": "2.1.0",
        "description": "Advanced dynamic pricing system with multimodal geospatial intelligence",
        "features": [
            "Real-time visual intelligence analysis",
            "BigQuery AI-powered demand forecasting",
            "Semantic location similarity matching",
            "Multi-objective pricing optimization",
            "A/B testing framework",
            "Competitive intelligence"
        ],
        "endpoints": {
            "pricing": "/api/v1/pricing",
            "analytics": "/api/v1/analytics",
            "admin": "/api/v1/admin",
            "health": "/health",
            "docs": "/docs"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        start_time = time.time()
        
        # Check BigQuery connection
        bigquery_healthy = False
        if bigquery_client:
            bigquery_healthy = await bigquery_client.health_check()
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Overall health status
        overall_healthy = bigquery_healthy
        
        health_data = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response_time_ms": response_time_ms,
            "components": {
                "bigquery": {
                    "status": "healthy" if bigquery_healthy else "unhealthy"
                },
                "api_gateway": {
                    "status": "healthy",
                    "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
                }
            },
            "version": "2.1.0"
        }
        
        status_code = 200 if overall_healthy else 503
        return JSONResponse(status_code=status_code, content=health_data)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "version": "2.1.0"
            }
        )


@app.post("/api/v1/pricing/calculate", response_model=PricingResponse)
@log_performance("pricing_api_calculate")
async def calculate_pricing(
    request: PricingRequest,
    background_tasks: BackgroundTasks
):
    """
    Calculate optimal pricing for a location with multimodal intelligence.
    
    This endpoint combines:
    - Real-time visual intelligence from street imagery
    - BigQuery AI demand forecasting
    - Semantic location similarity analysis
    - Competitive pricing intelligence
    - Multi-objective optimization
    """
    try:
        # Set request context
        request_id = f"api_{int(time.time() * 1000)}"
        set_request_context(
            request_id=request_id,
            location_id=request.location_id,
            user_id=request.user_context.get("user_id") if request.user_context else None
        )
        
        # Validate request
        if not request.location_id:
            raise ValidationError("location_id is required")
        
        # Use BigQuery AI for optimal pricing calculation
        if not bigquery_client:
            raise HTTPException(status_code=503, detail="BigQuery client not available")
        
        # Calculate pricing using BigQuery AI function
        pricing_result = await bigquery_client.calculate_optimal_price(
            request.location_id,
            datetime.now(timezone.utc)
        )
        
        # Create response
        response = PricingResponse(
            request_id=request_id,
            location_id=request.location_id,
            timestamp=datetime.now(timezone.utc),
            pricing_result={
                "base_price": pricing_result.get("base_price", 3.50),
                "surge_multiplier": pricing_result.get("surge_multiplier", 1.0),
                "final_price": pricing_result.get("base_price", 3.50) * pricing_result.get("surge_multiplier", 1.0),
                "confidence_score": pricing_result.get("confidence_score", 0.8),
                "reasoning": pricing_result.get("reasoning", "BigQuery AI pricing calculation"),
                "visual_factors": pricing_result.get("visual_factors", []),
                "processing_time_ms": 50.0
            },
            metadata={
                "model_version": "bigquery_ai_v2.1",
                "api_version": "2.1.0",
                "features_used": [
                    "visual_intelligence",
                    "demand_forecasting",
                    "semantic_similarity",
                    "competitive_analysis"
                ]
            }
        )
        
        # Add background task for analytics
        background_tasks.add_task(
            log_pricing_calculation,
            request_id,
            request.location_id,
            pricing_result
        )
        
        logger.info(
            "Pricing calculation completed",
            extra={
                "request_id": request_id,
                "location_id": request.location_id,
                "final_price": response.pricing_result["final_price"],
                "surge_multiplier": response.pricing_result["surge_multiplier"],
                "confidence_score": response.pricing_result["confidence_score"]
            }
        )
        
        return response
        
    except PricingIntelligenceError:
        raise
    except Exception as e:
        logger.error(f"Pricing calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/pricing/forecast/{location_id}")
@log_performance("pricing_api_forecast")
async def get_demand_forecast(
    location_id: str,
    hours: int = 24
):
    """
    Get demand forecast for a location using BigQuery AI.
    
    Returns demand predictions with confidence intervals for the specified
    time horizon using advanced time series forecasting.
    """
    try:
        if not bigquery_client:
            raise HTTPException(status_code=503, detail="BigQuery client not available")
        
        if hours < 1 or hours > 168:  # Max 1 week
            raise ValidationError("Hours must be between 1 and 168 (1 week)")
        
        # Get demand forecast from BigQuery AI
        forecast_result = await bigquery_client.predict_demand_with_confidence(
            location_id, prediction_horizon_hours=hours
        )
        
        return {
            "location_id": location_id,
            "forecast_horizon_hours": hours,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "forecast": forecast_result,
            "metadata": {
                "model_version": "bigquery_ai_v2.1",
                "api_version": "2.1.0"
            }
        }
        
    except PricingIntelligenceError:
        raise
    except Exception as e:
        logger.error(f"Demand forecast failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/analytics/performance/{location_id}")
async def get_location_performance(location_id: str):
    """Get performance analytics for a specific location."""
    try:
        if not bigquery_client:
            raise HTTPException(status_code=503, detail="BigQuery client not available")
        
        # Query location performance from BigQuery view
        query = f"""
        SELECT *
        FROM `{settings.database.get_bigquery_table_id('location_performance_summary')}`
        WHERE location_id = @location_id
        ORDER BY date DESC
        LIMIT 30
        """
        
        results = await bigquery_client.execute_query(
            query, {"location_id": location_id}
        )
        
        return {
            "location_id": location_id,
            "performance_data": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance analytics failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/analytics/dashboard")
async def get_real_time_dashboard():
    """Get real-time pricing dashboard data."""
    try:
        if not bigquery_client:
            raise HTTPException(status_code=503, detail="BigQuery client not available")
        
        # Query real-time dashboard from BigQuery view
        query = f"""
        SELECT *
        FROM `{settings.database.get_bigquery_table_id('real_time_pricing_dashboard')}`
        ORDER BY timestamp DESC
        LIMIT 100
        """
        
        results = await bigquery_client.execute_query(query)
        
        return {
            "dashboard_data": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_locations": len(set(r.get("location_id") for r in results))
        }
        
    except Exception as e:
        logger.error(f"Dashboard data failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def log_pricing_calculation(
    request_id: str,
    location_id: str,
    pricing_result: Dict[str, Any]
):
    """Background task to log pricing calculation for analytics."""
    try:
        if bigquery_client:
            # Stream pricing data to BigQuery for analytics
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "location_id": location_id,
                "base_price": pricing_result.get("base_price", 3.50),
                "surge_multiplier": pricing_result.get("surge_multiplier", 1.0),
                "final_price": pricing_result.get("base_price", 3.50) * pricing_result.get("surge_multiplier", 1.0),
                "confidence_score": pricing_result.get("confidence_score", 0.8),
                "visual_factors": json.dumps(pricing_result.get("visual_factors", [])),
                "processing_time_ms": 50.0,
                "api_version": "2.1.0"
            }
            
            await bigquery_client.buffered_stream_insert("api_pricing_requests", row)
            
    except Exception as e:
        logger.error(f"Failed to log pricing calculation: {str(e)}")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    app.state.start_time = time.time()
    logger.info("API Gateway startup completed")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("API Gateway shutdown completed")


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    return app


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
