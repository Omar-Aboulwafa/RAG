"""
RAG API Application with FastAPI
Supports metadata filtering, hybrid search, and citation handling
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
import os
from contextlib import asynccontextmanager

# Import your routers
from routes.base import base_router
from routes.data import data_router
from helpers.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_rag_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 RAG API starting up...")
    
    # Load settings
    settings = get_settings()
    logger.info(f"📝 Loaded settings: {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Test database connection
    try:
        from controllers.ProcessController import ProcessController
        test_controller = ProcessController("health_check")
        stats = test_controller.get_database_statistics()
        logger.info("✅ Database connection verified")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
    
    logger.info("🎯 RAG API ready for requests!")
    
    yield
    
    # Shutdown
    logger.info("🛑 RAG API shutting down...")

# Create FastAPI app
app = FastAPI(
    title="RAG API",
    description="Advanced RAG API with metadata filtering and hybrid search",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"🔵 {request.method} {request.url.path} - Client: {request.client.host}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"🟢 {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
    
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    """System health check"""
    try:
        settings = get_settings()
        
        # Test database
        db_status = "unknown"
        try:
            from controllers.ProcessController import ProcessController
            test_controller = ProcessController("health_check")
            test_controller.get_database_statistics()
            db_status = "healthy"
        except Exception:
            db_status = "unhealthy"
        
        return {
            "status": "healthy",
            "app": {
                "name": settings.APP_NAME,
                "version": settings.APP_VERSION
            },
            "services": {
                "database": db_status
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )

# Root endpoint
@app.get("/")
async def root():
    """Welcome endpoint"""
    settings = get_settings()
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "description": "Enhanced RAG API with metadata filtering",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "upload": "/api/v1/data/upload/{project_id}",
            "process": "/api/v1/data/process/{project_id}",
            "query": "/api/v1/data/query/{project_id}",
            "search": "/api/v1/data/search/{project_id}",
            "stats": "/api/v1/data/stats/{project_id}"
        }
    }

# Include routers
app.include_router(base_router)
app.include_router(data_router)

if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info",
        access_log=True,
        use_colors=True
    )
