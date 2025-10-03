from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
import psycopg2
from fastapi import FastAPI, Request, HTTPException
import json
import urllib.parse
import os
import time
import logging
import hashlib

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ✅ SINGLE PHOENIX INITIALIZATION WITH SINGLETON PATTERN
logger.info("🔥 Initializing Phoenix...")
phoenix_manager = None
phoenix_enabled = False

try:
    # Prevent multiple initializations
    if not os.environ.get("PHOENIX_INITIALIZED"):
        from services.phoenix_client import initialize_phoenix
        phoenix_manager = initialize_phoenix()
        os.environ["PHOENIX_INITIALIZED"] = "true"
        phoenix_enabled = True
        logger.info("✅ Phoenix initialized successfully")
    else:
        from services.phoenix_client import get_phoenix_manager
        phoenix_manager = get_phoenix_manager()
        phoenix_enabled = True
        logger.info("✅ Phoenix manager retrieved (already initialized)")
        
except Exception as e:
    logger.warning(f"⚠️ Phoenix initialization failed: {e}")
    phoenix_manager = None
    phoenix_enabled = False

# Import settings and routes
from config import get_settings
settings = get_settings()

# Import routes
from routes.data import data_router

# Create base router for basic endpoints
from fastapi import APIRouter
base_router = APIRouter()

@base_router.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "phoenix_tracing": phoenix_enabled,
        "phoenix_sessions": len(phoenix_manager.active_sessions) if phoenix_manager and hasattr(phoenix_manager, 'active_sessions') else 0,
        "endpoints": {
            "docs": "/docs",
            "health": "/health", 
            "data_api": "/api/v1/data/health",
            "phoenix": "http://localhost:6006" if phoenix_enabled else None,
            "sessions": "/api/v1/phoenix/sessions" if phoenix_enabled else None
        }
    }

@base_router.get("/health")
async def health_check():
    """Comprehensive health check with Phoenix session info"""
    health_status = {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "database": "unknown",
        "ollama": "unknown",
        "phoenix": {
            "enabled": phoenix_enabled,
            "active_sessions": len(phoenix_manager.active_sessions) if phoenix_manager and hasattr(phoenix_manager, 'active_sessions') else 0
        }
    }
    
    # Database check
    try:
        conn = psycopg2.connect(settings.DB_CONNECTION_STRING, connect_timeout=3)
        conn.close()
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = "disconnected"
        health_status["status"] = "degraded"
        logger.warning(f"Database health check failed: {e}")

    # Ollama check
    try:
        import requests
        resp = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3)
        health_status["ollama"] = "connected" if resp.status_code == 200 else "disconnected"
    except Exception as e:
        health_status["ollama"] = "disconnected"
        logger.warning(f"Ollama health check failed: {e}")

    return health_status

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} starting up...")
    if phoenix_enabled:
        logger.info("📊 Phoenix tracing is active - check http://localhost:6006")
        logger.info(f"📋 Phoenix manager type: {type(phoenix_manager).__name__}")
    yield
    logger.info("🛑 Application shutting down...")
    
    # Clean up Phoenix sessions
    if phoenix_manager and hasattr(phoenix_manager, 'active_sessions'):
        logger.info(f"🔒 Closing {len(phoenix_manager.active_sessions)} active Phoenix sessions...")
        for session_id in list(phoenix_manager.active_sessions.keys()):
            try:
                phoenix_manager.close_session(session_id)
            except Exception as e:
                logger.warning(f"Error closing session {session_id}: {e}")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise RAG Backend with CrewAI and Phoenix Observability",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(base_router)
app.include_router(data_router)

# OpenWebUI compatible endpoints
@app.get("/v1/models")
def list_models():
    """OpenAI-compatible endpoint to list available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "rag",
                "object": "model",
                "created": 1677652288,
                "owned_by": "regulatory-rag",
                "permission": [],
                "root": "regulatory-rag",
                "parent": None,
                "max_tokens": 131072,
                "context_length": 131072
            }
        ]
    }

def generate_session_id(request: Request) -> str:
    """Generate deterministic session ID based on client info"""
    client_host = request.client.host if request.client else "unknown"
    # Create 5-minute session windows
    time_window = int(time.time() / 300)  # 5-minute blocks
    session_data = f"{client_host}_{time_window}"
    return hashlib.md5(session_data.encode()).hexdigest()[:12]

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenWebUI compatible chat completions with Phoenix session tracing"""
    try:
        # Get the raw body
        body = await request.body()
        
        # Try to parse as JSON first
        try:
            if isinstance(body, bytes):
                body_str = body.decode('utf-8')
            else:
                body_str = str(body)
                
            # Handle URL-encoded data
            if body_str.startswith('%7B') or '=' in body_str:
                # URL decode the body
                decoded_body = urllib.parse.unquote(body_str)
                # Remove trailing '=' if present
                if decoded_body.endswith('='):
                    decoded_body = decoded_body[:-1]
                request_data = json.loads(decoded_body)
            else:
                # Direct JSON parsing
                request_data = json.loads(body_str)
                
        except json.JSONDecodeError:
            # Fallback: try to get from form data
            form = await request.form()
            if form:
                # Convert form data to dict
                request_data = dict(form)
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid request format - expected JSON"}
                )

        # Extract required fields
        messages = request_data.get("messages", [])
        if not messages:
            return JSONResponse(
                status_code=400,
                content={"error": "No messages provided"}
            )

        user_message = messages[-1].get("content", "").strip()
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty user message"}
            )

        logger.info(f"💬 Processing chat query: {user_message[:100]}...")

        # GENERATE SESSION ID FROM REQUEST
        session_id = generate_session_id(request)
        
        # PHOENIX MANAGER TO CREW SETUP
        from agents.crew_setup import process_persistent_query
        result = process_persistent_query(
            query=user_message,
            session_id=session_id,
            request_context={
                "user_message": user_message, 
                "session_id": session_id,
                "client_host": request.client.host if request.client else "unknown",
                "model": request_data.get("model", "rag")
            },
            phoenix_manager=phoenix_manager  # ✅ Pass the working Phoenix manager
        )

        # Return OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{session_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model", "regulatory-rag"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": str(result)
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(str(result).split()),
                "total_tokens": len(user_message.split()) + len(str(result).split())
            }
        }

        return response

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

# Phoenix Session Management Endpoints
@app.get("/api/v1/phoenix/sessions")
async def list_phoenix_sessions():
    """List all active Phoenix sessions with stats"""
    if not phoenix_manager or not hasattr(phoenix_manager, 'active_sessions'):
        return JSONResponse(
            status_code=503,
            content={"error": "Phoenix not available or sessions not supported"}
        )
    
    sessions_info = []
    for session_id, session_data in phoenix_manager.active_sessions.items():
        sessions_info.append({
            "session_id": session_id,
            "user_id": session_data.get("user_id", "anonymous"),
            "query_count": session_data.get("query_count", 0),
            "total_tokens": session_data.get("total_tokens", 0),
            "created_at": session_data.get("created_at", 0),
            "duration": time.time() - session_data.get("start_time", time.time()),
            "status": "active"
        })
    
    return {
        "sessions": sessions_info,
        "total_sessions": len(sessions_info),
        "phoenix_url": "http://localhost:6006"
    }

@app.get("/api/v1/phoenix/sessions/{session_id}")
async def get_phoenix_session(session_id: str):
    """Get detailed info for a specific Phoenix session"""
    if not phoenix_manager or not hasattr(phoenix_manager, 'get_session_stats'):
        return JSONResponse(
            status_code=503,
            content={"error": "Phoenix not available"}
        )
    
    session_stats = phoenix_manager.get_session_stats(session_id)
    if "error" in session_stats:
        return JSONResponse(
            status_code=404,
            content=session_stats
        )
    
    return session_stats

@app.delete("/api/v1/phoenix/sessions/{session_id}")
async def close_phoenix_session(session_id: str):
    """Manually close a Phoenix session"""
    if not phoenix_manager or not hasattr(phoenix_manager, 'close_session'):
        return JSONResponse(
            status_code=503,
            content={"error": "Phoenix not available"}
        )
    
    try:
        phoenix_manager.close_session(session_id)
        return {"message": f"Session {session_id} closed successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to close session: {str(e)}"}
        )

# Existing project management endpoints
from services.project_manager import get_project_manager

@app.get("/api/v1/projects")
async def list_projects():
    """List all available projects with statistics"""
    project_manager = get_project_manager()
    projects = project_manager.get_projects_with_stats()
    return {
        "projects": projects,
        "total_projects": len(projects),
        "active_projects": len([p for p in projects.values() if p['status'] == 'active'])
    }

@app.get("/api/v1/projects/{project_id}/stats")
async def get_project_stats(project_id: str):
    """Get detailed statistics for a specific project"""
    project_manager = get_project_manager()
    projects = project_manager.get_projects_with_stats()
    
    if project_id not in projects:
        return JSONResponse(
            status_code=404,
            content={"error": f"Project '{project_id}' not found"}
        )
    
    return projects[project_id]

# Phoenix redirect
@app.get("/phoenix")
async def phoenix_redirect():
    """Redirect to Phoenix UI"""
    if not phoenix_enabled:
        return JSONResponse(
            status_code=503,
            content={"error": "Phoenix tracing is not enabled"}
        )
    
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="http://localhost:6006")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )