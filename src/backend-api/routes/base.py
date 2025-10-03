from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

base_router = APIRouter(
    prefix="/api/v1",
    tags=["base"]
)

@base_router.get("/status")
async def status():
    """API status endpoint"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "features": {
            "phoenix_tracing": True,
            "crewai_agents": True,
            "openwebui_compatible": True,
            "regulatory_rag": True
        }
    }

@base_router.get("/memory/sessions")
async def list_memory_sessions():
    """List available memory sessions"""
    # Placeholder for session management
    return {
        "sessions": [
            {
                "id": "default",
                "created": "2025-09-29T00:00:00Z",
                "project": "omar_aboulwafa",
                "message_count": 0
            }
        ]
    }

@base_router.post("/memory/sessions/{session_id}/clear")
async def clear_session_memory(session_id: str):
    """Clear memory for specific session"""
    logger.info(f"Clearing memory for session: {session_id}")
    return {"status": "success", "session_id": session_id, "message": "Memory cleared"}
