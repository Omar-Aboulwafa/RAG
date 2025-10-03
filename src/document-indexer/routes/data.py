from fastapi import FastAPI, APIRouter, Depends, UploadFile, status, Query
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController
import aiofiles
from models import ResponseSignal
import logging
from .schemas.data import ProcessRequest
import time
import uuid
from typing import List, Dict, Any, Optional

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"],
)

@data_router.post("/upload/{project_id}")
async def upload_data(project_id: str, file: UploadFile,
                     app_settings: Settings = Depends(get_settings)):
    """Upload document file to project using configured upload directory"""
    
    # Validate project ID
    if not project_id or project_id.strip() == "":
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": "invalid_project_id",
                "message": "Project ID is required and cannot be empty"
            }
        )
    
    # Sanitize project ID
    import re
    project_id = re.sub(r'[^a-zA-Z0-9_-]', '_', project_id.strip())
    
    logger.info(f"Upload request for project: {project_id}, file: {file.filename}")
    
    # Validate the file properties
    data_controller = DataController()
    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)
    
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal,
                "message": "File validation failed",
                "project_id": project_id
            }
        )

    # Use the configured upload directory
    try:
        project_upload_dir = app_settings.get_project_upload_path(project_id)
        logger.info(f"Project upload directory: {project_upload_dir}")
        
        # Ensure directory exists
        os.makedirs(project_upload_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        import time
        timestamp = int(time.time())
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{os.path.splitext(file.filename)[0]}_{timestamp}{file_extension}"
        
        # Full file path
        file_path = os.path.join(project_upload_dir, unique_filename)
        
        logger.info(f"Full file path: {file_path}")
        
    except Exception as e:
        logger.error(f"Error setting up upload directory: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": "directory_setup_error",
                "message": f"Could not set up upload directory: {str(e)}"
            }
        )

    # Save the file
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
                
        # Verify file was saved
        if not os.path.exists(file_path):
            raise Exception("File was not saved successfully")
            
        file_size = os.path.getsize(file_path)
        logger.info(f"File saved successfully: {file_path} ({file_size} bytes)")
        
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value,
                "message": f"File save failed: {str(e)}"
            }
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
            "file_id": unique_filename,  # Use the unique filename as file_id
            "project_id": project_id,
            "file_path": file_path,
            "upload_directory": project_upload_dir,
            "file_size": file_size,
            "original_filename": file.filename,
            "message": f"File uploaded successfully to {project_upload_dir}"
        }
    )


@data_router.post("/process/{project_id}")
async def process_endpoint(project_id: str, process_request: ProcessRequest):
    """Process document with enhanced metadata extraction and chunking for indexing"""
    file_id = process_request.file_id
    try:
        process_controller = ProcessController(project_id=project_id)
        
        # Use enhanced document processing pipeline
        success = process_controller.process_document_pipeline(
            file_id=file_id,
            use_enhanced_chunking=True  # Enable enhanced chunking
        )
        
        if not success:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.PROCESSING_FAILED.value,
                    "message": "Document processing failed"
                }
            )
        
        # Get processing statistics
        stats = process_controller.get_database_statistics()
        return JSONResponse(
            content={
                "signal": "processing_success",
                "message": "Document processed and indexed with enhanced metadata extraction",
                "file_id": file_id,
                "project_id": project_id,
                "indexing_stats": {
                    "total_chunks": stats.get('total_chunks', 0),
                    "document_types_detected": stats.get('doc_type_distribution', {}),
                    "priority_distribution": stats.get('priority_distribution', {}),
                    "unique_files": stats.get('unique_files', 0)
                }
            }
        )
        
    except Exception as e:
        # ✅ CRITICAL FIX: Show full traceback
        import traceback
        full_traceback = traceback.format_exc()
        logger.error(f"Error in processing document {file_id}:")
        logger.error(full_traceback)  # ✅ Log full trace
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": ResponseSignal.PROCESSING_FAILED.value,
                "error": str(e),
                "error_type": type(e).__name__,  # ✅ NEW: Show error type
                "traceback": full_traceback,  # ✅ NEW: Include in response (for debugging only!)
                "message": "Internal processing error"
            }
        )


@data_router.get("/stats/{project_id}")
async def get_project_statistics(project_id: str):
    """Get comprehensive indexing statistics for project"""
    try:
        process_controller = ProcessController(project_id=project_id)
        stats = process_controller.get_database_statistics()
        
        return JSONResponse(
            content={
                "status": "success",
                "project_id": project_id,
                "indexing_statistics": {
                    "total_chunks_indexed": stats.get('total_chunks', 0),
                    "unique_files_processed": stats.get('unique_files', 0),
                    "document_types": stats.get('doc_type_distribution', {}),
                    "priority_levels": stats.get('priority_distribution', {}),
                    "average_chunk_length": stats.get('avg_chunk_length', 0),
                    "database_table": stats.get('table_name', '')
                },
                "message": "Indexing statistics retrieved successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting project statistics: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": f"Statistics retrieval error: {str(e)}"
            }
        )

@data_router.get("/health/{project_id}")
async def check_project_health(project_id: str):
    """Check indexing system health for specific project"""
    try:
        process_controller = ProcessController(project_id=project_id)
        
        # Test database connectivity
        stats = process_controller.get_database_statistics()
        
        # Check if project has indexed content
        has_content = stats.get('total_chunks', 0) > 0
        
        return JSONResponse(
            content={
                "status": "healthy",
                "project_id": project_id,
                "has_indexed_content": has_content,
                "database_accessible": True,
                "total_indexed_chunks": stats.get('total_chunks', 0),
                "message": "Indexing system is operational"
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed for project {project_id}: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "project_id": project_id,
                "error": str(e),
                "message": "Indexing system has issues"
            }
        )

@data_router.delete("/project/{project_id}")
async def clear_project_index(project_id: str):
    """Clear all indexed data for a project (useful for re-indexing)"""
    try:
        # This would require a method in ProcessController to clear project data
        # For now, return a placeholder response
        
        return JSONResponse(
            content={
                "status": "success",
                "project_id": project_id,
                "message": "Project index clearing requested",
                "note": "Implementation depends on your data retention policies"
            }
        )
        
    except Exception as e:
        logger.error(f"Error clearing project index: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": f"Index clearing error: {str(e)}"
            }
        )
