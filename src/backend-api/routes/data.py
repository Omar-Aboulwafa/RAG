from fastapi import FastAPI, APIRouter, Depends, UploadFile, status, Query
from fastapi.responses import JSONResponse
import os
from config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController, QueryController
import aiofiles
from models import ResponseSignal
import logging
from .schemas.data import QueryRequest, RegulatoryQueryRequest, ProcessRequest
import time
import uuid
import re
from typing import List, Dict, Any, Optional
from agents.crew_setup import process_persistent_query
from controllers.QueryController import QueryController, RegulatoryDocumentQueryController

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"],
)

@data_router.post("/upload/{project_id}")
async def upload_data(project_id: str, file: UploadFile,
                      app_settings: Settings = Depends(get_settings)):
    
    # validate the file properties
    data_controller = DataController()
    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": result_signal}
        )

    project_dir_path = ProjectController().get_project_path(project_id=project_id)
    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:
        logger.error(f"Error while uploading file: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.FILE_UPLOAD_FAILED.value}
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
            "file_id": file_id
        }
    )

@data_router.post("/process/{project_id}")
async def process_endpoint(project_id: str, process_request: ProcessRequest):
    file_id = process_request.file_id
    
    process_controller = ProcessController(project_id=project_id)
    
    # Use the existing pipeline method
    success = process_controller.process_document_pipeline(
        file_id=file_id,
        generate_context=True
    )

    if not success:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROCESSING_FAILED.value}
        )

    return JSONResponse(
        content={
            "signal": "processing_success",
            "message": "Document processed and stored successfully"
        }
    )

@data_router.post("/query/{project_id}")
async def enhanced_query_endpoint(project_id: str, query_request: QueryRequest):
    """Enhanced RAG query with metadata filtering"""
    try:
        query_controller = QueryController(project_id=project_id)
        
        # Build filters from request
        filters = query_request.filters or {}
        if query_request.doc_type:
            filters['doc_type'] = query_request.doc_type
        if query_request.priority:
            filters['priority'] = query_request.priority
        if query_request.process_group:
            filters['process_group'] = query_request.process_group
        if query_request.control_id:
            filters['control_id'] = query_request.control_id
        
        result = query_controller.query(
            query_text=query_request.query,
            include_sources=query_request.include_sources,
            filters=filters if filters else None
        )
        
        return JSONResponse(
            content={
                "status": "success",
                **result
            }
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": f"Query processing error: {str(e)}"
            }
        )

@data_router.get("/search/{project_id}")
async def vector_search_endpoint(
    project_id: str,
    query: str,
    top_k: int = Query(5, description="Number of results"),
    doc_type: Optional[str] = Query(None, description="Filter by document type")
):
    """Direct vector search without LLM"""
    try:
        query_controller = QueryController(project_id=project_id)
        
        filters = {}
        if doc_type:
            filters['doc_type'] = doc_type
            
        result = query_controller.simple_vector_search(
            query_str=query,
            top_k=top_k,
            filters=filters if filters else None
        )
        
        return JSONResponse(
            content={
                "status": "success",
                **result
            }
        )
        
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": f"Search error: {str(e)}"
            }
        )

@data_router.post("/retrieve/{project_id}")
async def query_with_retriever_details(project_id: str, query_request: QueryRequest):
    """Simple vector search without LLM generation"""
    try:
        query_controller = QueryController(project_id=project_id)
        result = query_controller.query(query_request.query)  # Use basic query method
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Retrieve failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Retrieve failed: {str(e)}"}
        )

@data_router.get("/health/{project_id}")
async def database_health(project_id: str):
    """Simple database health check"""
    import psycopg2
    from sqlalchemy import make_url
    
    try:
        settings = get_settings()
        
        # Test basic connection
        conn = psycopg2.connect(settings.DB_CONNECTION_STRING, connect_timeout=5)
        cursor = conn.cursor()
        
        # Test 1: Basic query
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()[0]
        
        # Test 2: Check if table exists
        table_name = f"data_document_chunks_project_{project_id}"
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
        """, (table_name,))
        table_exists = cursor.fetchone()[0]
        
        # Test 3: Count records if table exists
        data_count = 0
        embedding_dim = None
        if table_exists:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            data_count = cursor.fetchone()[0]
            
            # Check embedding dimensions
            cursor.execute(f"""
                SELECT array_length(embedding, 1) as dim_count
                FROM {table_name} 
                WHERE embedding IS NOT NULL 
                LIMIT 1;
            """)
            result = cursor.fetchone()
            embedding_dim = result[0] if result else None
        
        # Test 4: Vector extension check
        cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM pg_extension 
                WHERE extname = 'vector'
            );
        """)
        pgvector_installed = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return JSONResponse(content={
            "status": "healthy",
            "database_version": db_version,
            "table_name": table_name,
            "table_exists": table_exists,
            "data_count": data_count,
            "embedding_dimension": embedding_dim,
            "pgvector_extension": pgvector_installed,
            "connection": "success"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "connection": "failed"
            }
        )




@data_router.post("/advanced-crew/{project_id}")
async def advanced_regulatory_crew_query(project_id: str, query_request: RegulatoryQueryRequest):
    """Advanced ReAct-pattern regulatory query with comprehensive memory and validation"""
    try:
        from agents.crew import create_advanced_regulatory_crew
        
        # Auto-detect query type if not specified
        query_type = query_request.query_type or "general"
        if query_type == "general":
            if re.search(r'[MT]\d{1}\.\d{1}\.\d{1}', query_request.query):
                query_type = "ia_control"
            elif re.search(r'\d{1}\.\d{1}\.\d{1}\.\([IVX]+\)', query_request.query):
                query_type = "procurement_process"
            elif any(term in query_request.query.lower() for term in ['standard', 'bylaw', 'policy']):
                query_type = "standards_bylaws"
        
        # Create advanced crew with ReAct pattern
        crew = create_advanced_regulatory_crew(query_request.query, query_type)
        
        # Execute with detailed monitoring
        start_time = time.time()
        
        # Capture ReAct steps
        react_steps = []
        def step_monitor(step):
            react_steps.append({
                "agent": getattr(step, 'agent_name', 'unknown'),
                "thought": getattr(step, 'thought', ''),
                "action": getattr(step, 'action', ''),
                "observation": getattr(step, 'observation', ''),
                "timestamp": time.time() - start_time
            })
        
        # Execute crew
        result = crew.kickoff(inputs={
            'query': query_request.query,
            'query_type': query_type
        })
        
        processing_time = time.time() - start_time
        
        return JSONResponse(content={
            "status": "success",
            "query": query_request.query,
            "query_type": query_type,
            "response": str(result),
            "processing_time": round(processing_time, 3),
            "crew_type": "advanced_react_pattern",
            "agents_used": [
                "router_filter_agent",
                "ia_specialist" if query_type in ["ia_control", "general"] else None,
                "procurement_specialist" if query_type in ["procurement_process", "general"] else None,
                "compliance_synthesizer"
            ],
            "react_steps_count": len(react_steps),
            "memory_types": ["conversational", "working", "semantic"],
            "validation_enabled": True
        })
        
    except Exception as e:
        logger.error(f"Advanced crew error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


