from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ProcessRequest(BaseModel):
    """Schema for requesting document processing (chunking and embedding)."""
    file_id: str
    chunk_size: Optional[int] = 100
    overlap_size: Optional[int] = 20
    do_reset: Optional[int] = 0

class QueryRequest(BaseModel):
    """Schema for requesting a RAG query against the knowledge base.
    project_id is now MANDATORY."""
    query: str
    project_id: str # Made mandatory
    top_k: Optional[int] = 5
    include_sources: Optional[bool] = True
    similarity_threshold: Optional[float] = 0.7
    
class QueryResponse(BaseModel):
    """Schema for the RAG query response."""
    answer: str
    contexts: Optional[List[str]] = None
    sources: Optional[List[Dict[str, str]]] = None

class MessageResponse(BaseModel):
    """Generic response for success/status messages."""
    status: str
    message: str
