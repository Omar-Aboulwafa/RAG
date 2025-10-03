from pydantic import BaseModel
from typing import Optional, Dict, Any

class QueryRequest(BaseModel):
    """Schema for query requests"""
    query: str
    top_k: Optional[int] = 5
    include_sources: Optional[bool] = True
    filters: Optional[Dict[str, Any]] = None
    doc_type: Optional[str] = None
    priority: Optional[str] = None
    process_group: Optional[str] = None
    control_id: Optional[str] = None

class ProcessRequest(BaseModel):
    """Schema for processing requests"""
    file_id: str
    use_enhanced_chunking: Optional[bool] = True

class RegulatoryQueryRequest(BaseModel):
    """Schema for regulatory-specific queries"""
    query: str
    query_type: Optional[str] = "general"
    entity_id: Optional[str] = None
    top_k: Optional[int] = 5
    filters: Optional[Dict[str, Any]] = None
