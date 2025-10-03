from enum import Enum

class ResponseSignal(Enum):
    # File Validation & Upload
    FILE_VALIDATED_SUCCESS = "file_validate_successfully"
    FILE_TYPE_NOT_SUPPORTED = "file_type_not_supported"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    FILE_UPLOAD_SUCCESS = "file_upload_success"
    FILE_UPLOAD_FAILED = "file_upload_failed"
    
    # Document Processing
    PROCESSING_SUCCESS = "processing_success"
    PROCESSING_FAILED = "processing_failed"
    
    # Enhanced Processing Signals
    METADATA_EXTRACTION_SUCCESS = "metadata_extraction_success"
    METADATA_EXTRACTION_FAILED = "metadata_extraction_failed"
    DOCUMENT_TYPE_DETECTED = "document_type_detected"
    DOCUMENT_TYPE_UNKNOWN = "document_type_unknown"
    ENHANCED_CHUNKING_SUCCESS = "enhanced_chunking_success"
    ENHANCED_CHUNKING_FAILED = "enhanced_chunking_failed"
    
    # File & Project Management
    NO_FILES_ERROR = "not_found_files"
    FILE_ID_ERROR = "no_file_found_with_this_id"
    PROJECT_NOT_FOUND_ERROR = "project_not_found"
    
    # Vector Database Operations
    INSERT_INTO_VECTORDB_ERROR = "insert_into_vectordb_error"
    INSERT_INTO_VECTORDB_SUCCESS = "insert_into_vectordb_success"
    VECTORDB_COLLECTION_RETRIEVED = "vectordb_collection_retrieved"
    VECTORDB_SEARCH_ERROR = "vectordb_search_error"
    VECTORDB_SEARCH_SUCCESS = "vectordb_search_success"
    
    # Enhanced Search & Filtering
    METADATA_FILTER_SUCCESS = "metadata_filter_success"
    METADATA_FILTER_ERROR = "metadata_filter_error"
    METADATA_FILTER_NO_RESULTS = "metadata_filter_no_results"
    HYBRID_SEARCH_SUCCESS = "hybrid_search_success"
    HYBRID_SEARCH_ERROR = "hybrid_search_error"
    
    # RAG Query Operations
    RAG_ANSWER_ERROR = "rag_answer_error"
    RAG_ANSWER_SUCCESS = "rag_answer_success"
    RAG_QUERY_FILTERED_SUCCESS = "rag_query_filtered_success"
    RAG_QUERY_FILTERED_ERROR = "rag_query_filtered_error"
    
    # Citation & Source Attribution
    CITATION_GENERATED_SUCCESS = "citation_generated_success"
    CITATION_METADATA_MISSING = "citation_metadata_missing"
    SOURCE_ATTRIBUTION_SUCCESS = "source_attribution_success"
    
    # Configuration & Settings
    SETTINGS_LOADED_SUCCESS = "settings_loaded_success"
    SETTINGS_VALIDATION_ERROR = "settings_validation_error"
    DATABASE_CONNECTION_SUCCESS = "database_connection_success"
    DATABASE_CONNECTION_FAILED = "database_connection_failed"
    
    # Document Type Specific Signals
    IA_STANDARD_PROCESSED = "ia_standard_processed"
    PROCUREMENT_MANUAL_PROCESSED = "procurement_manual_processed"
    PROCUREMENT_STANDARD_PROCESSED = "procurement_standard_processed"
    HR_BYLAW_PROCESSED = "hr_bylaw_processed"
    
    # Priority & Classification
    HIGH_PRIORITY_CONTROL_DETECTED = "high_priority_control_detected"
    PROCESS_GROUP_IDENTIFIED = "process_group_identified"
    BUSINESS_ROLE_EXTRACTED = "business_role_extracted"
    
    # Statistics & Monitoring
    STATISTICS_RETRIEVED_SUCCESS = "statistics_retrieved_success"
    STATISTICS_RETRIEVAL_ERROR = "statistics_retrieval_error"
    CHUNK_COUNT_UPDATED = "chunk_count_updated"
    METADATA_STATS_GENERATED = "metadata_stats_generated"

#

class DocumentType(Enum):
    IA_STANDARD = "IA Standard"
    PROCUREMENT_MANUAL = "Procurement Manual"
    PROCUREMENT_STANDARD = "Procurement Standard"
    HR_BYLAW = "HR Bylaw"
    UNKNOWN = "Unknown"

class Priority(Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"

class ProcessGroup(Enum):
    DCM = "DCM"
    S2C = "S2C"
    CLM = "CLM"
    R2P = "R2P"
    SPRM = "SPRM"
    R_AND_R = "R&R"
    MDM = "MDM"

class SearchType(Enum):
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"
    METADATA_FILTERED = "metadata_filtered"
    BM25_ONLY = "bm25_only"

class ChunkType(Enum):
    IA_STANDARD_CHUNK = "chunk_ia_standard"
    PROCUREMENT_MANUAL_CHUNK = "chunk_procurement_manual"
    PROCUREMENT_STANDARD_CHUNK = "chunk_procurement_standard"
    HR_BYLAW_CHUNK = "chunk_hr_bylaw"
    UNKNOWN_CHUNK = "chunk_unknown"
