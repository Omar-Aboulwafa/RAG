from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List  

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.indexer")
    
    # Application Settings
    APP_NAME: str
    APP_VERSION: str
    
    # File Processing Settings  
    FILE_ALLOWED_TYPES: str = "pdf,docx,txt"
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int
    
    # Upload Settings 
    UPLOAD_DIRECTORY_PATH: str = "/media/omaraboulwafa/Data Science/MY_WORK/asessment-RAG-1/RAG/src/document-indexer/assets/files"
    
    # Database Settings
    DB_CONNECTION_STRING: str
    
    # Ollama Settings
    OLLAMA_BASE_URL: str
    DEFAULT_LLM_MODEL: str
    DEFAULT_EMBEDDING_MODEL: str
    LITELLM_PROVIDER: str
    

    ENABLE_RERANKING: bool = False
    RETRIEVAL_TOP_K: int = 20
    RETRIEVAL_TOP_N: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    RERANKER_MODEL: str = "mixedbread-ai/mxbai-rerank-base-v1"
    
    # Enhanced Metadata Settings
    ENABLE_ENHANCED_METADATA: bool = True
    METADATA_EXTRACTION_TIMEOUT: int = 30
    
    # Document-Specific Chunking Settings 
    IA_STANDARD_CHUNK_SIZE: int = 2048
    IA_STANDARD_OVERLAP: int = 200
    PROCUREMENT_MANUAL_CHUNK_SIZE: int = 1024
    PROCUREMENT_MANUAL_OVERLAP: int = 512
    PROCUREMENT_STANDARD_CHUNK_SIZE: int = 1536
    PROCUREMENT_STANDARD_OVERLAP: int = 256
    HR_BYLAW_CHUNK_SIZE: int = 2000
    HR_BYLAW_OVERLAP: int = 400
    
    # Citation and Source Attribution Settings
    ENABLE_CITATION_TRACKING: bool = True
    CITATION_FORMAT: str = "regulatory"
    INCLUDE_METADATA_IN_CITATION: bool = True
    
    # Contextual Enhancement Settings
    ENABLE_CONTEXTUAL_ENHANCEMENT: bool = True
    CONTEXT_WINDOW_SIZE: int = 3
    
    # Advanced Filtering Settings
    ENABLE_METADATA_FILTERING: bool = True
    
    def get_allowed_types(self) -> List[str]:
        """Get list of allowed file extensions"""
        return [ext.strip().lower() for ext in self.FILE_ALLOWED_TYPES.split(',')]
    
    def get_project_upload_path(self, project_id: str) -> str:
        """Get the full upload path for a specific project"""
        import os
        return os.path.join(self.UPLOAD_DIRECTORY_PATH, project_id)
    
    def get_chunk_settings_for_doc_type(self, doc_type: str) -> dict:
        """Get chunk size and overlap settings based on document type"""
        chunk_settings = {
            "IA Standard": {
                "chunk_size": self.IA_STANDARD_CHUNK_SIZE,
                "chunk_overlap": self.IA_STANDARD_OVERLAP
            },
            "Procurement Manual": {
                "chunk_size": self.PROCUREMENT_MANUAL_CHUNK_SIZE,
                "chunk_overlap": self.PROCUREMENT_MANUAL_OVERLAP
            },
            "Procurement Standard": {
                "chunk_size": self.PROCUREMENT_STANDARD_CHUNK_SIZE,
                "chunk_overlap": self.PROCUREMENT_STANDARD_OVERLAP
            },
            "HR Bylaw": {
                "chunk_size": self.HR_BYLAW_CHUNK_SIZE,
                "chunk_overlap": self.HR_BYLAW_OVERLAP
            },
            "Unknown": {
                "chunk_size": self.FILE_DEFAULT_CHUNK_SIZE,
                "chunk_overlap": 200
            }
        }
        return chunk_settings.get(doc_type, chunk_settings["Unknown"])

def get_settings() -> Settings:
    return Settings()
