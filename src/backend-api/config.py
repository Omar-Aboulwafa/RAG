from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.backend",
        extra="allow"
    )
    
    # Core settings
    APP_NAME: str = "Agentic_RAG_Backend"
    APP_VERSION: str = "1.0.0"
    DEFAULT_PROJECT_ID: str = "rag"
    PORT: int = 8000
    
    # Database
    DB_CONNECTION_STRING: str
    
    
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    DEFAULT_LLM_MODEL: str = "qwen3:0.6b-q4_K_M"
    DEFAULT_EMBEDDING_MODEL: str = "mxbai-embed-large:latest"
    ROUTER_LLM_MODEL: str = "llama3.1:8b"
    
    # Phoenix
    PHOENIX_COLLECTOR_ENDPOINT: str = "http://phoenix:6006/v1/traces"
    
    # File processing
    FILE_ALLOWED_TYPES: str = "pdf,docx,txt"
    FILE_MAX_SIZE: int = 10485760  # 10MB
    FILE_DEFAULT_CHUNK_SIZE: int = 1024
    
    # Reranking settings - ✅ ENABLED
    ENABLE_RERANKING: bool = True
    RETRIEVAL_TOP_K: int = 20      # Retrieve 20 candidates
    RETRIEVAL_TOP_N: int = 5       # Rerank to top 5
    SIMILARITY_THRESHOLD: float = 0.7
    RERANKER_MODEL: str = "mixedbread-ai/mxbai-rerank-base-v1"
    
    # CrewAI Memory Configuration
    CREWAI_STORAGE_DIR: str = "./storage/crewai_memory"
    CREWAI_MEMORY_BACKEND: str = "chroma"


def get_settings() -> Settings:
    """Singleton settings instance"""
    return Settings()
