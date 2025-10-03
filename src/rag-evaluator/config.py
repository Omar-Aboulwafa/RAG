import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvaluationSettings(BaseSettings):
    """Configuration for RAGAs evaluation with Gemini"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True
    )
    
    # Backend API Configuration
    BACKEND_API_URL: str = "http://localhost:8000"
    API_CHAT_ENDPOINT: str = "/v1/chat/completions"
    
    # Google Gemini Configuration
    GOOGLE_API_KEY: str = "AIzaSyC1PMI35riaAvnoouw3UYuuWc93awzdLjo"
    
    # Use Gemini 2.5 Flash - fastest and latest stable model
    EVALUATOR_LLM_MODEL: str = "models/gemini-2.5-flash" 
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    
    # Evaluation Metrics (start with 2 to avoid rate limits)
    METRICS_TO_EVALUATE: List[str] = [
        "faithfulness",
        "context_precision"
    ]
    
    # Dataset Configuration
    TEST_DATASET_PATH: str = "./eval_dataset.jsonl"
    RESULTS_DIR: str = "./results"
    
    # Settings
    BATCH_SIZE: int = 5
    TIMEOUT_SECONDS: int = 60
    DOCUMENT_TYPES: List[str] = ["HR Bylaw"]


def get_settings() -> EvaluationSettings:
    return EvaluationSettings()
