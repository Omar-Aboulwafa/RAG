import os
import logging
from typing import Optional, List, Any
import google.generativeai as genai
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig
from langchain_core.outputs import LLMResult, Generation

logger = logging.getLogger(__name__)


class GeminiLLM(BaseRagasLLM):
    """Direct Gemini SDK wrapper for RAGAs"""
    
    def __init__(
        self, 
        model_name: str = "models/gemini-2.5-flash",
        api_key: Optional[str] = None,
        run_config: Optional[RunConfig] = None
    ):
        super().__init__(run_config=run_config or RunConfig())
        
        # Configure Gemini
        if api_key:
            genai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        else:
            raise ValueError("GOOGLE_API_KEY must be set")
        
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        self.generation_config = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        
        logger.info(f"Initialized Gemini LLM: {model_name}")
    
    def is_finished(self, response: Any) -> bool:
        """Check if generation is finished"""
        return True
    
    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        """Generate text synchronously"""
        config = self.generation_config.copy()
        if temperature is not None:
            config["temperature"] = temperature
        
        # Convert prompt to string
        if hasattr(prompt, 'to_string'):
            prompt_str = prompt.to_string()
        elif hasattr(prompt, 'prompt_str'):
            prompt_str = prompt.prompt_str
        elif isinstance(prompt, str):
            prompt_str = prompt
        else:
            prompt_str = str(prompt)
        
        try:
            response = self.model.generate_content(
                prompt_str,
                generation_config=config
            )
            
            # Create Generation objects properly
            generation = Generation(text=response.text)
            
            return LLMResult(
                generations=[[generation]],
                llm_output=None
            )
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            # Return error as Generation object
            error_gen = Generation(text=f"Error: {str(e)}")
            return LLMResult(
                generations=[[error_gen]],
                llm_output={"error": str(e)}
            )
    
    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        """Generate text asynchronously"""
        config = self.generation_config.copy()
        if temperature is not None:
            config["temperature"] = temperature
        
        # Convert prompt to string
        if hasattr(prompt, 'to_string'):
            prompt_str = prompt.to_string()
        elif hasattr(prompt, 'prompt_str'):
            prompt_str = prompt.prompt_str
        elif isinstance(prompt, str):
            prompt_str = prompt
        else:
            prompt_str = str(prompt)
        
        try:
            response = await self.model.generate_content_async(
                prompt_str,
                generation_config=config
            )
            
            # Create Generation objects properly
            generation = Generation(text=response.text)
            
            return LLMResult(
                generations=[[generation]],
                llm_output=None
            )
            
        except Exception as e:
            logger.error(f"Gemini async generation error: {e}")
            # Return error as Generation object
            error_gen = Generation(text=f"Error: {str(e)}")
            return LLMResult(
                generations=[[error_gen]],
                llm_output={"error": str(e)}
            )


class GeminiEmbeddings:
    """Direct Gemini SDK embeddings wrapper"""
    
    def __init__(
        self,
        model_name: str = "models/text-embedding-004",
        api_key: Optional[str] = None
    ):
        if api_key:
            genai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        
        self.model_name = model_name
        logger.info(f"Initialized Gemini Embeddings: {model_name}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if not text or len(text.strip()) == 0:
            text = "empty"
        
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [0.0] * 768
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        embeddings = []
        for text in texts:
            if not text or len(text.strip()) == 0:
                text = "empty"
            
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                embeddings.append([0.0] * 768)
        
        return embeddings
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async embed query"""
        return self.embed_query(text)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed documents"""
        return self.embed_documents(texts)
