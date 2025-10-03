# services/phoenix_client.py - COMPLETE WORKING VERSION
import os
import logging
from typing import Optional, Dict, Any
import time


logger = logging.getLogger(__name__)


class SessionEnabledPhoenixManager:
    """Phoenix manager with proper session lifecycle management"""
    
    def __init__(self):
        self.phoenix_client = None
        self.tracer_provider = None
        self.active_sessions = {}  # Track active sessions
        self.prompts = {}
        self._initialize_phoenix()
    
    def _initialize_phoenix(self) -> None:
        """Initialize Phoenix with session support"""
        try:
            from phoenix.otel import register
            from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
            
            # Register with project name
            self.tracer_provider = register(
                project_name="regulatory-rag-backend",
                endpoint="http://localhost:6006/v1/traces"
            )
            
            # Instrument components
            LlamaIndexInstrumentor().instrument(tracer_provider=self.tracer_provider)
            
            # Try CrewAI instrumentation
            try:
                from openinference.instrumentation.crewai import CrewAIInstrumentor
                CrewAIInstrumentor().instrument(tracer_provider=self.tracer_provider)
                logger.info("✅ CrewAI instrumentation enabled")
            except ImportError:
                logger.info("CrewAI instrumentation not available")
            
            self._initialize_regulatory_prompts()
            logger.info("✅ Phoenix initialized with session support")
            
        except Exception as e:
            logger.error(f"Phoenix initialization failed: {e}")
            self.tracer_provider = None
    
    def create_session(self, session_id: str, user_id: str = None) -> Dict[str, Any]:
        """Create a new Phoenix session with proper metadata"""
        try:
            session_info = {
                "session_id": session_id,
                "user_id": user_id or "anonymous",
                "created_at": time.time(),
                "query_count": 0,
                "total_tokens": 0,
                "start_time": time.time()
            }
            
            self.active_sessions[session_id] = session_info
            
            # Create session span in Phoenix
            if self.tracer_provider:
                from opentelemetry import trace
                tracer = trace.get_tracer(__name__)
                
                # Create a session-level span
                session_span = tracer.start_span(f"session_{session_id}")
                session_span.set_attribute("session_id", session_id)
                session_span.set_attribute("user_id", user_id or "anonymous")
                session_span.set_attribute("session_type", "regulatory_rag")
                session_span.set_attribute("created_at", session_info["created_at"])
                
                # Store span for later completion
                session_info["span"] = session_span
            
            logger.info(f"🎯 Created Phoenix session: {session_id}")
            return session_info
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            return {"error": str(e)}
    
    def trace_query_with_session(self, query: str, session_id: str, agent_name: str = None):
        """Trace query execution within a session context"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.tracer_provider:
                    return func(*args, **kwargs)
                
                try:
                    # Ensure session exists
                    if session_id not in self.active_sessions:
                        self.create_session(session_id)
                    
                    # Update session stats
                    session = self.active_sessions[session_id]
                    session["query_count"] += 1
                    
                    from opentelemetry import trace
                    tracer = trace.get_tracer(__name__)
                    
                    # Create query span within session context
                    with tracer.start_as_current_span(f"query_{agent_name or 'unknown'}") as span:
                        # Set comprehensive attributes
                        span.set_attribute("session_id", session_id)
                        span.set_attribute("query", query[:200])
                        span.set_attribute("agent", agent_name or "unknown")
                        span.set_attribute("query_number", session["query_count"])
                        span.set_attribute("timestamp", time.time())
                        
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        execution_time = time.time() - start_time
                        
                        # Add result attributes
                        span.set_attribute("execution_time", execution_time)
                        span.set_attribute("result_length", len(str(result)))
                        span.set_attribute("status", "success")
                        
                        # Update session totals
                        session["total_tokens"] += len(query.split()) + len(str(result).split())
                        
                        return result
                        
                except Exception as e:
                    logger.error(f"Query tracing error: {e}")
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def log_agent_interaction_with_session(self, session_id: str, agent_name: str, action: str, inputs: Dict, outputs: Dict):
        """Log agent interaction within session context"""
        try:
            # Ensure session exists
            if session_id not in self.active_sessions:
                self.create_session(session_id)
            
            if not self.tracer_provider:
                logger.info(f"Session {session_id} - Agent {agent_name} - {action}")
                return
            
            from opentelemetry import trace
            tracer = trace.get_tracer(__name__)
            
            with tracer.start_as_current_span(f"agent_interaction_{agent_name}_{action}") as span:
                span.set_attribute("session_id", session_id)
                span.set_attribute("agent", agent_name)
                span.set_attribute("action", action)
                span.set_attribute("timestamp", time.time())
                
                # Log input/output details
                for key, value in inputs.items():
                    if isinstance(value, (str, int, float)):
                        span.set_attribute(f"input_{key}", str(value)[:1000])
                
                for key, value in outputs.items():
                    if isinstance(value, (str, int, float)):
                        span.set_attribute(f"output_{key}", str(value)[:1000])
            
            logger.info(f"✅ Logged agent interaction for session {session_id}")
            
        except Exception as e:
            logger.warning(f"Agent interaction logging failed: {e}")
    
    def close_session(self, session_id: str):
        """Close a Phoenix session"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                
                # Close session span if it exists
                if "span" in session:
                    span = session["span"]
                    span.set_attribute("total_queries", session["query_count"])
                    span.set_attribute("total_tokens", session["total_tokens"])
                    span.set_attribute("duration", time.time() - session["start_time"])
                    span.end()
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                logger.info(f"🔒 Closed Phoenix session: {session_id}")
            
        except Exception as e:
            logger.error(f"Session closure failed: {e}")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found", "session_id": session_id}
            
            session = self.active_sessions[session_id]
            return {
                "session_id": session_id,
                "query_count": session["query_count"],
                "total_tokens": session["total_tokens"],
                "duration": time.time() - session["start_time"],
                "created_at": session["created_at"],
                "user_id": session.get("user_id", "anonymous"),
                "status": "active"
            }
            
        except Exception as e:
            logger.error(f"Session stats error: {e}")
            return {"error": str(e)}
    
    def _initialize_regulatory_prompts(self):
        """Initialize regulatory prompt templates"""
        self.prompts = {
            "ia_control_analysis": {
                "template": """Analyze IA control requirements for {control_id}.
Context: {context}
Provide detailed compliance analysis.""",
                "variables": ["control_id", "context"]
            },
            "procurement_process_analysis": {
                "template": """Analyze procurement process {process_id}.
Context: {context}  
Provide comprehensive process guidance.""",
                "variables": ["process_id", "context"]
            }
        }
    
    def get_prompt_template(self, prompt_name: str, variables: Dict[str, Any] = None) -> str:
        """Get formatted prompt template"""
        try:
            if prompt_name not in self.prompts:
                return f"Regulatory analysis query: {variables.get('query', 'Please analyze the regulatory requirements.')}"
            
            template = self.prompts[prompt_name]["template"]
            if variables:
                return template.format(**variables)
            return template
            
        except Exception as e:
            logger.error(f"Prompt template error: {e}")
            return "Please provide regulatory analysis."

# Global instance
phoenix_manager = SessionEnabledPhoenixManager()

def get_phoenix_manager():
    return phoenix_manager

def initialize_phoenix():
    return phoenix_manager
