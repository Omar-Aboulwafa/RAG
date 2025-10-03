import os
import logging
import time
from typing import Dict, Any
from controllers.QueryController import QueryController
from services.project_manager import get_project_manager
from openinference.instrumentation import using_session
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace

logger = logging.getLogger(__name__)
_session_counters = {}

def process_persistent_query(query: str, session_id: str = "default", request_context: dict = None, phoenix_manager=None):
    """Process query with Phoenix session tracking - MINIMAL INTEGRATION"""
    try:
        # ✅ SIMPLE SESSION TRACKING 
        if session_id not in _session_counters:
            _session_counters[session_id] = {"count": 0, "start_time": time.time()}
        _session_counters[session_id]["count"] += 1
        query_number = _session_counters[session_id]["count"]
        
        # ✅ PHOENIX SESSION WRAPPER 
        tracer = trace.get_tracer(__name__)
        
        try:
            # Create Phoenix session span
            with tracer.start_as_current_span(
                name="regulatory_assistant", 
                attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: "agent"}
            ) as session_span:
                
                # Set session attributes
                session_span.set_attribute(SpanAttributes.SESSION_ID, session_id)
                session_span.set_attribute(SpanAttributes.INPUT_VALUE, query)
                session_span.set_attribute("query.number", query_number)
                session_span.set_attribute("project.id", "regulatory-rag")
                
                # ✅ PHOENIX SESSION CONTEXT
                with using_session(session_id):
                    
                    project_manager = get_project_manager()
                    
                    if request_context:
                        project_id, routing_method = project_manager.extract_project_from_request(request_context)
                        logger.info(f"🎯 Project routed to: {project_id} (method: {routing_method})")
                    else:
                        project_id = project_manager._get_fallback_project()
                        routing_method = "fallback"
                        logger.info(f"🎯 Using fallback project: {project_id}")
                    
                    logger.info(f"Processing query for project: {project_id}, session: {session_id}")
                    
                    # ✅ PHOENIX LOGGING
                    if phoenix_manager:
                        try:
                            if hasattr(phoenix_manager, 'log_agent_interaction'):
                                phoenix_manager.log_agent_interaction(
                                    "Regulatory_Specialist",
                                    "query_processing",
                                    {"query": query[:100], "session": session_id, "query_number": query_number},
                                    {"project": project_id, "status": "processing"}
                                )
                        except Exception as e:
                            logger.warning(f"Phoenix logging failed: {e}")
                    
                    
                    result = _execute_enhanced_crew_query(
                        query=query,
                        session_id=session_id,
                        phoenix_manager=phoenix_manager,
                        query_number=query_number
                    )
                
                # ✅ SET PHOENIX OUTPUT ATTRIBUTES
                session_span.set_attribute(SpanAttributes.OUTPUT_VALUE, result[:500])
                session_span.set_attribute("response.length", len(result))
                session_span.set_attribute("query.status", "completed")
                
                return result
                
        except Exception as phoenix_error:
            logger.warning(f"Phoenix session error: {phoenix_error}")
            # ✅ FALLBACK 
            project_manager = get_project_manager()
            project_id = project_manager._get_fallback_project()
            
            return _execute_enhanced_crew_query(
                query=query,
                session_id=session_id,
                phoenix_manager=phoenix_manager,
                query_number=query_number
            )
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return f"I encountered an error processing your regulatory query: {str(e)}"

def _execute_enhanced_crew_query(query: str, session_id: str, phoenix_manager, query_number: int = 1) -> str:
    """YOUR EXISTING WORKING FUNCTION - ADD PHOENIX RAG SPAN"""
    
    # ✅ ADD RAG PROCESSING SPAN
    tracer = trace.get_tracer(__name__)
    
    try:
        with tracer.start_as_current_span(
            name="rag_processing",
            attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: "retriever"}
        ) as rag_span:
            
            rag_span.set_attribute(SpanAttributes.SESSION_ID, session_id)
            rag_span.set_attribute("rag.query_number", query_number)
            
            project_id = extract_project_from_session(session_id) or "rag"
            query_controller = QueryController(project_id)
            
            # QUERY PROCESSING 
            retrieval_result = query_controller.regulatory_hybrid_query(
                query_str=query,
                top_k=5
            )
            
            # SET RAG METRICS
            sources_count = len(retrieval_result.get('source_nodes', []))
            rag_span.set_attribute("rag.sources_found", sources_count)
            rag_span.set_attribute("rag.search_type", "regulatory_hybrid")
            
            # CLEAN RESPONSE EXTRACTION 
            if retrieval_result and retrieval_result.get('source_nodes'):
                source_nodes = retrieval_result.get('source_nodes', [])
                response_parts = []
                
                for i, source in enumerate(source_nodes[:3], 1):
                    try:
                        if isinstance(source, dict):
                            content = source.get('content', '') or source.get('text', '')
                            metadata = source.get('metadata', {})
                        else:
                            content = str(source)
                            metadata = {}
                        
                        # Clean up JSON-like content
                        if '"text":' in content or '"content":' in content:
                            import re
                            text_match = re.search(r'"text":\s*"([^"]*)"', content)
                            if text_match:
                                content = text_match.group(1)
                            else:
                                content = re.sub(r'[{}"_\[\]:,]', ' ', content)
                                content = re.sub(r'\s+', ' ', content).strip()
                        
                        if content and len(content.strip()) > 10:
                            doc_type = metadata.get('doc_type', 'Document')
                            response_parts.append(f"\n**{doc_type}:**\n{content[:500]}")
                            
                    except Exception as e:
                        logger.warning(f"Error processing source {i}: {e}")
                        continue
                
                if response_parts:
                    final_response = "\n".join(response_parts)
                else:
                    final_response = "I found relevant information in the regulatory documents, but couldn't extract clean content. Please try rephrasing your question."
            
            else:
                final_response = "I couldn't find specific information about your query in the regulatory documents."
            
            # ADD SESSION INFO
            final_response += f"\n\n*📊 Session: {session_id} - Query #{query_number}*"
            
            # SET RAG SUCCESS ATTRIBUTES
            rag_span.set_attribute("rag.response_generated", True)
            rag_span.set_attribute("rag.status", "success")
            
            return final_response
    
    except Exception as e:
        logger.error(f"Enhanced CrewAI query execution error: {e}")
        # ✅ FALLBACK WITHOUT PHOENIX
        try:
            project_id = extract_project_from_session(session_id) or "rag"
            query_controller = QueryController(project_id)
            
            retrieval_result = query_controller.regulatory_hybrid_query(query_str=query, top_k=5)
            
            if retrieval_result and retrieval_result.get('source_nodes'):
                return "I found some regulatory information, but encountered an error during processing."
            else:
                return "I couldn't find specific information about your query in the regulatory documents."
                
        except Exception as fallback_error:
            logger.error(f"Fallback processing error: {fallback_error}")
            return f"I encountered an error while processing your regulatory query: {str(e)}"

def extract_project_from_session(session_id: str) -> str:
    return "rag"

def detect_query_type(query: str) -> str:
    query_lower = query.lower()
    import re
    
    if re.search(r'[mt]\d+\.\d+\.\d+', query_lower):
        return "ia_control"
    if re.search(r'\d+\.\d+\.\d+\.\([ivx]+\)', query_lower):
        return "procurement_process"
    
    return "general"

def extract_control_id(query: str) -> str:
    import re
    match = re.search(r'[MT]\d+\.\d+\.\d+', query, re.IGNORECASE)
    return match.group(0) if match else None

def extract_process_id(query: str) -> str:
    import re
    match = re.search(r'\d+\.\d+\.\d+\.\([IVX]+\)', query, re.IGNORECASE)
    return match.group(0) if match else None
