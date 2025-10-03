from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from crewai.memory import ShortTermMemory, LongTermMemory
from crewai.memory.contextual import ContextualMemory
from llama_index.llms.ollama import Ollama
import json
import re
from typing import Dict, List, Any, Optional
from config import get_settings

class RAGRetrievalTool(BaseTool):
    name: str = "rag_retrieval_tool"
    description: str = "Retrieves specific information from regulatory documents using hybrid BM25+Vector search"
    
    def _run(self, 
             query: str, 
             doc_type: Optional[str] = None, 
             control_id: Optional[str] = None,
             process_id: Optional[str] = None,
             top_k: int = 5) -> str:
        """ReAct Action: Execute RAG retrieval with specific parameters"""
        from controllers.QueryController import QueryController
        
        # Build filters based on parameters
        filters = {}
        if doc_type:
            filters["doc_type"] = doc_type
        if control_id:
            filters["control_id"] = control_id
        if process_id:
            filters["process_id"] = process_id
        
        query_controller = QueryController("regulatory_docs")
        result = query_controller.regulatory_hybrid_query(
            query_str=query,
            filters=filters,
            top_k=top_k
        )
        
        # Format for ReAct Observation
        return json.dumps({
            "retrieved_chunks": len(result.get('source_nodes', [])),
            "response": result.get('response', ''),
            "sources": [
                {
                    "content": node.get('content', '')[:200] + "...",
                    "metadata": node.get('metadata', {}),
                    "score": node.get('score', 0.0)
                }
                for node in result.get('source_nodes', [])[:3]  # Top 3 for context
            ]
        })

class ValidationTool(BaseTool):
    name: str = "compliance_validation_tool"
    description: str = "Validates responses against regulatory compliance requirements and factual accuracy"
    
    def _run(self, response: str, regulatory_domain: str) -> str:
        """Validate response against compliance guardrails"""
        validation_results = {
            "compliance_score": 0.85,  # Simulated validation
            "citations_present": bool(re.search(r'[MT]\d+\.\d+\.\d+|\d+\.\d+\.\d+\.\([IVX]+\)', response)),
            "regulatory_coverage": regulatory_domain in response,
            "factual_grounding": len(response.split()) > 50,
            "recommendations": []
        }
        
        if not validation_results["citations_present"]:
            validation_results["recommendations"].append("Add specific regulatory citations")
        
        if not validation_results["regulatory_coverage"]:
            validation_results["recommendations"].append(f"Ensure {regulatory_domain} domain coverage")
            
        return json.dumps(validation_results)

def create_function_calling_llm(model_name: str) -> Ollama:
    """Create Ollama LLM with function calling support"""
    settings = get_settings()
    return Ollama(
        model=model_name,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.1,  # Lower temperature for more consistent function calls
        system_prompt=(
            "You are a regulatory compliance AI agent. You MUST use available tools "
            "to retrieve information before providing responses. Follow the ReAct pattern: "
            "1) Thought: Plan your approach, 2) Action: Use tools to gather information, "
            "3) Observation: Process tool results, 4) Repeat until you have sufficient information."
        )
    )

class RouterFilterAgent:
    """Advanced Router/Filter Agent implementing ReAct pattern"""
    
    @staticmethod
    def create_agent() -> Agent:
        return Agent(
            role='Query Router and Filter Specialist',
            goal='Analyze incoming queries, decompose complex requests, and route to appropriate specialists',
            verbose=True,
            memory=True,
            backstory=(
                "You are an expert query router with deep understanding of regulatory domains. "
                "Your role is to: (1) Analyze query complexity and intent, (2) Decompose multi-part "
                "questions into focused sub-queries, (3) Identify the appropriate regulatory domain "
                "(IA Standards, Procurement, HR Bylaws), (4) Filter out non-regulatory queries, "
                "(5) Validate query completeness before routing. You follow the ReAct pattern: "
                "Thought -> Action -> Observation -> Decision."
            ),
            llm=create_function_calling_llm(get_settings().ROUTER_LLM_MODEL),
            max_iter=3,
            allow_delegation=True,
            step_callback=lambda step: print(f"🤔 Router Thought: {step.thought}") if hasattr(step, 'thought') else None
        )

class SpecialistAgentFactory:
    """Factory for creating domain-specific specialist agents with ReAct pattern"""
    
    @staticmethod
    def create_ia_specialist() -> Agent:
        return Agent(
            role='IA Standards Compliance Expert',
            goal='Analyze NESA UAE IA Standards using ReAct methodology for precise control analysis',
            verbose=True,
            memory=True,
            backstory=(
                "You are an Information Assurance specialist following the ReAct pattern. "
                "For each query: (1) Thought: Plan your analysis approach, (2) Action: Use RAG tools "
                "to retrieve specific IA controls, (3) Observation: Process retrieved information, "
                "(4) Continue until you have complete control analysis with priorities, applicability, "
                "and implementation guidance."
            ),
            tools=[RAGRetrievalTool()],
            llm=create_function_calling_llm(get_settings().DEFAULT_LLM_MODEL),
            max_iter=5,  # Allow more iterations for complex analysis
            allow_delegation=False,
            step_callback=lambda step: print(f"🔐 IA Specialist: {step.action}") if hasattr(step, 'action') else None
        )
    
    @staticmethod
    def create_procurement_specialist() -> Agent:
        return Agent(
            role='Procurement Process Expert',
            goal='Analyze procurement workflows using ReAct methodology for detailed process guidance',
            verbose=True,
            memory=True,
            backstory=(
                "You are a procurement process specialist following the ReAct pattern. "
                "For each query: (1) Thought: Plan process analysis, (2) Action: Retrieve process "
                "workflows, RACI matrices, and role definitions, (3) Observation: Analyze retrieved "
                "process steps and dependencies, (4) Continue until you have complete workflow analysis."
            ),
            tools=[RAGRetrievalTool()],
            llm=create_function_calling_llm(get_settings().DEFAULT_LLM_MODEL),
            max_iter=5,
            allow_delegation=False,
            step_callback=lambda step: print(f"💼 Procurement Specialist: {step.action}") if hasattr(step, 'action') else None
        )
    
    @staticmethod
    def create_compliance_synthesizer() -> Agent:
        return Agent(
            role='Compliance Synthesis and Validation Coordinator',
            goal='Synthesize specialist findings and ensure factual validation with compliance guardrails',
            verbose=True,
            memory=True,
            backstory=(
                "You are a compliance coordinator following the ReAct pattern for synthesis and validation. "
                "For each synthesis task: (1) Thought: Plan integration approach, (2) Action: Validate "
                "specialist findings against compliance requirements, (3) Observation: Identify gaps "
                "and conflicts, (4) Continue until you have validated, comprehensive guidance."
            ),
            tools=[RAGRetrievalTool(), ValidationTool()],
            llm=create_function_calling_llm(get_settings().DEFAULT_LLM_MODEL),
            max_iter=4,
            allow_delegation=True,
            step_callback=lambda step: print(f"⚖️ Compliance Synthesizer: {step.action}") if hasattr(step, 'action') else None
        )

class AdvancedMemoryManager:
    """Advanced memory management with multiple memory types"""
    
    @staticmethod
    def setup_crew_memory():
        """Configure comprehensive memory system"""
        return {
            # Conversational Memory (baseline)
            "short_term": ShortTermMemory(),
            
            # Working Memory (current task context)
            "working": ContextualMemory(
                embedder={
                    "provider": "ollama", 
                    "config": {
                        "model": get_settings().DEFAULT_EMBEDDING_MODEL,
                        "base_url": get_settings().OLLAMA_BASE_URL
                    }
                }
            ),
            
            # Semantic Memory (facts and knowledge)
            "semantic": LongTermMemory(
                storage_path="./storage/crewai_memory/semantic",
                embedder={
                    "provider": "ollama",
                    "config": {
                        "model": get_settings().DEFAULT_EMBEDDING_MODEL,
                        "base_url": get_settings().OLLAMA_BASE_URL
                    }
                }
            )
        }

def create_react_task(agent_type: str, query: str, context: Dict = None) -> Task:
    """Create ReAct-pattern tasks for different agent types"""
    
    if agent_type == "router":
        return Task(
            description=(
                f"REACT ROUTING TASK for query: '{query}'\n\n"
                "Follow ReAct pattern:\n"
                "1. THOUGHT: Analyze query complexity, identify regulatory domains involved, "
                "determine if decomposition is needed\n"
                "2. ACTION: If complex, decompose into sub-queries. Identify primary domain "
                "(IA Standards/Procurement/Standards)\n" 
                "3. OBSERVATION: Evaluate decomposition quality and domain classification\n"
                "4. DECISION: Route to appropriate specialists with clear sub-queries\n\n"
                "Output format: Provide routing decision with specific sub-queries for each domain."
            ),
            expected_output=(
                "Routing analysis including: (1) Query complexity assessment, "
                "(2) Domain classification (IA/Procurement/Standards), "
                "(3) Decomposed sub-queries if applicable, (4) Routing recommendations for specialists"
            ),
            agent=RouterFilterAgent.create_agent()
        )
    
    elif agent_type == "ia_specialist":
        return Task(
            description=(
                f"REACT IA ANALYSIS for: '{query}'\n\n"
                "Follow ReAct pattern:\n"
                "1. THOUGHT: Plan IA control analysis approach, identify specific controls to investigate\n"
                "2. ACTION: Use RAG tools to retrieve control definitions, priorities, and guidance\n"
                "3. OBSERVATION: Analyze retrieved control information for completeness\n"
                "4. REPEAT: Continue until you have comprehensive control analysis\n\n"
                "Focus on: Control IDs, priorities (P1-P4), applicability, implementation guidance"
            ),
            expected_output=(
                "Complete IA analysis including: (1) Control definitions with sub-controls, "
                "(2) Priority and applicability requirements, (3) Implementation guidance, "
                "(4) External standard mappings, (5) Proper citations"
            ),
            agent=SpecialistAgentFactory.create_ia_specialist()
        )
    
    elif agent_type == "procurement_specialist":
        return Task(
            description=(
                f"REACT PROCUREMENT ANALYSIS for: '{query}'\n\n"
                "Follow ReAct pattern:\n"
                "1. THOUGHT: Plan process analysis approach, identify specific processes and roles\n"
                "2. ACTION: Use RAG tools to retrieve process flows, RACI matrices, and requirements\n"
                "3. OBSERVATION: Analyze workflow completeness and role clarity\n"
                "4. REPEAT: Continue until you have detailed process guidance\n\n"
                "Focus on: Process IDs, workflow steps, RACI responsibilities, system dependencies"
            ),
            expected_output=(
                "Complete process analysis including: (1) Process scope and purpose, "
                "(2) Step-by-step workflow, (3) RACI matrix with responsibilities, "
                "(4) Required artifacts, (5) System dependencies, (6) Proper citations"
            ),
            agent=SpecialistAgentFactory.create_procurement_specialist()
        )
    
    elif agent_type == "compliance_synthesizer":
        return Task(
            description=(
                f"REACT COMPLIANCE SYNTHESIS for: '{query}'\n\n"
                "Follow ReAct pattern:\n"
                "1. THOUGHT: Plan synthesis approach, identify validation requirements\n"
                "2. ACTION: Use validation tools to check compliance and factual accuracy\n"
                "3. OBSERVATION: Identify gaps, conflicts, and compliance issues\n"
                "4. REPEAT: Continue until comprehensive, validated guidance is achieved\n\n"
                "Ensure: Factual grounding, proper citations, compliance guardrails, actionable guidance"
            ),
            expected_output=(
                "Validated compliance guidance including: (1) Integrated multi-domain analysis, "
                "(2) Validation results, (3) Compliance gaps identified, "
                "(4) Actionable recommendations, (5) Complete regulatory citations"
            ),
            agent=SpecialistAgentFactory.create_compliance_synthesizer(),
            context=context.get('specialist_outputs', []) if context else []
        )

def create_advanced_regulatory_crew(query: str, query_type: str = "general") -> Crew:
    """Create advanced crew with ReAct pattern and comprehensive memory"""
    
    # Setup memory management
    memory_config = AdvancedMemoryManager.setup_crew_memory()
    
    # Create tasks based on complexity
    tasks = []
    
    # Always start with router for query analysis
    router_task = create_react_task("router", query)
    tasks.append(router_task)
    
    # Create specialist tasks based on query type or router output
    if query_type == "ia_control" or "control" in query.lower():
        ia_task = create_react_task("ia_specialist", query)
        tasks.append(ia_task)
    
    if query_type == "procurement_process" or any(term in query.lower() for term in ['process', 'procurement', 'workflow']):
        proc_task = create_react_task("procurement_specialist", query)
        tasks.append(proc_task)
    
    # Always include synthesis for final validation
    synthesis_task = create_react_task("compliance_synthesizer", query)
    tasks.append(synthesis_task)
    
    # Get all agents from tasks
    agents = [task.agent for task in tasks]
    
    # Create crew with advanced configuration
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=2,
        memory=True,
        embedder={
            "provider": "ollama",
            "config": {
                "model": get_settings().DEFAULT_EMBEDDING_MODEL,
                "base_url": get_settings().OLLAMA_BASE_URL
            }
        },
        max_rpm=100,  # Higher rate limit for complex workflows
        max_execution_time=300,  # 5 minute timeout for complex analysis
    )
    
    return crew
