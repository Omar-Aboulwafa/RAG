# controllers/QueryController.py - ENHANCED VERSION WITH ALL FIXES
from .BaseController import BaseController
from llama_index.core import get_response_synthesizer, VectorStoreIndex, StorageContext
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
from llama_index.core.schema import BaseNode, TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore
from typing import List, Dict, Any, Optional
from sqlalchemy import make_url
import json
import logging
import time
import Stemmer
import re


class QueryController(BaseController):
    
    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id
        
        # Initialize components
        self._llm = None
        self._vector_store = None
        self._index = None
        self._embed_model = None
        self._bm25_retriever = None
        self._fusion_retriever = None
        self._query_count = 0  # Track queries per session
        
        # Initialize stemmer for BM25
        self.stemmer = Stemmer.Stemmer('english')
        
        self.logger.info(f"QueryController initialized for project: {project_id}")


    @property
    def embed_model(self) -> OllamaEmbedding:
        """Lazy initialization of Ollama embedding model"""
        if self._embed_model is None:
            self._embed_model = OllamaEmbedding(
                model_name=self.app_settings.DEFAULT_EMBEDDING_MODEL,
                base_url=self.app_settings.OLLAMA_BASE_URL,
                embed_batch_size=10
            )
            self.logger.info(f"Initialized Ollama embedding model: {self.app_settings.DEFAULT_EMBEDDING_MODEL}")
        return self._embed_model


    @property
    def llm(self) -> Ollama:
        """Lazy initialization of LLM - UPGRADED VERSION"""
        if self._llm is None:
            
            model_name = self.app_settings.DEFAULT_LLM_MODEL
            self.logger.info(f"🔍 Using LLM model from config: {model_name}")
            
            # Only use fallback if config is invalid
            if not model_name or model_name.strip() == "":
                model_name = "qwen3:0.6b-q4_K_M" 
                self.logger.warning(f"⚠️ Config model empty, using fallback: {model_name}")
            
            self._llm = Ollama(
                model=model_name,  
                base_url=self.app_settings.OLLAMA_BASE_URL
            )
            
            self.logger.info(f"✅ Initialized Ollama LLM: {model_name}")
        return self._llm


    @property
    def vector_store(self) -> PGVectorStore:
        """Lazy initialization of PGVectorStore connecting to indexer's database"""
        if self._vector_store is None:
            url = make_url(self.app_settings.DB_CONNECTION_STRING)
            self._vector_store = PGVectorStore.from_params(
                database=url.database,
                host=url.host,
                password=url.password,
                port=url.port,
                user=url.username,
                table_name=f"data_document_chunks_project_{self.project_id}",
                embed_dim=1024,  # mxbai-embed-large uses 1024 dimensions
                hybrid_search=True,
                text_search_config="english",
            )
            self.logger.info(f"Connected to existing vector store for project {self.project_id}")
        return self._vector_store


    @property
    def index(self) -> VectorStoreIndex:
        """Load existing VectorStoreIndex from indexer's data"""
        if self._index is None:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            try:
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=storage_context,
                    embed_model=self.embed_model
                )
                self.logger.info(f"Loaded existing index for project {self.project_id}")
            except Exception as e:
                self.logger.error(f"Could not load existing index: {e}")
                self._index = VectorStoreIndex(
                    [],
                    storage_context=storage_context,
                    embed_model=self.embed_model
                )
                self.logger.warning(f"Created empty index for project {self.project_id}")
        return self._index


    def get_all_nodes(self) -> List[BaseNode]:
        """Get ALL nodes from the vector database - NO LIMIT"""
        try:
            import psycopg2
            from llama_index.core.schema import TextNode
            from sqlalchemy import make_url
            
            url = make_url(self.app_settings.DB_CONNECTION_STRING)
            conn = psycopg2.connect(
                host=url.host,
                port=url.port,
                database=url.database,
                user=url.username,
                password=url.password
            )
            cursor = conn.cursor()
            
            table_name = f"data_document_chunks_project_{self.project_id}"
            
            # ✅ CRITICAL FIX: REMOVED LIMIT 1000 - Now fetches ALL nodes
            cursor.execute(f"""
                SELECT text, metadata_ 
                FROM {table_name}
                WHERE text IS NOT NULL 
                AND metadata_ IS NOT NULL
                ORDER BY id
            """)
            
            rows = cursor.fetchall()
            nodes = []
            
            for text_content, metadata in rows:
                node = TextNode(
                    text=text_content,
                    metadata=metadata if isinstance(metadata, dict) else {}
                )
                nodes.append(node)
            
            cursor.close()
            conn.close()
            
            self.logger.info(f"✅ Retrieved {len(nodes)} nodes from database (all chunks)")
            return nodes
            
        except Exception as e:
            self.logger.error(f"Error getting nodes from database: {e}")
            return []


    def create_regulatory_bm25_retriever(self, nodes: List[BaseNode]) -> BM25Retriever:
        """Create BM25 optimized for regulatory document patterns"""
        
        if not nodes:
            self.logger.warning("No nodes available for BM25 retriever")
            return None
        
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=15,  # Higher for regulatory precision
            stemmer=self.stemmer,
            language="english"
        )
        
        self.logger.info(f"Regulatory-optimized BM25 retriever created with {len(nodes)} nodes")
        return bm25_retriever


    def create_fusion_retriever(self, top_k: int = 10, filters: Dict = None) -> QueryFusionRetriever:
        """Create hybrid fusion retriever with optional reranking"""
        
        # Get all nodes for BM25
        all_nodes = self.get_all_nodes()
        
        if not all_nodes:
            self.logger.warning("No nodes available for fusion retriever")
            return None
        
        # ✅ NEW: Apply filters to BM25 nodes if provided
        if filters:
            filtered_nodes = []
            for node in all_nodes:
                match = True
                for key, value in filters.items():
                    if node.metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_nodes.append(node)
            
            if filtered_nodes:
                self.logger.info(f"🔍 Filtered BM25 corpus to {len(filtered_nodes)} nodes matching {filters}")
                all_nodes = filtered_nodes
            else:
                self.logger.warning(f"⚠️ No nodes match filters {filters}, using all nodes")
        
        # Create BM25 retriever
        bm25_retriever = self.create_regulatory_bm25_retriever(all_nodes)
        
        if not bm25_retriever:
            self.logger.warning("Could not create BM25 retriever, using vector-only")
            return VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k
            )
        
        # Create Vector retriever
        # ✅ CHANGED: Increased from 15 to 20 for more candidates before reranking
        vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.app_settings.RETRIEVAL_TOP_K,  # Uses config value (20)
        )
        
        
        
        # Create fusion retriever
        fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=self.app_settings.RETRIEVAL_TOP_K,  
            num_queries=3,  # Generate multiple query variations
            mode="reciprocal_rerank",
            use_async=False,
            verbose=True,
            llm=self.llm
        )
        
        self.logger.info(f"Fusion retriever created: retrieval_top_k={self.app_settings.RETRIEVAL_TOP_K}")
        return fusion_retriever


    def expand_query_for_hr_bylaws(self, query_str: str) -> str:
        """Expand query with HR Bylaw-specific terms to improve retrieval"""
        
        # Detect article numbers in query
        article_pattern = r'Article\s+(\d+)'
        article_match = re.search(article_pattern, query_str, re.IGNORECASE)
        
        # Detect clause numbers
        clause_pattern = r'Clause\s+(\d+)'
        clause_match = re.search(clause_pattern, query_str, re.IGNORECASE)
        
        expansions = []
        
        # Add article/clause if detected
        if article_match:
            article_num = article_match.group(1)
            expansions.append(f"Article {article_num}")
            # Add clause context if asking about maximum/limits
            if 'maximum' in query_str.lower() or 'limit' in query_str.lower():
                expansions.append(f"Article {article_num} Clause 3")  # Clause 3 often has limits
        
        # Add domain-specific synonyms
        if 'salary deduction' in query_str.lower() or 'deduct' in query_str.lower():
            expansions.extend(['basic salary', 'deduction from salary', 'salary reduction', 'deducted'])
        
        if 'maximum' in query_str.lower():
            expansions.extend(['cannot exceed', 'not exceeding', 'limit', 'ceiling', 'cap', 'does not exceed'])
        
        if 'year' in query_str.lower() or 'annual' in query_str.lower():
            expansions.extend(['per year', 'annually', 'during one year', 'in one year', 'single year'])
        
        if 'disciplinary' in query_str.lower():
            expansions.extend(['penalty', 'penalties', 'disciplinary action', 'disciplinary measure'])
        
        # Build expanded query
        if expansions:
            expanded = f"{query_str} {' '.join(expansions)}"
            self.logger.info(f"📝 Query expanded with: {', '.join(expansions[:5])}...")
            return expanded
        
        return query_str


    def detect_and_apply_hr_filters(self, query_str: str, filters: Dict = None) -> Dict:
        """Detect HR Bylaw patterns and auto-apply metadata filters"""
        
        if filters is None:
            filters = {}
        
        # Auto-detect HR Bylaw query
        if 'hr bylaw' in query_str.lower() or 'article' in query_str.lower():
            filters['doc_type'] = 'HR Bylaw'
            self.logger.info("🏷️ Auto-applied filter: doc_type='HR Bylaw'")
        
        # Detect and filter by article number
        article_match = re.search(r'Article\s+(\d+)', query_str, re.IGNORECASE)
        if article_match:
            article_num = article_match.group(1)
            filters['primary_article_number'] = article_num
            self.logger.info(f"🏷️ Auto-applied filter: primary_article_number='{article_num}'")
        
        # Detect clause (for logging, not filtering unless metadata supports it)
        clause_match = re.search(r'Clause\s+(\d+)', query_str, re.IGNORECASE)
        if clause_match:
            clause_num = clause_match.group(1)
            self.logger.info(f"🏷️ Detected Clause {clause_num} in query (not applied as filter)")
        
        return filters


    def regulatory_hybrid_query(self, query_str: str, filters: Dict = None, top_k: int = 10) -> Dict[str, Any]:
        """Enhanced regulatory hybrid search with fusion retrieval, reranking, and intelligent filtering
        
        Args:
            query_str: The search query
            filters: Metadata filters (doc_type, control_id, primary_article_number, etc.)
            top_k: Number of final results (default: 10, up from 5)
        
        Returns:
            Dict with response, source_nodes, metadata about retrieval
        """
        try:
            start_time = time.time()
            self._query_count += 1
            
            # ✅ NEW: Auto-detect and apply filters from query
            filters = self.detect_and_apply_hr_filters(query_str, filters)
            
            # ✅ NEW: Expand query if it's about HR Bylaws
            original_query = query_str
            if filters and filters.get('doc_type') == 'HR Bylaw':
                query_str = self.expand_query_for_hr_bylaws(query_str)
            
            # Get fusion retriever with filters applied to BM25
            fusion_retriever = self.create_fusion_retriever(top_k=top_k, filters=filters)
            
            # ✅ ENHANCED: Try fusion retrieval with fallback chain
            try:
                retrieved_nodes = fusion_retriever.retrieve(query_str)
                self.logger.info(f"🔍 Fusion retrieval returned {len(retrieved_nodes)} nodes")
                
                if not retrieved_nodes or len(retrieved_nodes) == 0:
                    self.logger.warning("⚠️ Fusion retrieval returned 0 nodes, trying direct vector search")
                    
                    # Fallback: Direct vector search
                    vector_retriever = VectorIndexRetriever(
                        index=self.index,
                        similarity_top_k=top_k,
                    )
                    retrieved_nodes = vector_retriever.retrieve(query_str)
                    self.logger.info(f"🔍 Direct vector search returned {len(retrieved_nodes)} nodes")
                    
                    if not retrieved_nodes or len(retrieved_nodes) == 0:
                        self.logger.warning("⚠️ Direct vector search also failed, using sample nodes")
                        
                        # Last resort: Sample nodes from database
                        all_nodes = self.get_all_nodes()
                        if all_nodes and len(all_nodes) > 0:
                            retrieved_nodes = all_nodes[:top_k]
                            self.logger.info(f"🔍 Using {len(retrieved_nodes)} sample nodes from database")
            
            except Exception as retrieval_error:
                self.logger.error(f"❌ Fusion retrieval failed: {retrieval_error}")
                
                # Emergency fallback
                all_nodes = self.get_all_nodes()
                retrieved_nodes = all_nodes[:top_k] if all_nodes else []
                self.logger.info(f"🔧 Emergency fallback: Using {len(retrieved_nodes)} nodes")
            
            # ✅ IMPROVED: Build comprehensive response from retrieved nodes
            if retrieved_nodes and len(retrieved_nodes) > 0:
                source_nodes = []
                for i, node in enumerate(retrieved_nodes):
                    source_info = {
                        'content': getattr(node, 'text', str(node)),
                        'metadata': getattr(node, 'metadata', {}),
                        'score': getattr(node, 'score', 0.0),
                        'rank': i + 1
                    }
                    source_nodes.append(source_info)
                
                # Build response content with intelligent truncation
                response_parts = []
                for i, source in enumerate(source_nodes[:5], 1):  # ✅ Use top 5 (was 3)
                    content = source['content']
                    doc_type = source['metadata'].get('doc_type', 'Document')
                    
                    # Add citation info
                    citation = ""
                    if doc_type == "HR Bylaw":
                        article = source['metadata'].get('primary_article_number', '')
                        citation = f" [Article {article}]" if article else ""
                    elif doc_type == "IA Standard":
                        control = source['metadata'].get('control_id', '')
                        citation = f" [{control}]" if control else ""
                    
                    # ✅ IMPROVED: Intelligent truncation (don't cut mid-sentence)
                    max_len = 800  # Increased from 500
                    if len(content) > max_len:
                        truncated = content[:max_len]
                        last_period = truncated.rfind('.')
                        if last_period > max_len * 0.7:
                            content = truncated[:last_period + 1]
                        else:
                            content = truncated + "..."
                    
                    response_parts.append(f"{doc_type}{citation}:\n{content}")
                
                response_content = "\n\n".join(response_parts)
                
                # ✅ NEW: Add session tracking info
                session_id = self.project_id[:12]
                response_content += f"\n\n📊 Session: {session_id} - Query #{self._query_count}"
                
                processing_time = time.time() - start_time
                
                return {
                    'response': response_content,
                    'source_nodes': source_nodes,
                    'query': original_query,
                    'expanded_query': query_str if query_str != original_query else None,
                    'filters_applied': filters if filters else {},
                    'total_sources': len(source_nodes),
                    'processing_time': processing_time,
                    'search_type': 'hybrid_fusion_reranked',
                    'fusion_method': 'bm25_vector_fusion_reranked'
                }
            
            else:
                self.logger.error("❌ No nodes retrieved by any method")
                return {
                    'response': 'No regulatory content found for this query.',
                    'source_nodes': [],
                    'query': original_query,
                    'expanded_query': query_str if query_str != original_query else None,
                    'filters_applied': filters if filters else {},
                    'total_sources': 0,
                    'processing_time': time.time() - start_time,
                    'search_type': 'failed',
                    'fusion_method': 'none'
                }
                
        except Exception as e:
            self.logger.error(f"❌ Regulatory hybrid query error: {e}", exc_info=True)
            return {
                'response': f'Query processing error: {str(e)}',
                'source_nodes': [],
                'query': query_str,
                'filters_applied': filters if filters else {},
                'total_sources': 0,
                'processing_time': 0,
                'search_type': 'error',
                'fusion_method': 'none'
            }


    def simple_vector_search(self, query_str: str, top_k: int = 5, filters: dict = None) -> Dict[str, Any]:
        """Simple vector search fallback"""
        try:
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k
            )
            
            nodes = retriever.retrieve(query_str)
            
            source_nodes = []
            for node in nodes:
                source_nodes.append({
                    "content": node.node.get_content()[:300] + "..." if len(node.node.get_content()) > 300 else node.node.get_content(),
                    "score": float(node.score) if hasattr(node, 'score') else 0.0,
                    "metadata": node.node.metadata,
                    "retrieval_method": "vector_only"
                })
            
            return {
                "response": f"Found {len(source_nodes)} relevant documents for: {query_str}",
                "source_nodes": source_nodes,
                "query": query_str,
                "total_sources": len(source_nodes),
                "search_type": "vector_search"
            }
            
        except Exception as e:
            self.logger.error(f"Error in vector search: {e}")
            return {
                "response": f"Search error: {str(e)}",
                "source_nodes": [],
                "query": query_str,
                "total_sources": 0,
                "error": str(e)
            }


    # ✅ NEW: Diagnostic tool for debugging retrieval issues
    def diagnose_retrieval(self, query_str: str, top_k: int = 20) -> Dict[str, Any]:
        """Diagnostic tool to inspect database contents and retrieval results"""
        
        self.logger.info("=" * 80)
        self.logger.info(f"🔬 DIAGNOSTIC MODE: Analyzing query")
        self.logger.info(f"Query: {query_str}")
        self.logger.info("=" * 80)
        
        # Check total nodes
        all_nodes = self.get_all_nodes()
        self.logger.info(f"📊 Total nodes in database: {len(all_nodes)}")
        
        # Check document type distribution
        doc_types = {}
        for node in all_nodes:
            dt = node.metadata.get('doc_type', 'Unknown')
            doc_types[dt] = doc_types.get(dt, 0) + 1
        
        self.logger.info("📊 Document types:")
        for dt, count in doc_types.items():
            self.logger.info(f"  - {dt}: {count} nodes")
        
        # Check HR Bylaw nodes
        hr_nodes = [n for n in all_nodes if n.metadata.get('doc_type') == 'HR Bylaw']
        self.logger.info(f"📊 HR Bylaw nodes: {len(hr_nodes)}")
        
        # Check Article 110 specifically
        article_110_nodes = [n for n in hr_nodes if n.metadata.get('primary_article_number') == '110']
        self.logger.info(f"📊 Article 110 nodes: {len(article_110_nodes)}")
        
        # Sample Article 110 content
        if article_110_nodes:
            self.logger.info("📄 Sample Article 110 content:")
            for i, node in enumerate(article_110_nodes[:3], 1):
                preview = node.text[:200] + "..." if len(node.text) > 200 else node.text
                self.logger.info(f"  Chunk {i}: {preview}")
                # Check for "60" or "sixty"
                if '60' in node.text or 'sixty' in node.text.lower():
                    self.logger.info(f"    ✅ Contains '60' or 'sixty'")
        
        # Test retrieval
        self.logger.info("\n🔍 Testing retrieval...")
        results = self.regulatory_hybrid_query(query_str, top_k=top_k)
        
        self.logger.info(f"Retrieved {results['total_sources']} sources:")
        for i, source in enumerate(results['source_nodes'][:10], 1):
            meta = source['metadata']
            article = meta.get('primary_article_number', 'N/A')
            score = source.get('score', 0.0)
            preview = source['content'][:100] + "..."
            self.logger.info(f"  #{i}: Article {article}, score={score:.3f}")
            self.logger.info(f"       {preview}")
        
        self.logger.info("=" * 80)
        
        return results


    # Specialized queries for document types
    def query_control_by_id(self, control_id: str) -> Dict[str, Any]:
        """Query specific IA control by ID (e.g., M1.1.1)"""
        return self.regulatory_hybrid_query(
            f"control {control_id} requirements implementation guidance",
            filters={"doc_type": "IA Standard", "control_id": control_id}
        )


    def query_process_by_id(self, process_id: str) -> Dict[str, Any]:
        """Query procurement process by ID (e.g., 2.3.3.(IX))"""
        return self.regulatory_hybrid_query(
            f"process {process_id} procedure steps RACI responsibilities",
            filters={"doc_type": "Procurement Manual", "level3_process_id": process_id}
        )


    def query_by_role_responsibilities(self, role: str, raci_type: str = None) -> Dict[str, Any]:
        """Query by business role responsibilities"""
        query = f"role {role} responsibilities duties tasks"
        if raci_type:
            query += f" {raci_type} accountable responsible"
        
        return self.regulatory_hybrid_query(
            query,
            filters={"business_role": role}
        )


# For backward compatibility
RegulatoryDocumentQueryController = QueryController
