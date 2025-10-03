from .BaseController import BaseController
from .ProjectController import ProjectController
from config import get_settings
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
from llama_index.core.schema import BaseNode, TextNode, Document
import os
from typing import List, Optional, Dict, Any
import psycopg2
import json
from llama_index.vector_stores.postgres import PGVectorStore
import textwrap
from llama_index.embeddings.ollama import OllamaEmbedding
from contextlib import contextmanager
from sqlalchemy import make_url
import logging
import re

class ProcessController(BaseController):


    def __init__(self, project_id: str, default_llm_model: str = None):
        super().__init__()
        
        if default_llm_model is None:
            default_llm_model = self.app_settings.DEFAULT_LLM_MODEL
        
        self.default_llm_model = default_llm_model
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)
        
        self.reader = DoclingReader(
            export_type=DoclingReader.ExportType.JSON,
            keep_tables=True,
            keep_figures=True
        )
        
        self.default_embedding_model = self.app_settings.DEFAULT_EMBEDDING_MODEL
        
        # Lazy initialization
        self._llm = None
        self._vector_store = None
        self._index = None
        self._embed_model = None
        
        self.logger.info(f"ProcessController initialized for project: {project_id}")

    @property
    def embed_model(self) -> OllamaEmbedding:
        """Lazy initialization of Ollama embedding model"""
        if self._embed_model is None:
            self._embed_model = OllamaEmbedding(
                model_name=self.default_embedding_model,
                base_url=self.app_settings.OLLAMA_BASE_URL,
                embed_batch_size=10
            )
            self.logger.info(f"Initialized Ollama embedding model: {self.default_embedding_model}")
        return self._embed_model

    @property
    def vector_store(self) -> PGVectorStore:
        """Lazy initialization of PGVectorStore following LlamaIndex documentation"""
        if self._vector_store is None:
            url = make_url(self.app_settings.DB_CONNECTION_STRING)
            self._vector_store = PGVectorStore.from_params(
                database=url.database,
                host=url.host,
                password=url.password,
                port=url.port,
                user=url.username,
                table_name=f"data_document_chunks_project_{self.project_id}",  # Project-specific table
                embed_dim=1024,  # mxbai-embed-large uses 1024 dimensions
                hybrid_search=True,  # Enable hybrid search as per documentation
                text_search_config="english",  # Configure for English text search
                hnsw_kwargs={
                    "hnsw_m": 50,  # Higher number of connections per node for better accuracy (16-64)
                    "hnsw_ef_construction": 350,  # search depth during construction (50-500)
                    "hnsw_ef_search": 40,  # search depth during querying (50-500)
                    "hnsw_dist_method": "cosine",
                },
            )
            self.logger.info(f"Initialized PGVectorStore for project {self.project_id}")
        return self._vector_store

    @property
    def index(self) -> VectorStoreIndex:
        """Get or create VectorStoreIndex with proper embed_model parameter"""
        if self._index is None:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            # Try to load existing index first
            try:
                # CRITICAL: Pass embed_model when creating from vector store
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=storage_context,
                    embed_model=self.embed_model  # REQUIRED for Ollama embeddings
                )
                self.logger.info(f"Loaded existing index for project {self.project_id}")
            except Exception as e:
                self.logger.warning(f"Could not load existing index: {e}")
                # Create new empty index with embed_model
                self._index = VectorStoreIndex(
                    [],
                    storage_context=storage_context,
                    embed_model=self.embed_model,  # REQUIRED for Ollama embeddings
                    show_progress=True
                )
                self.logger.info(f"Created new index for project {self.project_id}")
        return self._index

    @property
    def llm(self) -> Ollama:
        """Lazy initialization of LLM"""
        if self._llm is None:
            self._llm = self.initialize_ollama_llm(self.default_llm_model)
        return self._llm

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        db_connection_string = self.app_settings.DB_CONNECTION_STRING
        conn = None
        try:
            conn = psycopg2.connect(db_connection_string)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()

    def load_documents(self, file_id: str) -> List[Document]:
        """Load document from file using the correct upload directory path"""
        
        app_settings = get_settings()
        
        # Use the SAME path logic as upload endpoint
        project_upload_dir = app_settings.get_project_upload_path(self.project_id)
        file_path = os.path.join(project_upload_dir, file_id)
        
        self.logger.info(f"Project ID: {self.project_id}")
        self.logger.info(f"File ID: {file_id}")  
        self.logger.info(f"Upload directory: {project_upload_dir}")
        self.logger.info(f"Full file path: {file_path}")
        self.logger.info(f"Absolute path: {os.path.abspath(file_path)}")
        self.logger.info(f"Current working directory: {os.getcwd()}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            self.logger.error(f"File not found at: {file_path}")
            
            # Debug: Show what's actually in the directory
            if os.path.exists(project_upload_dir):
                try:
                    available_files = os.listdir(project_upload_dir)
                    self.logger.info(f"Available files in {project_upload_dir}:")
                    for f in available_files:
                        full_path = os.path.join(project_upload_dir, f)
                        size = os.path.getsize(full_path) if os.path.isfile(full_path) else "DIR"
                        self.logger.info(f"  - {f} ({size} bytes)")
                except Exception as e:
                    self.logger.error(f"Error listing directory contents: {e}")
            else:
                self.logger.error(f"Project upload directory doesn't exist: {project_upload_dir}")
                
                # Check if the base upload directory exists
                base_upload_dir = app_settings.UPLOAD_DIRECTORY_PATH
                if os.path.exists(base_upload_dir):
                    self.logger.info(f"Base upload directory exists: {base_upload_dir}")
                    try:
                        subdirs = [d for d in os.listdir(base_upload_dir) if os.path.isdir(os.path.join(base_upload_dir, d))]
                        self.logger.info(f"Available project directories: {subdirs}")
                    except Exception as e:
                        self.logger.error(f"Error listing base directory: {e}")
                else:
                    self.logger.error(f"Base upload directory doesn't exist: {base_upload_dir}")
            
            return []

        try:
            self.logger.info(f"Loading documents from: {file_path}")
            documents = self.reader.load_data(file_path)
            
            # Enhanced metadata extraction
            enhanced_documents = []
            for doc in documents:
                # Add basic metadata
                doc.metadata["source_file"] = file_id
                doc.metadata["filename"] = file_id
                doc.metadata["file_path"] = file_path
                doc.metadata["project_id"] = self.project_id
                
                # Apply comprehensive metadata extraction
                enhanced_doc = self.extract_comprehensive_metadata(doc)
                enhanced_documents.append(enhanced_doc)
            
            self.logger.info(f"Successfully loaded and enhanced {len(enhanced_documents)} documents from '{file_id}'")
            return enhanced_documents
            
        except Exception as e:
            self.logger.error(f"Error loading document {file_id}: {str(e)}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def extract_comprehensive_metadata(self, document: Document) -> Document:
        """Extract comprehensive metadata from document based on type detection"""
        try:
            content = document.text.lower()
            filename = document.metadata.get('filename', '').lower()
            
            # Document type detection
            doc_type = self.detect_document_type(content, filename)
            document.metadata['doc_type'] = doc_type
            
            # Extract type-specific metadata
            if doc_type == "IA Standard":
                self.extract_ia_standard_metadata(document)
            elif doc_type == "Procurement Manual":
                self.extract_procurement_manual_metadata(document)
            elif doc_type == "Procurement Standard":
                self.extract_procurement_standard_metadata(document)
            elif doc_type == "HR Bylaw":
                self.extract_hr_bylaw_metadata(document)
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            return document

    def detect_document_type(self, content: str, filename: str) -> str:
        """Detect document type based on content patterns and filename"""
        # Check filename patterns first
        if any(term in filename for term in ['ia', 'nesa', 'information assurance', 'security']):
            return "IA Standard"
        elif any(term in filename for term in ['procurement manual', 'ariba', 'process']):
            return "Procurement Manual"
        elif any(term in filename for term in ['procurement standard', 'standard']):
            return "Procurement Standard"
        elif any(term in filename for term in ['hr', 'bylaw', 'human resource']):
            return "HR Bylaw"
        
        # Check content patterns
        if re.search(r'[MT]\d{1}\.\d{1}\.\d{1}', content):
            return "IA Standard"
        elif re.search(r'\d{1}\.\d{1}\.\d{1}\.\([IVXLCDM]+\)', content):
            return "Procurement Manual"
        elif 'procurement standard' in content and re.search(r'\d+\.\d+', content):
            return "Procurement Standard"
        elif re.search(r'article\s*\(\d+\)', content):
            return "HR Bylaw"
        
        return "Unknown"

    def extract_ia_standard_metadata(self, document: Document):
        """Extract metadata specific to IA Standards"""
        content = document.text
        
        # Extract Control IDs
        control_ids = re.findall(r'[MT]\d{1}\.\d{1}\.\d{1}', content)
        if control_ids:
            document.metadata['control_ids'] = list(set(control_ids))
            document.metadata['primary_control_id'] = control_ids[0]
        
        # Extract Priority
        priority_match = re.search(r'PRIORITY\s+(P[1-4])', content)
        if priority_match:
            document.metadata['priority'] = priority_match.group(1)
        
        # Extract Applicability
        applicability_match = re.search(r'APPLICABILITY\s+(ALWAYS APPLICABLE|BASED ON RISK ASSESSMENT)', content)
        if applicability_match:
            document.metadata['applicability'] = applicability_match.group(1)
        
        # Set section path and citation
        if 'primary_control_id' in document.metadata:
            control_id = document.metadata['primary_control_id']
            chapter = control_id.split('.')[0]
            document.metadata['section_path'] = f"IA/Ch{chapter[1:]}/{control_id}"
            document.metadata['citation_id'] = control_id

    def extract_procurement_manual_metadata(self, document: Document):
        """Extract metadata specific to Procurement Manuals"""
        content = document.text
        
        # Extract L3 Process IDs
        l3_process_matches = re.findall(r'\d{1}\.\d{1}\.\d{1}\.\(([IVXLCDM]+)\)', content)
        if l3_process_matches:
            full_matches = re.findall(r'\d{1}\.\d{1}\.\d{1}\.\([IVXLCDM]+\)', content)
            document.metadata['l3_process_ids'] = list(set(full_matches))
            document.metadata['primary_l3_process_id'] = full_matches[0]
        
        # Extract Process Groups
        process_groups = re.findall(r'\b(DCM|S2C|CLM|R2P|SPRM|R&R|MDM)\b', content)
        if process_groups:
            document.metadata['process_groups'] = list(set(process_groups))
            document.metadata['primary_process_group'] = process_groups[0]
        
        # Extract Business Roles
        business_roles = re.findall(r'\*([A-Za-z\s\/\-]+)\*', content)
        if business_roles:
            document.metadata['business_roles'] = list(set(business_roles))
        
        # Extract System Documents
        system_docs_double = re.findall(r'"([^"]+)"', content)
        system_docs_single = re.findall(r"'([^']+)'", content)
        all_system_docs = system_docs_double + system_docs_single
        if all_system_docs:
            document.metadata['artifact_names'] = list(set(all_system_docs))
        
        # Extract Software Systems
        software_systems = re.findall(r'\b(SAP Ariba|ORACLE ADERP|Ariba|Oracle)\b', content, re.IGNORECASE)
        if software_systems:
            document.metadata['software_systems'] = list(set(software_systems))
        
        # Set section path and citation
        if 'primary_process_group' in document.metadata and 'primary_l3_process_id' in document.metadata:
            document.metadata['section_path'] = f"PM/{document.metadata['primary_process_group']}/{document.metadata['primary_l3_process_id']}"
            document.metadata['citation_id'] = document.metadata['primary_l3_process_id']

    def extract_procurement_standard_metadata(self, document: Document):
        """Extract metadata specific to Procurement Standards"""
        content = document.text
        
        # Extract Standard Numbers
        standard_numbers = re.findall(r'\d+\.\d+(?:\.\d+)?', content)
        if standard_numbers:
            document.metadata['standard_numbers'] = list(set(standard_numbers))
            document.metadata['primary_standard_number'] = standard_numbers[0]
            document.metadata['section_path'] = f"Std/Sec{standard_numbers[0].split('.')[0]}/{standard_numbers[0]}"
            document.metadata['citation_id'] = standard_numbers[0]

    def extract_hr_bylaw_metadata(self, document: Document):
        """Extract metadata specific to HR Bylaws"""
        content = document.text
        
        # Extract Article Numbers
        article_numbers = re.findall(r'Article\s*\((\d+)\)', content, re.IGNORECASE)
        if article_numbers:
            document.metadata['article_numbers'] = list(set(article_numbers))
            document.metadata['primary_article_number'] = article_numbers[0]
            document.metadata['section_path'] = f"HR/Article{article_numbers[0]}"
            document.metadata['citation_id'] = f"Article {article_numbers[0]}"


    def chunk_documents(self, documents: List[Document]) -> List[BaseNode]:
        """Create document-aware chunks with per-document-type size and overlap"""
        try:
            from llama_index.core.schema import TextNode  # Import TextNode instead of BaseNode
            enhanced_nodes = []

            for doc in documents:
                # Determine document type and chunk settings
                doc_type = doc.metadata.get("doc_type", "Unknown")
                chunk_cfg = self.app_settings.get_chunk_settings_for_doc_type(doc_type)
                if not chunk_cfg:
                    # Fallback to default settings
                    chunk_cfg = {"chunk_size": 1024, "chunk_overlap": 200}
                
                size = chunk_cfg["chunk_size"]
                overlap = chunk_cfg["chunk_overlap"]
                raw_text = doc.text

                # Sliding window chunking
                start = 0
                text_len = len(raw_text)
                while start < text_len:
                    end = min(start + size, text_len)
                    chunk_text = raw_text[start:end]

                    # Create a TextNode for this chunk (NOT BaseNode)
                    new_node = TextNode(
                        text=chunk_text,
                        metadata=doc.metadata.copy()  # Copy metadata directly in constructor
                    )
                    
                    # Add additional metadata
                    new_node.metadata["project_id"] = self.project_id
                    
                    # Annotate chunk type based on document type
                    chunk_type = doc_type.replace(" ", "_").lower()
                    new_node.metadata["chunk_type"] = f"chunk_{chunk_type}"
                    
                    # Mark citation ready and ensure citation_id exists
                    new_node.metadata["citation_ready"] = True
                    if "citation_id" not in new_node.metadata:
                        for key in ("primary_control_id", "primary_l3_process_id",
                                    "primary_standard_number", "primary_article_number"):
                            if key in new_node.metadata:
                                cid = new_node.metadata[key]
                                new_node.metadata["citation_id"] = (
                                    f"Article {cid}" if key == "primary_article_number" else cid
                                )
                                break

                    enhanced_nodes.append(new_node)
                    
                    # Advance window (avoid infinite loop for small texts)
                    if overlap >= size:
                        start += 1  # Move by at least 1 character
                    else:
                        start += size - overlap

            self.logger.info(f"Created {len(enhanced_nodes)} document-aware chunks")
            return enhanced_nodes
            
        except Exception as e:
            self.logger.error(f"Error in document-aware chunking: {str(e)}")
            import traceback
            self.logger.error(f"Chunking traceback: {traceback.format_exc()}")
            return []


    def metadata_filtered_search(self, query: str, filters: dict = None, top_k: int = 10) -> List[dict]:
        """Enhanced metadata-filtered search for high-precision retrieval"""
        try:
            # Build metadata filters
            metadata_filter_list = []
            if filters:
                # Document type filtering
                if 'doc_type' in filters:
                    metadata_filter_list.append(
                        MetadataFilter(key="doc_type", value=filters['doc_type'], operator=FilterOperator.EQ)
                    )
                
                # Priority filtering (for IA Standards)
                if 'priority' in filters:
                    if isinstance(filters['priority'], list):
                        metadata_filter_list.append(
                            MetadataFilter(key="priority", value=filters['priority'], operator=FilterOperator.IN)
                        )
                    else:
                        metadata_filter_list.append(
                            MetadataFilter(key="priority", value=filters['priority'], operator=FilterOperator.EQ)
                        )
                
                # Process group filtering (for Procurement Manuals)
                if 'process_group' in filters:
                    if isinstance(filters['process_group'], list):
                        metadata_filter_list.append(
                            MetadataFilter(key="primary_process_group", value=filters['process_group'], operator=FilterOperator.IN)
                        )
                    else:
                        metadata_filter_list.append(
                            MetadataFilter(key="primary_process_group", value=filters['process_group'], operator=FilterOperator.EQ)
                        )
                
                # Control ID filtering (exact match)
                if 'control_id' in filters:
                    metadata_filter_list.append(
                        MetadataFilter(key="primary_control_id", value=filters['control_id'], operator=FilterOperator.EQ)
                    )
                
                # Citation ID filtering
                if 'citation_id' in filters:
                    metadata_filter_list.append(
                        MetadataFilter(key="citation_id", value=filters['citation_id'], operator=FilterOperator.EQ)
                    )
                
                # Business role filtering (contains search)
                if 'business_role' in filters:
                    metadata_filter_list.append(
                        MetadataFilter(key="business_roles", value=filters['business_role'], operator=FilterOperator.CONTAINS)
                    )
                
                # Software system filtering
                if 'software_system' in filters:
                    metadata_filter_list.append(
                        MetadataFilter(key="software_systems", value=filters['software_system'], operator=FilterOperator.CONTAINS)
                    )
            
            # Create MetadataFilters object
            metadata_filters = None
            if metadata_filter_list:
                metadata_filters = MetadataFilters(
                    filters=metadata_filter_list,
                    condition=FilterCondition.AND
                )
            
            # Create retriever with metadata filters
            retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                filters=metadata_filters
            )
            
            # Retrieve nodes
            retrieved_nodes = retriever.retrieve(query)
            
            # Format results with enhanced metadata
            results = []
            for node in retrieved_nodes:
                result = {
                    'text': node.text,
                    'score': node.score if hasattr(node, 'score') else 0.0,
                    'metadata': node.metadata,
                    'citation_id': node.metadata.get('citation_id', ''),
                    'doc_type': node.metadata.get('doc_type', 'Unknown'),
                    'section_path': node.metadata.get('section_path', ''),
                    'priority': node.metadata.get('priority', ''),
                    'process_group': node.metadata.get('primary_process_group', ''),
                    'control_id': node.metadata.get('primary_control_id', ''),
                    'business_roles': node.metadata.get('business_roles', []),
                    'software_systems': node.metadata.get('software_systems', [])
                }
                results.append(result)
            
            self.logger.info(f"Retrieved {len(results)} filtered results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in metadata-filtered search: {str(e)}")
            return []

    def initialize_ollama_llm(self, model_name: str) -> Ollama:
        """Initialize and return an Ollama LLM instance with the specified model."""
        try:
            ollama_llm = Ollama(
                model=model_name,
                base_url=self.app_settings.OLLAMA_BASE_URL
            )
            self.logger.info(f"Ollama LLM initialized with model '{model_name}'.")
            return ollama_llm
        except Exception as e:
            self.logger.error(f"Error initializing Ollama LLM with model '{model_name}': {e}")
            raise

    def embed_and_store_chunks(self, nodes: List[BaseNode]) -> bool:
        """Generate embeddings for chunks and store them in pgvector database with detailed logging"""
        if not nodes:
            self.logger.warning("No nodes provided for embedding.")
            return False

        try:
            self.logger.info(f"Starting to embed and store {len(nodes)} chunks...")
            
            # Log first node details for debugging
            first_node = nodes[0]
            self.logger.info(f"First node preview: {first_node.text[:100]}...")
            self.logger.info(f"First node metadata keys: {list(first_node.metadata.keys())}")
            
            # Ensure vector store is accessible
            self.logger.info(f"Vector store type: {type(self.vector_store)}")
            self.logger.info(f"Vector store table: data_document_chunks_project_{self.project_id}")
            
            # Test database connection
            try:
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    self.logger.info("Database connection test: SUCCESS")
            except Exception as db_error:
                self.logger.error(f"Database connection test: FAILED - {db_error}")
                return False
            
            # Process in small batches
            batch_size = 3  # Very small batches for reliability
            successful_chunks = 0
            
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i+batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(nodes) + batch_size - 1) // batch_size
                
                self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                
                try:
                    # Insert nodes using the index
                    self.logger.info(f"Inserting batch {batch_num} into index...")
                    self.index.insert_nodes(batch)
                    successful_chunks += len(batch)
                    self.logger.info(f"Batch {batch_num} inserted successfully")
                    
                    # Verify insertion by checking database
                    try:
                        with self.get_db_connection() as conn:
                            cursor = conn.cursor()
                            table_name = f"data_document_chunks_project_{self.project_id}"
                            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                            current_count = cursor.fetchone()[0]
                            self.logger.info(f"Current database count after batch {batch_num}: {current_count} chunks")
                    except Exception as verify_error:
                        self.logger.warning(f"Could not verify insertion for batch {batch_num}: {verify_error}")
                    
                except Exception as batch_error:
                    self.logger.error(f"Error processing batch {batch_num}: {batch_error}")
                    import traceback
                    self.logger.error(f"Batch error traceback: {traceback.format_exc()}")
                    # Continue with next batch

            # Final verification
            try:
                stats = self.get_database_statistics()
                final_count = stats.get('total_chunks', 0)
                self.logger.info(f"Final database verification: {final_count} total chunks")
                
                if final_count > 0:
                    self.logger.info(f"SUCCESS: Embedded and stored {final_count} chunks in vector database")
                    return True
                else:
                    self.logger.error(f"FAILURE: No chunks found in database after processing")
                    return False
                    
            except Exception as verify_error:
                self.logger.error(f"Final verification failed: {verify_error}")
                # Return True if we processed batches successfully, even if verification failed
                return successful_chunks > 0

        except Exception as e:
            self.logger.error(f"Error embedding and storing chunks: {e}")
            import traceback
            self.logger.error(f"Embedding traceback: {traceback.format_exc()}")
            return False

    def create_index_from_documents(self, documents: List[Document]) -> VectorStoreIndex:
        """Create index directly from documents following LlamaIndex documentation"""
        try:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # CRITICAL: Must pass embed_model parameter for Ollama embeddings
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model,  # REQUIRED for Ollama
                show_progress=True
            )
            
            self.logger.info(f"Created index from {len(documents)} documents")
            return index

        except Exception as e:
            self.logger.error(f"Error creating index from documents: {e}")
            raise

    def process_document_pipeline(self, file_id: str, use_enhanced_chunking: bool = True) -> bool:
        """Complete enhanced document processing pipeline with detailed logging"""
        try:
            self.logger.info(f"Starting document processing pipeline for {file_id}")
            
            # Step 1: Load documents with metadata extraction
            self.logger.info("Step 1: Loading documents...")
            documents = self.load_documents(file_id)
            if not documents:
                self.logger.error("Step 1 FAILED: No documents loaded")
                return False
            
            self.logger.info(f"Step 1 SUCCESS: Loaded {len(documents)} documents")
            for i, doc in enumerate(documents):
                self.logger.info(f"  Document {i+1}: {len(doc.text)} characters, doc_type: {doc.metadata.get('doc_type', 'Unknown')}")

            if use_enhanced_chunking:
                # Step 2: Enhanced chunking with metadata
                self.logger.info("Step 2: Enhanced chunking...")
                nodes = self.chunk_documents(documents)
                if not nodes:
                    self.logger.error("Step 2 FAILED: No nodes created from chunking")
                    return False
                
                self.logger.info(f"Step 2 SUCCESS: Created {len(nodes)} chunks")
                
                # Step 3: Store the enhanced nodes
                self.logger.info("Step 3: Storing chunks in vector database...")
                storage_success = self.embed_and_store_chunks(nodes)
                if not storage_success:
                    self.logger.error("Step 3 FAILED: Vector storage failed")
                    return False
                
                self.logger.info("Step 3 SUCCESS: Chunks stored successfully")
                
            else:
                # Direct approach: create index from documents
                self.logger.info("Using direct document indexing...")
                self._index = self.create_index_from_documents(documents)
            
            self.logger.info(f"Document processing pipeline completed successfully for {file_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error in document processing pipeline for {file_id}: {e}")
            import traceback
            self.logger.error(f"Pipeline traceback: {traceback.format_exc()}")
            return False


    def get_database_statistics(self) -> dict:
        """Get comprehensive database statistics with correct column names"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                table_name = f"data_document_chunks_project_{self.project_id}"
                
                # Check if table exists first
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    );
                """, (table_name,))
                
                table_exists = cursor.fetchone()[0]
                
                if not table_exists:
                    self.logger.warning(f"Table {table_name} does not exist yet")
                    return {
                        'table_exists': False,
                        'table_name': table_name,
                        'total_chunks': 0,
                        'unique_files': 0,
                        'document_types': 0,
                        'doc_type_distribution': {},
                        'priority_distribution': {},
                        'avg_chunk_length': 0
                    }
                
                # Get basic statistics using correct column name (metadata_)
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(DISTINCT(metadata_->>'source_file')) as unique_files,
                        COUNT(DISTINCT(metadata_->>'doc_type')) as document_types,
                        AVG(LENGTH(text)) as avg_chunk_length
                    FROM {table_name}
                """)
                
                basic_stats = cursor.fetchone()
                
                # Convert Decimal to float for JSON serialization
                def convert_decimal(value):
                    if hasattr(value, '__float__'):  # Check if it's a Decimal-like object
                        return float(value) if value is not None else 0.0
                    return value if value is not None else 0
                
                # Get document type distribution
                cursor.execute(f"""
                    SELECT 
                        COALESCE(metadata_->>'doc_type', 'Unknown') as doc_type,
                        COUNT(*) as count
                    FROM {table_name}
                    GROUP BY metadata_->>'doc_type'
                    ORDER BY count DESC
                """)
                
                doc_type_stats = cursor.fetchall()
                
                # Get priority distribution (for IA Standards)
                cursor.execute(f"""
                    SELECT 
                        metadata_->>'priority' as priority,
                        COUNT(*) as count
                    FROM {table_name}
                    WHERE metadata_->>'priority' IS NOT NULL
                    GROUP BY metadata_->>'priority'
                    ORDER BY priority
                """)
                
                priority_stats = cursor.fetchall()
                
                return {
                    'table_exists': True,
                    'total_chunks': int(basic_stats[0]) if basic_stats[0] else 0,
                    'unique_files': int(basic_stats[1]) if basic_stats[1] else 0,
                    'document_types': int(basic_stats[2]) if basic_stats[2] else 0,
                    'avg_chunk_length': round(convert_decimal(basic_stats[3]), 2),
                    'doc_type_distribution': {k: int(v) for k, v in doc_type_stats},
                    'priority_distribution': {k: int(v) for k, v in priority_stats},
                    'table_name': table_name
                }
                    
        except Exception as e:
            self.logger.error(f"Error getting database statistics: {e}")
            import traceback
            self.logger.error(f"Statistics error traceback: {traceback.format_exc()}")
            return {
                'error': str(e),
                'table_name': f"data_document_chunks_project_{self.project_id}"
            }
