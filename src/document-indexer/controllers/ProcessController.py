from .BaseController import BaseController
from .ProjectController import ProjectController
from .MetadataExtractor import MetadataExtractor  
from helpers.config import get_settings
from llama_index.readers.docling import DoclingReader
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
)
import os
from typing import List
import psycopg2
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from contextlib import contextmanager
from sqlalchemy import make_url
import logging


class ProcessController(BaseController):

    def __init__(self, project_id: str, default_llm_model: str = None):
        super().__init__()

        if default_llm_model is None:
            default_llm_model = self.app_settings.DEFAULT_LLM_MODEL

        self.default_llm_model = default_llm_model
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)

        # Initialize metadata extractor instance
        self.metadata_extractor = MetadataExtractor()

        # Initialize DoclingReader with your previous configuration
        self.reader = DoclingReader(
            export_type=DoclingReader.ExportType.JSON,
            keep_tables=True,
            keep_figures=True,
            table_structure_mode="accurate"
        )

        self.default_embedding_model = self.app_settings.DEFAULT_EMBEDDING_MODEL

        # Lazy initialized attributes
        self._llm = None
        self._vector_store = None
        self._index = None
        self._embed_model = None

        self.logger.info(f"ProcessController initialized for project: {project_id}")

    @property
    def embed_model(self) -> OllamaEmbedding:
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
        if self._vector_store is None:
            url = make_url(self.app_settings.DB_CONNECTION_STRING)
            self._vector_store = PGVectorStore.from_params(
                database=url.database,
                host=url.host,
                password=url.password,
                port=url.port,
                user=url.username,
                table_name=f"document_chunks_project_{self.project_id}",
                embed_dim=1024,
                hybrid_search=True,
                text_search_config="english",
                hnsw_kwargs={
                    "hnsw_m": 40,
                    "hnsw_ef_construction": 350,
                    "hnsw_ef_search": 350,
                    "hnsw_dist_method": "cosine",
                },
            )
            self.logger.info(f"Initialized PGVectorStore for project {self.project_id}")
        return self._vector_store

    @property
    def index(self) -> VectorStoreIndex:
        if self._index is None:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            try:
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=storage_context,
                    embed_model=self.embed_model
                )
                self.logger.info(f"Loaded existing index for project {self.project_id}")
            except Exception as ex:
                self.logger.warning(f"Could not load existing index: {ex}")
                self._index = VectorStoreIndex(
                    [],
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                    show_progress=True
                )
                self.logger.info(f"Created new index for project {self.project_id}")
        return self._index

    @property
    def llm(self) -> Ollama:
        if self._llm is None:
            self._llm = self.initialize_ollama_llm(self.default_llm_model)
        return self._llm

    @contextmanager
    def get_db_connection(self):
        db_connection_string = self.app_settings.DB_CONNECTION_STRING
        conn = None
        try:
            conn = psycopg2.connect(db_connection_string)
            yield conn
        except Exception:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def load_documents(self, file_id: str) -> List[Document]:
        app_settings = get_settings()
        project_upload_dir = app_settings.get_project_upload_path(self.project_id)
        file_path = os.path.join(project_upload_dir, file_id)

        self.logger.info(f"📄 Loading file: {file_path}")

        if not os.path.exists(file_path):
            self.logger.error(f"❌ File not found at: {file_path}")
            return []

        try:
            self.logger.info("📖 Reading document with DoclingReader...")
            documents = self.reader.load_data(file_path)
            self.logger.info(f"✅ DoclingReader loaded {len(documents)} raw documents")

            enhanced_docs = []
            for idx, doc in enumerate(documents):
                self.logger.info(f"🔧 Processing document {idx + 1}/{len(documents)}")

                # Add essential metadata
                doc.metadata["source_file"] = file_id
                doc.metadata["filename"] = file_id
                doc.metadata["file_path"] = file_path
                doc.metadata["project_id"] = self.project_id

                self.logger.info(f"📝 Text length: {len(doc.text)} chars")
                self.logger.info("🔍 Calling MetadataExtractor...")

                try:
                    enhanced_doc = self.metadata_extractor.extract_comprehensive_metadata(doc, self.project_id)
                    doc_type = enhanced_doc.metadata.get('doc_type', 'Unknown')
                    article = enhanced_doc.metadata.get('primary_article_number', 'None')
                    self.logger.info(f"✅ Metadata extracted - Type: {doc_type}, Article: {article}")
                    enhanced_docs.append(enhanced_doc)
                except Exception as e:
                    self.logger.error(f"❌ Metadata extraction failed: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    enhanced_docs.append(doc)

            self.logger.info(f"✅ Successfully enhanced {len(enhanced_docs)} documents")
            return enhanced_docs

        except Exception as e:
            self.logger.error(f"❌ Error loading document {file_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    def chunk_documents(self, documents: List[Document]) -> List[BaseNode]:
        from llama_index.core.schema import TextNode
        import re
        
        enhanced_nodes = []

        for doc in documents:
            doc_type = doc.metadata.get("doc_type", "Unknown")
            chunk_cfg = self.app_settings.get_chunk_settings_for_doc_type(doc_type) or {
                "chunk_size": 2000, "chunk_overlap": 400
            }

            size = chunk_cfg["chunk_size"]
            overlap = chunk_cfg["chunk_overlap"]
            raw_text = doc.text

            start = 0
            text_len = len(raw_text)
            while start < text_len:
                end = min(start + size, text_len)
                chunk_text = raw_text[start:end]

                new_node = TextNode(text=chunk_text, metadata=doc.metadata.copy())
                new_node.metadata["project_id"] = self.project_id

                # Extract article number FROM THIS CHUNK'S TEXT
                if doc_type == "HR Bylaw":
                    article_match = re.search(r'Article\s*\(?(\d+)\)?', chunk_text, re.IGNORECASE)
                    if article_match:
                        article_num = article_match.group(1)
                        new_node.metadata["primary_article_number"] = article_num
                        new_node.metadata["citation_id"] = f"Article {article_num}"
                        
                        # Check for sixty days flag
                        if re.search(r'sixty\s+days?|60\s+days?', chunk_text, re.IGNORECASE):
                            new_node.metadata["contains_maximum_deduction"] = True

                chunk_type = doc_type.replace(" ", "_").lower()
                new_node.metadata["chunk_type"] = f"chunk_{chunk_type}"
                new_node.metadata["citation_ready"] = True

                enhanced_nodes.append(new_node)
                start += max(1, size - overlap)

            self.logger.info(f"Created {len(enhanced_nodes)} document-aware chunks")
            return enhanced_nodes


    def metadata_filtered_search(self, query: str, filters: dict = None, top_k: int = 10) -> List[dict]:
        try:
            metadata_filter_list = []
            if filters:
                if 'doc_type' in filters:
                    metadata_filter_list.append(MetadataFilter(key="doc_type", value=filters['doc_type'], operator=FilterOperator.EQ))
                if 'priority' in filters:
                    op = FilterOperator.IN if isinstance(filters['priority'], list) else FilterOperator.EQ
                    metadata_filter_list.append(MetadataFilter(key="priority", value=filters['priority'], operator=op))
                if 'process_group' in filters:
                    op = FilterOperator.IN if isinstance(filters['process_group'], list) else FilterOperator.EQ
                    metadata_filter_list.append(MetadataFilter(key="primary_process_group", value=filters['process_group'], operator=op))
                if 'control_id' in filters:
                    metadata_filter_list.append(MetadataFilter(key="primary_control_id", value=filters['control_id'], operator=FilterOperator.EQ))
                if 'citation_id' in filters:
                    metadata_filter_list.append(MetadataFilter(key="citation_id", value=filters['citation_id'], operator=FilterOperator.EQ))
                if 'business_role' in filters:
                    metadata_filter_list.append(MetadataFilter(key="business_roles", value=filters['business_role'], operator=FilterOperator.CONTAINS))
                if 'software_system' in filters:
                    metadata_filter_list.append(MetadataFilter(key="software_systems", value=filters['software_system'], operator=FilterOperator.CONTAINS))

            metadata_filters = MetadataFilters(filters=metadata_filter_list, condition=FilterCondition.AND) if metadata_filter_list else None

            retriever = self.index.as_retriever(similarity_top_k=top_k, filters=metadata_filters)
            retrieved_nodes = retriever.retrieve(query)

            results = [{
                'text': node.text,
                'score': getattr(node, 'score', 0.0),
                'metadata': node.metadata,
                'citation_id': node.metadata.get('citation_id', ''),
                'doc_type': node.metadata.get('doc_type', 'Unknown'),
                'section_path': node.metadata.get('section_path', ''),
                'priority': node.metadata.get('priority', ''),
                'process_group': node.metadata.get('primary_process_group', ''),
                'control_id': node.metadata.get('primary_control_id', ''),
                'business_roles': node.metadata.get('business_roles', []),
                'software_systems': node.metadata.get('software_systems', [])
            } for node in retrieved_nodes]

            self.logger.info(f"Retrieved {len(results)} filtered results for query: {query[:50]}...")
            return results

        except Exception as e:
            self.logger.error(f"Error in metadata-filtered search: {e}")
            return []

    def initialize_ollama_llm(self, model_name: str) -> Ollama:
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
        if not nodes:
            self.logger.warning("No nodes provided for embedding.")
            return False
        try:
            self.logger.info(f"Starting to embed and store {len(nodes)} chunks...")
            first_node = nodes[0]
            self.logger.info(f"First node preview: {first_node.text[:100]}...")
            self.logger.info(f"First node metadata keys: {list(first_node.metadata.keys())}")
            self.logger.info(f"Vector store type: {type(self.vector_store)}")
            self.logger.info(f"Vector store table: document_chunks_project_{self.project_id}")
            try:
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    self.logger.info("Database connection test: SUCCESS")
            except Exception as db_error:
                self.logger.error(f"Database connection test: FAILED - {db_error}")
                return False
            batch_size = 10 
            successful_chunks = 0
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i+batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(nodes) + batch_size - 1) // batch_size
                self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                try:
                    self.logger.info(f"Inserting batch {batch_num} into index...")
                    self.index.insert_nodes(batch)
                    successful_chunks += len(batch)
                    self.logger.info(f"Batch {batch_num} inserted successfully")
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
                return successful_chunks > 0
        except Exception as e:
            self.logger.error(f"Error embedding and storing chunks: {e}")
            import traceback
            self.logger.error(f"Embedding traceback: {traceback.format_exc()}")
            return False

    def create_index_from_documents(self, documents: List[Document]) -> VectorStoreIndex:
        try:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True
            )
            self.logger.info(f"Created index from {len(documents)} documents")
            return index
        except Exception as e:
            self.logger.error(f"Error creating index from documents: {e}")
            raise

    def process_document_pipeline(self, file_id: str, use_enhanced_chunking: bool = True) -> bool:
        try:
            self.logger.info(f"Starting document processing pipeline for {file_id}")
            self.logger.info("Step 1: Loading documents...")
            documents = self.load_documents(file_id)
            if not documents:
                self.logger.error("Step 1 FAILED: No documents loaded")
                return False
            self.logger.info(f"Step 1 SUCCESS: Loaded {len(documents)} documents")
            for i, doc in enumerate(documents):
                self.logger.info(f"  Document {i+1}: {len(doc.text)} characters, doc_type: {doc.metadata.get('doc_type', 'Unknown')}")
            if use_enhanced_chunking:
                self.logger.info("Step 2: Enhanced chunking...")
                nodes = self.chunk_documents(documents)
                if not nodes:
                    self.logger.error("Step 2 FAILED: No nodes created from chunking")
                    return False
                self.logger.info(f"Step 2 SUCCESS: Created {len(nodes)} chunks")
                self.logger.info("Step 3: Storing chunks in vector database...")
                storage_success = self.embed_and_store_chunks(nodes)
                if not storage_success:
                    self.logger.error("Step 3 FAILED: Vector storage failed")
                    return False
                self.logger.info("Step 3 SUCCESS: Chunks stored successfully")
            else:
                self.logger.info("Using direct document indexing...")
                self._index = self.create_index_from_documents(documents)
            self.logger.info(f"Document processing pipeline completed successfully for {file_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error in document processing pipeline for {file_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def get_database_statistics(self) -> dict:
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                table_name = f"data_document_chunks_project_{self.project_id}"
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
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(DISTINCT(metadata_->>'source_file')) as unique_files,
                        COUNT(DISTINCT(metadata_->>'doc_type')) as document_types,
                        AVG(LENGTH(text)) as avg_chunk_length
                    FROM {table_name}
                """)
                basic_stats = cursor.fetchone()

                def convert_decimal(value):
                    if hasattr(value, '__float__'):
                        return float(value) if value is not None else 0.0
                    return value if value is not None else 0

                cursor.execute(f"""
                    SELECT 
                        COALESCE(metadata_->>'doc_type', 'Unknown') as doc_type,
                        COUNT(*) as count
                    FROM {table_name}
                    GROUP BY metadata_->>'doc_type'
                    ORDER BY count DESC
                """)
                doc_type_stats = cursor.fetchall()
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
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'table_name': f"data_document_chunks_project_{self.project_id}"
            }
