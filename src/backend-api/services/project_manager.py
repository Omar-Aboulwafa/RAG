# services/project_manager.py - Dynamic Project Discovery & Routing
import re
import logging
import psycopg2
from typing import List, Dict, Optional, Tuple
from config import get_settings

logger = logging.getLogger(__name__)

class ProjectManager:
    """Dynamic project discovery and routing service"""
    
    def __init__(self):
        self.settings = get_settings()
        self._cached_projects = None
        self._cache_ttl = 300  # 5 minutes cache
        self._last_cache_time = 0
        
    def discover_projects(self) -> List[str]:
        """Auto-discover all available projects from database tables"""
        try:
            conn = psycopg2.connect(self.settings.DB_CONNECTION_STRING)
            cursor = conn.cursor()
            
            # Query for all project tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'data_document_chunks_project_%'
            """)
            
            tables = cursor.fetchall()
            projects = []
            
            # Extract project names from table names
            for (table_name,) in tables:
                match = re.match(r'data_document_chunks_project_(.+)', table_name)
                if match:
                    projects.append(match.group(1))
            
            cursor.close()
            conn.close()
            
            logger.info(f"🔍 Discovered projects: {projects}")
            return projects
            
        except Exception as e:
            logger.error(f"Error discovering projects: {e}")
            return []
    
    def get_projects_with_stats(self) -> Dict[str, Dict]:
        """Get all projects with document counts and metadata"""
        projects_info = {}
        projects = self.discover_projects()
        
        if not projects:
            return {}
        
        try:
            conn = psycopg2.connect(self.settings.DB_CONNECTION_STRING)
            cursor = conn.cursor()
            
            for project_id in projects:
                table_name = f"data_document_chunks_project_{project_id}"
                
                cursor.execute(f"""
                SELECT
                    COUNT(*) as chunk_count,
                    COUNT(DISTINCT COALESCE(metadata_->>'file_name', 'unnamed')) as file_count,
                    COUNT(DISTINCT metadata_->>'doc_type') as doc_types  
                FROM {table_name}
                WHERE metadata_ IS NOT NULL
                """)
                
                stats = cursor.fetchone()
                
                cursor.execute(f"""
                SELECT
                    metadata_->>'doc_type' as doc_type,
                    COUNT(*) as count
                FROM {table_name} 
                WHERE metadata_->>'doc_type' IS NOT NULL
                GROUP BY metadata_->>'doc_type'
                LIMIT 10
                """)
                
                doc_types = dict(cursor.fetchall())
                
                projects_info[project_id] = {
                    "chunk_count": stats[0] if stats else 0,
                    "file_count": stats[1] if stats else 0, 
                    "doc_type_count": stats[2] if stats else 0,
                    "document_types": doc_types,
                    "status": "active" if stats and stats[0] > 0 else "empty"
                }
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error getting project stats: {e}")
            
        return projects_info


    
    def extract_project_from_request(self, request: Dict) -> Tuple[str, str]:
        """Smart project extraction with multiple fallback strategies"""
        
        # Strategy 1: Direct project parameter
        if 'project_id' in request:
            project_id = request['project_id']
            return project_id, "direct_parameter"
        
        # Strategy 2: Extract from model name
        model_name = request.get('model', 'regulatory-rag')
        project_from_model = self._extract_project_from_model(model_name)
        if project_from_model:
            return project_from_model, "model_name"
            
        # Strategy 3: Extract from session ID
        session_id = request.get('session_id', '')
        if session_id and '_' in session_id:
            potential_project = session_id.split('_')[0]
            if self._is_valid_project(potential_project):
                return potential_project, "session_id"
        
        # Strategy 4: Extract from user ID or custom headers
        user_id = request.get('user', request.get('user_id', ''))
        if user_id and self._is_valid_project(user_id):
            return user_id, "user_id"
            
        # Strategy 5: Intelligent fallback - use most active project
        fallback_project = self._get_fallback_project()
        return fallback_project, "intelligent_fallback"
    
    def _extract_project_from_model(self, model_name: str) -> Optional[str]:
        """Extract project from model name using patterns"""
        patterns = [
            r'(.+)-rag$',           # project-rag
            r'rag-(.+)$',           # rag-project  
            r'(.+)-model$',         # project-model
            r'model-(.+)$',         # model-project
            r'^(.+)$'               # exact project name
        ]
        
        for pattern in patterns:
            match = re.match(pattern, model_name, re.IGNORECASE)
            if match:
                potential_project = match.group(1).lower()
                if self._is_valid_project(potential_project):
                    return potential_project
        
        return None
    
    def _is_valid_project(self, project_id: str) -> bool:
        """Check if project exists in database"""
        if not project_id:
            return False
            
        available_projects = self.get_cached_projects()
        return project_id in available_projects
    
    def get_cached_projects(self) -> List[str]:
        """Get projects with caching for performance"""
        import time
        current_time = time.time()
        
        if (self._cached_projects is None or 
            current_time - self._last_cache_time > self._cache_ttl):
            
            self._cached_projects = self.discover_projects()
            self._last_cache_time = current_time
            
        return self._cached_projects or []
    
    def _get_fallback_project(self) -> str:
        """Intelligent fallback - return most active project"""
        projects_info = self.get_projects_with_stats()
        
        if not projects_info:
            return "default"
        
        # Return project with most documents
        best_project = max(
            projects_info.items(),
            key=lambda x: x[1].get('chunk_count', 0)
        )
        
        logger.info(f"🎯 Using fallback project: {best_project[0]} ({best_project[1]['chunk_count']} chunks)")
        return best_project[0]
    
    def create_project_if_needed(self, project_id: str) -> bool:
        """Auto-create project table if it doesn't exist (for indexing)"""
        try:
            conn = psycopg2.connect(self.settings.DB_CONNECTION_STRING)
            cursor = conn.cursor()
            
            table_name = f"data_document_chunks_project_{project_id}"
            
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                )
            """, (table_name,))
            
            exists = cursor.fetchone()[0]
            
            if not exists:
                logger.info(f"📋 Creating new project table: {table_name}")
                # Table creation would happen in your indexer service
                cursor.close()
                conn.close()
                return False  # Indicate table needs creation
            
            cursor.close()
            conn.close()
            return True  # Table exists
            
        except Exception as e:
            logger.error(f"Error checking/creating project {project_id}: {e}")
            return False

# Global project manager instance
project_manager = ProjectManager()

def get_project_manager() -> ProjectManager:
    """Get global project manager instance"""
    return project_manager
