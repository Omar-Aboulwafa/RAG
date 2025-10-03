"""
STANDALONE APPLICATION: Document Indexer Console
Works with your existing ProcessController and metadata extraction
"""

import sys
import os
import argparse
import shutil
from pathlib import Path
import logging
from typing import List

# Local imports
from helpers.config import get_settings
from controllers.ProcessController import ProcessController
from controllers.DataController import DataController

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StandaloneDocumentIndexer:
    """Standalone document indexer using ProcessController"""
    
    def __init__(self, project_id: str = "rag"):
        self.settings = get_settings()
        self.project_id = project_id
        self.process_controller = ProcessController(project_id=project_id)
        
        # Ensure project upload directory exists
        self.upload_dir = self.settings.get_project_upload_path(project_id)
        os.makedirs(self.upload_dir, exist_ok=True)
        
        logger.info(f"✅ Initialized Document Indexer for project: {project_id}")
        logger.info(f"📁 Upload directory: {self.upload_dir}")
        
    def validate_database_connection(self) -> bool:
        """Test database connectivity"""
        try:
            import psycopg2
            conn = psycopg2.connect(self.settings.DB_CONNECTION_STRING)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            logger.info("✅ Database connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def copy_to_project_dir(self, file_path: Path) -> str:
        """Copy file to project upload directory and return new filename"""
        import time
        timestamp = int(time.time())
        new_filename = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        destination = Path(self.upload_dir) / new_filename
        
        shutil.copy2(file_path, destination)
        logger.info(f"📋 Copied to: {new_filename}")
        return new_filename
    
    def index_directory(self, directory_path: str) -> bool:
        """Index all supported documents in directory"""
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"❌ Directory not found: {directory_path}")
            return False
        
        # Find all PDF documents (primary format for HR Bylaws, etc.)
        supported_extensions = ['.pdf', '.docx', '.txt']
        document_files = []
        
        for ext in supported_extensions:
            document_files.extend(directory.glob(f"**/*{ext}"))
        
        if not document_files:
            logger.warning(f"⚠️  No supported documents found in {directory_path}")
            logger.info(f"Supported extensions: {supported_extensions}")
            return False
        
        logger.info(f"📚 Found {len(document_files)} documents to process")
        
        # Process each document
        success_count = 0
        failed_files = []
        
        for doc_file in document_files:
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"📄 Processing: {doc_file.name}")
                logger.info(f"{'='*70}")
                
                # Copy file to project directory
                file_id = self.copy_to_project_dir(doc_file)
                
                # Use ProcessController.process_document_pipeline()
                success = self.process_controller.process_document_pipeline(
                    file_id=file_id,
                    use_enhanced_chunking=True
                )
                
                if success:
                    success_count += 1
                    logger.info(f"✅ Successfully indexed: {doc_file.name}")
                else:
                    logger.error(f"❌ Failed to index: {doc_file.name}")
                    failed_files.append(doc_file.name)
                    
            except Exception as e:
                logger.error(f"❌ Error processing {doc_file.name}: {e}")
                failed_files.append(doc_file.name)
                import traceback
                logger.error(traceback.format_exc())
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 INDEXING COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"✅ Success: {success_count}/{len(document_files)} documents")
        logger.info(f"❌ Failed: {len(failed_files)}/{len(document_files)} documents")
        logger.info(f"📈 Success rate: {(success_count/len(document_files)*100):.1f}%")
        
        if failed_files:
            logger.info(f"\n❌ Failed files:")
            for filename in failed_files:
                logger.info(f"   - {filename}")
        
        return success_count > 0
    
    def index_single_file(self, file_path: str) -> bool:
        """Index a single document file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            return False
        
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"📄 Processing single file: {file_path.name}")
            logger.info(f"{'='*70}")
            
            # Copy to project directory
            file_id = self.copy_to_project_dir(file_path)
            
            # Process document
            success = self.process_controller.process_document_pipeline(
                file_id=file_id,
                use_enhanced_chunking=True
            )
            
            if success:
                logger.info(f"✅ Successfully indexed: {file_path.name}")
            else:
                logger.error(f"❌ Failed to index: {file_path.name}")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def show_statistics(self):
        """Display indexing statistics"""
        try:
            stats = self.process_controller.get_database_statistics()
            
            logger.info(f"\n{'='*70}")
            logger.info(f"📊 DATABASE STATISTICS - Project: {self.project_id}")
            logger.info(f"{'='*70}")
            
            if stats.get('table_exists'):
                logger.info(f"✅ Table: {stats.get('table_name')}")
                logger.info(f"📦 Total chunks: {stats.get('total_chunks', 0)}")
                logger.info(f"📁 Unique files: {stats.get('unique_files', 0)}")
                logger.info(f"📚 Document types: {stats.get('document_types', 0)}")
                logger.info(f"📏 Avg chunk length: {stats.get('avg_chunk_length', 0)} chars")
                
                # Document type distribution
                doc_types = stats.get('doc_type_distribution', {})
                if doc_types:
                    logger.info(f"\n📋 Document Type Distribution:")
                    for doc_type, count in doc_types.items():
                        logger.info(f"   - {doc_type}: {count} chunks")
                
                # Priority distribution (for IA Standards)
                priorities = stats.get('priority_distribution', {})
                if priorities:
                    logger.info(f"\n⚡ Priority Distribution:")
                    for priority, count in priorities.items():
                        logger.info(f"   - {priority}: {count} chunks")
            else:
                logger.info(f"⚠️  Table does not exist: {stats.get('table_name')}")
                logger.info(f"💡 Run indexing to create the table")
                
        except Exception as e:
            logger.error(f"❌ Error querying database: {e}")
            import traceback
            logger.error(traceback.format_exc())


def main():
    """Main console application entry point"""
    parser = argparse.ArgumentParser(
        description='Standalone Document Indexer Console Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test database connection
  python main.py --test-db

  # Index all documents in a directory
  python main.py --directory /path/to/documents --project-id rag

  # Index a single file
  python main.py --file HR_Bylaws.pdf --project-id rag

  # Show statistics
  python main.py --stats --project-id rag
        """
    )
    
    parser.add_argument('--directory', '-d', 
                       help='Directory containing documents to index')
    parser.add_argument('--file', '-f', 
                       help='Single file to index')
    parser.add_argument('--stats', '-s', action='store_true',
                       help='Show indexing statistics')
    parser.add_argument('--project-id', default='rag',
                       help='Project ID for indexing (default: rag)')
    parser.add_argument('--test-db', action='store_true',
                       help='Test database connection')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🗂️  STANDALONE DOCUMENT INDEXER APPLICATION")
    print("=" * 70)
    print(f"📋 Project ID: {args.project_id}")
    print("=" * 70 + "\n")
    
    # Initialize indexer
    try:
        indexer = StandaloneDocumentIndexer(project_id=args.project_id)
    except Exception as e:
        logger.error(f"❌ Failed to initialize indexer: {e}")
        sys.exit(1)
    
    # Handle different operations
    if args.test_db:
        print("🔍 Testing database connection...\n")
        if indexer.validate_database_connection():
            print("✅ Database connection test PASSED\n")
        else:
            print("❌ Database connection test FAILED\n")
            sys.exit(1)
            
    elif args.stats:
        print(f"📊 Showing statistics for project: {args.project_id}\n")
        indexer.show_statistics()
        
    elif args.directory:
        print(f"📁 Indexing directory: {args.directory}\n")
        success = indexer.index_directory(args.directory)
        sys.exit(0 if success else 1)
        
    elif args.file:
        print(f"📄 Indexing file: {args.file}\n")
        success = indexer.index_single_file(args.file)
        sys.exit(0 if success else 1)
        
    else:
        print("⚠️  No operation specified.")
        print("\nQuick start commands:")
        print("  python main.py --test-db")
        print("  python main.py --file HR_Bylaws.pdf")
        print("  python main.py --directory ../documents")
        print("  python main.py --stats")
        print("\nUse --help for full usage instructions\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
