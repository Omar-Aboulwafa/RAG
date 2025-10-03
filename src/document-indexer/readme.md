# Document Indexer Application

## Overview
Standalone console application for indexing documents into PostgreSQL with pgvector. This application processes PDF, DOCX, and TXT files using Docling and LlamaIndex.

## Features
- ✅ **Multi-format Support**: PDF, DOCX, TXT documents
- ✅ **Batch Processing**: Index entire directories
- ✅ **Database Integration**: PostgreSQL with pgvector
- ✅ **Progress Tracking**: Detailed logging and statistics
- ✅ **Error Handling**: Robust error recovery and reporting

## Prerequisites
- Python 3.11+
- PostgreSQL with pgvector extension
- Ollama running with embedding model
- Environment variables configured in parent `.env` file

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Database Connection
```bash
python main.py --test-db
```

## Usage

### Index Directory of Documents
```bash
# Index all documents in assets/documents
python main.py --directory ../assets/documents

# Index documents for specific project
python main.py --directory ../assets/documents --project-id procurement
```

### Index Single File
```bash
# Index a single PDF file
python main.py --file /path/to/document.pdf

# Index single file with project ID
python main.py --file document.docx --project-id compliance
```

### View Statistics
```bash
# Show indexing statistics for default project
python main.py --stats

# Show statistics for specific project
python main.py --stats --project-id procurement
```

### Test Database Connection
```bash
python main.py --test-db
```

## Configuration

The application reads configuration from `../env` file:

```env
# Database
DB_CONNECTION_STRING=postgresql://postgres:postgres@localhost:5400/rag_db

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_EMBEDDING_MODEL=mxbai-embed-large:latest

# File Processing
FILE_ALLOWED_TYPES=pdf,docx,txt
FILE_MAX_SIZE=10485760
```

## Output

The application provides detailed progress information:

```
🗂️  STANDALONE DOCUMENT INDEXER APPLICATION
======================================================================
Project ID: default

📁 Indexing directory: ../assets/documents
📄 Processing: policy-document.pdf
✅ Successfully indexed: policy-document.pdf
📄 Processing: procurement-guidelines.docx  
✅ Successfully indexed: procurement-guidelines.docx

📊 INDEXING COMPLETE
Success: 2/2 documents
Success rate: 100.0%
```

## Error Handling

The application handles various error scenarios:
- Invalid file paths
- Database connection issues  
- Unsupported file formats
- Processing failures
- Network connectivity problems

## Integration

This application integrates with:
- **Application 2**: Backend API queries indexed documents
- **Application 3**: RAG Evaluator tests retrieval quality
- **PostgreSQL**: Vector storage with pgvector
- **Ollama**: Embedding generation

## Database Schema

Documents are stored in table format:
```sql
Table: data_{project_id}
- id: UUID (primary key)
- content: TEXT (document chunk)
- metadata: JSONB (file info, section, etc.)
- embedding: VECTOR(1024) (pgvector embedding)
```

## Troubleshooting

### Database Connection Failed
- Verify PostgreSQL is running on port 5400
- Check credentials in `.env` file
- Ensure pgvector extension is installed

### Ollama Connection Failed  
- Verify Ollama is running on port 11434
- Check if embedding model is available: `ollama list`
- Pull model if needed: `ollama pull mxbai-embed-large:latest`

### No Documents Found
- Check file extensions are supported (pdf, docx, txt)
- Verify directory path exists
- Ensure files are readable

### Processing Failures
- Check file permissions
- Verify file isn't corrupted
- Monitor disk space
- Review application logs for specific errors

## Performance

Typical processing performance:
- **PDF files**: 50-100 pages/minute
- **DOCX files**: 100-200 pages/minute  
- **TXT files**: 500+ pages/minute
- **Memory usage**: 200-500MB depending on file size
- **Database writes**: 10-50 chunks/second

## Support

For issues or questions:
1. Check application logs for detailed error messages
2. Verify all prerequisites are installed
3. Test database and Ollama connectivity
4. Review configuration settings




#For Console/Batch Operations

# Test database connection
python main.py --test-db --project-id test

# Index a directory of documents
python main.py --directory ./assets/documents --project-id regulatory

# Index a single file
python main.py --file document.pdf --project-id test

# Show statistics
python main.py --stats --project-id test



# Start the API server
python api_main.py

# Or with uvicorn directly
uvicorn api_main:app --reload --port 5000


