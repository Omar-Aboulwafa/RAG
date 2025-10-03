#!/bin/bash

echo " Starting Agentic RAG with Native Reranking"
echo "============================================"

# Start infrastructure
echo " Starting infrastructure services..."
docker-compose up -d pgvector phoenix ollama

# Wait for services
echo " Waiting for services..."
sleep 20

# Only pull LLM and embedding models (no reranker needed)
echo " Pulling required Ollama models..."
echo "   Pulling LLM model: qwen2.5:0.5b"
docker exec rag_ollama ollama pull qwen2.5:0.5b

echo "   Pulling embedding model: mxbai-embed-large:latest"
docker exec rag_ollama ollama pull mxbai-embed-large:latest

echo " Ollama models downloaded!"

# Start backend API (will download reranker model on first build)
echo " Building and starting backend API..."
docker-compose up -d --build backend-api

# Wait for backend
echo " Waiting for backend API..."
sleep 30

# Start OpenWebUI
echo " Starting OpenWebUI..."
docker-compose up -d openwebui

echo ""
echo " SYSTEM READY!"
echo "============================================"
echo " Phoenix Observability: http://localhost:6006"
echo " Backend API:          http://localhost:8000"
echo " API Documentation:    http://localhost:8000/docs"
echo " OpenWebUI:            http://localhost:3001"
echo "  PostgreSQL:          localhost:5400"
echo ""
echo " Reranking Features:"
echo "  • Native reranking:     mixedbread-ai/mxbai-rerank-base-v1"
echo "  • Initial retrieval:    20 documents"
echo "  • Final results:        5 best documents"
echo "  • No Ollama dependency: Uses Hugging Face transformers"
echo ""
echo " Next steps:"
echo "1. Index documents:      cd 01-document-indexer && python main.py --directory ../assets/documents"
echo "2. Test reranking:       Open http://localhost:3001"
echo "3. Run evaluation:       cd 03-rag-evaluator && python main.py"
