#!/bin/bash

# Create necessary directories
mkdir -p docs 

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

echo "Starting Course Materials RAG System..."
echo "Make sure Ollama is running and the configured model is pulled (e.g. 'ollama pull llama3.1')"

# Change to backend directory and start the server
cd backend && uv run uvicorn app:app --reload --port 8000