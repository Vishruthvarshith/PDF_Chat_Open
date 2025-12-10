#!/bin/bash

echo "Starting Document Analysis RAG Application..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  Ollama is not running. Please start Ollama first:"
    echo "   ollama serve"
    echo ""
    echo "   And make sure you have the required models:"
    echo "   ollama pull llama3.1:8b"
    echo "   ollama pull nomic-embed-text"
    exit 1
fi

echo "✅ Ollama is running"

# Activate virtual environment
source venv/bin/activate

echo "🚀 Starting FastAPI application..."
echo "📱 Web interface will be available at: http://localhost:8000"
echo "📚 API documentation at: http://localhost:8000/docs"
echo ""

# Start the application
python app/main.py