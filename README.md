# 🚀 Document Analysis RAG Application

A powerful FastAPI + LangChain + VectorDB application that allows users to upload documents, chat about them using Llama 3.1:8b, and visualize data patterns.

## ✨ Features

- **📄 Document Upload**: Support for PDF, CSV, Excel, TXT, JSON, Markdown, and Word files
- **🤖 Intelligent Chat**: RAG-powered conversations with document context and source citation
- **📊 Data Visualization**: Automatic chart generation and data analysis
- **💬 Conversation Memory**: Session-based chat history with context awareness
- **🔍 Semantic Search**: Advanced document retrieval using ChromaDB
- **📱 Modern UI**: Responsive web interface with real-time updates

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.12+
- **LLM**: Llama 3.1:8b via Ollama
- **Vector Database**: ChromaDB with persistent storage
- **Document Processing**: LangChain + Unstructured
- **Visualization**: Plotly.js + Matplotlib
- **Frontend**: HTML/CSS/JavaScript with Tailwind CSS

## 📋 Prerequisites

### 1. Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Verify Python version (3.12+ recommended)
python --version
```

### 2. Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Start Ollama server
ollama serve
```

### 3. Application Dependencies
```bash
# Install Python packages
pip install -r requirements.txt
```

## 🚀 Quick Start

### Option 1: Using Start Script (Recommended)
```bash
# Make sure Ollama is running first
./start.sh
```

### Option 2: Manual Start
```bash
# 1. Start Ollama (if not already running)
ollama serve

# 2. In another terminal, start the application
mac: source venv/bin/activate #for windows ./venv/Scripts/activate
python app/main.py
```

## 🌐 Access Points

Once running, access the application at:

- **📱 Web Interface**: http://localhost:8000
- **📚 API Documentation**: http://localhost:8000/docs
- **🔍 Interactive API**: http://localhost:8000/redoc
- **💚 Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
document-chat-app/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── core/
│   │   └── config.py       # Configuration management
│   ├── services/
│   │   ├── llm_service.py      # LLM integration
│   │   ├── document_service.py # Document processing
│   │   ├── rag_service.py      # RAG pipeline
│   │   └── visualization_service.py # Chart generation
│   ├── db/
│   │   └── vector_store.py   # ChromaDB integration
│   ├── api/routes/
│   │   ├── chat.py          # Chat endpoints
│   │   ├── upload.py        # Document upload
│   │   └── visualization.py # Data viz endpoints
│   ├── models.py              # Pydantic models
│   └── utils/
│       └── helpers.py       # Utility functions
├── static/
│   └── index.html          # Web interface
├── uploads/                  # Document storage
├── vector_db/               # ChromaDB storage
├── requirements.txt          # Dependencies
├── .env                   # Environment variables
├── start.sh               # Startup script
└── test_setup.py          # Setup verification
```

## 🔧 Configuration

### Environment Variables (.env)
```env
# API Configuration
API_V1_STR=/api/v1
PROJECT_NAME=Document Analysis RAG
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text

# Vector Database
CHROMA_PERSIST_DIRECTORY=./vector_db
CHROMA_COLLECTION_NAME=documents

# File Upload
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=52428800  # 50MB

# Security
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"]
```

## 📚 API Endpoints

### Document Management
- `POST /api/v1/upload` - Upload documents
- `GET /api/v1/documents` - List uploaded documents
- `GET /api/v1/documents/{id}` - Get document details
- `DELETE /api/v1/documents/{id}` - Delete document

### Chat & Query
- `POST /api/v1/chat` - Chat with documents
- `POST /api/v1/chat/stream` - Streaming chat
- `GET /api/v1/chat/history/{session_id}` - Get chat history
- `DELETE /api/v1/chat/history/{session_id}` - Clear chat history

### System
- `GET /api/v1/stats` - System statistics
- `GET /health` - Health check

## 💡 Usage Examples

### Upload Documents
```bash
# Upload single file
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@document.pdf"

# Upload multiple files
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@doc1.pdf" \
  -F "file=@doc2.csv"
```

### Chat with Documents
```bash
# Send chat message
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the key findings in the uploaded documents?"}'
```

### Stream Chat
```bash
# Streaming chat response
curl -X POST "http://localhost:8000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize the main points"}' \
  --no-buffer
```

## 🧪 Testing

### Run Setup Tests
```bash
# Verify all components are working
python test_setup.py
```

Expected output:
```
Document Analysis RAG - Setup Test
==================================================
Configuration loaded successfully
LLM Service initialized successfully
Vector Store Service initialized successfully
Document Processing Service initialized successfully
RAG Service initialized successfully
Ollama is running
Llama 3.1:8b model is available
Nomic-embed-text model is available
Upload directory exists and is writable
Vector DB directory exists and is writable
LLM test response: Hello World
Token count test: 10 tokens

==================================================
Test Results: 4/4 tests passed
All tests passed! The application is ready to run.
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Ollama Connection Failed
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start Ollama
ollama serve

# Check available models
ollama list
```

#### 2. Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
which python
python --version
```

#### 3. Permission Errors
```bash
# Check directory permissions
ls -la uploads/
ls -la vector_db/

# Fix permissions if needed
chmod 755 uploads/
chmod 755 vector_db/
```

#### 4. Model Not Found
```bash
# Pull missing models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Verify models
ollama list
```

### Debug Mode
Enable debug mode in `.env`:
```env
DEBUG=true
```

This provides detailed error logs and stack traces.

## 🐳 Docker Deployment (Optional)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Create directories
RUN mkdir -p uploads vector_db

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "app/main.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./vector_db:/app/vector_db
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
```

## 📊 Performance Optimization

### Recommended Settings
- **Chunk Size**: 1000 characters with 200 overlap
- **Retrieval**: Top 4 most relevant documents
- **Model Temperature**: 0.3 for balanced creativity
- **Max File Size**: 50MB per document

### Scaling Considerations
- Use Redis for session storage in production
- Implement document queuing for batch processing
- Add caching for frequent queries
- Consider GPU acceleration for embeddings

## 🔒 Security Features

- **File Validation**: Type and size restrictions
- **Input Sanitization**: XSS and injection prevention
- **CORS Configuration**: Restricted origins
- **Rate Limiting**: Prevent abuse (implement as needed)
- **Secure Headers**: Security headers included

## 📈 Monitoring & Logging

### Application Logs
```bash
# View application logs
tail -f logs/app.log

# Enable debug logging
export DEBUG=true
python main.py
python run.py
```

### Health Monitoring
- `/health` endpoint provides system status
- `/api/v1/stats` shows usage statistics
- Monitor Ollama service separately

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For issues and questions:
- Check the troubleshooting section
- Review application logs
- Verify Ollama and model status
- Create an issue with detailed information

---

**🎉 Ready to build your document analysis RAG application!**
