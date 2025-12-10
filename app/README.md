# Document Analysis RAG Application

## 🚀 Quick Start

### Option 1: Run from app directory (Recommended)
```bash
cd app
python run.py
```

### Option 2: Run from project root
```bash
source venv/bin/activate
PYTHONPATH=/Users/vishruth/Desktop/virtual python app/main.py
```

### Option 3: Use the start script
```bash
./start.sh
```

## 🌐 Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📁 What this does

The `run.py` script automatically:
- ✅ Finds and activates the virtual environment
- ✅ Sets up the correct Python path
- ✅ Starts the FastAPI server with hot reload
- ✅ Serves the web interface and API

## 🛑 To Stop

Press `Ctrl+C` in the terminal where the server is running.

## 🔧 Troubleshooting

If you get import errors, make sure you're running from the `app/` directory:
```bash
cd /Users/vishruth/Desktop/virtual/app
python run.py
```