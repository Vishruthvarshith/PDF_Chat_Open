from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from typing import List, Optional
import os
import shutil
import uuid
import sys
from datetime import datetime

# Add parent directory to path so we can run from app/ folder
# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.core.config import settings
from app.models import (
    UploadResponse,
    DocumentInfo,
    ChatRequest,
    ChatResponse,
    VisualizationRequest,
    VisualizationResponse,
    AnalysisRequest,
    AnalysisResponse,
    DocumentMetadata,
    DocumentType,
)
from app.services.document_service import DocumentProcessingService
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from app.services.visualization_service import VisualizationService
from app.services.analysis_service import AnalysisService
from app.db.vector_store import VectorStoreService
from app.utils.helpers import (
    validate_file,
    generate_document_id,
    get_file_extension,
    format_file_size,
    sanitize_filename,
    create_upload_directory,
    get_document_path,
    extract_text_preview,
)


def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    import os

    return os.path.splitext(filename)[1]


# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Document Analysis RAG Application with Llama 3.1:8b",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (handle both running from root and app/ directory)
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
static_dir = os.path.abspath(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize services lazily
document_service = None
rag_service = None
llm_service = None
visualization_service = None
analysis_service = None
vector_store = None


def get_document_service():
    global document_service
    if document_service is None:
        document_service = DocumentProcessingService()
    return document_service


def get_rag_service():
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service


def get_llm_service():
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
    return llm_service


def get_vector_store():
    global vector_store
    if vector_store is None:
        vector_store = VectorStoreService()
    return vector_store


def get_visualization_service():
    global visualization_service
    if visualization_service is None:
        visualization_service = VisualizationService()
    return visualization_service


def get_analysis_service():
    global analysis_service
    if analysis_service is None:
        analysis_service = AnalysisService()
    return analysis_service


# Create upload directory
create_upload_directory()

# In-memory storage for document metadata (in production, use a database)
document_store = {}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML interface"""
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    index_path = os.path.join(static_dir, "index.html")
    return FileResponse(index_path)


@app.api_route(
    "/api/v1/upload", methods=["POST", "OPTIONS"], response_model=UploadResponse
)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for processing"""
    try:
        print(
            f"Upload started for file: {file.filename}, content_type: {file.content_type}"
        )

        # Validate file
        validate_file(file)
        print("File validation passed")

        # Generate document ID and filename
        document_id = generate_document_id(file.filename or "unnamed")
        safe_filename = sanitize_filename(file.filename or "unnamed.txt")
        file_path = get_document_path(document_id, safe_filename)

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("File saved successfully")

        # Process document
        doc_service = get_document_service()
        documents = doc_service.load_document(file_path, document_id)
        print(f"Document processed: {len(documents)} chunks")

        # Add to vector store
        vec_store = get_vector_store()
        vec_store.add_documents(documents)
        print("Added to vector store successfully")

        # Extract metadata
        raw_metadata = doc_service.extract_metadata(file_path, documents)
        print(f"Metadata extracted: {raw_metadata}")

        # Create proper DocumentMetadata object
        file_extension = get_file_extension(safe_filename)
        doc_type_str = file_extension.lstrip(".").lower()
        if doc_type_str not in ["pdf", "txt", "csv", "xlsx", "docx", "md", "json"]:
            doc_type_str = "txt"  # Default fallback

        doc_type = DocumentType(doc_type_str)

        metadata = DocumentMetadata(
            filename=safe_filename,
            file_type=doc_type,
            file_size=raw_metadata.get("file_size", 0),
            page_count=raw_metadata.get("page_count"),
            row_count=raw_metadata.get("row_count"),
            column_count=raw_metadata.get("column_count"),
        )

        # Store document info
        document_store[document_id] = {
            "id": document_id,
            "filename": safe_filename,
            "metadata": metadata.dict(),
            "upload_time": datetime.now().isoformat(),
        }

        return UploadResponse(
            success=True,
            document_id=document_id,
            filename=safe_filename,
            message="Document uploaded and processed successfully",
            metadata=metadata,
        )

    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )


@app.get("/api/v1/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents"""
    documents = []
    for doc_id, doc_info in document_store.items():
        metadata = doc_info["metadata"]

        # Get content preview from vector store
        vec_store = get_vector_store()
        docs = vec_store.get_document_by_id(doc_id)
        content_preview = ""
        if docs and docs.get("documents"):
            content_preview = extract_text_preview(docs["documents"][0][:200])

        documents.append(
            DocumentInfo(
                id=doc_id,
                metadata=metadata,
                content_preview=content_preview,
                chunk_count=metadata.get("total_chunks", 0),
            )
        )

    return documents


@app.get("/api/v1/documents/{document_id}")
async def get_document(document_id: str):
    """Get details of a specific document"""
    if document_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_info = document_store[document_id]
    return doc_info


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    if document_id not in document_store:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        # Delete from vector store
        vec_store = get_vector_store()
        vec_store.delete_document(document_id)

        # Delete file
        doc_info = document_store[document_id]
        file_path = get_document_path(document_id, doc_info["filename"])
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove from store
        del document_store[document_id]

        return {"message": "Document deleted successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting document: {str(e)}"
        )


@app.delete("/api/v1/documents")
async def delete_all_documents():
    """Delete all uploaded documents and clear vector database"""
    try:
        deleted_files = []
        deleted_count = 0

        # Get all documents from document store
        doc_ids = list(document_store.keys())

        # Delete individual documents
        for document_id in doc_ids:
            try:
                # Delete file
                doc_info = document_store[document_id]
                file_path = get_document_path(document_id, doc_info["filename"])
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(doc_info["filename"])

                # Remove from store
                del document_store[document_id]
                deleted_count += 1

            except Exception as e:
                print(f"Error deleting document {document_id}: {e}")
                continue

        # Clear the entire vector database collection
        try:
            vec_store = get_vector_store()
            vectorstore = vec_store.get_or_create_vectorstore()

            # Get all document IDs in the collection and delete them
            try:
                all_docs = vectorstore.get()
                if all_docs.get("ids"):
                    vectorstore.delete(ids=all_docs["ids"])
                    print(
                        f"Deleted {len(all_docs['ids'])} documents from vector database"
                    )
                else:
                    print("No documents found in vector database")
            except Exception as e:
                print(f"Error getting documents from vectorstore: {e}")
                # Try alternative method
                try:
                    vectorstore.delete()  # Try without parameters
                    print("Vector database cleared (alternative method)")
                except Exception as e2:
                    print(f"Error clearing vector database: {e2}")

        except Exception as e:
            print(f"Error accessing vector store: {e}")

        # Clear all chat sessions
        cleared_sessions = 0
        try:
            rag_svc = get_rag_service()
            cleared_sessions = rag_svc.clear_all_sessions()
            print(f"Chat history cleared: {cleared_sessions} sessions")
        except Exception as e:
            print(f"Error clearing chat history: {e}")

        return {
            "message": f"Successfully deleted {deleted_count} documents and cleared all data",
            "deleted_files": deleted_files,
            "deleted_count": deleted_count,
            "vector_db_cleared": True,
            "chat_history_cleared": True,
            "sessions_cleared": cleared_sessions,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting all documents: {str(e)}"
        )


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """Chat with uploaded documents"""
    try:
        print(
            f"🔍 Chat request: '{request.message[:50]}...', web_search: {request.include_web_search}"
        )
        rag_svc = get_rag_service()
        result = rag_svc.chat_with_documents(
            question=request.message,
            session_id=request.session_id,
            document_ids=request.document_ids,
            include_web_search=request.include_web_search,
        )

        return ChatResponse(
            response=result["answer"],
            session_id=result["session_id"],
            sources=result["sources"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")


@app.post("/api/v1/chat/stream")
async def stream_chat_with_documents(request: ChatRequest):
    """Stream chat with uploaded documents"""
    try:
        from fastapi.responses import StreamingResponse

        async def generate():
            rag_svc = get_rag_service()
            for chunk in rag_svc.stream_chat_with_documents(
                question=request.message,
                session_id=request.session_id,
                document_ids=request.document_ids,
            ):
                yield f"data: {chunk}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in streaming chat: {str(e)}"
        )


@app.get("/api/v1/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        rag_svc = get_rag_service()
        history = rag_svc.get_chat_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting chat history: {str(e)}"
        )


@app.delete("/api/v1/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    try:
        rag_svc = get_rag_service()
        success = rag_svc.clear_chat_history(session_id)
        if success:
            return {"message": "Chat history cleared"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error clearing chat history: {str(e)}"
        )


@app.delete("/api/v1/chat/sessions")
async def clear_all_sessions():
    """Clear all chat sessions"""
    try:
        rag_svc = get_rag_service()
        cleared_count = rag_svc.clear_all_sessions()
        return {
            "message": f"Successfully cleared {cleared_count} chat sessions",
            "sessions_cleared": cleared_count,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error clearing sessions: {str(e)}"
        )


@app.get("/api/v1/sessions")
async def get_all_sessions():
    """Get all chat sessions"""
    try:
        rag_svc = get_rag_service()
        sessions = rag_svc.get_all_sessions()

        # Get session details including last message preview and name
        session_details = []
        for session_id in sessions:
            try:
                history = rag_svc.get_chat_history(session_id)
                last_message = ""
                message_count = 0
                if history:
                    # Get the last user message as preview
                    user_messages = [
                        msg for msg in history if msg.get("role") == "human"
                    ]
                    if user_messages:
                        last_message = user_messages[-1].get("content", "")[:50] + (
                            "..."
                            if len(user_messages[-1].get("content", "")) > 50
                            else ""
                        )
                    message_count = len(history)

                session_details.append(
                    {
                        "id": session_id,
                        "name": rag_svc.session_names.get(session_id, "Chat Session"),
                        "message_count": message_count,
                        "last_message": last_message,
                        "last_access": rag_svc.session_timestamps.get(session_id, 0),
                    }
                )
            except Exception as e:
                print(f"Error getting details for session {session_id}: {e}")
                continue

        return {"sessions": session_details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sessions: {str(e)}")


@app.get("/api/v1/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        vec_store = get_vector_store()
        collection_stats = vec_store.get_collection_stats()
        rag_svc = get_rag_service()
        sessions = rag_svc.get_all_sessions()

        return {
            "documents": len(document_store),
            "chat_sessions": len(sessions),
            "vector_store": collection_stats,
            "system_info": {
                "model": settings.LLM_MODEL,
                "embedding_model": settings.EMBEDDING_MODEL,
                "upload_dir": settings.UPLOAD_DIR,
                "vector_db_dir": settings.CHROMA_PERSIST_DIRECTORY,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.post("/api/v1/visualize", response_model=VisualizationResponse)
async def visualize_document(request: VisualizationRequest):
    """Create visualizations from document data"""
    try:
        # Get document info
        if request.document_id not in document_store:
            raise HTTPException(status_code=404, detail="Document not found")

        doc_info = document_store[request.document_id]
        filename = doc_info["filename"]

        # Create visualization
        viz_service = get_visualization_service()
        result = viz_service.create_visualization(
            document_id=request.document_id,
            filename=filename,
            chart_type=request.chart_type,
            columns=request.columns,
            filters=request.filters,
        )

        return VisualizationResponse(
            chart_data=result["chart_data"],
            chart_type=result["chart_type"],
            insights=result["insights"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating visualization: {str(e)}"
        )


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_document(request: AnalysisRequest):
    """Analyze document for insights and statistics"""
    try:
        # Get document info
        if request.document_id not in document_store:
            raise HTTPException(status_code=404, detail="Document not found")

        doc_info = document_store[request.document_id]
        filename = doc_info["filename"]

        # Analyze document
        analysis_svc = get_analysis_service()
        result = analysis_svc.analyze_document(
            document_id=request.document_id,
            filename=filename,
            analysis_type=request.analysis_type,
        )

        return AnalysisResponse(
            document_id=result["document_id"],
            analysis_type=result["analysis_type"],
            summary=result["summary"],
            statistics=result["statistics"],
            insights=result["insights"],
            recommendations=result["recommendations"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing document: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
