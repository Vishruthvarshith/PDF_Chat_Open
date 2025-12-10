#!/usr/bin/env python3
"""
Test script to verify the Document Analysis RAG application setup
"""

import sys
import os
import requests


def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")

    try:
        from app.core.config import settings

        print("✅ Configuration loaded successfully")
        print(f"   - Model: {settings.LLM_MODEL}")
        print(f"   - Embedding Model: {settings.EMBEDDING_MODEL}")
        print(f"   - Upload Dir: {settings.UPLOAD_DIR}")
    except Exception as e:
        print(f"❌ Configuration import failed: {e}")
        return False

    try:
        from app.services.llm_service import LLMService

        llm_service = LLMService()
        print("✅ LLM Service initialized successfully")
    except Exception as e:
        print(f"❌ LLM Service import failed: {e}")
        return False

    try:
        from app.db.vector_store import VectorStoreService

        vector_service = VectorStoreService()
        print("✅ Vector Store Service initialized successfully")
    except Exception as e:
        print(f"❌ Vector Store Service import failed: {e}")
        return False

    try:
        from app.services.document_service import DocumentProcessingService

        doc_service = DocumentProcessingService()
        print("✅ Document Processing Service initialized successfully")
    except Exception as e:
        print(f"❌ Document Processing Service import failed: {e}")
        return False

    try:
        from app.services.rag_service import RAGService

        rag_service = RAGService()
        print("✅ RAG Service initialized successfully")
    except Exception as e:
        print(f"❌ RAG Service import failed: {e}")
        return False

    return True


def test_ollama_connection():
    """Test connection to Ollama"""
    print("\n🔗 Testing Ollama connection...")

    try:
        import requests

        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]

            print("✅ Ollama is running")
            print(f"   - Available models: {', '.join(model_names)}")

            # Check for required models
            if "llama3.1:8b" in model_names:
                print("✅ Llama 3.1:8b model is available")
            else:
                print("❌ Llama 3.1:8b model not found")
                return False

            if any("nomic-embed-text" in model for model in model_names):
                print("✅ Nomic-embed-text model is available")
            else:
                print("❌ Nomic-embed-text model not found")
                return False

            return True
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing Ollama: {e}")
        return False


def test_directories():
    """Test if required directories exist and are writable"""
    print("\n📁 Testing directories...")

    from app.core.config import settings

    # Test upload directory
    if os.path.exists(settings.UPLOAD_DIR):
        if os.access(settings.UPLOAD_DIR, os.W_OK):
            print(f"✅ Upload directory exists and is writable: {settings.UPLOAD_DIR}")
        else:
            print(f"❌ Upload directory is not writable: {settings.UPLOAD_DIR}")
            return False
    else:
        print(f"❌ Upload directory does not exist: {settings.UPLOAD_DIR}")
        return False

    # Test vector DB directory
    if os.path.exists(settings.CHROMA_PERSIST_DIRECTORY):
        if os.access(settings.CHROMA_PERSIST_DIRECTORY, os.W_OK):
            print(
                f"✅ Vector DB directory exists and is writable: {settings.CHROMA_PERSIST_DIRECTORY}"
            )
        else:
            print(
                f"❌ Vector DB directory is not writable: {settings.CHROMA_PERSIST_DIRECTORY}"
            )
            return False
    else:
        print(
            f"❌ Vector DB directory does not exist: {settings.CHROMA_PERSIST_DIRECTORY}"
        )
        return False

    return True


def test_basic_llm():
    """Test basic LLM functionality"""
    print("\n🤖 Testing basic LLM functionality...")

    try:
        from app.services.llm_service import LLMService

        llm_service = LLMService()

        # Test simple generation
        response = llm_service.generate_response(
            "Say 'Hello World' in exactly two words."
        )
        print(f"✅ LLM test response: {response}")

        # Test token count
        token_count = llm_service.get_token_count(
            "This is a test sentence for token counting."
        )
        print(f"✅ Token count test: {token_count} tokens")

        return True

    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Document Analysis RAG - Setup Test")
    print("=" * 50)

    tests_passed = 0
    total_tests = 4

    # Test imports
    if test_imports():
        tests_passed += 1

    # Test Ollama connection
    if test_ollama_connection():
        tests_passed += 1

    # Test directories
    if test_directories():
        tests_passed += 1

    # Test basic LLM functionality
    if test_basic_llm():
        tests_passed += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! The application is ready to run.")
        print("\n🚀 Start the application with:")
        print("   ./start.sh")
        print("\n📱 Or run directly with:")
        print("   python app/main.py")
        print("\n🌐 Web interface will be available at:")
        print("   http://localhost:8000")
        return True
    else:
        print(
            "❌ Some tests failed. Please fix the issues before running the application."
        )
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
