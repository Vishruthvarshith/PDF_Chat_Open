from typing import Dict, List, Any, Optional
import uuid
import os
import json
from datetime import datetime
from app.services.llm_service import LLMService
from app.db.vector_store import VectorStoreService
from app.core.config import settings
import requests


class RAGService:
    def __init__(self):
        self.llm_service = LLMService()
        self.vector_store_service = VectorStoreService()
        self.chat_histories: Dict[str, List[Dict[str, str]]] = {}
        self.session_timestamps: Dict[
            str, float
        ] = {}  # Track when sessions were last used
        self.session_names: Dict[str, str] = {}  # Store session names
        # Use absolute path for chat history directory
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.chat_history_dir = os.path.join(base_dir, "uploads", "chat_histories")
        print(f"Chat history directory: {self.chat_history_dir}")
        os.makedirs(self.chat_history_dir, exist_ok=True)
        self._load_chat_histories()

    def _generate_session_name(self, session_id: str) -> str:
        """Generate a session name based on the first user message"""
        history = self.chat_histories.get(session_id, [])
        if not history:
            return "New Chat"

        # Find the first user message
        for msg in history:
            if msg.get("role") == "human":
                content = msg.get("content", "").strip()
                if content:
                    # Use first 30 characters of the message as name, or up to first sentence
                    if len(content) <= 30:
                        return content
                    else:
                        # Try to find first sentence
                        sentences = content.split(".")
                        if sentences and len(sentences[0]) <= 30:
                            return sentences[0] + "."
                        else:
                            return content[:27] + "..."
        return "Chat Session"

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one"""
        import time

        if not session_id:
            session_id = str(uuid.uuid4())

        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []
            self.session_timestamps[session_id] = time.time()
            # Generate initial name for new session
            self.session_names[session_id] = "New Chat"
        else:
            # Update last access time
            self.session_timestamps[session_id] = time.time()

        # Clean up old sessions (older than 1 hour)
        self._cleanup_old_sessions()

        return session_id

    def _load_chat_histories(self):
        """Load chat histories from disk"""
        try:
            for filename in os.listdir(self.chat_history_dir):
                if filename.endswith(".json"):
                    session_id = filename[:-5]  # Remove .json extension
                    filepath = os.path.join(self.chat_history_dir, filename)
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        self.chat_histories[session_id] = data.get("history", [])
                        self.session_timestamps[session_id] = data.get("last_access", 0)
                        self.session_names[session_id] = data.get(
                            "session_name", self._generate_session_name(session_id)
                        )
        except Exception as e:
            print(f"Warning: Failed to load chat histories: {e}")

    def _save_chat_history(self, session_id: str):
        """Save chat history to disk"""
        try:
            filepath = os.path.join(self.chat_history_dir, f"{session_id}.json")
            data = {
                "history": self.chat_histories.get(session_id, []),
                "last_access": self.session_timestamps.get(session_id, 0),
                "session_name": self.session_names.get(session_id, "Chat Session"),
                "last_updated": datetime.now().isoformat(),
            }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            print(f"✅ Saved chat history for session {session_id} to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save chat history for {session_id}: {e}")
            raise  # Re-raise the exception to see it

    def _cleanup_old_sessions(self):
        """Clean up sessions that haven't been used in the last hour"""
        import time

        current_time = time.time()
        session_timeout = 3600  # 1 hour

        sessions_to_remove = []
        for session_id, last_access in self.session_timestamps.items():
            if current_time - last_access > session_timeout:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            if session_id in self.chat_histories:
                del self.chat_histories[session_id]
            if session_id in self.session_timestamps:
                del self.session_timestamps[session_id]
            print(f"🧹 Cleaned up old session: {session_id}")

    def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web for information"""
        print(f"🌐 Searching web for: {query}")

        # For demonstration, return mock results for common queries
        mock_results = {
            "what is the capital of france": [
                {
                    "title": "Paris - Capital of France",
                    "content": "Paris is the capital and most populous city of France. It is located in the north-central part of the country, along the Seine River.",
                    "url": "https://en.wikipedia.org/wiki/Paris",
                    "source": "Wikipedia",
                }
            ],
            "what is the population of tokyo": [
                {
                    "title": "Tokyo Population Statistics",
                    "content": "Tokyo is the capital and largest city of Japan with a population of approximately 14 million people in the city proper.",
                    "url": "https://en.wikipedia.org/wiki/Tokyo",
                    "source": "Wikipedia",
                }
            ],
            "what is machine learning": [
                {
                    "title": "Machine Learning",
                    "content": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence.",
                    "url": "https://en.wikipedia.org/wiki/Machine_learning",
                    "source": "Wikipedia",
                }
            ],
        }

        # Check for mock results first
        query_lower = query.lower().strip()
        for key, results in mock_results.items():
            if key in query_lower:
                print(f"🎯 Found mock results for: {key}")
                return results[:num_results]

        # Try real web search
        try:
            # Using a simple web search API
            search_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"

            print(f"📡 Making request to DuckDuckGo API")
            response = requests.get(search_url, timeout=3)  # Very short timeout
            response.raise_for_status()

            data = response.json()
            print(f"📄 Received data from DuckDuckGo")

            results = []
            if data.get("AbstractText"):
                results.append(
                    {
                        "title": data.get("Heading", "Web Result"),
                        "content": data.get("AbstractText", ""),
                        "url": data.get("AbstractURL", ""),
                        "source": "DuckDuckGo",
                    }
                )

            # Add related topics if available
            for topic in data.get("RelatedTopics", [])[: num_results - 1]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(
                        {
                            "title": topic.get("FirstURL", "Related Topic"),
                            "content": topic.get("Text", ""),
                            "url": topic.get("FirstURL", ""),
                            "source": "DuckDuckGo",
                        }
                    )

            print(f"📊 Returning {len(results)} web results")
            return results[:num_results]

        except requests.exceptions.Timeout:
            print(f"⏰ Web search timeout - using fallback")
            return [
                {
                    "title": f"Web Search Result for: {query}",
                    "content": f'I searched the web for "{query}" but the search service is currently unavailable. Please try again later.',
                    "url": "",
                    "source": "Web Search Service",
                }
            ]
        except Exception as e:
            print(f"❌ Web search error: {e} - using fallback")
            return [
                {
                    "title": f"Web Search Result for: {query}",
                    "content": f'I attempted to search the web for "{query}" but encountered an error. The web search feature may be temporarily unavailable.',
                    "url": "",
                    "source": "Web Search Service",
                }
            ]
        except requests.exceptions.RequestException as e:
            print(f"🌐 Web search network error: {e}")
            return []
        except Exception as e:
            print(f"❌ Web search error: {e}")
            return []

    def chat_with_documents(
        self,
        question: str,
        session_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        collection_name: str = "documents",
        include_web_search: bool = False,
    ) -> Dict[str, Any]:
        """Chat with documents using RAG"""
        # Get or create session
        session_id = self.get_or_create_session(session_id)

        # Update session name if this is the first user message
        if len(self.chat_histories[session_id]) == 0:
            self.session_names[session_id] = self._generate_session_name(session_id)

        # Create filter for specific documents if provided
        filter_dict = None
        if document_ids:
            filter_dict = {"document_id": {"$in": document_ids}}

        # Get relevant documents
        docs = self.vector_store_service.similarity_search(
            question, k=4, collection_name=collection_name, filter_dict=filter_dict
        )

        # Get web search results if requested
        web_results = []
        if include_web_search:
            print(f"🔍 Web search requested for question: {question}")
            web_results = self.search_web(question, num_results=3)
            print(f"📊 Found {len(web_results)} web results")
        else:
            print(f"📄 Using only document search (web search disabled)")

        # Combine document and web contexts
        all_sources = []

        # Add document sources
        if docs:
            for i, doc in enumerate(docs):
                all_sources.append(
                    {
                        "type": "document",
                        "title": f"Document {i + 1}",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "document_id": doc.metadata.get("document_id"),
                        "file_type": doc.metadata.get("file_type"),
                        "source_file": doc.metadata.get("source_file"),
                    }
                )

        # Add web sources
        if web_results:
            for result in web_results:
                all_sources.append(
                    {
                        "type": "web",
                        "title": result.get("title", "Web Result"),
                        "content": result.get("content", ""),
                        "url": result.get("url", ""),
                        "source": result.get("source", "Web"),
                    }
                )

        if not all_sources:
            response = {
                "answer": "I couldn't find relevant information in the uploaded documents or on the web to answer your question.",
                "sources": [],
                "session_id": session_id,
            }

            # Add to chat history
            self.chat_histories[session_id].append(
                {"role": "human", "content": question}
            )
            self.chat_histories[session_id].append(
                {"role": "ai", "content": response["answer"]}
            )

            # Save chat history to disk
            print(f"💾 Attempting to save chat history for session {session_id}")
            self._save_chat_history(session_id)

            return response

        # Create context from all sources
        context_parts = []
        for source in all_sources:
            if source["type"] == "document":
                context_parts.append(
                    f"Document ({source['source_file']}): {source['content']}"
                )
            elif source["type"] == "web":
                context_parts.append(
                    f"Web ({source['source']}): {source['title']} - {source['content']}"
                )

        context = "\n\n".join(context_parts)

        # Get chat history
        chat_history = self.chat_histories.get(session_id, [])
        history_text = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in chat_history[-10:]  # Last 10 messages
            ]
        )

        # Create prompt
        prompt = self.llm_service.create_rag_prompt(context, question, chat_history)

        # Generate response
        response_text = self.llm_service.generate_response(prompt)

        # Add to chat history
        self.chat_histories[session_id].append({"role": "human", "content": question})
        self.chat_histories[session_id].append({"role": "ai", "content": response_text})

        # Update session name based on first user message if not already set
        if (
            len(
                [
                    msg
                    for msg in self.chat_histories[session_id]
                    if msg["role"] == "human"
                ]
            )
            == 1
        ):
            self.session_names[session_id] = self._generate_session_name(session_id)

        # Prepare sources in the new format
        sources = []
        for source in all_sources:
            if source["type"] == "document":
                sources.append(
                    {
                        "type": "document",
                        "document_id": source.get("document_id"),
                        "file_type": source.get("file_type"),
                        "source_file": source.get("source_file"),
                        "content": source["content"][:200] + "..."
                        if len(source["content"]) > 200
                        else source["content"],
                    }
                )
            elif source["type"] == "web":
                sources.append(
                    {
                        "type": "web",
                        "title": source["title"],
                        "content": source["content"][:200] + "..."
                        if len(source["content"]) > 200
                        else source["content"],
                        "url": source.get("url", ""),
                        "source": source.get("source", "Web"),
                    }
                )

        return {
            "answer": response_text,
            "sources": sources,
            "session_id": session_id,
            "web_search_used": include_web_search,
        }

    def stream_chat_with_documents(
        self,
        question: str,
        session_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        collection_name: str = "documents",
    ):
        """Stream chat with documents using RAG"""
        # Get or create session
        session_id = self.get_or_create_session(session_id)

        # Create filter for specific documents if provided
        filter_dict = None
        if document_ids:
            filter_dict = {"document_id": {"$in": document_ids}}

        # Get relevant documents
        docs = self.vector_store_service.similarity_search(
            question, k=4, collection_name=collection_name, filter_dict=filter_dict
        )

        if not docs:
            yield {
                "type": "answer",
                "content": "I couldn't find relevant information in the uploaded documents to answer your question.",
                "session_id": session_id,
                "sources": [],
            }
            return

        # Create context from retrieved documents
        context = "\n\n".join(
            [f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
        )

        # Get chat history
        chat_history = self.chat_histories.get(session_id, [])
        history_text = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in chat_history[-10:]  # Last 10 messages
            ]
        )

        # Create prompt
        prompt = self.llm_service.create_rag_prompt(context, question, chat_history)

        # Stream response
        full_response = ""
        for chunk in self.llm_service.stream_response(prompt):
            full_response += chunk
            yield {"type": "chunk", "content": chunk, "session_id": session_id}

        # Add to chat history
        self.chat_histories[session_id].append({"role": "human", "content": question})
        self.chat_histories[session_id].append({"role": "ai", "content": full_response})

        # Send final message with sources
        sources = [
            {
                "document_id": doc.metadata.get("document_id"),
                "file_type": doc.metadata.get("file_type"),
                "source_file": doc.metadata.get("source_file"),
                "content": doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content,
            }
            for doc in docs
        ]

        yield {"type": "sources", "sources": sources, "session_id": session_id}

    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get chat history for a session"""
        return self.chat_histories.get(session_id, [])

    def clear_chat_history(self, session_id: str) -> bool:
        """Clear chat history for a session"""
        try:
            if session_id in self.chat_histories:
                self.chat_histories[session_id] = []
                # Delete the chat history file
                filepath = os.path.join(self.chat_history_dir, f"{session_id}.json")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return True
            return False
        except Exception as e:
            print(f"Error clearing chat history for session {session_id}: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session"""
        try:
            if session_id in self.chat_histories:
                del self.chat_histories[session_id]
            if session_id in self.session_timestamps:
                del self.session_timestamps[session_id]
            if session_id in self.session_names:
                del self.session_names[session_id]
            return True
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            return False

    def clear_all_sessions(self) -> int:
        """Clear all chat sessions and return the number cleared"""
        try:
            cleared_count = len(self.chat_histories)
            self.chat_histories.clear()
            self.session_timestamps.clear()
            self.session_names.clear()

            # Delete all chat history files
            for filename in os.listdir(self.chat_history_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.chat_history_dir, filename)
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        print(f"Warning: Failed to delete {filepath}: {e}")

            print(
                f"🧹 Cleared all {cleared_count} chat sessions and deleted history files"
            )
            return cleared_count
        except Exception as e:
            print(f"Error clearing all sessions: {e}")
            return 0
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            return False

    def get_all_sessions(self) -> List[str]:
        """Get all session IDs"""
        return list(self.chat_histories.keys())

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information including name"""
        if session_id not in self.chat_histories:
            return {"error": "Session not found"}

        history = self.chat_histories[session_id]
        human_messages = [msg for msg in history if msg["role"] == "human"]

        return {
            "id": session_id,
            "name": self.session_names.get(session_id, "Chat Session"),
            "total_messages": len(history),
            "human_messages": len(human_messages),
            "last_activity": history[-1].get("timestamp") if history else None,
        }

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        if session_id not in self.chat_histories:
            return {"error": "Session not found"}

        history = self.chat_histories[session_id]
        human_messages = [msg for msg in history if msg["role"] == "human"]
        ai_messages = [msg for msg in history if msg["role"] == "ai"]

        return {
            "session_id": session_id,
            "total_messages": len(history),
            "human_messages": len(human_messages),
            "ai_messages": len(ai_messages),
            "last_activity": history[-1].get("timestamp") if history else None,
        }
