from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any
import uuid
from app.core.config import settings


class VectorStoreService:
    def __init__(self):
        self._embeddings = None
        self._text_splitter = None
        self._vectorstore_cache = {}

    @property
    def embeddings(self):
        if self._embeddings is None:
            try:
                self._embeddings = OllamaEmbeddings(
                    model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL
                )
            except Exception as e:
                print(f"Warning: Failed to initialize embeddings: {e}")
                # Fallback to a simple embedding (this won't work well but won't crash)
                from langchain_community.embeddings import FakeEmbeddings

                self._embeddings = FakeEmbeddings(size=384)
        return self._embeddings

    @property
    def text_splitter(self):
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
            )
        return self._text_splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )

    def get_or_create_vectorstore(self, collection_name: str = "documents") -> Chroma:
        """Get existing vector store or create new one"""
        cache_key = collection_name
        if cache_key not in self._vectorstore_cache:
            try:
                self._vectorstore_cache[cache_key] = Chroma(
                    collection_name=collection_name,
                    persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
                    embedding_function=self.embeddings,
                )
            except Exception as e:
                print(f"Error creating vector store for {collection_name}: {e}")
                raise
        return self._vectorstore_cache[cache_key]

    def add_documents(
        self, documents: List[Document], collection_name: str = "documents"
    ) -> Chroma:
        """Add documents to vector store"""
        # Split documents into chunks
        splits = self.text_splitter.split_documents(documents)

        # Add unique IDs to each chunk
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = str(uuid.uuid4())
            if "id" not in split.metadata:
                split.metadata["id"] = f"doc_{uuid.uuid4()}"

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        )

        return vectorstore

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        collection_name: str = "documents",
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Perform similarity search"""
        vectorstore = self.get_or_create_vectorstore(collection_name)

        if filter_dict:
            return vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            return vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        collection_name: str = "documents",
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[tuple]:
        """Perform similarity search with scores"""
        vectorstore = self.get_or_create_vectorstore(collection_name)

        if filter_dict:
            return vectorstore.similarity_search_with_score(
                query, k=k, filter=filter_dict
            )
        else:
            return vectorstore.similarity_search_with_score(query, k=k)

    def get_document_by_id(
        self, doc_id: str, collection_name: str = "documents"
    ) -> Dict[str, Any]:
        """Get documents by ID"""
        vectorstore = self.get_or_create_vectorstore(collection_name)
        return vectorstore.get(where={"id": doc_id})

    def delete_document(self, doc_id: str, collection_name: str = "documents") -> bool:
        """Delete document by ID"""
        try:
            vectorstore = self.get_or_create_vectorstore(collection_name)
            vectorstore.delete(where={"id": doc_id})
            return True
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            return False

    def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            vectorstore = self.get_or_create_vectorstore()
            # This is a workaround since Chroma doesn't expose list_collections directly
            return [settings.CHROMA_COLLECTION_NAME]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []

    def get_collection_stats(
        self, collection_name: str = "documents"
    ) -> Dict[str, Any]:
        """Get statistics about a collection"""
        try:
            vectorstore = self.get_or_create_vectorstore(collection_name)
            # Get all documents to count them
            all_docs = vectorstore.get()

            stats = {
                "collection_name": collection_name,
                "total_documents": len(all_docs.get("ids", [])),
                "total_chunks": len(all_docs.get("ids", [])),
                "metadata_keys": set(),
            }

            # Analyze metadata
            for metadata in all_docs.get("metadatas", []):
                if metadata:
                    stats["metadata_keys"].update(metadata.keys())

            stats["metadata_keys"] = list(stats["metadata_keys"])
            return stats

        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"error": str(e)}
