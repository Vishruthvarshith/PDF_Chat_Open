from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredCSVLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import os
import pandas as pd
import json
from app.core.config import settings


def get_file_ext(filename: str) -> str:
    """Get file extension from filename"""
    return os.path.splitext(filename)[1].lower()


class DocumentProcessingService:
    def __init__(self):
        self.supported_formats = {
            ".pdf": self._load_pdf,
            ".txt": self._load_text,
            ".csv": self._load_csv,
            ".xlsx": self._load_excel,
            ".md": self._load_markdown,
            ".pptx": self._load_powerpoint,
            ".docx": self._load_word,
            ".json": self._load_json,
        }

    def load_document(self, file_path: str, document_id: str) -> List[Document]:
        """Load document based on file type"""
        file_extension = get_file_ext(file_path)

        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")

        loader_func = self.supported_formats[file_extension]
        documents = loader_func(file_path)

        # Filter complex metadata that ChromaDB can't handle
        documents = filter_complex_metadata(documents)

        # Add metadata to all documents
        for doc in documents:
            doc.metadata.update(
                {
                    "document_id": document_id,
                    "file_type": file_extension,
                    "source_file": os.path.basename(file_path),
                }
            )

        return documents

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF document"""
        loader = PyMuPDFLoader(file_path)
        return loader.load()

    def _load_text(self, file_path: str) -> List[Document]:
        """Load text document"""
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    def _load_csv(self, file_path: str) -> List[Document]:
        """Load CSV document"""
        # First analyze CSV structure
        df = pd.read_csv(file_path)

        # Create a summary document
        summary = self._create_dataframe_summary(df, file_path)

        # Use unstructured loader for detailed content
        loader = UnstructuredCSVLoader(file_path)
        documents = loader.load()

        # Add summary as first document
        summary_doc = Document(
            page_content=summary,
            metadata={
                "source": file_path,
                "document_type": "csv_summary",
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
            },
        )

        return [summary_doc] + documents

    def _load_excel(self, file_path: str) -> List[Document]:
        """Load Excel document"""
        # First analyze Excel structure
        df = pd.read_excel(file_path)

        # Create a summary document
        summary = self._create_dataframe_summary(df, file_path)

        # Use unstructured loader for detailed content
        loader = UnstructuredExcelLoader(file_path)
        documents = loader.load()

        # Add summary as first document
        summary_doc = Document(
            page_content=summary,
            metadata={
                "source": file_path,
                "document_type": "excel_summary",
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
            },
        )

        return [summary_doc] + documents

    def _load_markdown(self, file_path: str) -> List[Document]:
        """Load Markdown document"""
        loader = UnstructuredMarkdownLoader(file_path)
        return loader.load()

    def _load_powerpoint(self, file_path: str) -> List[Document]:
        """Load PowerPoint document"""
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()

    def _load_word(self, file_path: str) -> List[Document]:
        """Load Word document"""
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()

    def _load_json(self, file_path: str) -> List[Document]:
        """Load JSON document"""
        try:
            # First analyze JSON structure
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Create a summary document
            summary = self._create_json_summary(data, file_path)

            # Load as text for detailed content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Add summary as first document
            summary_doc = Document(
                page_content=summary,
                metadata={
                    "source": file_path,
                    "document_type": "json_summary",
                    "json_structure": self._analyze_json_structure(data),
                },
            )

            # Add content document
            content_doc = Document(
                page_content=content,
                metadata={"source": file_path, "document_type": "json_content"},
            )

            return [summary_doc, content_doc]

        except Exception as e:
            # Fallback to text loading
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            return [
                Document(
                    page_content=content,
                    metadata={"source": file_path, "document_type": "json_text"},
                )
            ]

    def _create_dataframe_summary(self, df: pd.DataFrame, file_path: str) -> str:
        """Create summary for DataFrame"""
        summary = f"""
        Data Summary for {os.path.basename(file_path)}:
        
        Shape: {df.shape[0]} rows, {df.shape[1]} columns
        
        Columns:
        {", ".join(df.columns.tolist())}
        
        Data Types:
        {df.dtypes.to_string()}
        
        Basic Statistics:
        {df.describe().to_string()}
        
        First few rows:
        {df.head().to_string()}
        
        Missing Values:
        {df.isnull().sum().to_string()}
        """

        return summary

    def _create_json_summary(self, data: Any, file_path: str) -> str:
        """Create summary for JSON data"""
        structure = self._analyze_json_structure(data)

        summary = f"""
        JSON Summary for {os.path.basename(file_path)}:
        
        Structure: {structure["type"]}
        
        """

        if structure["type"] == "dict":
            summary += f"Keys: {', '.join(structure['keys'])}\n"
            if structure["nested_levels"]:
                summary += f"Nested levels: {structure['nested_levels']}\n"

        elif structure["type"] == "list":
            summary += f"List length: {structure['length']}\n"
            if structure["item_types"]:
                summary += f"Item types: {', '.join(structure['item_types'])}\n"

        summary += f"\nSample data:\n{json.dumps(data, indent=2)[:1000]}..."

        return summary

    def _analyze_json_structure(self, data: Any, max_depth: int = 3) -> Dict[str, Any]:
        """Analyze JSON structure"""

        def analyze_recursive(obj, depth=0):
            if depth > max_depth:
                return {"type": "nested_too_deep"}

            if isinstance(obj, dict):
                return {
                    "type": "dict",
                    "keys": list(obj.keys()),
                    "nested_levels": max(
                        [
                            analyze_recursive(v, depth + 1).get("nested_levels", 0)
                            for v in obj.values()
                        ]
                        + [depth]
                    )
                    if obj
                    else [depth],
                }
            elif isinstance(obj, list):
                if not obj:
                    return {"type": "list", "length": 0, "item_types": []}

                item_types = list(set(type(item).__name__ for item in obj[:10]))
                return {
                    "type": "list",
                    "length": len(obj),
                    "item_types": item_types,
                    "sample_structure": analyze_recursive(obj[0], depth + 1)
                    if obj
                    else None,
                }
            else:
                return {"type": type(obj).__name__, "nested_levels": depth}

        return analyze_recursive(data)

    def extract_metadata(
        self, file_path: str, documents: List[Document]
    ) -> Dict[str, Any]:
        """Extract metadata from loaded documents"""
        file_extension = get_file_ext(file_path)
        file_size = os.path.getsize(file_path)

        metadata = {
            "file_type": file_extension,
            "file_size": file_size,
            "total_chunks": len(documents),
            "upload_time": pd.Timestamp.now().isoformat(),
        }

        # Add specific metadata based on file type
        if documents and documents[0].metadata:
            doc_metadata = documents[0].metadata

            if file_extension == ".pdf":
                metadata["page_count"] = len(documents)
            elif file_extension in [".csv", ".xlsx"]:
                metadata.update(
                    {
                        "row_count": doc_metadata.get("row_count"),
                        "column_count": doc_metadata.get("column_count"),
                        "columns": doc_metadata.get("columns", []),
                    }
                )
            elif file_extension == ".json":
                metadata["json_structure"] = doc_metadata.get("json_structure", {})

        return metadata
