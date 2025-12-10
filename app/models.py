from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    CSV = "csv"
    XLSX = "xlsx"
    DOCX = "docx"
    MD = "md"
    JSON = "json"


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    sources: Optional[List[Dict[str, Any]]] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    include_web_search: bool = False


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class DocumentMetadata(BaseModel):
    filename: str
    file_type: DocumentType
    file_size: int
    upload_time: datetime = Field(default_factory=datetime.now)
    page_count: Optional[int] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None


class DocumentInfo(BaseModel):
    id: str
    metadata: DocumentMetadata
    content_preview: str
    chunk_count: int


class UploadResponse(BaseModel):
    success: bool
    document_id: Optional[str] = None
    filename: str
    message: str
    metadata: Optional[DocumentMetadata] = None


class VisualizationRequest(BaseModel):
    document_id: str
    chart_type: Optional[str] = None
    columns: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None


class VisualizationResponse(BaseModel):
    chart_data: Dict[str, Any]
    chart_type: str
    insights: List[str]


class AnalysisRequest(BaseModel):
    document_id: str
    analysis_type: Optional[str] = "summary"


class AnalysisResponse(BaseModel):
    document_id: str
    analysis_type: str
    summary: str
    statistics: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
