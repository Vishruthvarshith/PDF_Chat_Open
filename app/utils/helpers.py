import os
import hashlib
import uuid
from typing import List, Optional
from fastapi import UploadFile, HTTPException
from app.core.config import settings


def validate_file(file: UploadFile) -> bool:
    """Validate file type and size"""
    if not file.filename:
        # If no filename, assume it's valid for now (will be handled later)
        return True

    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = [".pdf", ".txt", ".csv", ".xlsx", ".docx", ".md", ".json"]
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not allowed. Allowed types: {allowed_extensions}",
        )

    # Check file size (skip if size is not available)
    if (
        hasattr(file, "size")
        and file.size is not None
        and file.size > settings.MAX_FILE_SIZE
    ):
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes",
        )

    return True


def generate_document_id(filename: str) -> str:
    """Generate unique document ID"""
    timestamp = str(uuid.uuid4())
    content = f"{filename}_{timestamp}"
    return hashlib.md5(content.encode()).hexdigest()


def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return os.path.splitext(filename)[1].lower()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size_float = float(size_bytes)
    while size_float >= 1024 and i < len(size_names) - 1:
        size_float /= 1024.0
        i += 1

    return f"{size_float:.1f}{size_names[i]}"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace unsafe characters
    unsafe_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    for char in unsafe_chars:
        filename = filename.replace(char, "_")

    # Remove leading/trailing spaces and dots
    filename = filename.strip(" .")

    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"

    return filename


def create_upload_directory() -> None:
    """Create upload directory if it doesn't exist"""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


def get_document_path(document_id: str, filename: str) -> str:
    """Get full path for stored document"""
    return os.path.join(
        settings.UPLOAD_DIR, f"{document_id}_{sanitize_filename(filename)}"
    )


def extract_text_preview(text: str, max_length: int = 200) -> str:
    """Extract text preview for display"""
    if len(text) <= max_length:
        return text

    preview = text[:max_length]
    # Try to end at a word boundary
    last_space = preview.rfind(" ")
    if last_space > max_length * 0.8:  # If we have at least 80% of max_length
        preview = preview[:last_space]

    return preview + "..."


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If this isn't the last chunk, try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence endings
            sentence_end = text.rfind(".", start, end)
            if sentence_end > start + chunk_size * 0.7:  # At least 70% of chunk_size
                end = sentence_end + 1
            else:
                # Look for word boundary
                word_end = text.rfind(" ", start, end)
                if word_end > start + chunk_size * 0.8:  # At least 80% of chunk_size
                    end = word_end

        chunks.append(text[start:end].strip())

        # Calculate next start position with overlap
        start = max(start + 1, end - overlap)

    return chunks
