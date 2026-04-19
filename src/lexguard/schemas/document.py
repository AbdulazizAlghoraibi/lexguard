from pydantic import BaseModel
from typing import Optional


class DocumentChunk(BaseModel):
    document_id: str
    document_title: str
    document_type: str
    page_number: int
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    clause_id: str
    chunk_text: str
    char_start: int
    char_end: int