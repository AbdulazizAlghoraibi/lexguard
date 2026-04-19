import re
from typing import List, Optional
from lexguard.schemas.document import DocumentChunk


SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(Section|Article|Clause)\s+(\d+[\.\d]*)\s*[-:\.]?\s*(.*)$",
    re.IGNORECASE
)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def detect_section_header(paragraph: str):
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return None

    first_line = lines[0]
    match = SECTION_HEADER_PATTERN.match(first_line)
    if match:
        section_id = match.group(2).strip()
        section_title = match.group(3).strip() or first_line
        return {
            "section_id": section_id,
            "section_title": section_title,
            "raw_header": first_line,
        }
    return None


def build_chunks(document_id: str, title: str, pages) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    current_section_id: Optional[str] = None
    current_section_title: Optional[str] = None

    for page in pages:
        normalized = normalize_text(page["text"])
        paragraphs = split_paragraphs(normalized)

        for i, para in enumerate(paragraphs):
            header_info = detect_section_header(para)

            if header_info:
                current_section_id = header_info["section_id"]
                current_section_title = header_info["section_title"]

                lines = [line.strip() for line in para.splitlines() if line.strip()]
                body = "\n".join(lines[1:]).strip()

                if not body:
                    continue

                chunk_text = body
            else:
                chunk_text = para

            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    document_title=title,
                    document_type="policy",
                    page_number=page["page_number"],
                    section_id=current_section_id,
                    section_title=current_section_title,
                    clause_id=f"{page['page_number']}_{i}",
                    chunk_text=chunk_text,
                    char_start=0,
                    char_end=len(chunk_text),
                )
            )

    return chunks