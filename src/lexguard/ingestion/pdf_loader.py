import fitz  # PyMuPDF


def load_pdf(path: str):
    doc = fitz.open(path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page_number": i + 1,
            "text": text
        })

    return pages