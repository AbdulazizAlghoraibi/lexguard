from pathlib import Path


def load_text(path: str):
    text = Path(path).read_text(encoding="utf-8")
    return [
        {
            "page_number": 1,
            "text": text
        }
    ]