from pypdf import PdfReader
from pathlib import Path

def read_pdf(file_path: str | Path) -> str:
    """Extract and return text from every page of a PDF file."""
    try:
        reader = PdfReader(str(file_path))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Unable to read PDF file: {file_path}"
        ) from exc

    pages_text: list[str] = []
    for page in reader.pages:
        extracted_text = page.extract_text() or ""
        if extracted_text.strip():
            pages_text.append(extracted_text.strip())

    return "\n\n".join(pages_text)