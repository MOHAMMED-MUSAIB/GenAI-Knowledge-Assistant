
"""Extract text from Microsoft Word DOCX files using Python's standard library."""

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NAMESPACE = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
}


def read_docx(file_path: str | Path) -> str:
    """Read paragraph text from a DOCX file.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        Extracted paragraph text separated by newlines.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the extension is wrong, the DOCX is invalid,
            or the document XML cannot be parsed.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")

    if path.suffix.lower() != ".docx":
        raise ValueError(f"Expected a .docx file, got: {path}")

    try:
        with zipfile.ZipFile(path, "r") as docx_zip:
            if "word/document.xml" not in docx_zip.namelist():
                raise ValueError(f"Invalid DOCX file: {path}")

            document_xml = docx_zip.read("word/document.xml")

        root = ET.fromstring(document_xml)

    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid DOCX archive: {path}") from exc
    except ET.ParseError as exc:
        raise ValueError(f"Invalid DOCX document XML: {path}") from exc

    paragraphs: list[str] = []

    for paragraph in root.findall(".//w:p", NAMESPACE):
        text = "".join(
            node.text or ""
            for node in paragraph.findall(".//w:t", NAMESPACE)
        )

        if text.strip():
            paragraphs.append(text)

    return "\n".join(paragraphs)


# Convenience alias
extract_text_from_docx = read_docx


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python {Path(sys.argv[0]).name} <path-to-docx>")
        raise SystemExit(1)

    print(read_docx(sys.argv[1]))
