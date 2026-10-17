
"""Unit tests for the DOCX reader."""

import zipfile

import pytest
from docx import Document

from src.ingestion.docx_reader import read_docx
import subprocess
import sys


@pytest.fixture
def paragraph_docx(tmp_path):
    """Create a DOCX file containing two paragraphs."""
    path = tmp_path / "paragraphs.docx"

    document = Document()
    document.add_paragraph("First paragraph")
    document.add_paragraph("Second paragraph")
    document.save(path)

    return path


@pytest.fixture
def table_docx(tmp_path):
    """Create a DOCX file containing a table."""
    path = tmp_path / "table.docx"

    document = Document()
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Column"
    table.cell(0, 1).text = "Value"
    document.save(path)

    return path


def test_reads_all_paragraphs(paragraph_docx):
    """Extract all non-empty paragraphs."""
    text = read_docx(paragraph_docx)

    assert "First paragraph" in text
    assert "Second paragraph" in text
    assert text.index("First paragraph") < text.index("Second paragraph")


def test_reads_table_cells(table_docx):
    """Extract text contained in table cells."""
    text = read_docx(table_docx)

    assert "Column" in text
    assert "Value" in text


def test_reads_empty_document(tmp_path):
    """Return an empty string when no text is present."""
    path = tmp_path / "empty.docx"
    Document().save(path)

    assert read_docx(path) == ""


def test_missing_file_raises(tmp_path):
    """Raise FileNotFoundError for a missing file."""
    path = tmp_path / "missing.docx"

    with pytest.raises(FileNotFoundError, match="File not found"):
        read_docx(path)


def test_directory_path_raises(tmp_path):
    """Reject a directory path instead of a file."""
    path = tmp_path / "directory"
    path.mkdir()

    with pytest.raises(ValueError, match="Expected a file"):
        read_docx(path)


def test_wrong_extension_raises(tmp_path):
    """Reject files that do not have a DOCX extension."""
    path = tmp_path / "document.txt"
    path.write_text("This is not a DOCX file.", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Expected a \.docx file"):
        read_docx(path)


def test_corrupt_docx_archive_raises(tmp_path):
    """Reject a file that is not a valid ZIP archive."""
    path = tmp_path / "corrupt.docx"
    path.write_bytes(b"not a valid DOCX archive")

    with pytest.raises(ValueError, match="Invalid DOCX archive"):
        read_docx(path)


def test_missing_document_xml_raises(tmp_path):
    """Reject a ZIP archive without the required document XML."""
    path = tmp_path / "missing_xml.docx"

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("example.txt", "Not a Word document")

    with pytest.raises(ValueError, match="Invalid DOCX file"):
        read_docx(path)


def test_invalid_document_xml_raises(tmp_path):
    """Reject malformed XML inside a DOCX archive."""
    path = tmp_path / "invalid_xml.docx"

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", "<document><broken></document>")

    with pytest.raises(ValueError, match="Invalid DOCX document XML"):
        read_docx(path)


def test_extract_text_from_docx_alias(paragraph_docx):
    """Verify that the optional alias points to the same reader."""
    from src.ingestion.docx_reader import extract_text_from_docx

    assert extract_text_from_docx(paragraph_docx) == read_docx(paragraph_docx)


def test_cli_without_file_argument():
    """Show usage and exit with an error when no file path is supplied."""
    result = subprocess.run(
        [sys.executable, "-m", "src.ingestion.docx_reader"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Usage:" in result.stdout + result.stderr
