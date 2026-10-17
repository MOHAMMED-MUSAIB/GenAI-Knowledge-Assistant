import pytest

from src.ingestion.pdf_reader import read_pdf

from unittest.mock import MagicMock, patch

def test_read_pdf_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_pdf("this_file_does_not_exist.pdf")

def test_read_pdf_with_no_extractable_text():
    file_path = "tests/fixtures/test.pdf"

    result = read_pdf(file_path)

    assert result == ""

def test_read_pdf_extracts_text_from_pages():
    mock_reader = MagicMock()
    mock_reader.pages = [
        MagicMock(),
        MagicMock(),
    ]
    mock_reader.pages[0].extract_text.return_value = "First page text"
    mock_reader.pages[1].extract_text.return_value = "Second page text"

    with patch("src.ingestion.pdf_reader.PdfReader", return_value=mock_reader):
        result = read_pdf("sample.pdf")

    assert result == "First page text\n\nSecond page text"

def test_read_pdf_page_with_no_text_returns_empty_string():
    mock_reader = MagicMock()
    mock_reader.pages = [MagicMock()]
    mock_reader.pages[0].extract_text.return_value = None

    with patch("src.ingestion.pdf_reader.PdfReader", return_value=mock_reader):
        result = read_pdf("sample.pdf")

    assert result == ""

def test_read_pdf_page_with_whitespace_only_text_returns_empty_string():
    mock_reader = MagicMock()
    mock_reader.pages = [MagicMock()]
    mock_reader.pages[0].extract_text.return_value = "   \n  "

    with patch("src.ingestion.pdf_reader.PdfReader", return_value=mock_reader):
        result = read_pdf("sample.pdf")

    assert result == ""

def test_read_pdf_invalid_file(tmp_path):
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_text("This is not a valid PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to read PDF file"):
        read_pdf(invalid_pdf)