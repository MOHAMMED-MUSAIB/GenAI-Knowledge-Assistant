import pytest

from src.ingestion.upload_validator import (
    validate_file_extension,
    validate_file_size,
)


@pytest.mark.parametrize(
    "filename",
    [
        "notes.txt",
        "report.pdf",
        "document.docx",
        "REPORT.PDF",
    ],
)
def test_accepts_supported_extensions(filename):
    assert validate_file_extension(filename) is None


def test_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file extension"):
        validate_file_extension("program.exe")
def test_accepts_file_within_size_limit():
    assert validate_file_size(500, 1000) is None


def test_accepts_file_at_size_limit():
    assert validate_file_size(1000, 1000) is None


def test_rejects_file_over_size_limit():
    with pytest.raises(ValueError, match="File size exceeds"):
        validate_file_size(1001, 1000)


@pytest.mark.parametrize(
    "size_bytes,max_size_bytes",
    [
        (-1, 1000),
        (100, 0),
        (100, -1),
    ],
)
def test_rejects_invalid_size_values(size_bytes, max_size_bytes):
    with pytest.raises(ValueError):
        validate_file_size(size_bytes, max_size_bytes)

@pytest.mark.parametrize(
    "filename",
    [
        "",
        "   ",
    ],
)
def test_rejects_empty_filename(filename):
    with pytest.raises(ValueError, match="Filename cannot be empty"):
        validate_file_extension(filename)