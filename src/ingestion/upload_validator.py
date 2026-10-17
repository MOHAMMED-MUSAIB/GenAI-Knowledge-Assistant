from pathlib import Path


def validate_file_extension(filename: str) -> None:
    """Validate that a filename has a supported extension."""
    if not filename or not filename.strip():
        raise ValueError("Filename cannot be empty")

    extension = Path(filename).suffix.lower()
    if extension not in {".txt", ".pdf", ".docx"}:
        raise ValueError(f"Unsupported file extension: {extension}")


def validate_file_size(size_bytes: int, max_size_bytes: int) -> None:
    """Validate the uploaded file size against the configured limit."""
    if size_bytes < 0:
        raise ValueError("File size cannot be negative")

    if max_size_bytes <= 0:
        raise ValueError("Maximum file size must be positive")

    if size_bytes > max_size_bytes:
        raise ValueError("File size exceeds the maximum allowed size")