def read_txt(file_path: str) -> str:
    """Read and return the contents of a UTF-8 text file."""
    with open(file_path, "r", encoding="utf-8", errors="strict") as file:
        return file.read()
