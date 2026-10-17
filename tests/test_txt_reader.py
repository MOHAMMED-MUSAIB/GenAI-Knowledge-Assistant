import pytest

from src.ingestion.txt_reader import read_txt


def test_read_txt(tmp_path):
	text_file = tmp_path / "example.txt"
	sample_text = "This is sample text written into the file."
	text_file.write_text(sample_text, encoding="utf-8")

	assert read_txt(text_file) == sample_text


def test_read_empty_txt(tmp_path):
	text_file = tmp_path / "empty.txt"
	text_file.write_text("", encoding="utf-8")

	assert read_txt(text_file) == ""


def test_read_txt_file_not_found():
	with pytest.raises(FileNotFoundError):
		read_txt("this_file_does_not_exist.txt")

def test_read_txt_invalid_utf8(tmp_path):
    text_file = tmp_path / "invalid.txt"
    text_file.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(UnicodeDecodeError):
        read_txt(text_file)