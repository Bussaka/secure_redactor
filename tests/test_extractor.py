import shutil
import fitz
import pytest

def has_tesseract():
    return shutil.which("tesseract") is not None

def test_utf8_text(extractor, tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Malipo ya ankara — KSH 5,000", encoding="utf-8")
    assert extractor.process_file(str(path)) == "Malipo ya ankara — KSH 5,000"

def test_utf8_bom_is_stripped(extractor, tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Invoice", encoding="utf-8-sig")
    assert extractor.process_file(str(path)) == "Invoice"

def test_windows_encoded_text(extractor, tmp_path):
    path = tmp_path / "doc.txt"
    path.write_bytes("Café invoice – total due".encode("cp1252"))
    assert extractor.process_file(str(path)) == "Café invoice – total due"

def test_digital_pdf(extractor, tmp_path):
    path = tmp_path / "doc.pdf"
    with fitz.open() as doc:
        doc.new_page().insert_text((72, 72), "Tax Invoice total due KSH 12,000")
        doc.save(str(path))
    assert "Tax Invoice total due" in extractor.process_file(str(path))

@pytest.mark.skipif(not has_tesseract(), reason="Tesseract OCR binary is not installed")
def test_scanned_pdf_uses_ocr(extractor, tmp_path):
    # Build an image-only PDF: render text to a bitmap, then embed only the bitmap
    with fitz.open() as text_doc:
        page = text_doc.new_page()
        page.insert_text((72, 72), "SCANNED INVOICE", fontsize=28)
        png = page.get_pixmap(dpi=200).tobytes("png")
    path = tmp_path / "scan.pdf"
    with fitz.open() as doc:
        doc.new_page().insert_image(fitz.Rect(0, 0, 595, 842), stream=png)
        doc.save(str(path))

    assert "SCANNED INVOICE" in extractor.process_file(str(path)).upper()

def test_unsupported_extension(extractor, tmp_path):
    path = tmp_path / "notes.docx"
    path.write_bytes(b"PK")
    assert extractor.process_file(str(path)) == ""
