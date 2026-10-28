import io
import logging
import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Handles secure extraction of text from incoming files directly into volatile memory (RAM)."""
    
    @staticmethod
    def extract_from_digital_pdf(pdf_path: str) -> str:
        """Extracts text embedding directly from a digital PDF."""
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text.strip()
        except Exception as e:
            logger.error("Error reading digital PDF %s: %s", pdf_path, e)
            return ""

    @staticmethod
    def extract_from_scanned_image(image_path: str) -> str:
        """Extracts text from scanned files (JPEG/PNG) using local Tesseract OCR."""
        try:
            # Context manager releases the file handle so the original can be deleted afterwards on Windows
            with Image.open(image_path) as img:
                text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            # Friendly warning if Tesseract isn't fully set up on the host OS yet
            logger.warning("OCR skipped or unavailable: %s", e)
            return ""

    @staticmethod
    def extract_from_scanned_pdf(pdf_path: str) -> str:
        """Renders each page of an image-only PDF to a bitmap in memory and runs Tesseract OCR on it."""
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    pix = page.get_pixmap(dpi=300)
                    with Image.open(io.BytesIO(pix.tobytes("png"))) as img:
                        text += pytesseract.image_to_string(img) + "\n"
            return text.strip()
        except Exception as e:
            logger.warning("OCR skipped or unavailable: %s", e)
            return ""

    def process_file(self, file_path: str) -> str:
        """Determines the file extension and safely extracts text into a string variable."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            extracted_text = self.extract_from_digital_pdf(file_path)
            # Fallback to OCR if the PDF contains flat scanned images instead of digital text
            if not extracted_text.strip():
                logger.info("%s has no digital text. Attempting image OCR fallback...", os.path.basename(file_path))
                return self.extract_from_scanned_pdf(file_path)
            return extracted_text
            
        elif ext in ['.jpg', '.jpeg', '.png']:
            return self.extract_from_scanned_image(file_path)
            
        elif ext == '.txt':
            # Try UTF-8 first (with or without BOM), then fall back to the Windows default code page
            for encoding in ('utf-8-sig', 'cp1252'):
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read().strip()
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.error("Error reading text file %s: %s", file_path, e)
                    return ""
            logger.error("Could not decode text file %s", file_path)
            return ""
            
        else:
            logger.warning("Unsupported file extension: %s", ext)
            return ""

if __name__ == "__main__":
    extractor = DocumentExtractor()
    print("[+] Phase 1: DocumentExtractor successfully initialized.")