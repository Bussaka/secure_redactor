import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

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
            print(f"[-] Error reading digital PDF {pdf_path}: {e}")
            return ""

    @staticmethod
    def extract_from_scanned_image(image_path: str) -> str:
        """Extracts text from scanned files (JPEG/PNG) using local Tesseract OCR."""
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            # Friendly warning if Tesseract isn't fully set up on the host OS yet
            print(f"[!] System OCR fallback skipped or unavailable: {e}")
            return ""

    def process_file(self, file_path: str) -> str:
        """Determines the file extension and safely extracts text into a string variable."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            extracted_text = self.extract_from_digital_pdf(file_path)
            # Fallback to OCR if the PDF contains flat scanned images instead of digital text
            if not extracted_text.strip():
                print(f"[!] {os.path.basename(file_path)} has no digital text. Attempting image OCR fallback...")
                return self.extract_from_scanned_image(file_path)
            return extracted_text
            
        elif ext in ['.jpg', '.jpeg', '.png']:
            return self.extract_from_scanned_image(file_path)
            
        elif ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"[-] Error reading text file {file_path}: {e}")
                return ""
            
        else:
            print(f"[-] Unsupported file extension: {ext}")
            return ""

if __name__ == "__main__":
    extractor = DocumentExtractor()
    print("[+] Phase 1: DocumentExtractor successfully initialized.")