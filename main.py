import os
import time
from src.extractor import DocumentExtractor
from src.classifier import DocumentClassifier
from src.redactor import HybridRedactor
from src.secure_store import CryptographicVault

def run_secure_pipeline():
    # 1. Establish the local operational directory layouts
    INCOMING_DIR = os.path.join("data", "incoming")
    SAFE_DIR = os.path.join("data", "processed_safe")
    VAULT_DIR = os.path.join("data", "encrypted_vault")
    
    # Ensure the storage folders exist physically on the machine before starting
    os.makedirs(INCOMING_DIR, exist_ok=True)
    os.makedirs(SAFE_DIR, exist_ok=True)
    os.makedirs(VAULT_DIR, exist_ok=True)
    
    # 2. Boot up our Phase 1-4 modular engines into RAM memory
    print("[*] Booting Secure SME AI Document Classifier & Redaction Engine...")
    extractor = DocumentExtractor()
    classifier = DocumentClassifier()
    redactor = HybridRedactor()
    vault = CryptographicVault()
    print("[+] Entire Security Pipeline Online. Watching for folder additions...\n")

    # 3. Start the continuous real-time file monitoring loop
    while True:
        # Scan the incoming directory for any new files (ignoring hidden system files)
        files = [f for f in os.listdir(INCOMING_DIR) if not f.startswith('.')]
        
        if files:
            print(f"[!] Found {len(files)} new target asset(s) to process.")
            
            for file_name in files:
                full_path = os.path.join(INCOMING_DIR, file_name)
                print(f"\n[*] Commencing pipeline ingestion on: {file_name}")
                
                # --- Phase 1: Text Extraction ---
                raw_text = extractor.process_file(full_path)
                if not raw_text.strip():
                    print(f"[-] Skip or Extraction failure on {file_name}. Moving on.")
                    continue
                
                # --- Phase 2: AI Classification ---
                doc_type = classifier.predict_type(raw_text)
                print(f"[+] AI Classification Result: [ Document Class -> '{doc_type}' ]")
                
                # --- Phase 3: Hybrid Memory Redaction ---
                print("[*] Running hybrid masking engines over memory registers...")
                sanitized_text = redactor.redact_text(raw_text)
                
                # Write out the clean, redacted plain text copy for daily business operations
                safe_txt_name = f"sanitized_{os.path.splitext(file_name)[0]}.txt"
                safe_target_path = os.path.join(SAFE_DIR, safe_txt_name)
                
                with open(safe_target_path, "w", encoding="utf-8") as safe_file:
                    safe_file.write(f"--- DOCUMENT CATEGORY: {doc_type.upper()} ---\n\n")
                    safe_file.write(sanitized_text)
                print(f"[+] Clean, compliant text copy generated at: {safe_target_path}")
                
                # --- Phase 4: Cryptographic Vaulting ---
                vault.encrypt_and_vault(full_path, VAULT_DIR)
                
                # Clean Up: Delete the raw, unredacted file from the exposed incoming directory
                os.remove(full_path)
                print(f"[+] Pipeline task finalized. Ingestion workspace cleared for {file_name}.")
                
        # Wait 3 seconds before checking the incoming directory again (prevents CPU overloading)
        time.sleep(3)

if __name__ == "__main__":
    run_secure_pipeline()