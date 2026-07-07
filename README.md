# SecureRedactor: Automated PII Masking & Cryptographic Document Vault

SecureRedactor is a localized, enterprise-grade Data Loss Prevention (DLP) and document ingestion pipeline designed to help Small and Medium Enterprises (SMEs) comply with strict data privacy frameworks, such as **Section 25 of the Kenya Data Protection Act 2019**.

The system automates the ingestion of corporate documents, extracts text content safely into volatile memory, uses Machine Learning to classify document types, contextually sanitizes Personally Identifiable Information (PII), and archives original copies inside an AES-256 bit encrypted cryptographic vault.

## Core Security Features & Architecture

To prevent data remnants and unauthorized data exposure, this project implements a rigorous **Defense-in-Depth** strategy broken into 4 distinct phases:

1. **Zero-Remnant Ingestion (RAM Only):** Files are read into memory streams directly using `PyMuPDF` and localized OCR fallback via `pytesseract`. Unredacted plain-text strings are stored strictly within volatile memory (`RAM`) and are never written to unencrypted log paths or temporary disk caches.
2. **Context-Aware Machine Learning Classification:** An offline, high-efficiency `scikit-learn` pipeline utilizes TF-IDF text vectorization and a Multinomial Naive Bayes classifier to predict document types (e.g., Invoices, Contracts, National IDs).
3. **Hybrid Masking Engine:** Combines the contextual power of Named Entity Recognition (NER via `spaCy`) to dynamically redact human names with the deterministic precision of Regular Expressions (`RegEx`) to strip out localized structural patterns like Kenyan National ID formats, phone numbers, and email addresses.
4. **AES-256 Symmetric Cryptographic Vault:** Original compliance copies are preserved securely by reading their raw byte arrays, encrypting them via standard Fernet block specifications, flushing the encrypted payload to a protected archive directory, and securely purging the original unencrypted entry from disk.

---

## Directory Structure

```text
secure_redactor/
│
├── data/
│   ├── incoming/        # Hot folder monitored in real-time for new documents
│   ├── processed_safe/  # Destination for safe, sanitized, compliant text files
│   └── encrypted_vault/ # Sealed cryptographic folder storing AES-256 encrypted .enc blocks
│
├── src/
│   ├── extractor.py     # Phase 1: In-memory document parser & OCR tool
│   ├── classifier.py    # Phase 2: ML pipeline for text category prediction
│   ├── redactor.py      # Phase 3: Hybrid NLP & RegEx redaction rules
│   └── secure_store.py  # Phase 4: Symmetrical cryptographic file archiver
│
├── main.py              # Central Orchestrator & continuous workspace tracking monitor
├── requirements.txt     # Locked production software library dependencies
└── vault_master.key     # Cryptographic root key variable (Keep strictly hidden!)