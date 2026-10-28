# SecureRedactor: Automated PII Masking & Cryptographic Document Vault

SecureRedactor is a localized, enterprise-grade Data Loss Prevention (DLP) and document ingestion pipeline designed to help Small and Medium Enterprises (SMEs) comply with strict data privacy frameworks, such as **Section 25 of the Kenya Data Protection Act 2019**.

The system automates the ingestion of corporate documents, extracts text content safely into volatile memory, uses Machine Learning to classify document types, contextually sanitizes Personally Identifiable Information (PII), and archives original copies inside an encrypted, tamper-evident cryptographic vault.

## Core Security Features & Architecture

To prevent data remnants and unauthorized data exposure, this project implements a rigorous **Defense-in-Depth** strategy broken into 4 distinct phases:

1. **Zero-Remnant Ingestion (RAM Only):** Files are read into memory streams directly using `PyMuPDF` and localized OCR fallback via `pytesseract`. Unredacted plain-text strings are kept in volatile memory (`RAM`) and are never written to log files. *Note:* `pytesseract` passes images to the Tesseract binary through short-lived temp files, so OCR'd documents do briefly touch the system temp folder.
2. **Context-Aware Machine Learning Classification:** An offline, high-efficiency `scikit-learn` pipeline utilizes TF-IDF text vectorization and a Multinomial Naive Bayes classifier to predict document types (e.g., Invoices, Contracts, National IDs). Low-confidence predictions are reported as `Unknown` rather than forced into a category. A small synthetic demo model is built in; train your own on real labelled documents with `python main.py train` (see below).
3. **Hybrid Masking Engine:** Combines the contextual power of Named Entity Recognition (NER via `spaCy`) to dynamically redact human names with the deterministic precision of Regular Expressions (`RegEx`) to strip out localized structural patterns like Kenyan National ID formats, KRA PINs, phone numbers, and email addresses. RegEx runs first so NER can never claim part of a structured value. Kenyan name rules then catch the names spaCy misses (see [Name redaction](#name-redaction)).
4. **Symmetric Cryptographic Vault:** Original compliance copies are encrypted with Fernet (AES-128-CBC + HMAC-SHA256, so tampering is detected on decrypt), written atomically to the vault, and only then is the original deleted from the incoming folder. A standard delete does not wipe the underlying disk blocks, so run the host on a full-disk-encrypted volume (e.g. BitLocker).

**Operational safeguards**

- **Locked-down secrets:** the master key and every folder that can hold unredacted data (vault, quarantine, restored, audit, logs) are restricted to the account running the pipeline (plus SYSTEM on Windows). A key file found readable by other accounts is tightened automatically on startup.
- **Tamper-evident audit trail:** every processed, quarantined and restored document is recorded in `data/audit/audit_log.jsonl` with its SHA-256 fingerprint, document class and per-type redaction counts, never its contents. Entries are hash-chained and HMAC-signed with a key derived from the master key, so edited, deleted or reordered entries are detected by `python main.py verify-audit`.
- **Event-driven ingestion:** the incoming folder is watched with OS file-system events (`watchdog`), so documents are picked up within about a second of finishing copying. A file is only ingested once its size and modified time have stopped changing, and a periodic rescan covers drives that do not emit events.
- **Structured logging:** console output plus a rotating log at `data/logs/pipeline.log`. Logs record file names and counts only. Note that file names themselves can contain personal data.

---

## Directory Structure

```text
secure_redactor/
│
├── data/
│   ├── incoming/        # Hot folder monitored in real-time for new documents
│   ├── processed_safe/  # Destination for safe, sanitized, compliant text files
│   ├── encrypted_vault/ # Sealed cryptographic folder storing encrypted .enc archives
│   ├── quarantine/      # Unsupported or failed files (may be UNREDACTED)
│   ├── restored/        # Originals decrypted on request via `main.py restore` (UNREDACTED)
│   ├── training/        # Labelled example documents, one sub-folder per category (UNREDACTED)
│   ├── models/          # Signed, trained classifier (document_classifier.joblib + .meta.json)
│   ├── audit/           # Tamper-evident, signed audit trail (audit_log.jsonl)
│   └── logs/            # Rotating application logs
│
├── src/
│   ├── extractor.py     # Phase 1: In-memory document parser & OCR tool
│   ├── classifier.py    # Phase 2: ML pipeline for text category prediction
│   ├── trainer.py       # Trains, evaluates and saves the classifier from labelled documents
│   ├── redactor.py      # Phase 3: Hybrid NLP & RegEx redaction rules
│   ├── name_rules.py    # Kenyan name list + context cues that back up spaCy NER
│   ├── resources/
│   │   └── kenyan_names.txt  # Common Kenyan names, one per line
│   ├── secure_store.py  # Phase 4: Symmetrical cryptographic file archiver
│   ├── audit_log.py     # Signed, hash-chained audit trail
│   └── permissions.py   # Owner-only file/folder permissions (icacls / chmod)
│
├── benchmarks/
│   └── name_redaction.py  # Compares spaCy models with/without the name rules
├── tests/               # pytest suite
├── main.py              # Central Orchestrator & continuous workspace tracking monitor
├── requirements.txt     # Locked production software library dependencies
└── vault_master.key     # Cryptographic root key variable (Keep strictly hidden!)
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

This installs spaCy's large English model `en_core_web_lg` (about 560 MB) plus the small `en_core_web_sm` as a fallback. Scanned PDFs and images additionally need the [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) binary installed and on your `PATH`.

## Usage

```bash
# Watch data/incoming/ and process new documents
python main.py

# Decrypt an original from the vault (written to data/restored/)
python main.py restore <archive-name>.enc

# Check the audit trail has not been tampered with
python main.py verify-audit

# Train the document classifier on your own labelled documents
python main.py train
```

## Name redaction

spaCy's English models are trained mostly on Western text and miss many Kenyan names, such as "Dear Kazungu" or "between the Employer and Moraa Nyakundi". Two rule-based signals in `src/name_rules.py` close the gap:

- **Name list:** a capitalised word from `src/resources/kenyan_names.txt` marks a name, and the match grows over neighbouring capitalised words so an unlisted surname is caught too. Names that are also everyday words (Grace, Faith, Baraka, Imani) are left off the list on purpose. To add names, put one per line in the file; lines starting with `#` are comments.
- **Context cues:** whatever capitalised words follow phrases like `Mr.`, `Dear`, `Employee:`, `Prepared by` or `SURNAME:` are treated as a name, even if neither spaCy nor the list knows it.

Once a name is found, every other capitalised mention of it in the document is redacted too (including possessives like "Moraa's"). Company words such as `Ltd` or `Holdings` stop a name, so "Otieno Holdings Ltd" only loses "Otieno". The rules also stop spaCy from redacting business terms it mislabels, such as `KSH`, `NSSF` or `M-Pesa`.

**Choosing a spaCy model.** The default is `en_core_web_lg`. Set the `SPACY_MODEL` environment variable to use another installed model, e.g. `SPACY_MODEL=en_core_web_sm` on a machine with little disk space. If the chosen model isn't installed, the redactor falls back to `en_core_web_sm` and logs a warning.

Results from `python -m benchmarks.name_redaction` (165 sentences with Kenyan names, 15 business sentences with no names):

| Setup | People fully protected | Business sentences over-redacted | Size |
|---|---|---|---|
| `en_core_web_sm` alone | 78.8% | 25.9% | 12 MB |
| `en_core_web_sm` + name rules | 98.8% | 16.7% | 12 MB |
| `en_core_web_lg` alone | 89.7% | 18.5% | 560 MB |
| **`en_core_web_lg` + name rules (default)** | **99.4%** | **11.1%** | 560 MB |
| `en_core_web_trf` + name rules | 98.8% | 14.8% | ~460 MB + PyTorch, ~20x slower |

**Known gaps:** ALL-CAPS address lines with unlisted names (e.g. `WAIREGI NGATIA / P.O. BOX ...`) can still leak, and spaCy redacts some department names (Finance Department, Audit Committee) as organisations. Bank account numbers are not redacted yet. Spot-check the output in `data/processed_safe/` before sharing it.

To rerun the comparison on other installed models, use `python -m benchmarks.name_redaction --models en_core_web_sm en_core_web_lg --show-leaks`.

## Training the classifier on your own documents

1. Put example documents (PDF, JPG/PNG, TXT) in one sub-folder per category. The folder name becomes the label:

   ```text
   data/training/
   ├── Invoice/      invoice_001.pdf, invoice_002.pdf, ...
   ├── Contract/     ...
   ├── Payslip/      ...
   └── National ID/  ...
   ```

   Use at least 2 documents per category (the minimum) and ideally 10 or more. Try to keep categories roughly balanced.
2. Run `python main.py train`. It reports cross-validated accuracy, per-category precision and recall, and how many documents would be labelled `Unknown` at the confidence threshold. Use `--min-confidence 0.4` (or similar) to trade coverage against accuracy. With many categories, the top probability is naturally lower, so a lower threshold is often right.
3. Restart the watcher. It loads the trained model on startup and falls back to the built-in demo model if none exists.

**How the model is protected**

- Training text is **redacted before training**, the vocabulary keeps letters only (no digits, so no IDs, phone numbers or amounts), and a word must appear in at least 3 training documents to be kept. Together these keep personal data out of the saved model file.
- The model and its settings are **signed with a key derived from the vault master key**. The signature is checked *before* the file is unpickled, because loading a pickle can execute code. A modified, swapped or foreign model is refused, and the built-in model is used instead.
- A model trained with a different scikit-learn version is refused with a prompt to retrain.
- Every training run is recorded in the audit log (`model_trained`).
- `data/training/` holds real, unredacted documents and is restricted to the pipeline's account. Delete or archive the examples once training is done if you don't need to retrain.

## Master key

The master key is read from the `VAULT_MASTER_KEY` environment variable if set, otherwise from `vault_master.key` in the project root (generated on first run). **Back this key up.** Without it the vault cannot be decrypted, and the pipeline refuses to generate a new key while encrypted archives exist.

## Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```
