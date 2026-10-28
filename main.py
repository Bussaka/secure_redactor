import argparse
import logging
import os
import sys
import threading
import time
import uuid
from logging.handlers import RotatingFileHandler

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.audit_log import AuditLog, sha256_of_file
from src.extractor import DocumentExtractor
from src.classifier import DocumentClassifier
from src.permissions import restrict_to_owner
from src.redactor import HybridRedactor
from src.secure_store import CryptographicVault
from src.trainer import TrainingDataError, train_classifier

logger = logging.getLogger("secure_redactor")

# Anchor every path to the project folder so the pipeline behaves the same from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INCOMING_DIR = os.path.join(BASE_DIR, "data", "incoming")
SAFE_DIR = os.path.join(BASE_DIR, "data", "processed_safe")
VAULT_DIR = os.path.join(BASE_DIR, "data", "encrypted_vault")
QUARANTINE_DIR = os.path.join(BASE_DIR, "data", "quarantine")
RESTORED_DIR = os.path.join(BASE_DIR, "data", "restored")
AUDIT_DIR = os.path.join(BASE_DIR, "data", "audit")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
TRAINING_DIR = os.path.join(BASE_DIR, "data", "training")
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "document_classifier.joblib")

# A file must keep the same size and modified time for this long before we treat it as fully copied
SETTLE_SECONDS = 1.0
# Safety-net rescan even without filesystem events (some network drives never deliver them)
RESCAN_SECONDS = 30
# Used instead of events if the OS file watcher cannot be started
POLL_INTERVAL_SECONDS = 3

def audit_path() -> str:
    return os.path.join(AUDIT_DIR, "audit_log.jsonl")

def configure_logging():
    """Console output plus a rotating log file. Logs carry file names and counts, never document text."""
    os.makedirs(LOG_DIR, exist_ok=True)
    restrict_to_owner(LOG_DIR)
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    log_file = RotatingFileHandler(os.path.join(LOG_DIR, "pipeline.log"),
                                   maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    log_file.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(log_file)
    logging.getLogger("watchdog").setLevel(logging.WARNING)

class _ChangeSignal(FileSystemEventHandler):
    def __init__(self, changed: threading.Event):
        self._changed = changed

    def on_any_event(self, event):
        self._changed.set()

class FolderWatcher:
    """Wakes the pipeline the moment anything changes in the incoming folder, instead of blind polling."""

    def __init__(self, path: str):
        self._changed = threading.Event()
        self._observer = Observer()
        try:
            self._observer.schedule(_ChangeSignal(self._changed), path, recursive=False)
            self._observer.start()
            self.is_live = True
        except Exception as e:
            logger.warning("File watcher unavailable (%s); falling back to polling every %ss.", e, POLL_INTERVAL_SECONDS)
            self.is_live = False

    def wait(self, timeout: float) -> bool:
        """Blocks until a change event arrives or the timeout passes. Returns True if something changed."""
        if not self.is_live:
            timeout = min(timeout, POLL_INTERVAL_SECONDS)
        changed = self._changed.wait(timeout)
        self._changed.clear()
        return changed

    def stop(self):
        if self.is_live:
            self._observer.stop()
            self._observer.join(timeout=5)

def quarantine_file(full_path: str, file_name: str, reason: str, audit: AuditLog):
    """Moves a file we could not process out of the hot folder so it is not retried forever."""
    try:
        file_hash = sha256_of_file(full_path)
    except OSError:
        file_hash = None
    target = os.path.join(QUARANTINE_DIR, f"{uuid.uuid4().hex[:8]}_{file_name}")
    try:
        os.replace(full_path, target)
        logger.warning("%s. Moved %s to quarantine: %s", reason, file_name, target)
        audit.record("quarantined", source_name=file_name, sha256=file_hash, reason=reason,
                     quarantine_file=os.path.basename(target))
    except OSError as e:
        logger.error("%s. Could not quarantine %s either: %s", reason, file_name, e)
        audit.record("quarantine_failed", source_name=file_name, sha256=file_hash, reason=f"{reason}; {e}")

def process_single_file(full_path, file_name, extractor, classifier, redactor, vault, audit):
    """Runs one document through Phases 1-4. The original is only deleted once it is safely vaulted."""
    file_hash = sha256_of_file(full_path)
    size_bytes = os.path.getsize(full_path)

    # --- Phase 1: Text Extraction ---
    raw_text = extractor.process_file(full_path)
    if not raw_text.strip():
        quarantine_file(full_path, file_name, "Unsupported file or extraction failure", audit)
        return

    # --- Phase 2: Hybrid Memory Redaction ---
    sanitized_text, redaction_counts = redactor.redact_with_report(raw_text)
    logger.info("Redacted %d item(s) from %s", sum(redaction_counts.values()), file_name)

    # --- Phase 3: AI Classification ---
    # Classify the redacted text: it matches what trained models learn from, and keeps PII away from the model
    doc_type, confidence = classifier.predict_with_confidence(sanitized_text)
    logger.info("Classification result for %s: %s (confidence %.2f)", file_name, doc_type, confidence)

    # Unique run ID keeps same-named uploads from overwriting each other's outputs
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # --- Phase 4: Cryptographic Vaulting ---
    # Vault before writing any outputs, so a failure here leaves no orphaned sanitized copy behind
    vault_path = vault.encrypt_and_vault(full_path, VAULT_DIR, target_name=f"{run_id}_{file_name}")
    if not vault_path:
        quarantine_file(full_path, file_name, "Vaulting failed, original NOT deleted", audit)
        return

    # Write out the clean, redacted plain text copy for daily business operations
    safe_txt_name = f"sanitized_{os.path.splitext(file_name)[0]}_{run_id}.txt"
    safe_target_path = os.path.join(SAFE_DIR, safe_txt_name)

    with open(safe_target_path, "w", encoding="utf-8") as safe_file:
        safe_file.write(f"--- DOCUMENT CATEGORY: {doc_type.upper()} ---\n\n")
        safe_file.write(sanitized_text)
    logger.info("Sanitized copy written to %s", safe_target_path)

    # Clean Up: Delete the raw, unredacted file from the exposed incoming directory
    os.remove(full_path)
    audit.record("processed", source_name=file_name, sha256=file_hash, size_bytes=size_bytes,
                 doc_type=doc_type, confidence=round(confidence, 3), redactions=dict(sorted(redaction_counts.items())),
                 vault_archive=os.path.basename(vault_path), sanitized_file=safe_txt_name)
    logger.info("Pipeline finished for %s; original removed from the incoming folder.", file_name)

def scan_incoming(previous: dict, now: float):
    """Returns (ready_files, snapshot). A file is ready once its size and mtime have held still for SETTLE_SECONDS,
    so documents still being copied into the hot folder are never ingested half-written."""
    snapshot = {}
    for f in os.listdir(INCOMING_DIR):
        path = os.path.join(INCOMING_DIR, f)
        # Ignore hidden system files and sub-folders
        if f.startswith('.') or not os.path.isfile(path):
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue  # File vanished or is locked; try again next scan
        signature = (st.st_size, st.st_mtime_ns)
        old = previous.get(f)
        stable_since = old[1] if old and old[0] == signature else now
        snapshot[f] = (signature, stable_since)

    ready = [f for f, (sig, since) in snapshot.items()
             if sig[0] > 0 and f in previous and previous[f][0] == sig and now - since >= SETTLE_SECONDS]
    return ready, snapshot

def run_secure_pipeline():
    # 1. Ensure the storage folders exist, and lock down every folder that can hold unredacted data
    for directory in (INCOMING_DIR, SAFE_DIR, VAULT_DIR, QUARANTINE_DIR, AUDIT_DIR):
        os.makedirs(directory, exist_ok=True)
    for directory in (VAULT_DIR, QUARANTINE_DIR, AUDIT_DIR):
        restrict_to_owner(directory)

    # 2. Boot up our Phase 1-4 modular engines into RAM memory
    logger.info("Booting Secure SME AI Document Classifier & Redaction Engine...")
    vault = CryptographicVault(vault_dir=VAULT_DIR)
    audit = AuditLog(audit_path(), vault.key)
    extractor = DocumentExtractor()
    classifier = DocumentClassifier(model_path=MODEL_PATH, master_key=vault.key)
    redactor = HybridRedactor()
    watcher = FolderWatcher(INCOMING_DIR)
    logger.info("Pipeline online. Watching %s for new documents.", INCOMING_DIR)

    snapshot = {}
    # 3. Start the continuous real-time file monitoring loop
    try:
        while True:
            ready_files, snapshot = scan_incoming(snapshot, time.monotonic())

            for file_name in ready_files:
                full_path = os.path.join(INCOMING_DIR, file_name)
                logger.info("Commencing pipeline ingestion on: %s", file_name)

                # One bad file must never take down the whole monitoring service
                try:
                    process_single_file(full_path, file_name, extractor, classifier, redactor, vault, audit)
                except Exception:
                    logger.exception("Unexpected error while processing %s", file_name)
                    if os.path.exists(full_path):
                        quarantine_file(full_path, file_name, "Pipeline error", audit)
                snapshot.pop(file_name, None)

            # Files still settling need a quick re-check; otherwise sleep until the next filesystem event
            watcher.wait(SETTLE_SECONDS if snapshot else RESCAN_SECONDS)
    except KeyboardInterrupt:
        logger.info("Shutdown requested. Stopping the folder watcher.")
    finally:
        watcher.stop()

def restore_from_vault(vault_file: str, output_dir: str) -> int:
    """Decrypts a single archive out of the vault. Accepts a full path or a bare file name inside the vault."""
    vault_path = vault_file if os.path.exists(vault_file) else os.path.join(VAULT_DIR, vault_file)
    if not os.path.isfile(vault_path):
        logger.error("Vault archive not found: %s", vault_file)
        return 1

    vault = CryptographicVault(vault_dir=VAULT_DIR)
    audit = AuditLog(audit_path(), vault.key)
    os.makedirs(output_dir, exist_ok=True)
    restrict_to_owner(output_dir)
    try:
        restored_path = vault.decrypt_from_vault(vault_path, output_dir)
    except FileExistsError:
        logger.error("A restored copy already exists in %s. Remove it first.", output_dir)
        return 1
    except ValueError as e:
        logger.error("%s", e)
        audit.record("restore_failed", vault_archive=os.path.basename(vault_path), reason=str(e))
        return 1
    audit.record("restored", vault_archive=os.path.basename(vault_path), restored_file=restored_path)
    logger.warning("The restored file is UNREDACTED. Delete it as soon as it is no longer needed.")
    return 0

def verify_audit_log() -> int:
    """Checks that no audit entry has been edited, removed or reordered."""
    vault = CryptographicVault(vault_dir=VAULT_DIR)
    ok, count, problem = AuditLog(audit_path(), vault.key).verify()
    if ok:
        logger.info("Audit log intact: %d entries verified.", count)
        return 0
    logger.error("AUDIT LOG TAMPERING DETECTED after %d valid entries: %s", count, problem)
    return 1

def train_model(training_dir: str, min_confidence: float) -> int:
    """Trains the document classifier on data/training/<Label>/ folders and saves a signed model."""
    os.makedirs(training_dir, exist_ok=True)
    # Training documents are real, unredacted originals: keep them owner-only like the vault
    restrict_to_owner(training_dir)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    restrict_to_owner(os.path.dirname(MODEL_PATH))

    vault = CryptographicVault(vault_dir=VAULT_DIR)
    audit = AuditLog(audit_path(), vault.key)
    try:
        metadata = train_classifier(training_dir, MODEL_PATH, vault.key, DocumentExtractor(), HybridRedactor(),
                                    min_confidence=min_confidence)
    except TrainingDataError as e:
        logger.error("%s", e)
        logger.error("Put example documents in one sub-folder per category, e.g. %s",
                     os.path.join(training_dir, "Invoice", "invoice_001.pdf"))
        return 1

    evaluation = metadata["evaluation"]
    logger.info("Cross-validated accuracy: %.1f%% (%d folds)", evaluation["accuracy"] * 100, evaluation["folds"])
    logger.info("At confidence threshold %.2f, %.1f%% of documents get a label (accuracy on those: %s)",
                min_confidence, evaluation["coverage_at_threshold"] * 100,
                f"{evaluation['accuracy_when_confident'] * 100:.1f}%" if evaluation["accuracy_when_confident"] is not None else "n/a")
    logger.info("Per-category results:\n%s", metadata["evaluation_report"])
    logger.info("Signed model saved to %s. Restart the watcher to start using it.", MODEL_PATH)

    audit.record("model_trained", labels=metadata["labels"], samples_per_label=metadata["samples_per_label"],
                 skipped_documents=metadata["skipped_documents"], min_confidence=min_confidence,
                 evaluation=evaluation, model_sha256=metadata["model_sha256"])
    return 0

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SecureRedactor document redaction pipeline")
    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser("watch", help="Monitor the incoming folder and process new documents (default)")
    restore_parser = subcommands.add_parser("restore", help="Decrypt an original document from the vault")
    restore_parser.add_argument("vault_file", help="Path or file name of the .enc archive inside the vault")
    restore_parser.add_argument("--out", default=RESTORED_DIR, help=f"Output folder (default: {RESTORED_DIR})")
    subcommands.add_parser("verify-audit", help="Check the audit log for tampering")
    train_parser = subcommands.add_parser("train", help="Train the document classifier on labelled examples")
    train_parser.add_argument("--data", default=TRAINING_DIR,
                              help=f"Folder with one sub-folder of examples per category (default: {TRAINING_DIR})")
    train_parser.add_argument("--min-confidence", type=float, default=0.5,
                              help="Predictions below this confidence are labelled Unknown (default: 0.5)")
    return parser.parse_args(argv)

if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    if args.command == "restore":
        sys.exit(restore_from_vault(args.vault_file, args.out))
    if args.command == "verify-audit":
        sys.exit(verify_audit_log())
    if args.command == "train":
        sys.exit(train_model(args.data, args.min_confidence))
    run_secure_pipeline()
