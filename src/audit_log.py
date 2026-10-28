import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64

def sha256_of_file(path: str) -> str:
    """Fingerprints a document so the audit trail can prove which exact file was processed, without storing it."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

class AuditLog:
    """Append-only, tamper-evident JSON Lines audit trail of every pipeline action.

    Each entry carries the signature of the previous one (a hash chain), and every signature is an
    HMAC keyed from the vault master key. Editing, deleting or reordering any entry therefore breaks
    verification, and nobody without the master key can forge a replacement chain.
    Entries record file names, hashes and counts only, never document contents or the PII itself.
    """

    def __init__(self, path: str, master_key: bytes):
        self.path = path
        # Derive a separate key so the audit signatures never reuse the encryption key directly
        self._key = hmac.new(master_key, b"secure-redactor-audit-log-v1", hashlib.sha256).digest()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._last_hash = self._read_last_hash()

    def _sign(self, entry: dict) -> str:
        body = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hmac.new(self._key, body, hashlib.sha256).hexdigest()

    def _read_last_hash(self) -> str:
        if not os.path.exists(self.path):
            return GENESIS_HASH
        last_line = ""
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line
        if not last_line:
            return GENESIS_HASH
        return json.loads(last_line)["entry_hash"]

    def record(self, event: str, **fields) -> dict:
        """Appends one signed entry and flushes it to disk immediately."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            **fields,
            "prev_hash": self._last_hash,
        }
        entry["entry_hash"] = self._sign(entry)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._last_hash = entry["entry_hash"]
        return entry

    def verify(self):
        """Re-checks every signature and chain link. Returns (is_valid, entry_count, problem_description)."""
        if not os.path.exists(self.path):
            return True, 0, None

        expected_prev = GENESIS_HASH
        count = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    claimed = entry.pop("entry_hash")
                except (ValueError, KeyError):
                    return False, count, f"line {line_no} is not a valid audit entry"
                if entry.get("prev_hash") != expected_prev:
                    return False, count, f"line {line_no} breaks the chain (an entry was removed, reordered or inserted)"
                if not hmac.compare_digest(claimed, self._sign(entry)):
                    return False, count, f"line {line_no} signature mismatch (entry was modified or wrong master key)"
                expected_prev = claimed
                count += 1
        return True, count, None
