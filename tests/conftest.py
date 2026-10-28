import pytest
from cryptography.fernet import Fernet

import main
from src.audit_log import AuditLog
from src.classifier import DocumentClassifier
from src.extractor import DocumentExtractor
from src.redactor import HybridRedactor
from src.secure_store import CryptographicVault

# Loading spaCy and training the classifier is slow, so share one instance across the whole run
@pytest.fixture(scope="session")
def redactor():
    r = HybridRedactor()
    assert r.nlp is not None, "en_core_web_sm must be installed for the test suite (see requirements.txt)"
    return r

@pytest.fixture(scope="session")
def classifier():
    return DocumentClassifier()

@pytest.fixture
def extractor():
    return DocumentExtractor()

@pytest.fixture(autouse=True)
def no_env_key(monkeypatch):
    """Make sure a VAULT_MASTER_KEY set on the developer's machine never leaks into tests."""
    monkeypatch.delenv("VAULT_MASTER_KEY", raising=False)

@pytest.fixture
def vault(tmp_path):
    return CryptographicVault(key_path=str(tmp_path / "test.key"), vault_dir=str(tmp_path / "vault"))

@pytest.fixture
def pipeline_dirs(tmp_path, monkeypatch):
    """Points main.py at a throwaway data/ layout so tests never touch the real vault or key."""
    dirs = {}
    for name in ("INCOMING_DIR", "SAFE_DIR", "VAULT_DIR", "QUARANTINE_DIR", "RESTORED_DIR", "AUDIT_DIR", "LOG_DIR"):
        path = tmp_path / name.lower()
        path.mkdir()
        monkeypatch.setattr(main, name, str(path))
        dirs[name] = path

    key_path = str(tmp_path / "test.key")
    real_vault = main.CryptographicVault
    monkeypatch.setattr(main, "CryptographicVault",
                        lambda vault_dir=None: real_vault(key_path=key_path, vault_dir=vault_dir))
    monkeypatch.setattr(main, "TRAINING_DIR", str(tmp_path / "training_dir"))
    monkeypatch.setattr(main, "MODEL_PATH", str(tmp_path / "models_dir" / "document_classifier.joblib"))
    dirs["TRAINING_DIR"] = tmp_path / "training_dir"
    dirs["MODEL_PATH"] = tmp_path / "models_dir" / "document_classifier.joblib"
    # Treat a file as settled as soon as two consecutive scans agree, so tests don't need real waiting
    monkeypatch.setattr(main, "SETTLE_SECONDS", 0)
    dirs["key_path"] = key_path
    return dirs

@pytest.fixture
def audit(tmp_path):
    return AuditLog(str(tmp_path / "audit" / "audit_log.jsonl"), Fernet.generate_key())
