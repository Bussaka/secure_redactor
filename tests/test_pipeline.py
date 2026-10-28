import json
import os
import pytest

import main

DOC = "Supplier onboarding: John Kamau, email john@kamau.co.ke, phone 0712345678, ID 34567890."

class StopLoop(Exception):
    pass

def list_dir(path):
    return sorted(os.listdir(path))

def read_audit(dirs):
    path = dirs["AUDIT_DIR"] / "audit_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def run_loop(monkeypatch, scans, on_scan=None):
    """Runs the real watch loop for a fixed number of scans, replacing the filesystem watcher with a stub."""
    count = {"n": 0}

    class FakeWatcher:
        def __init__(self, path):
            self.stopped = False
        def wait(self, timeout):
            count["n"] += 1
            if on_scan:
                on_scan(count["n"])
            if count["n"] >= scans:
                raise StopLoop
            return True
        def stop(self):
            pass

    monkeypatch.setattr(main, "FolderWatcher", FakeWatcher)
    with pytest.raises(StopLoop):
        main.run_secure_pipeline()

def test_document_is_redacted_vaulted_and_removed(pipeline_dirs, monkeypatch):
    (pipeline_dirs["INCOMING_DIR"] / "doc.txt").write_text(DOC, encoding="utf-8")
    run_loop(monkeypatch, scans=2)

    assert list_dir(pipeline_dirs["INCOMING_DIR"]) == []
    [safe_name] = list_dir(pipeline_dirs["SAFE_DIR"])
    safe_text = (pipeline_dirs["SAFE_DIR"] / safe_name).read_text(encoding="utf-8")
    for leaked in ("John", "Kamau", "john@", "0712345678", "34567890"):
        assert leaked not in safe_text

    [enc_name] = list_dir(pipeline_dirs["VAULT_DIR"])
    assert enc_name.endswith("_doc.txt.enc")

def test_processing_is_audited_without_pii(pipeline_dirs, monkeypatch):
    import hashlib
    (pipeline_dirs["INCOMING_DIR"] / "doc.txt").write_text(DOC, encoding="utf-8")
    run_loop(monkeypatch, scans=2)

    [entry] = read_audit(pipeline_dirs)
    assert entry["event"] == "processed"
    assert entry["source_name"] == "doc.txt"
    assert entry["sha256"] == hashlib.sha256(DOC.encode("utf-8")).hexdigest()
    assert entry["redactions"]["REDACTED_EMAIL"] == 1
    assert entry["redactions"]["REDACTED_PHONE_NUMBER"] == 1
    assert entry["redactions"]["REDACTED_NATIONAL_ID"] == 1
    assert entry["redactions"]["REDACTED_PERSON"] >= 1
    assert entry["vault_archive"] in list_dir(pipeline_dirs["VAULT_DIR"])

    audit_text = (pipeline_dirs["AUDIT_DIR"] / "audit_log.jsonl").read_text(encoding="utf-8")
    for leaked in ("Kamau", "john@", "0712345678", "34567890"):
        assert leaked not in audit_text
    assert main.verify_audit_log() == 0

def test_same_name_uploads_do_not_overwrite(pipeline_dirs, monkeypatch):
    incoming = pipeline_dirs["INCOMING_DIR"]
    (incoming / "doc.txt").write_text(DOC, encoding="utf-8")

    def drop_second_copy(scan):
        if scan == 2:
            (incoming / "doc.txt").write_text(DOC, encoding="utf-8")
    run_loop(monkeypatch, scans=4, on_scan=drop_second_copy)

    assert len(list_dir(pipeline_dirs["SAFE_DIR"])) == 2
    assert len(list_dir(pipeline_dirs["VAULT_DIR"])) == 2
    assert [e["event"] for e in read_audit(pipeline_dirs)] == ["processed", "processed"]

def test_file_still_being_written_is_not_processed(pipeline_dirs, monkeypatch):
    path = pipeline_dirs["INCOMING_DIR"] / "growing.txt"
    path.write_text("Invoice", encoding="utf-8")

    def keep_growing(scan):
        with open(path, "a", encoding="utf-8") as f:
            f.write(" more")
    run_loop(monkeypatch, scans=3, on_scan=keep_growing)

    assert list_dir(pipeline_dirs["INCOMING_DIR"]) == ["growing.txt"]
    assert list_dir(pipeline_dirs["VAULT_DIR"]) == []

def test_file_must_settle_for_the_configured_time(pipeline_dirs, monkeypatch):
    (pipeline_dirs["INCOMING_DIR"] / "doc.txt").write_text(DOC, encoding="utf-8")
    monkeypatch.setattr(main, "SETTLE_SECONDS", 60)

    ready, snapshot = main.scan_incoming({}, now=1000.0)
    assert ready == []
    ready, snapshot = main.scan_incoming(snapshot, now=1030.0)
    assert ready == []
    ready, _ = main.scan_incoming(snapshot, now=1061.0)
    assert ready == ["doc.txt"]

def test_unsupported_and_folders(pipeline_dirs, monkeypatch):
    incoming = pipeline_dirs["INCOMING_DIR"]
    (incoming / "notes.docx").write_bytes(b"PK")
    (incoming / "subfolder").mkdir()
    run_loop(monkeypatch, scans=2)

    assert list_dir(incoming) == ["subfolder"]
    [quarantined] = list_dir(pipeline_dirs["QUARANTINE_DIR"])
    assert quarantined.endswith("_notes.docx")
    [entry] = read_audit(pipeline_dirs)
    assert entry["event"] == "quarantined" and entry["quarantine_file"] == quarantined

def test_unexpected_error_does_not_stop_the_loop(pipeline_dirs, monkeypatch):
    incoming = pipeline_dirs["INCOMING_DIR"]
    (incoming / "a.txt").write_text(DOC, encoding="utf-8")
    (incoming / "b.txt").write_text(DOC, encoding="utf-8")

    real_predict = main.DocumentClassifier.predict_with_confidence
    def flaky_predict(self, text):
        if flaky_predict.calls == 0:
            flaky_predict.calls += 1
            raise RuntimeError("boom")
        return real_predict(self, text)
    flaky_predict.calls = 0
    monkeypatch.setattr(main.DocumentClassifier, "predict_with_confidence", flaky_predict)
    run_loop(monkeypatch, scans=2)

    assert len(list_dir(pipeline_dirs["QUARANTINE_DIR"])) == 1
    assert len(list_dir(pipeline_dirs["VAULT_DIR"])) == 1

def test_vault_failure_keeps_original(pipeline_dirs, extractor, classifier, redactor, audit):
    path = pipeline_dirs["INCOMING_DIR"] / "doc.txt"
    path.write_text(DOC, encoding="utf-8")

    class BrokenVault:
        def encrypt_and_vault(self, *args, **kwargs):
            return None
    main.process_single_file(str(path), "doc.txt", extractor, classifier, redactor, BrokenVault(), audit)

    [quarantined] = list_dir(pipeline_dirs["QUARANTINE_DIR"])
    assert (pipeline_dirs["QUARANTINE_DIR"] / quarantined).read_text(encoding="utf-8") == DOC
    assert list_dir(pipeline_dirs["SAFE_DIR"]) == []

def test_restore_command(pipeline_dirs, monkeypatch):
    (pipeline_dirs["INCOMING_DIR"] / "doc.txt").write_text(DOC, encoding="utf-8")
    run_loop(monkeypatch, scans=2)
    [enc_name] = list_dir(pipeline_dirs["VAULT_DIR"])

    args = main.parse_args(["restore", enc_name, "--out", str(pipeline_dirs["RESTORED_DIR"])])
    assert main.restore_from_vault(args.vault_file, args.out) == 0
    [restored] = list_dir(pipeline_dirs["RESTORED_DIR"])
    assert (pipeline_dirs["RESTORED_DIR"] / restored).read_text(encoding="utf-8") == DOC

    # A second restore must refuse instead of overwriting
    assert main.restore_from_vault(enc_name, str(pipeline_dirs["RESTORED_DIR"])) == 1
    assert [e["event"] for e in read_audit(pipeline_dirs)] == ["processed", "restored"]

def test_restore_missing_archive(pipeline_dirs):
    assert main.restore_from_vault("nope.enc", str(pipeline_dirs["RESTORED_DIR"])) == 1

def test_verify_audit_detects_tampering(pipeline_dirs, monkeypatch):
    (pipeline_dirs["INCOMING_DIR"] / "doc.txt").write_text(DOC, encoding="utf-8")
    run_loop(monkeypatch, scans=2)
    log = pipeline_dirs["AUDIT_DIR"] / "audit_log.jsonl"
    log.write_text(log.read_text(encoding="utf-8").replace('"Unknown"', '"Invoice"').replace("doc.txt", "other.txt"),
                   encoding="utf-8")
    assert main.verify_audit_log() == 1

def test_real_folder_watcher_wakes_on_new_file(tmp_path):
    watcher = main.FolderWatcher(str(tmp_path))
    try:
        assert watcher.is_live
        assert watcher.wait(0.2) is False  # nothing happened yet
        (tmp_path / "new.txt").write_text("x", encoding="utf-8")
        assert watcher.wait(5) is True
    finally:
        watcher.stop()
