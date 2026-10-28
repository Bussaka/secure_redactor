import json
from cryptography.fernet import Fernet

from src.audit_log import AuditLog, GENESIS_HASH, sha256_of_file

def write_entries(audit, n=3):
    for i in range(n):
        audit.record("processed", source_name=f"doc{i}.txt", sha256="ab" * 32)

def lines(audit):
    with open(audit.path, encoding="utf-8") as f:
        return f.read().splitlines()

def save(audit, new_lines):
    with open(audit.path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

def test_empty_log_is_valid(audit):
    assert audit.verify() == (True, 0, None)

def test_entries_form_a_chain(audit):
    write_entries(audit)
    entries = [json.loads(l) for l in lines(audit)]
    assert entries[0]["prev_hash"] == GENESIS_HASH
    assert entries[1]["prev_hash"] == entries[0]["entry_hash"]
    assert entries[2]["prev_hash"] == entries[1]["entry_hash"]
    assert audit.verify() == (True, 3, None)

def test_chain_continues_across_restarts(tmp_path):
    key = Fernet.generate_key()
    path = str(tmp_path / "audit.jsonl")
    write_entries(AuditLog(path, key), n=2)
    reopened = AuditLog(path, key)
    reopened.record("restored", vault_archive="x.enc")
    assert reopened.verify() == (True, 3, None)

def test_modified_entry_is_detected(audit):
    write_entries(audit)
    content = lines(audit)
    content[1] = content[1].replace("doc1.txt", "innocent.txt")
    save(audit, content)
    ok, count, problem = audit.verify()
    assert not ok and count == 1 and "signature mismatch" in problem

def test_deleted_entry_is_detected(audit):
    write_entries(audit)
    content = lines(audit)
    save(audit, [content[0], content[2]])
    ok, count, problem = audit.verify()
    assert not ok and "breaks the chain" in problem

def test_reordered_entries_are_detected(audit):
    write_entries(audit)
    content = lines(audit)
    save(audit, [content[1], content[0], content[2]])
    assert audit.verify()[0] is False

def test_forged_chain_without_key_is_detected(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    write_entries(AuditLog(path, Fernet.generate_key()))
    # An attacker rebuilding the whole chain with their own key still fails against the real key
    attacker = AuditLog(str(tmp_path / "forged.jsonl"), Fernet.generate_key())
    write_entries(attacker)
    real_key_check = AuditLog(attacker.path, Fernet.generate_key())
    assert real_key_check.verify()[0] is False

def test_garbage_line_is_detected(audit):
    write_entries(audit, n=1)
    save(audit, lines(audit) + ["not json"])
    ok, count, problem = audit.verify()
    assert not ok and count == 1 and "not a valid audit entry" in problem

def test_sha256_of_file(tmp_path):
    import hashlib
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    assert sha256_of_file(str(p)) == hashlib.sha256(b"hello").hexdigest()
