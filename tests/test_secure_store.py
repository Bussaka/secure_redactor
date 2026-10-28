import os
import pytest
from cryptography.fernet import Fernet

from src.secure_store import CryptographicVault

ORIGINAL = b"CONFIDENTIAL: National ID 34567890, phone 0712345678"

@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "original.pdf"
    path.write_bytes(ORIGINAL)
    return str(path)

def test_round_trip(vault, source_file, tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    enc_path = vault.encrypt_and_vault(source_file, str(vault_dir))

    assert enc_path.endswith("original.pdf.enc")
    assert ORIGINAL not in open(enc_path, "rb").read()
    assert not [f for f in os.listdir(vault_dir) if f.endswith(".tmp")]

    restored = vault.decrypt_from_vault(enc_path, str(tmp_path / "restored"))
    assert os.path.basename(restored) == "original.pdf"
    assert open(restored, "rb").read() == ORIGINAL

def test_target_name_is_used(vault, source_file, tmp_path):
    enc_path = vault.encrypt_and_vault(source_file, str(tmp_path), target_name="run1_original.pdf")
    assert os.path.basename(enc_path) == "run1_original.pdf.enc"

def test_encrypt_failure_returns_none(vault, source_file, tmp_path):
    assert vault.encrypt_and_vault(source_file, str(tmp_path / "does_not_exist")) is None
    assert os.path.exists(source_file)

def test_wrong_key_is_rejected(vault, source_file, tmp_path):
    enc_path = vault.encrypt_and_vault(source_file, str(tmp_path))
    other = CryptographicVault(key_path=str(tmp_path / "other.key"))
    with pytest.raises(ValueError, match="wrong master key"):
        other.decrypt_from_vault(enc_path, str(tmp_path / "restored"))

def test_tampered_archive_is_rejected(vault, source_file, tmp_path):
    enc_path = vault.encrypt_and_vault(source_file, str(tmp_path))
    data = bytearray(open(enc_path, "rb").read())
    data[len(data) // 2] ^= 0x01
    open(enc_path, "wb").write(bytes(data))
    with pytest.raises(ValueError, match="tampered"):
        vault.decrypt_from_vault(enc_path, str(tmp_path / "restored"))

def test_restore_never_overwrites(vault, source_file, tmp_path):
    enc_path = vault.encrypt_and_vault(source_file, str(tmp_path))
    out_dir = str(tmp_path / "restored")
    vault.decrypt_from_vault(enc_path, out_dir)
    with pytest.raises(FileExistsError):
        vault.decrypt_from_vault(enc_path, out_dir)

def test_key_is_reused_across_instances(tmp_path, source_file):
    key_path = str(tmp_path / "master.key")
    first = CryptographicVault(key_path=key_path)
    enc_path = first.encrypt_and_vault(source_file, str(tmp_path))
    second = CryptographicVault(key_path=key_path)
    restored = second.decrypt_from_vault(enc_path, str(tmp_path / "restored"))
    assert open(restored, "rb").read() == ORIGINAL

def test_refuses_new_key_over_existing_vault(tmp_path, source_file):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    key_path = tmp_path / "master.key"
    CryptographicVault(key_path=str(key_path)).encrypt_and_vault(source_file, str(vault_dir))

    key_path.unlink()
    with pytest.raises(RuntimeError, match="already contains encrypted files"):
        CryptographicVault(key_path=str(key_path), vault_dir=str(vault_dir))
    assert not key_path.exists()

def test_env_key_takes_priority(tmp_path, monkeypatch):
    env_key = Fernet.generate_key()
    monkeypatch.setenv("VAULT_MASTER_KEY", env_key.decode())
    vault = CryptographicVault(key_path=str(tmp_path / "unused.key"))
    assert vault.key == env_key
    assert not (tmp_path / "unused.key").exists()
