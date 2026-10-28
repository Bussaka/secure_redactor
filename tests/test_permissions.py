import os
import stat
import subprocess

from src.permissions import is_restricted_to_owner, restrict_to_owner
from src.secure_store import CryptographicVault

def test_restrict_file(tmp_path):
    path = tmp_path / "secret.key"
    path.write_text("x")
    assert restrict_to_owner(str(path))
    assert is_restricted_to_owner(str(path))
    if os.name == "nt":
        acl = subprocess.run(["icacls", str(path)], capture_output=True, text=True).stdout
        assert "BUILTIN\\Users" not in acl and "Everyone" not in acl and "(I)" not in acl
    else:
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

def test_restricted_folder_applies_to_new_files(tmp_path):
    folder = tmp_path / "vault"
    folder.mkdir()
    assert restrict_to_owner(str(folder))
    (folder / "later.enc").write_text("x")
    if os.name == "nt":
        assert is_restricted_to_owner(str(folder / "later.enc"))

def test_new_key_is_created_restricted(tmp_path):
    key_path = tmp_path / "master.key"
    CryptographicVault(key_path=str(key_path))
    assert is_restricted_to_owner(str(key_path))

def test_existing_open_key_is_tightened_on_load(tmp_path):
    key_path = tmp_path / "master.key"
    key_path.write_bytes(CryptographicVault(key_path=str(tmp_path / "other.key")).key)
    if os.name != "nt":
        os.chmod(key_path, 0o644)
    assert not is_restricted_to_owner(str(key_path))  # inherits the temp folder's broader ACL on Windows
    CryptographicVault(key_path=str(key_path))
    assert is_restricted_to_owner(str(key_path))
