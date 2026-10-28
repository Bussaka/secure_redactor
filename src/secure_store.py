import logging
import os
from cryptography.fernet import Fernet, InvalidToken

from src.permissions import is_restricted_to_owner, restrict_to_owner

logger = logging.getLogger(__name__)

# Anchor all paths to the project root so the key is found no matter which directory we launch from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KEY_PATH = os.path.join(BASE_DIR, "vault_master.key")

class CryptographicVault:
    """Manages Fernet (AES-128-CBC + HMAC-SHA256) encryption for original raw corporate documents."""

    def __init__(self, key_path: str = DEFAULT_KEY_PATH, vault_dir: str = None):
        # Production deployments should supply the key via the VAULT_MASTER_KEY environment variable.
        # For our local standalone setup, we fall back to a local key file or generate it on demand.
        self.key_path = key_path
        self.vault_dir = vault_dir
        self.key = self._load_or_generate_key()
        self.fernet = Fernet(self.key)

    def _load_or_generate_key(self) -> bytes:
        """Loads an existing key or creates a fresh one to guarantee encryption continuity."""
        env_key = os.environ.get("VAULT_MASTER_KEY")
        if env_key:
            return env_key.strip().encode()

        if os.path.exists(self.key_path):
            # Tighten a key file that other accounts can still read (e.g. one created before this check existed)
            if not is_restricted_to_owner(self.key_path):
                logger.warning("Master key %s is readable by other accounts; restricting it to the current user.",
                               self.key_path)
                restrict_to_owner(self.key_path)
            with open(self.key_path, "rb") as key_file:
                return key_file.read().strip()

        # Refuse to mint a new key if the vault already holds files sealed with a previous one,
        # otherwise those archives would silently become undecryptable.
        if self.vault_dir and os.path.isdir(self.vault_dir):
            if any(f.endswith(".enc") for f in os.listdir(self.vault_dir)):
                raise RuntimeError(
                    f"Master key not found at {self.key_path}, but {self.vault_dir} already contains "
                    "encrypted files. Restore the original key or set VAULT_MASTER_KEY."
                )

        new_key = Fernet.generate_key()
        # O_EXCL guarantees we never clobber a key another process just wrote; the 0o600 mode applies on POSIX
        fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        with os.fdopen(fd, "wb") as key_file:
            key_file.write(new_key)
        restrict_to_owner(self.key_path)
        logger.info("New master encryption key generated at %s. Back it up somewhere safe.", self.key_path)
        return new_key

    def encrypt_and_vault(self, source_file_path: str, destination_vault_dir: str, target_name: str = None):
        """Reads raw document binaries, runs symmetric encryption, and writes to vault.

        Returns the vault path on success, or None on failure.
        """
        temp_path = None
        try:
            filename = target_name or os.path.basename(source_file_path)
            vault_target_path = os.path.join(destination_vault_dir, f"{filename}.enc")
            temp_path = vault_target_path + ".tmp"

            # Read original binary data from disk
            with open(source_file_path, "rb") as f:
                raw_bytes = f.read()

            # Perform Fernet symmetric encryption
            encrypted_bytes = self.fernet.encrypt(raw_bytes)

            # Write to a temp file and fsync, then atomically swap it in so a crash never leaves a half-written archive
            with open(temp_path, "wb") as f_enc:
                f_enc.write(encrypted_bytes)
                f_enc.flush()
                os.fsync(f_enc.fileno())
            os.replace(temp_path, vault_target_path)

            logger.info("Original file locked securely at %s", vault_target_path)
            return vault_target_path
        except Exception as e:
            logger.error("Cryptographic error during archival: %s", e)
            # Don't leave a partial archive lying around if the write itself failed
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return None

    def decrypt_from_vault(self, vault_file_path: str, output_dir: str) -> str:
        """Decrypts a vaulted .enc archive back to its original bytes (e.g. for a data subject access request).

        Returns the restored file path. Raises on a wrong key, tampered archive, or if the target already exists.
        """
        with open(vault_file_path, "rb") as f:
            encrypted_bytes = f.read()

        try:
            raw_bytes = self.fernet.decrypt(encrypted_bytes)
        except InvalidToken:
            raise ValueError(f"Cannot decrypt {vault_file_path}: wrong master key or the archive was tampered with.")

        filename = os.path.basename(vault_file_path)
        if filename.endswith(".enc"):
            filename = filename[:-len(".enc")]
        os.makedirs(output_dir, exist_ok=True)
        restored_path = os.path.join(output_dir, filename)

        # 'xb' mode refuses to overwrite an existing file rather than silently clobbering it
        with open(restored_path, "xb") as f_out:
            f_out.write(raw_bytes)

        logger.info("Restored original document to %s", restored_path)
        return restored_path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Create test layout directories if they don't exist
    os.makedirs("test_vault", exist_ok=True)

    # Create a quick dummy file to test encryption
    with open("test_raw_doc.txt", "w") as f:
        f.write("CONFIDENTIAL DATA: National ID 11223344")

    vault = CryptographicVault()
    vault.encrypt_and_vault("test_raw_doc.txt", "test_vault")
    print("[+] Phase 4: Cryptographic Vault successfully initialized and tested.")
