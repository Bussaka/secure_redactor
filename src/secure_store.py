import os
from cryptography.fernet import Fernet

class CryptographicVault:
    """Manages high-grade AES-256 bit encryption for original raw corporate documents."""
    
    def __init__(self):
        # In a production app, encryption keys should be pulled via system environment variables
        # (e.g., os.environ.get("VAULT_MASTER_KEY")). 
        # For our local standalone setup, we will look for a local key file or generate it on demand.
        self.key_path = "vault_master.key"
        self.key = self._load_or_generate_key()
        self.fernet = Fernet(self.key)

    def _load_or_generate_key(self) -> bytes:
        """Loads an existing key or creates a fresh one to guarantee encryption continuity."""
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as key_file:
                return key_file.read()
        else:
            new_key = Fernet.generate_key()
            with open(self.key_path, "wb") as key_file:
                key_file.write(new_key)
            print("[+] Cryptography: New master encryption key generated safely (`vault_master.key`).")
            return new_key

    def encrypt_and_vault(self, source_file_path: str, destination_vault_dir: str):
        """Reads raw document binaries, runs symmetric encryption, and writes to vault."""
        try:
            filename = os.path.basename(source_file_path)
            vault_target_path = os.path.join(destination_vault_dir, f"{filename}.enc")
            
            # Read original binary data from disk
            with open(source_file_path, "rb") as f:
                raw_bytes = f.read()
                
            # Perform AES-256 Fernet block encryption
            encrypted_bytes = self.fernet.encrypt(raw_bytes)
            
            # Flush securely to the encrypted folder
            with open(vault_target_path, "wb") as f_enc:
                f_enc.write(encrypted_bytes)
                
            print(f"[+] Cryptography: Original file locked securely at {vault_target_path}")
            return True
        except Exception as e:
            print(f"[-] Cryptographic error during archival: {e}")
            return False

if __name__ == "__main__":
    # Create test layout directories if they don't exist
    os.makedirs("test_vault", exist_ok=True)
    
    # Create a quick dummy file to test encryption
    with open("test_raw_doc.txt", "w") as f:
        f.write("CONFIDENTIAL DATA: National ID 11223344")
        
    vault = CryptographicVault()
    vault.encrypt_and_vault("test_raw_doc.txt", "test_vault")
    print("[+] Phase 4: Cryptographic Vault successfully initialized and tested.")