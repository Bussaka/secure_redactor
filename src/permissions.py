import getpass
import logging
import os
import stat
import subprocess

logger = logging.getLogger(__name__)

def restrict_to_owner(path: str) -> bool:
    """Locks a file or folder down so only the current user (and SYSTEM on Windows) can access it.

    Returns True on success. Failure is logged but never fatal, so the pipeline still runs on
    filesystems that do not support permissions (e.g. FAT32 USB drives).
    """
    if os.name == "nt":
        return _restrict_windows(path)
    return _restrict_posix(path)

def _restrict_posix(path: str) -> bool:
    mode = stat.S_IRWXU if os.path.isdir(path) else stat.S_IRUSR | stat.S_IWUSR  # 700 / 600
    try:
        os.chmod(path, mode)
        return True
    except OSError as e:
        logger.warning("Could not restrict permissions on %s: %s", path, e)
        return False

def _restrict_windows(path: str) -> bool:
    # (OI)(CI) makes folder permissions apply to every file created inside it later
    rights = "(OI)(CI)F" if os.path.isdir(path) else "F"
    command = [
        "icacls", path,
        "/inheritance:r",                          # drop permissions inherited from parent folders
        "/grant:r", f"{getpass.getuser()}:{rights}",
        "/grant:r", f"*S-1-5-18:{rights}",         # SYSTEM, by well-known SID so it works in any language
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except (OSError, subprocess.CalledProcessError) as e:
        details = getattr(e, "stderr", "") or getattr(e, "stdout", "") or str(e)
        logger.warning("Could not restrict permissions on %s: %s", path, details.strip())
        return False

def is_restricted_to_owner(path: str) -> bool:
    """Best-effort check used to warn about a key file that other accounts can read."""
    if os.name == "nt":
        try:
            result = subprocess.run(["icacls", path], check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError):
            return False
        allowed = {getpass.getuser().lower(), "nt authority\\system"}
        for line in result.stdout.splitlines():
            entry = line.replace(path, "", 1).strip()
            if ":" not in entry or entry.startswith("Successfully"):
                continue
            account = entry.rsplit(":", 1)[0].strip().lower()
            if account not in allowed and account.split("\\")[-1] not in allowed:
                return False
        return True
    return stat.S_IMODE(os.stat(path).st_mode) & 0o077 == 0
