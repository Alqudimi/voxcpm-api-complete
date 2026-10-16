import hashlib
import os
import secrets
from pathlib import Path

from .errors import ValidationError


class LocalStorage:
    def __init__(self, root: Path, temp: Path):
        self.root = root.resolve()
        self.temp = temp.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.temp.mkdir(parents=True, exist_ok=True)

    def safe_key(self, key):
        if (
            not key
            or "/" in key
            or "\\" in key
            or ".." in key
            or key.startswith(".")
            or "\x00" in key
        ):
            raise ValidationError("Invalid storage key")
        return key

    def save_upload(
        self, source, suffix="", max_size=None
    ):  # source is a file-like object
        key = secrets.token_hex(16) + suffix
        tmp = self.temp / (key + ".part")
        final = self.root / key
        h = hashlib.sha256()
        size = 0
        try:
            with tmp.open("wb") as f:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if max_size is not None and size > max_size:
                        raise ValidationError("Upload is too large")
                    f.write(chunk)
                    h.update(chunk)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, final)
        except Exception:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            raise
        return key, size, h.hexdigest()

    def path(self, key):
        return self.root / self.safe_key(key)

    def delete(self, key):
        try:
            self.path(key).unlink()
        except FileNotFoundError:
            pass

    def cleanup_temp(self):
        for p in self.temp.glob("*.part"):
            try:
                p.unlink()
            except OSError:
                pass
