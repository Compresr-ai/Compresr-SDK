"""Persistent credentials store at ``~/.compresr/credentials``."""

from __future__ import annotations

import configparser
import os
import re
import stat
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_PROFILE = "default"

_REPLACE_RETRIES = 6
_REPLACE_BACKOFF_S = 0.1
_LOCK_DEADLINE_S = 10.0
_LOCK_POLL_S = 0.1

_API_KEY_RE = re.compile(r"^cmp_[A-Za-z0-9_-]{16,128}$")


class LockAcquisitionError(RuntimeError):
    """Raised when the credentials lock can't be acquired in time."""


def credentials_path() -> Path:
    override = os.environ.get("COMPRESR_CREDENTIALS_FILE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".compresr" / "credentials"


def _new_parser() -> configparser.ConfigParser:
    return configparser.ConfigParser(interpolation=None)


def _refuse_symlink(path: Path, kind: str) -> None:
    if path.is_symlink():
        raise PermissionError(f"Refusing to use {kind} {path}: symlink at that path.")


def _read_file() -> configparser.ConfigParser:
    parser = _new_parser()
    path = credentials_path()
    if not path.exists():
        return parser
    _refuse_symlink(path, "credentials file")
    if os.name == "posix":
        mode = path.lstat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError(
                f"Refusing to read {path}: file is group/world accessible. "
                "Run `chmod 600` on it."
            )
    parser.read(path)
    return parser


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")

    if os.name == "posix":
        import fcntl

        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        acquired = False
        try:
            deadline = time.monotonic() + _LOCK_DEADLINE_S
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        raise LockAcquisitionError(
                            f"Could not acquire {lock_path} within "
                            f"{_LOCK_DEADLINE_S:g}s — another process may be "
                            "writing, or a previous run left the lock stale."
                        )
                    time.sleep(_LOCK_POLL_S)
            yield
        finally:
            if acquired:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(fd)
        return

    try:
        import msvcrt  # type: ignore[import-not-found]
    except ImportError:
        print(
            "warning: file locking unavailable on this platform; "
            "concurrent `compresr-sdk` invocations may race.",
            file=sys.stderr,
        )
        yield
        return
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
    acquired = False
    try:
        deadline = time.monotonic() + _LOCK_DEADLINE_S
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise LockAcquisitionError(
                        f"Could not acquire {lock_path} within {_LOCK_DEADLINE_S:g}s."
                    )
                time.sleep(_LOCK_POLL_S)
        yield
    finally:
        if acquired:
            try:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            except OSError:
                pass
        os.close(fd)


def _atomic_write(parser: configparser.ConfigParser, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        default_dir = Path.home() / ".compresr"
        if path.parent == default_dir:
            try:
                os.chmod(path.parent, 0o700)
            except OSError:
                pass

    tmp_path: Optional[Path] = None
    replaced = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=str(path.parent), delete=False, prefix=".credentials-"
        ) as tmp:
            tmp_path = Path(tmp.name)
            parser.write(tmp)
            tmp.flush()
            os.fsync(tmp.fileno())

        if os.name == "posix":
            os.chmod(tmp_path, 0o600)

        if path.is_symlink():
            raise PermissionError(f"Refusing to replace symlink at {path}; remove it first.")

        last_err: Optional[OSError] = None
        for attempt in range(_REPLACE_RETRIES):
            try:
                os.replace(tmp_path, path)
                replaced = True
                break
            except OSError as e:
                last_err = e
                time.sleep(_REPLACE_BACKOFF_S * (2**attempt))
        if not replaced and last_err is not None:
            raise last_err
    finally:
        if not replaced and tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _safe_read_or_fresh(path: Path) -> configparser.ConfigParser:
    try:
        return _read_file()
    except configparser.Error as e:
        print(
            f"warning: credentials file at {path} is corrupt ({e}); recreating.",
            file=sys.stderr,
        )
        return _new_parser()


def load(profile: str = DEFAULT_PROFILE) -> Optional[str]:
    """Return the stored API key for ``profile``, or ``None``."""
    try:
        parser = _read_file()
    except configparser.Error:
        return None
    if profile not in parser:
        return None
    return parser[profile].get("api_key") or None


def save(
    api_key: str,
    *,
    profile: str = DEFAULT_PROFILE,
    base_url: Optional[str] = None,
    account_type: Optional[str] = None,
) -> Path:
    from datetime import datetime, timezone

    if not _API_KEY_RE.match(api_key):
        raise ValueError(
            "Refusing to save malformed api key: " "must match ^cmp_[A-Za-z0-9_-]{16,128}$."
        )

    path = credentials_path()
    with _file_lock(path):
        parser = _safe_read_or_fresh(path) if path.exists() else _new_parser()
        section = {
            "api_key": api_key,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if base_url:
            section["base_url"] = base_url
        if account_type:
            section["account_type"] = account_type
        parser[profile] = section
        _atomic_write(parser, path)
    return path


def clear(profile: str = DEFAULT_PROFILE) -> bool:
    path = credentials_path()
    if not path.exists():
        return False
    with _file_lock(path):
        try:
            parser = _read_file()
        except configparser.Error:
            return False
        if profile not in parser:
            return False
        parser.remove_section(profile)
        _atomic_write(parser, path)
    return True


def resolve_api_key(explicit: Optional[str], profile: Optional[str] = None) -> Optional[str]:
    """explicit > env > credentials file. An explicit non-None short-circuits
    fallbacks (even ``""``) so downstream validation can distinguish "user
    passed empty" from "no arg passed"."""
    if explicit is not None:
        return explicit
    env = os.environ.get("COMPRESR_API_KEY")
    if env:
        return env
    try:
        return load(profile or os.environ.get("COMPRESR_PROFILE", DEFAULT_PROFILE))
    except PermissionError:
        return None


__all__ = [
    "credentials_path",
    "load",
    "save",
    "clear",
    "resolve_api_key",
    "DEFAULT_PROFILE",
    "LockAcquisitionError",
]
