"""Persist original tool outputs under ``HERMES_HOME/cache/compresr/tool-output``
so recovery references resolve. Paths are handed out only when the active
backend can prove it can read them; otherwise callers fail open.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CACHE_SUBDIR = Path("cache") / "compresr" / "tool-output"
# Canonical string form of the cache subdir. Exposed so out-of-tree callers
# (e.g. a host plugin shim) import one source of truth instead of duplicating
# the literal and risking drift.
CACHE_SUBPATH = _CACHE_SUBDIR.as_posix()
_CONTAINER_HERMES_HOME = "/root/.hermes"
_DIR_MODE = 0o700
_FILE_MODE = 0o600
_MB = 1024 * 1024

_STORE_LOCK = threading.Lock()

# Recently-written entries are pinned against eviction so a footer path just
# handed to the model survives a parallel prune.
_PRUNE_PIN_SECONDS = 300.0


def cache_relpath() -> str:
    """Cache dir relative to HERMES_HOME, for host mount registration."""
    return CACHE_SUBPATH


def get_cache_root() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / _CACHE_SUBDIR


def ensure_cache_root() -> Path:
    root = get_cache_root()
    root.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    try:
        os.chmod(root, _DIR_MODE)
    except OSError:
        pass
    return root


def cache_file_path(cache_id: str) -> Path:
    return get_cache_root() / cache_id


def _get_active_env(task_id: str) -> Any:
    try:
        from tools.terminal_tool import get_active_env as _get

        return _get(task_id)
    except Exception:
        return None


def _container_file_visible(active_env: Any, container_path: str, host_path: Path) -> bool:
    """Prove the running container can read *container_path*: Docker bind-mounts
    are fixed at container creation, so a reused container may lack the mount
    even when path translation succeeds."""
    execute = getattr(active_env, "execute", None)
    if not callable(execute):
        return False
    try:
        expected = host_path.stat().st_size
    except OSError:
        return False
    try:
        res = execute(f"wc -c < {shlex.quote(str(container_path))}")
    except Exception as e:
        logger.debug("compresr: container visibility probe failed: %s", e)
        return False
    if not isinstance(res, dict) or res.get("returncode", 1) != 0:
        return False
    match = re.search(r"\d+", str(res.get("output", "")))
    return match is not None and int(match.group()) == expected


def _agent_visible_cache_path(cache_path: Path, task_id: str) -> Optional[str]:
    """Translate *cache_path* to a path the active backend can read, or None."""
    active_env = _get_active_env(task_id)
    probe_container = False

    if active_env is not None:
        env_name = active_env.__class__.__name__
        if env_name == "LocalEnvironment":
            try:
                return str(cache_path.resolve())
            except OSError:
                return str(cache_path)
        if env_name == "SingularityEnvironment" or "singularity" in env_name.lower():
            return None

        remote_home = getattr(active_env, "_remote_home", None)
        if isinstance(remote_home, str) and remote_home.strip():
            container_base = f"{remote_home.rstrip('/')}/.hermes"
        elif env_name in {"DockerEnvironment", "ModalEnvironment"}:
            container_base = _CONTAINER_HERMES_HOME
            probe_container = env_name == "DockerEnvironment"
        else:
            return None
    else:
        backend = (os.getenv("TERMINAL_ENV") or "local").strip().lower() or "local"
        if backend == "local":
            try:
                return str(cache_path.resolve())
            except OSError:
                return str(cache_path)
        if backend in {"docker", "modal"}:
            container_base = _CONTAINER_HERMES_HOME
        else:
            return None

    try:
        from tools.credential_files import map_cache_path_to_container

        translated = map_cache_path_to_container(str(cache_path), container_base=container_base)
        translated = str(translated) if translated else None
    except Exception as e:
        logger.debug("compresr: cache path mapping failed: %s", e)
        return None

    if (
        translated
        and probe_container
        and not _container_file_visible(active_env, translated, cache_path)
    ):
        logger.warning(
            "compresr: container cannot read cache path %s — failing open",
            translated,
        )
        return None
    return translated


def _force_sync_visible_cache(cache_path: Path, task_id: str) -> bool:
    active_env = _get_active_env(task_id)
    if active_env is None:
        return True

    env_name = active_env.__class__.__name__
    if env_name == "LocalEnvironment":
        return True
    if env_name == "SingularityEnvironment" or "singularity" in env_name.lower():
        return True

    sync_manager = None
    for attr in ("_sync_manager", "sync_manager", "_file_sync_manager"):
        candidate = getattr(active_env, attr, None)
        if candidate is not None and callable(getattr(candidate, "sync", None)):
            sync_manager = candidate
            break
    if sync_manager is None:
        return True

    try:
        try:
            sync_manager.sync(force=True, raise_on_error=True)
        except TypeError:
            sync_manager.sync(force=True)
        return True
    except Exception as e:
        logger.warning("compresr: force sync failed for %s: %s", cache_path, e)
        return False


def _prune_cache_dir(cache_dir: str, max_bytes: int, keep_path: str) -> None:
    if max_bytes <= 0:
        return
    root = Path(cache_dir)
    keep = Path(keep_path)
    now = time.time()
    try:
        entries = []
        total = 0
        for path in root.iterdir():
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            size = int(stat.st_size)
            total += size
            entries.append((float(stat.st_mtime), path, size))
        if total <= max_bytes:
            return
        for mtime, path, size in sorted(entries):
            try:
                if path.resolve() == keep.resolve():
                    continue
                if now - mtime < _PRUNE_PIN_SECONDS:
                    continue
                path.unlink()
                total -= size
            except OSError:
                continue
            if total <= max_bytes:
                break
    except Exception as e:
        logger.debug("compresr: cache prune failed for %s: %s", cache_dir, e)


def store_original(
    cache_id: str,
    content: str,
    task_id: str = "default",
    max_cache_mb: int = 256,
) -> Optional[str]:
    """Persist ``content`` under ``cache_id``; return an agent-visible path or
    None (caller must fail open)."""
    root = ensure_cache_root()
    cache_path = cache_file_path(cache_id)
    # O_NOFOLLOW + os.replace: symlink-safe, and concurrent readers see either
    # full old or full new bytes. Temp name is unique per call: identical
    # content-addressed ids written from two threads must never share a path.
    tmp_path = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        fd = os.open(
            str(tmp_path),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
            _FILE_MODE,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        try:
            os.chmod(tmp_path, _FILE_MODE)
        except OSError:
            pass
        os.replace(str(tmp_path), str(cache_path))
    except Exception as e:
        logger.warning("compresr: cache write failed for %s: %s", cache_path, e)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None

    sync_ok = _force_sync_visible_cache(cache_path, task_id)
    visible_path = _agent_visible_cache_path(cache_path, task_id) if sync_ok else None

    # Prune even when the path isn't visible so non-visible backends can't grow
    # the cache without bound. Never unlink on visibility failure: cache ids are
    # content-addressed and a sibling may already hold this path.
    try:
        with _STORE_LOCK:
            _prune_cache_dir(str(root), max(0, int(max_cache_mb)) * _MB, str(cache_path))
    except Exception as e:
        logger.debug("compresr: cache prune failed: %s", e)

    if visible_path is None:
        logger.warning(
            "compresr: cache path not visible to the active backend: %s",
            cache_path,
        )
        return None
    return visible_path
