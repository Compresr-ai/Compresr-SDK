"""Compresr tool-output compression hook for Hermes.

Compresses large tool outputs on Hermes's ``transform_tool_result`` hook and
appends a footer pointing at a cached copy of the original — with secrets/PII
masked — in Hermes's managed cache (recoverable via ``read_file``/``search_files``).
Fail-open throughout.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, cast

from compresr.credentials import resolve_api_key

from . import cache
from ._config import as_bool, as_float, as_int, opt, read_config_block
from ._security import resolve_base_url, sanitize_secret
from .recovery import (
    DEFAULT_TOOL_OUTPUT_MODEL,
    FOOTER_MARKER,
    compress_with_recovery,
    count_tokens,
)

try:
    from agent.redact import redact_sensitive_text as _redact
except Exception:
    import logging as _logging
    import re as _re

    # agent.redact unavailable — apply a conservative built-in pass so secrets
    # never fall through to the compression API or recovery cache.
    _logging.getLogger(__name__).warning(
        "compresr: agent.redact unavailable — using built-in fallback redactor"
    )
    _SECRET_PATTERNS = [
        # PEM private-key blocks (multiline) — must run before line-oriented rules.
        _re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            _re.DOTALL,
        ),
        _re.compile(r"\b(?:sk|pk|rk)[-_](?:live|test|proj)?[-_]?[A-Za-z0-9]{16,}\b"),
        _re.compile(r"\bcmp_[A-Za-z0-9_-]{16,}\b"),
        _re.compile(r"\b(?:ghp|gho|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}\b"),
        _re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
        _re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        _re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
        _re.compile(r"\bya29\.[0-9A-Za-z_-]{20,}\b"),
        _re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        _re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|bearer)\b\s*[:=]\s*\S+"),
        # URI credentials; username may be empty (redis://:pass@host).
        _re.compile(r"(?i)(?<=://)[^\s:/@]*:[^\s:/@]+(?=@)"),
        _re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ]

    def _redact(s: str) -> str:  # type: ignore[misc]
        if not isinstance(s, str):
            return s
        for _pat in _SECRET_PATTERNS:
            s = _pat.sub("[REDACTED]", s)
        return s


logger = logging.getLogger(__name__)

_DEFAULT_MIN_TOKENS = 1500
_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_CACHE_MB = 256
_DEFAULT_TARGET_RATIO = 2.0
_COOLDOWN_SECONDS = 30.0
_MAX_QUERY_CHARS = 600
_CACHE_ID_LEN = 32
_OK_STATUSES = ("", "ok", "success")
_FALLBACK_QUERY = (
    "Preserve the facts, paths, identifiers, errors, and results in this tool "
    "output that are needed to continue the task."
)
_QUERY_ARG_KEYS = ("query", "pattern", "command", "q", "search", "regex", "url")
_PATH_ARG_KEYS = ("file_path", "path", "file", "filename", "directory")

# Kept in sync with cache._CACHE_SUBDIR; present in host and container paths.
_CACHE_PATH_MARKER = "cache/compresr/tool-output"

# Tool → key holding the plain-text payload inside a JSON envelope.
_UNWRAPPABLE_JSON_TOOLS: Dict[str, str] = {
    "read_file": "content",
    "terminal": "output",
    "execute_code": "output",
    "search_files": "matches_text",
}

# Tools whose unwrapped payload is line-numbered ("N|code"); cached de-numbered
# so a recovery read_file re-adds exactly one clean gutter.
_NUMBERED_JSON_TOOLS = {"read_file"}
_LINE_GUTTER_RE = re.compile(r"^\d+\|")


def _max_recoverable_line_length() -> int:
    try:
        from tools.tool_output_limits import get_max_line_length

        return int(get_max_line_length())
    except Exception:
        return 2000


def _has_unrecoverable_long_line(text: str) -> bool:
    """Lines longer than the recovery tools' per-line cap can't be recovered
    byte-exact, so such outputs are left uncompressed."""
    limit = _max_recoverable_line_length()
    return any(len(ln) > limit for ln in text.split("\n"))


def _strip_line_gutter(text: str) -> str:
    return "\n".join(_LINE_GUTTER_RE.sub("", ln) for ln in text.split("\n"))


def _is_fully_guttered(text: str) -> bool:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return bool(lines) and all(_LINE_GUTTER_RE.match(ln) for ln in lines)


def _try_unwrap_json_tool_result(
    tool_name: str, result: str
) -> Tuple[Optional[str], Optional[Callable[[str], str]]]:
    key = _UNWRAPPABLE_JSON_TOOLS.get(tool_name)
    if key is None:
        return None, None
    if not result.lstrip().startswith("{"):
        return None, None
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, ValueError):
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    inner = parsed.get(key)
    if not isinstance(inner, str) or not inner.strip():
        return None, None

    def splice(new_inner: str) -> str:
        out = dict(parsed)
        out[key] = new_inner
        return json.dumps(out, ensure_ascii=False)

    return inner, splice


class ToolOutputCompressor:
    """Holds config + the SDK client and implements the transform hook."""

    def __init__(self) -> None:
        cfg = read_config_block()

        self.api_key = sanitize_secret(resolve_api_key(None) or "", "COMPRESR_API_KEY")
        self.base_url = resolve_base_url(opt(cfg, "COMPRESR_BASE_URL", "base_url", None))
        self.enabled = as_bool(opt(cfg, "COMPRESR_TOOL_OUTPUT_ENABLED", "tool_output_enabled", ""))
        self.model = str(
            opt(cfg, "COMPRESR_TOOL_OUTPUT_MODEL", "tool_output_model", DEFAULT_TOOL_OUTPUT_MODEL)
        )
        self.min_tokens = as_int(
            opt(
                cfg,
                "COMPRESR_TOOL_OUTPUT_MIN_TOKENS",
                "tool_output_min_tokens",
                _DEFAULT_MIN_TOKENS,
            ),
            _DEFAULT_MIN_TOKENS,
        )
        self.timeout = max(
            1,
            as_int(
                opt(cfg, "COMPRESR_TOOL_OUTPUT_TIMEOUT", "tool_output_timeout", _DEFAULT_TIMEOUT),
                _DEFAULT_TIMEOUT,
            ),
        )
        self.max_cache_mb = as_int(
            opt(
                cfg,
                "COMPRESR_TOOL_OUTPUT_MAX_CACHE_MB",
                "tool_output_max_cache_mb",
                _DEFAULT_MAX_CACHE_MB,
            ),
            _DEFAULT_MAX_CACHE_MB,
        )
        self.target_ratio: float = cast(
            float,
            as_float(
                opt(
                    cfg,
                    "COMPRESR_TOOL_OUTPUT_TARGET_RATIO",
                    "tool_output_target_ratio",
                    _DEFAULT_TARGET_RATIO,
                ),
                _DEFAULT_TARGET_RATIO,
            ),
        )

        # Built eagerly (when a key exists) so no construction race is possible
        # once Hermes fans out threads.
        self._client: Any = self._build_client() if self.api_key else None
        self.calls = 0
        self.errors = 0
        self.tokens_in = 0
        self.tokens_saved = 0
        self.recoveries = 0
        self._cooldown_until = 0.0

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.api_key)

    def _build_client(self) -> Any:
        from compresr import CompressionClient

        return CompressionClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @staticmethod
    def _derive_query(tool_name: str, args: Any) -> str:
        """Reconstruct the tool call's intent as a Compresr query from its args."""
        if isinstance(args, dict):
            for k in _QUERY_ARG_KEYS:
                v = args.get(k)
                if isinstance(v, str) and v.strip():
                    return f"{tool_name}: {v.strip()}"[:_MAX_QUERY_CHARS]
            for k in _PATH_ARG_KEYS:
                v = args.get(k)
                if isinstance(v, str) and v.strip():
                    return f"Relevant content of {v.strip()} for the current task"[
                        :_MAX_QUERY_CHARS
                    ]
        if tool_name:
            return f"Relevant output of the {tool_name} call for the current task"
        return _FALLBACK_QUERY

    @staticmethod
    def _is_recovery_read(args: Any) -> bool:
        """True if *args* points at a file under the compresr cache — recovery
        reads must come back verbatim, never re-compressed."""
        if not isinstance(args, dict):
            return False
        try:
            cache_root = str(cache.get_cache_root().resolve())
        except Exception:
            cache_root = ""
        for v in args.values():
            if not isinstance(v, str) or not v:
                continue
            if _CACHE_PATH_MARKER in v:
                return True
            if cache_root:
                try:
                    resolved = str(Path(v).resolve())
                    if resolved == cache_root or (
                        os.path.commonpath([resolved, cache_root]) == cache_root
                    ):
                        return True
                except (OSError, ValueError):
                    pass
        return False

    @staticmethod
    def _cache_id(content: str) -> str:
        # Content-addressed: distinct outputs can't collide, identical ones dedupe.
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:_CACHE_ID_LEN]

    def on_transform_tool_result(
        self,
        tool_name: str = "",
        args: Any = None,
        result: Any = None,
        task_id: str = "",
        tool_call_id: str = "",
        status: str = "",
        **_: Any,
    ) -> Optional[str]:
        """Return a compressed replacement string, or None to leave unchanged."""
        if not self.active or not isinstance(result, str):
            return None
        if status and status not in _OK_STATUSES:
            return None
        if FOOTER_MARKER in result:
            return None
        if self._is_recovery_read(args):
            self.recoveries += 1
            return None
        if count_tokens(result) < self.min_tokens:
            return None
        now = time.monotonic()
        if now < self._cooldown_until:
            return None

        # Redact before anything leaves the process to a third-party API.
        query = _redact(self._derive_query(tool_name, args))

        inner_text, splice = _try_unwrap_json_tool_result(tool_name, result)
        compress_target = inner_text if inner_text is not None else result
        if inner_text is not None and count_tokens(inner_text) < self.min_tokens:
            return None

        cache_content = compress_target
        if (
            inner_text is not None
            and tool_name in _NUMBERED_JSON_TOOLS
            and _is_fully_guttered(inner_text)
        ):
            cache_content = _strip_line_gutter(inner_text)

        if _has_unrecoverable_long_line(cache_content):
            return None

        # Durable on-disk artefact — redact it too; content-address post-redaction.
        compress_target = _redact(compress_target)
        cache_content = _redact(cache_content)

        cache_id = self._cache_id(compress_target)
        try:
            out, info = compress_with_recovery(
                query=query,
                content=compress_target,
                cache_content=cache_content,
                tool_name=tool_name,
                cache_id=cache_id,
                client=self._client,
                model=self.model,
                task_id=task_id,
                max_cache_mb=self.max_cache_mb,
                target_ratio=self.target_ratio,
            )
        except Exception as e:
            self.errors += 1
            self._cooldown_until = time.monotonic() + _COOLDOWN_SECONDS
            logger.warning("compresr: tool-output hook error (%s)", e)
            return None

        if info.get("error"):
            # Genuine failures arm the cooldown; a benign "no net win" carries
            # skipped_reason instead so it can't suppress later outputs.
            self.errors += 1
            self._cooldown_until = time.monotonic() + _COOLDOWN_SECONDS
        if not info.get("called_api"):
            return None
        if not info.get("shortened"):
            return None
        saved = max(0, info.get("base_tokens", 0) - info.get("out_tokens", 0))
        self.calls += 1
        self.tokens_in += info.get("base_tokens", 0)
        self.tokens_saved += saved
        logger.info(
            "compresr: %s %d→%d tokens (saved %d, unwrapped=%s, cache=%s)",
            tool_name,
            info.get("base_tokens", 0),
            info.get("out_tokens", 0),
            saved,
            splice is not None,
            info.get("cache_path", "?"),
        )
        if splice is not None:
            return splice(out)
        return out

    def get_status(self) -> Dict[str, Any]:
        return {
            "plugin": "compresr-tool-output",
            "active": self.active,
            "model": self.model,
            "min_tokens": self.min_tokens,
            "max_cache_mb": self.max_cache_mb,
            "calls": self.calls,
            "errors": self.errors,
            "tokens_in": self.tokens_in,
            "tokens_saved": self.tokens_saved,
            "recoveries": self.recoveries,
        }


__all__ = ["ToolOutputCompressor"]
