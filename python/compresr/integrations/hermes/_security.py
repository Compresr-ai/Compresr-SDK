"""Base-URL / host validation and secret sanitization for the Hermes integration."""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.compresr.ai"

_SECRET_CTL_RE = re.compile(r"[\r\n\x00]")
_NUMERIC_ONLY_HOST_RE = re.compile(r"\A(?:0x[0-9a-fA-F]+|[0-9]+)\Z")

_BLOCKED_METADATA_HOST_NAMES = frozenset({"metadata.google.internal", "metadata.goog", "metadata"})
_BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "169.254.0.0/16",
        "fd00::/8",
        "fe80::/10",
        "100.100.100.200/32",
        "168.63.129.16/32",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",
    )
)
_LOCALHOST_NAMES = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def sanitize_secret(raw: str, label: str) -> str:
    """Strip whitespace; reject CR/LF/NUL. Returns "" on rejection, never the value."""
    if not raw:
        return ""
    stripped = raw.strip()
    if _SECRET_CTL_RE.search(stripped):
        logger.error("compresr: %s contained CR/LF/NUL and was rejected", label)
        return ""
    return stripped


def _resolve_host_ips(host: str) -> Tuple:
    if not host:
        return ()
    ips: list = []
    try:
        ip = ipaddress.ip_address(host)
        mapped = getattr(ip, "ipv4_mapped", None)
        ips.append(mapped if mapped is not None else ip)
        return tuple(ips)
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, socket.herror, UnicodeError, OSError):
        return ()
    for _, _, _, _, sockaddr in infos:
        try:
            ip = ipaddress.ip_address(str(sockaddr[0]).split("%", 1)[0])
        except ValueError:
            continue
        mapped = getattr(ip, "ipv4_mapped", None)
        ips.append(mapped if mapped is not None else ip)
    return tuple(ips)


def _is_blocked_host(host: str) -> bool:
    """Canonical localhost forms are allow-listed before this check runs; any
    other loopback representation (127.0.0.2, octal/hex shorthand) is blocked."""
    h = (host or "").rstrip(".").lower()
    if h in _BLOCKED_METADATA_HOST_NAMES:
        return True
    return any(
        ip.is_loopback or any(ip in net for net in _BLOCKED_NETWORKS) for ip in _resolve_host_ips(h)
    )


def _safe_url_repr(parsed: Any) -> str:
    if parsed is None:
        return "?"
    try:
        return f"{parsed.scheme or '?'}://{parsed.hostname or '?'}"
    except Exception:
        return "?"


def secure_base_url(url: str, default: str = DEFAULT_BASE_URL) -> str:
    """Reject non-HTTPS (except localhost), numeric-shorthand IPv4 literals,
    cloud-metadata hosts, and hosts resolving to private/link-local nets.
    Logs only scheme://host — the raw URL may carry userinfo credentials."""
    try:
        parsed = urlparse(url)
    except Exception:
        parsed = None
    host = (parsed.hostname or "").lower() if parsed else ""
    safe = _safe_url_repr(parsed)

    if parsed and host and _NUMERIC_ONLY_HOST_RE.match(host):
        logger.warning("compresr: refusing numeric-shorthand IPv4 host %s; using %s", safe, default)
        return default
    if parsed and parsed.scheme in ("http", "https") and host in _LOCALHOST_NAMES:
        return url
    if parsed and _is_blocked_host(host):
        logger.warning(
            "compresr: refusing base_url %s (metadata/private host); using %s",
            safe,
            default,
        )
        return default
    if parsed and parsed.scheme == "https":
        return url
    logger.warning(
        "compresr: ignoring insecure base_url %s (must be https); using %s",
        safe,
        default,
    )
    return default


def resolve_base_url(raw: Optional[str]) -> Optional[str]:
    """Validate a user-supplied base URL for the SDK client; None when unset.
    A trailing /api is stripped — SDK endpoint paths already carry it."""
    if raw is None or not str(raw).strip():
        return None
    url = secure_base_url(str(raw).strip().rstrip("/"))
    if url.endswith("/api"):
        url = url[: -len("/api")]
    return url


__all__ = [
    "DEFAULT_BASE_URL",
    "sanitize_secret",
    "secure_base_url",
    "resolve_base_url",
]
