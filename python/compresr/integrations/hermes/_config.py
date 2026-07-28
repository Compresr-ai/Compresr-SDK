"""Config resolution for the Hermes integration: env var > config.yaml > default.

All coercions are tolerant — a typo'd value falls back to the default instead
of raising in a plugin ``__init__`` and silently disabling the feature.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def read_config_block(key: str = "compresr") -> Dict[str, Any]:
    """Best-effort read of a top-level *key* block from Hermes's config.yaml. Never raises."""
    try:
        from hermes_constants import get_hermes_home

        cfg_path = get_hermes_home() / "config.yaml"
    except Exception:
        return {}
    if not cfg_path.exists():
        return {}
    try:
        import yaml

        with open(cfg_path, encoding="utf-8-sig") as f:
            data = yaml.safe_load(f) or {}
        block = data.get(key, {})
        return block if isinstance(block, dict) else {}
    except Exception as e:
        logger.debug("compresr: could not read config block %r: %s", key, e)
        return {}


def opt(cfg: Dict[str, Any], env_key: str, cfg_key: str, default: Any) -> Any:
    val = os.environ.get(env_key)
    if val is not None and val != "":
        return val
    if cfg_key in cfg and cfg[cfg_key] not in (None, ""):
        return cfg[cfg_key]
    return default


def as_bool(v: Any) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on")


def as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        logger.warning("compresr: invalid numeric value %r, using %s", v, default)
        return default


def as_float(v: Any, default: Optional[float]) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        if v not in (None, ""):
            logger.warning("compresr: invalid numeric value %r, using %s", v, default)
        return default


__all__ = ["read_config_block", "opt", "as_bool", "as_int", "as_float"]
