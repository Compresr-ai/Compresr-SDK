"""Hermes plugin shim for Compresr.

All logic lives in the ``compresr`` PyPI package
(``compresr.integrations.hermes``); this directory only makes it installable
via ``hermes plugins install Compresr-ai/Compresr-SDK/hermes-plugin`` and
carries the manifest Hermes uses to prompt for the API key.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "compresr: the `compresr` Python package is not installed in Hermes's "
    "environment. Install it with `pip install compresr` (same interpreter "
    "that runs Hermes), then restart Hermes."
)


def register(ctx: Any) -> None:
    try:
        from compresr.integrations.hermes.plugin import register as _register
    except ImportError:
        logger.error(_INSTALL_HINT)
        return
    _register(ctx)
