"""Install / uninstall the Compresr discovery shim into the active litellm package.

Pre-merge of the upstream LiteLLM PR, the proxy only discovers built-in
guardrails by walking ``litellm/proxy/guardrails/guardrail_hooks/<name>/``
inside the installed litellm package. This module ships the 1-file shim
content (a string constant — no files outside the package distribution) and
exposes idempotent ``install`` / ``uninstall`` / ``is_installed``.

Two ways to trigger ``install``:

- The ``install-compresr-shim`` console script (manual, one command).
- Set ``COMPRESR_AUTO_INSTALL_SHIM=1`` in the environment and import
  ``compresr.integrations.litellm`` — install runs at import time.

The shim contents themselves contain no logic; they only re-export the
registries from this package, so updates ship via ``pip install -U compresr``
without touching the shim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SHIM_CONTENT = '''"""Compresr discovery shim for LiteLLM proxy.

Re-exports the registries owned by ``compresr.integrations.litellm`` so the
proxy can find the guardrail during its hooks-dir walk. All real logic lives
in the compresr package — this file should not be edited by hand. Manage it
via ``install-compresr-shim`` or ``compresr.integrations.litellm.shim_installer``.
"""

from compresr.integrations.litellm import (
    guardrail_class_registry,
    guardrail_initializer_registry,
    initialize_guardrail,
)

__all__ = [
    "guardrail_class_registry",
    "guardrail_initializer_registry",
    "initialize_guardrail",
]
'''


def _hooks_dir() -> Path:
    import litellm.proxy.guardrails.guardrail_hooks as g

    return Path(next(iter(g.__path__)))


def _target_init() -> Path:
    return _hooks_dir() / "compresr" / "__init__.py"


def is_installed() -> bool:
    return _target_init().exists()


def install(*, verbose: bool = True) -> bool:
    """Write the shim into the active litellm install. Returns True on write."""
    target = _target_init()
    if target.exists():
        if verbose:
            print(f"compresr shim already installed at {target}")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SHIM_CONTENT, encoding="utf-8")
    if verbose:
        print(f"installed compresr shim → {target}")
    return True


def uninstall(*, verbose: bool = True) -> bool:
    """Remove the shim from the active litellm install. Returns True on removal."""
    target = _target_init()
    if not target.exists():
        if verbose:
            print(f"no compresr shim found at {target}")
        return False
    target.unlink()
    try:
        target.parent.rmdir()
    except OSError:
        pass
    if verbose:
        print(f"removed compresr shim ← {target}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()
    if args.uninstall:
        uninstall()
    else:
        install()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
