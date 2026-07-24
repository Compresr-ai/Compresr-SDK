#!/usr/bin/env python3
"""Thin delegator to ``compresr.integrations.litellm.shim_installer``.

Kept so existing docs that point at this path still work; new instructions
should use the packaged ``install-compresr-shim`` console script instead.
"""

from __future__ import annotations

import sys


def main() -> int:
    from compresr.integrations.litellm.shim_installer import main as _main

    return _main()


if __name__ == "__main__":
    sys.exit(main())
