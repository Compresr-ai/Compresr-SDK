"""Name-based eligibility filter used by tool/middleware integrations."""

from __future__ import annotations

from typing import Callable, Iterable, Optional


def make_filter(
    allow: Optional[Iterable[str]] = None,
    ignore: Optional[Iterable[str]] = None,
) -> Callable[[Optional[str]], bool]:
    """Build a name predicate from allow- or ignore-lists.

    - ``allow``: only names in this set return True.
    - ``ignore``: every name except those in this set returns True.
    - Both ``None``: every name returns True (no-op).

    Passing both raises ``ValueError`` — pick one mode.
    """
    if allow is not None and ignore is not None:
        raise ValueError("Pass allow OR ignore, not both.")

    if allow is not None:
        allowed = frozenset(allow)
        return lambda name: name is not None and name in allowed

    if ignore is not None:
        ignored = frozenset(ignore)
        return lambda name: name is None or name not in ignored

    return lambda _name: True
