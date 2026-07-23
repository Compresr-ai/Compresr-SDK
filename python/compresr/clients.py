"""Back-compat re-export of public clients.

``compresr.CompressionClient`` is the canonical import path. This module
exists so that ``from compresr.clients import CompressionClient`` keeps
working for callers who pinned that older path.
"""

from .services.compression import CompressionClient

__all__ = ["CompressionClient"]
