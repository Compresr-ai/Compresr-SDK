"""
Compresr Client Classes

Two client types:
- CompressionClient: Token-level compression, customizable compression_ratio
- SearchClient: Agentic search over pre-indexed knowledge bases
"""

from .services.compression import CompressionClient
from .services.search import SearchClient

__all__ = [
    "CompressionClient",
    "SearchClient",
]
