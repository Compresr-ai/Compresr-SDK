"""
Compresr Client Classes

Two client types:
- CompressionClient: Token-level compression, customizable compression_ratio
- FilterClient: Coarse-grained chunk selection, keeps/drops entire chunks
"""

from .services.compression import CompressionClient
from .services.filter import FilterClient

__all__ = [
    "CompressionClient",
    "FilterClient",
]
