"""
Compresr Client Classes

Client type:
- CompressionClient: Token-level compression, customizable compression_ratio
"""

from .services.compression import CompressionClient

__all__ = [
    "CompressionClient",
]
