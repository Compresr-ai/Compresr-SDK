"""
Compresr Client Classes

Direct imports of client classes for easy access.
"""

from .services.compression_agnostic import CompressionClient
from .services.compression_qs import QSCompressionClient

__all__ = [
    "CompressionClient",
    "QSCompressionClient",
]
