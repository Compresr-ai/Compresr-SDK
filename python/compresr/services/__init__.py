"""
Compresr Services

Usage:
    from compresr import CompressionClient

    # CompressionClient - compression service only
    client = CompressionClient(api_key="cmp_...")
"""

from .compression import CompressionClient

__all__ = ["CompressionClient"]
