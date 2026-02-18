"""
Compresr Services

Usage:
    from compresr import CompressionClient, QSCompressionClient

    # Agnostic compression (no question needed)
    client = CompressionClient(api_key="cmp_...")
    result = client.compress(context="Your long context...")

    # Question-specific compression (requires question)
    qs_client = QSCompressionClient(api_key="cmp_...")
    result = qs_client.compress(context="Your context...", question="What is...?")
"""

from .compression_agnostic import CompressionClient
from .compression_qs import QSCompressionClient

__all__ = ["CompressionClient", "QSCompressionClient"]
