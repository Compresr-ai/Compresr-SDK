"""
Compresr Python SDK

Compress context to reduce LLM costs.

Quick Start - Agnostic Compression:
    from compresr import CompressionClient, MODELS

    client = CompressionClient(api_key="cmp_...")
    response = client.compress(
        context="Your long context...",
        compression_model_name=MODELS.DEFAULT  # or MODELS.FAST
    )
    print(response.data.compressed_context)

Quick Start - Question-Specific Compression:
    from compresr import QSCompressionClient, MODELS

    client = QSCompressionClient(api_key="cmp_...")
    response = client.compress(
        context="Your long context...",
        question="What is the main conclusion?"  # REQUIRED
    )
    print(response.data.compressed_context)
"""

from .clients import CompressionClient, QSCompressionClient
from .config import MODELS

__version__ = "1.0.22"
__all__ = ["CompressionClient", "QSCompressionClient", "MODELS"]
