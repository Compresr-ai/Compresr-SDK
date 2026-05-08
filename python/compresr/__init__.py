"""
Compresr Python SDK

Compress context to reduce LLM costs.

Quick Start - Compression:
    from compresr import CompressionClient, MODELS

    client = CompressionClient(api_key="cmp_...")

    # Agnostic compression (espresso_v1, default)
    response = client.compress(context="Your long context...")
    print(response.data.compressed_context)  # str

    # Query-specific compression (latte_v1)
    response = client.compress(
        context="Your long context...",
        query="What is the main conclusion?",
        compression_model_name="latte_v1",
    )

    # Coarse-grained compression (paragraph-level, faster)
    response = client.compress(
        context="Your long context...",
        query="What is the main conclusion?",
        compression_model_name="latte_v1",
        coarse=True,  # paragraph-level (faster)
    )
"""

from importlib.metadata import PackageNotFoundError, version

from .clients import CompressionClient
from .config import MODELS

try:
    __version__ = version("compresr")
except PackageNotFoundError:
    # Package not installed (e.g., running from source)
    __version__ = "0.0.0-dev"

__all__ = [
    "CompressionClient",
    "MODELS",
]
