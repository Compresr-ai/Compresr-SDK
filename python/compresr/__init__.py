"""
Compresr Python SDK

Compress context to reduce LLM costs.

Two client types:
1. CompressionClient - Token-level compression, customizable compression_ratio
2. SearchClient - Agentic search over pre-indexed knowledge bases

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

Quick Start - Agentic Search:
    from compresr import SearchClient, MODELS

    client = SearchClient(api_key="cmp_...")

    response = client.search(
        query="What is machine learning?",
        index_name="my-knowledge-base",
    )
    print(response.data.chunks)  # List[str]
"""

from .clients import CompressionClient, SearchClient
from .config import MODELS

__version__ = "2.1.0"
__all__ = [
    "CompressionClient",
    "SearchClient",
    "MODELS",
]
