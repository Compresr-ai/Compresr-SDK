"""
Compresr Services

Usage:
    from compresr import CompressionClient, SearchClient

    # Token-level compression
    client = CompressionClient(api_key="cmp_...")
    result = client.compress(context="Your long context...")

    # Query-specific compression
    result = client.compress(
        context="Your context...",
        query="What is...?",
        compression_model_name="latte_v1",
    )

    # Query-specific compression with coarse (faster)
    result = client.compress(
        context="Your context...",
        query="What is...?",
        compression_model_name="latte_v1",
        coarse=True,  # paragraph-level (faster)
    )

    # Agentic search
    search_client = SearchClient(api_key="cmp_...")
    result = search_client.search(
        query="What is machine learning?",
        index_name="my-knowledge-base",
    )
"""

from .compression import CompressionClient
from .search import SearchClient

__all__ = [
    "CompressionClient",
    "SearchClient",
]
