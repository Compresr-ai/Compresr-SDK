"""End-to-end tests for ``CompressionClient`` against the real backend."""

from __future__ import annotations

import pytest

LONG_TEXT = (
    "The Transformer architecture introduced by Vaswani et al. in 2017 in "
    "'Attention Is All You Need' replaced recurrent layers with multi-head "
    "self-attention. This enabled massive parallelization on GPUs and "
    "removed the sequential bottleneck of LSTMs and GRUs. The encoder-decoder "
    "design with stacked attention blocks became the foundation for BERT, "
    "GPT, T5, and most modern large language models. "
) * 6


def test_compress_basic(live_client):
    response = live_client.compress(
        context=LONG_TEXT,
        query="What replaced recurrence in the Transformer?",
        target_compression_ratio=0.5,
    )
    assert response.success is True
    assert response.data is not None
    assert response.data.compressed_tokens <= response.data.original_tokens
    assert response.data.tokens_saved >= 0


@pytest.mark.asyncio
async def test_compress_async(live_client):
    response = await live_client.compress_async(
        context=LONG_TEXT,
        query="What replaced recurrence in the Transformer?",
        target_compression_ratio=0.5,
    )
    assert response.success is True
    assert response.data is not None
    assert response.data.compressed_tokens <= response.data.original_tokens


def test_compress_batch_single_query(live_client):
    response = live_client.compress_batch(
        contexts=[LONG_TEXT, LONG_TEXT, LONG_TEXT],
        queries="What is self-attention?",
        target_compression_ratio=0.5,
    )
    assert response.success is True
    assert response.data is not None
    assert response.data.count == 3
    assert len(response.data.results) == 3


def test_compress_batch_per_context_queries(live_client):
    response = live_client.compress_batch(
        contexts=[LONG_TEXT, LONG_TEXT],
        queries=["What is ML?", "What is GPT?"],
        target_compression_ratio=0.5,
    )
    assert response.success is True
    assert response.data.count == 2


def test_compress_batch_query_mismatch_raises(live_client):
    from compresr.exceptions import ValidationError

    with pytest.raises(ValidationError):
        live_client.compress_batch(
            contexts=[LONG_TEXT, LONG_TEXT],
            queries=["only one query"],
        )
