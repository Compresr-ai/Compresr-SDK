"""
Performance Tests for SDK

Measures latency, throughput, and performance characteristics.

Run with: pytest tests/performance/ --env=dev -v -s
"""

import statistics
import time

import pytest


@pytest.fixture(scope="module")
def test_contexts():
    """Sample contexts for performance testing."""
    return [
        "Short context for testing.",
        "Medium length context that provides enough tokens for meaningful compression testing and analysis.",
        "Long context with substantial content. " * 50,
        "Technical documentation explaining machine learning concepts and implementations in detail. "
        * 20,
        "Business analysis report with market insights and strategic recommendations. " * 15,
    ]


class TestSingleCompressionPerformance:
    """Test performance of single compression operations."""

    @pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
    def test_compression_latency_by_ratio(self, admin_client, test_contexts, ratio):
        """Measure compression latency at different ratios."""
        context = test_contexts[2]  # Use medium-long context

        start = time.perf_counter()
        response = admin_client.compress(
            context=context,
            compression_model_name="cmprsr_v1",
            target_compression_ratio=ratio,
        )
        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        assert response.success is True
        print(
            f"\n  📊 Ratio {ratio}: {latency_ms:.0f}ms | "
            f"{response.data.original_tokens} → {response.data.compressed_tokens} tokens | "
            f"{response.data.actual_compression_ratio:.2%} compression"
        )

        # Reasonable latency expectations
        assert latency_ms < 10000, f"Latency {latency_ms}ms exceeds 10s threshold"

    @pytest.mark.parametrize("context_idx", [0, 1, 2, 3, 4])
    def test_latency_by_context_size(self, admin_client, test_contexts, context_idx):
        """Measure latency across different context sizes."""
        context = test_contexts[context_idx]

        start = time.perf_counter()
        response = admin_client.compress(context=context, compression_model_name="cmprsr_v1")
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        tokens_per_ms = response.data.compressed_tokens / latency_ms if latency_ms > 0 else 0

        assert response.success is True
        print(
            f"\n  ⚡ Context {context_idx}: {response.data.original_tokens} tokens | "
            f"{latency_ms:.0f}ms | {tokens_per_ms:.2f} tokens/ms"
        )

    def test_repeated_compressions_consistency(self, admin_client):
        """Test latency consistency across multiple runs."""
        context = "Repeated compression test to measure latency variance and consistency."
        latencies = []

        for i in range(5):
            start = time.perf_counter()
            response = admin_client.compress(context=context, compression_model_name="cmprsr_v1")
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            assert response.success is True

        avg_latency = statistics.mean(latencies)
        std_dev = statistics.stdev(latencies)

        print(
            f"\n  📈 Avg: {avg_latency:.0f}ms | StdDev: {std_dev:.0f}ms | "
            f"Min: {min(latencies):.0f}ms | Max: {max(latencies):.0f}ms"
        )

        # Latency should be reasonably consistent (coefficient of variation < 50%)
        cv = (std_dev / avg_latency) * 100 if avg_latency > 0 else 0
        assert cv < 50, f"Latency variance too high: {cv:.1f}%"


class TestBatchCompressionPerformance:
    """Test performance of batch compression operations."""

    @pytest.mark.parametrize("batch_size", [3, 5, 10, 20])
    def test_batch_throughput(self, admin_client, batch_size):
        """Measure batch compression throughput."""
        contexts = [f"Batch context number {i} for throughput testing." for i in range(batch_size)]

        start = time.perf_counter()
        response = admin_client.compress_batch(
            contexts=contexts, compression_model_name="cmprsr_v1"
        )
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        contexts_per_sec = (batch_size / latency_ms) * 1000 if latency_ms > 0 else 0

        assert response.success is True
        print(
            f"\n  🚀 Batch {batch_size}: {latency_ms:.0f}ms | "
            f"{contexts_per_sec:.2f} contexts/sec | "
            f"{response.data.total_tokens_saved} tokens saved"
        )

    def test_batch_vs_sequential_performance(self, admin_client, test_contexts):
        """Compare batch vs sequential compression performance."""
        contexts = test_contexts[:5]

        # Sequential compression
        seq_start = time.perf_counter()
        for ctx in contexts:
            admin_client.compress(context=ctx, compression_model_name="cmprsr_v1")
        seq_end = time.perf_counter()
        seq_latency = (seq_end - seq_start) * 1000

        # Batch compression
        batch_start = time.perf_counter()
        admin_client.compress_batch(contexts=contexts, compression_model_name="cmprsr_v1")
        batch_end = time.perf_counter()
        batch_latency = (batch_end - batch_start) * 1000

        speedup = seq_latency / batch_latency if batch_latency > 0 else 0

        print(
            f"\n  ⚡ Sequential: {seq_latency:.0f}ms | Batch: {batch_latency:.0f}ms | "
            f"Speedup: {speedup:.2f}x"
        )

        # Batch should be faster than sequential
        assert batch_latency < seq_latency, "Batch should be faster than sequential"


class TestStreamingPerformance:
    """Test performance of streaming compression."""

    def test_streaming_latency(self, admin_client):
        """Measure streaming compression latency."""
        context = (
            "Stream performance test with time-to-first-byte and total latency measurement. " * 10
        )

        start = time.perf_counter()
        first_chunk_time = None
        chunk_count = 0

        for chunk in admin_client.compress_stream(
            context=context, compression_model_name="cmprsr_v1"
        ):
            if first_chunk_time is None and chunk.content:
                first_chunk_time = time.perf_counter()
            chunk_count += 1

        end = time.perf_counter()

        total_latency = (end - start) * 1000
        ttfb = ((first_chunk_time - start) * 1000) if first_chunk_time else total_latency

        print(
            f"\n  🌊 TTFB: {ttfb:.0f}ms | Total: {total_latency:.0f}ms | " f"Chunks: {chunk_count}"
        )

        assert ttfb < total_latency, "Time to first byte should be less than total"

    def test_chunk_frequency(self, admin_client):
        """Measure chunk delivery frequency."""
        context = "Analyze streaming chunk delivery patterns and frequency. " * 20

        chunk_times = []

        for chunk in admin_client.compress_stream(
            context=context, compression_model_name="cmprsr_v1"
        ):
            if chunk.content:
                chunk_times.append(time.perf_counter())

        if len(chunk_times) > 1:
            intervals = [
                (chunk_times[i] - chunk_times[i - 1]) * 1000 for i in range(1, len(chunk_times))
            ]
            avg_interval = statistics.mean(intervals) if intervals else 0

            print(f"\n  📦 Chunks: {len(chunk_times)} | " f"Avg interval: {avg_interval:.1f}ms")


class TestTokenEfficiency:
    """Test token compression efficiency."""

    def test_compression_efficiency_by_ratio(self, admin_client, test_contexts):
        """Measure compression efficiency at different ratios."""
        context = test_contexts[3]  # Technical documentation

        results = []
        for ratio in [0.3, 0.5, 0.7]:
            response = admin_client.compress(
                context=context,
                compression_model_name="cmprsr_v1",
                target_compression_ratio=ratio,
            )

            efficiency = (
                response.data.tokens_saved / response.data.original_tokens
                if response.data.original_tokens > 0
                else 0
            )
            results.append(
                {
                    "target": ratio,
                    "actual": response.data.actual_compression_ratio,
                    "efficiency": efficiency,
                    "saved": response.data.tokens_saved,
                }
            )

        print("\n  💾 Compression Efficiency:")
        for r in results:
            print(
                f"    Target {r['target']:.1%}: Actual {r['actual']:.1%} | "
                f"Saved {r['saved']} tokens | Efficiency {r['efficiency']:.1%}"
            )

    def test_average_tokens_per_ms(self, admin_client, test_contexts):
        """Calculate average processing speed in tokens/ms."""
        measurements = []

        for context in test_contexts:
            start = time.perf_counter()
            response = admin_client.compress(context=context, compression_model_name="cmprsr_v1")
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            tokens_per_ms = response.data.compressed_tokens / latency_ms if latency_ms > 0 else 0
            measurements.append(tokens_per_ms)

        avg_throughput = statistics.mean(measurements)
        print(f"\n  ⚡ Avg throughput: {avg_throughput:.2f} tokens/ms")

        assert avg_throughput > 0, "Should process tokens efficiently"
