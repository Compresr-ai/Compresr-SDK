"""Unit tests for CompressionClient.compress_tool_output(_async)."""

from unittest.mock import AsyncMock, patch

import pytest

from compresr import CompressionClient
from compresr.config import ENDPOINTS
from compresr.exceptions import ValidationError

API_KEY = "cmp_" + "a" * 32

BACKEND_RESPONSE = {
    "success": True,
    "message": None,
    "data": {
        "compressed_output": "short",
        "original_tokens": 100,
        "compressed_tokens": 20,
        "compression_ratio": 5.0,
        "tool_name": "grep",
        "duration_ms": 42,
    },
}


@pytest.fixture
def client():
    return CompressionClient(api_key=API_KEY)


class TestCompressToolOutput:
    def test_posts_to_tool_output_endpoint(self, client):
        with patch.object(client, "post", return_value=BACKEND_RESPONSE) as post:
            resp = client.compress_tool_output(
                tool_output="long output", tool_name="grep", query="find x"
            )
        endpoint, payload = post.call_args.args
        assert endpoint == ENDPOINTS.COMPRESS_TOOL_OUTPUT
        assert payload["tool_output"] == "long output"
        assert payload["tool_name"] == "grep"
        assert payload["query"] == "find x"
        assert payload["compression_model_name"] == "toc_latte_v2"
        assert payload["source"] == "sdk:python"
        assert "target_compression_ratio" not in payload
        assert resp.data.compressed_output == "short"

    def test_ratio_and_model_forwarded(self, client):
        with patch.object(client, "post", return_value=BACKEND_RESPONSE) as post:
            client.compress_tool_output(
                tool_output="x",
                tool_name="grep",
                compression_model_name="toc_latte_v1",
                target_compression_ratio=3.0,
            )
        _, payload = post.call_args.args
        assert payload["compression_model_name"] == "toc_latte_v1"
        assert payload["target_compression_ratio"] == 3.0

    def test_list_input_forwarded(self, client):
        with patch.object(client, "post", return_value=BACKEND_RESPONSE) as post:
            client.compress_tool_output(tool_output=["a", "b"], tool_name="grep")
        _, payload = post.call_args.args
        assert payload["tool_output"] == ["a", "b"]

    def test_invalid_input_raises_sdk_validation_error(self, client):
        with pytest.raises(ValidationError):
            client.compress_tool_output(tool_output="x", tool_name="")

    @pytest.mark.asyncio
    async def test_async_invalid_input_raises_sdk_validation_error(self, client):
        with pytest.raises(ValidationError):
            await client.compress_tool_output_async(tool_output="x", tool_name="")

    def test_source_forwarded(self, client):
        with patch.object(client, "post", return_value=BACKEND_RESPONSE) as post:
            client.compress_tool_output(
                tool_output="x", tool_name="grep", source="integration:hermes"
            )
        _, payload = post.call_args.args
        assert payload["source"] == "integration:hermes"

    def test_source_defaults_to_sdk_python(self, client):
        with patch.object(client, "post", return_value=BACKEND_RESPONSE) as post:
            client.compress_tool_output(tool_output="x", tool_name="grep")
        _, payload = post.call_args.args
        assert payload["source"] == "sdk:python"

    @pytest.mark.asyncio
    async def test_async_posts_to_tool_output_endpoint(self, client):
        with patch.object(
            client, "post_async", new=AsyncMock(return_value=BACKEND_RESPONSE)
        ) as post:
            resp = await client.compress_tool_output_async(
                tool_output="long output", tool_name="grep", query="find x"
            )
        endpoint, payload = post.call_args.args
        assert endpoint == ENDPOINTS.COMPRESS_TOOL_OUTPUT
        assert payload["tool_name"] == "grep"
        assert resp.data.compression_ratio == 5.0
