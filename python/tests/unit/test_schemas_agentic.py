"""Unit tests for the tool-output (agentic) schemas."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from compresr.schemas import (
    CompressToolOutputRequest,
    CompressToolOutputResponse,
    CompressToolOutputResult,
)


class TestCompressToolOutputRequest:
    def test_defaults(self):
        req = CompressToolOutputRequest(tool_output="some output", tool_name="grep")
        assert req.tool_output == "some output"
        assert req.tool_name == "grep"
        assert req.query is None
        assert req.compression_model_name == "toc_latte_v2"
        assert req.target_compression_ratio is None
        assert req.source == "sdk:python"

    def test_wire_format_excludes_unset_optionals(self):
        req = CompressToolOutputRequest(tool_output="x", tool_name="grep")
        wire = req.model_dump(exclude_none=True)
        assert "query" not in wire
        assert "target_compression_ratio" not in wire
        assert wire["compression_model_name"] == "toc_latte_v2"
        assert wire["source"] == "sdk:python"

    def test_wire_format_includes_set_fields(self):
        req = CompressToolOutputRequest(
            tool_output="x",
            tool_name="web_search",
            query="find the answer",
            target_compression_ratio=5.0,
        )
        wire = req.model_dump(exclude_none=True)
        assert wire["query"] == "find the answer"
        assert wire["target_compression_ratio"] == 5.0

    def test_list_input_accepted(self):
        req = CompressToolOutputRequest(tool_output=["a", "b"], tool_name="grep")
        assert req.tool_output == ["a", "b"]

    def test_tool_name_required(self):
        with pytest.raises(PydanticValidationError):
            CompressToolOutputRequest(tool_output="x")

    def test_empty_tool_name_rejected(self):
        with pytest.raises(PydanticValidationError):
            CompressToolOutputRequest(tool_output="x", tool_name="")

    def test_negative_ratio_rejected(self):
        with pytest.raises(PydanticValidationError):
            CompressToolOutputRequest(
                tool_output="x", tool_name="grep", target_compression_ratio=-1.0
            )

    def test_ratio_above_backend_cap_rejected(self):
        with pytest.raises(PydanticValidationError):
            CompressToolOutputRequest(
                tool_output="x", tool_name="grep", target_compression_ratio=201.0
            )

    def test_empty_query_accepted_backend_mirror(self):
        req = CompressToolOutputRequest(tool_output="x", tool_name="grep", query="")
        assert req.query == ""


class TestCompressToolOutputResponse:
    def test_parses_backend_shape(self):
        resp = CompressToolOutputResponse.model_validate(
            {
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
        )
        assert resp.success is True
        assert resp.data is not None
        assert resp.data.compressed_output == "short"
        assert resp.data.original_tokens == 100
        assert resp.data.compression_ratio == 5.0
        assert resp.data.duration_ms == 42

    def test_list_output_mirrors_input_type(self):
        result = CompressToolOutputResult(
            compressed_output=["a", "b"],
            original_tokens=10,
            compressed_tokens=4,
            compression_ratio=2.5,
        )
        assert result.compressed_output == ["a", "b"]
        assert result.tool_name == ""
        assert result.duration_ms == 0

    def test_data_optional_on_error(self):
        resp = CompressToolOutputResponse.model_validate({"success": False, "message": "boom"})
        assert resp.success is False
        assert resp.data is None
