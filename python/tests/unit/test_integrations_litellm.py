"""
Compresr × LiteLLM guardrail — unit tests for ``compresr.integrations.litellm``.

The hook is imported from its home in the SDK; all Compresr API access is
mocked — no real network. Skipped entirely when ``litellm`` is not installed
(it's the ``compresr[litellm]`` optional peer dependency).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

pytest.importorskip("litellm")

from fastapi.exceptions import HTTPException  # noqa: E402
from litellm import DualCache  # noqa: E402
from litellm.proxy._types import UserAPIKeyAuth  # noqa: E402

from compresr.exceptions import (  # noqa: E402
    AuthenticationError,
    CompresrError,
    ValidationError,
)
from compresr.integrations.litellm import (  # noqa: E402
    CompresrGuardrail,
    CompresrGuardrailMissingSecrets,
)

LONG = "x" * 600  # exceeds the 500-char default min_chars_to_compress
SHORT = "hi there"


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------


def _single_response(text: str = "[COMPRESSED]") -> SimpleNamespace:
    return SimpleNamespace(
        success=True,
        message=None,
        data=SimpleNamespace(
            compressed_context=text,
            original_tokens=150,
            compressed_tokens=60,
            tokens_saved=90,
            actual_compression_ratio=0.6,
            duration_ms=120,
        ),
    )


def _batch_response(texts) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            results=[
                SimpleNamespace(
                    compressed_context=t,
                    original_tokens=150,
                    compressed_tokens=60,
                    tokens_saved=45,
                    actual_compression_ratio=0.5,
                    duration_ms=70,
                )
                for t in texts
            ],
            total_tokens_saved=45 * len(texts),
            average_compression_ratio=0.5,
            count=len(texts),
        )
    )


def _make_guardrail(**overrides) -> CompresrGuardrail:
    params = dict(
        guardrail_name="compresr-test",
        api_key="cmp_testkey",
        api_base="https://api.compresr.ai",
        default_on=True,
        event_hook="pre_call",
    )
    params.update(overrides)
    return CompresrGuardrail(**params)


@pytest.fixture
def user_api_key_dict():
    return UserAPIKeyAuth()


@pytest.fixture
def dual_cache():
    return DualCache()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("COMPRESR_API_KEY", raising=False)
    with pytest.raises(CompresrGuardrailMissingSecrets, match="Couldn't get Compresr API key"):
        CompresrGuardrail(guardrail_name="compresr-no-key")


def test_get_config_model():
    model = CompresrGuardrail.get_config_model()
    assert model is not None
    assert model.ui_friendly_name() == "Compresr (context compression)"


def test_api_base_routes_to_onprem_host():
    """On-prem: only api_base changes; SDK constructed with that base_url."""
    captured: dict = {}

    def _fake_ctor(api_key=None, base_url=None, timeout=None):
        captured.update(api_key=api_key, base_url=base_url, timeout=timeout)
        return Mock()

    guardrail = _make_guardrail(api_base="http://compresr-onprem:8000")
    with patch("compresr.CompressionClient", side_effect=_fake_ctor):
        guardrail._get_client()

    assert captured["base_url"] == "http://compresr-onprem:8000"
    assert captured["api_key"] == "cmp_testkey"
    assert isinstance(captured["timeout"], int)


# ---------------------------------------------------------------------------
# Compression behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compresses_tool_outputs_query_aware(user_api_key_dict, dual_cache):
    """Default behaviour: only tool/function outputs are compressed, query-aware.
    System, history, assistant, and the query itself are untouched."""
    guardrail = _make_guardrail()
    fake = Mock()
    fake.compress_batch_async = AsyncMock(return_value=_batch_response(["[T1]", "[T2]"]))
    guardrail._client = fake

    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": LONG},
            {"role": "user", "content": LONG},  # prior user history
            {"role": "assistant", "content": "calling tools"},
            {"role": "tool", "tool_call_id": "a", "content": LONG},  # tool out 1
            {"role": "function", "name": "search", "content": LONG},  # tool out 2
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }

    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )

    msgs = result["messages"]
    assert msgs[0]["content"] == LONG  # system untouched
    assert msgs[1]["content"] == LONG  # history untouched (off by default)
    assert msgs[2]["content"] == "calling tools"  # assistant untouched
    assert msgs[3]["content"] == "[T1]"  # tool output compressed
    assert msgs[4]["content"] == "[T2]"  # function output compressed
    assert msgs[5]["content"] == "What is the capital of France?"  # query verbatim

    _, kwargs = fake.compress_batch_async.call_args
    # Both tool outputs have no resolvable tool_call_id → fall back to the
    # last user message; queries is a list matched 1:1 with contexts.
    assert kwargs["queries"] == [
        "What is the capital of France?",
        "What is the capital of France?",
    ]
    assert kwargs["contexts"] == [LONG, LONG]
    stats = result["metadata"]["compresr_stats"]
    assert stats["messages_compressed"] == 2


@pytest.mark.asyncio
async def test_tool_call_intent_used_as_query(user_api_key_dict, dual_cache):
    """Each tool output is compressed with the intent of *its* originating
    tool call (name + arguments), not the user's overall question."""
    guardrail = _make_guardrail()
    fake = Mock()
    fake.compress_batch_async = AsyncMock(return_value=_batch_response(["[W]", "[H]"]))
    guardrail._client = fake

    data = {
        "messages": [
            {"role": "user", "content": "Plan a 3-day trip to Tokyo on a budget"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Tokyo","days":3}',
                        },
                    },
                    {
                        "id": "call_hotels",
                        "type": "function",
                        "function": {
                            "name": "search_hotels",
                            "arguments": '{"city":"Tokyo","max_price":200}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_weather", "content": LONG},
            {"role": "tool", "tool_call_id": "call_hotels", "content": LONG},
        ],
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )

    _, kwargs = fake.compress_batch_async.call_args
    assert kwargs["queries"] == [
        'get_weather: {"city":"Tokyo","days":3}',
        'search_hotels: {"city":"Tokyo","max_price":200}',
    ]
    assert kwargs["contexts"] == [LONG, LONG]


@pytest.mark.asyncio
async def test_tool_intent_falls_back_when_unresolvable(user_api_key_dict, dual_cache):
    """When the tool_call_id can't be linked back to any assistant tool_calls
    entry (malformed history), fall back to the last user message — don't
    hard-fail."""
    guardrail = _make_guardrail()
    fake = Mock()
    fake.compress_async = AsyncMock(return_value=_single_response("[T]"))
    guardrail._client = fake

    data = {
        "messages": [
            {"role": "user", "content": "What's the latest revenue?"},
            {"role": "assistant", "content": "let me check"},  # no tool_calls
            {"role": "tool", "tool_call_id": "orphan_id", "content": LONG},
        ],
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )

    _, kwargs = fake.compress_async.call_args
    assert kwargs["query"] == "What's the latest revenue?"


@pytest.mark.asyncio
async def test_legacy_function_call_intent(user_api_key_dict, dual_cache):
    """The legacy `function_call` shape (single function per assistant turn,
    role=function with `name`) is resolved by function name."""
    guardrail = _make_guardrail()
    fake = Mock()
    fake.compress_async = AsyncMock(return_value=_single_response("[L]"))
    guardrail._client = fake

    data = {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                },
            },
            {"role": "function", "name": "get_weather", "content": LONG},
        ],
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )

    _, kwargs = fake.compress_async.call_args
    assert kwargs["query"] == 'get_weather: {"city":"Paris"}'


@pytest.mark.asyncio
async def test_history_and_system_not_compressed_by_default(user_api_key_dict, dual_cache):
    """With no tool outputs, a system + history + question request has nothing
    to compress by default (history and system are opt-in)."""
    guardrail = _make_guardrail()
    fake = Mock()
    fake.compress_async = AsyncMock(return_value=_single_response())
    fake.compress_batch_async = AsyncMock(return_value=_batch_response(["x"]))
    guardrail._client = fake

    data = {
        "messages": [
            {"role": "system", "content": LONG},
            {"role": "user", "content": LONG},  # prior history
            {"role": "user", "content": "short question?"},
        ],
    }

    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )

    assert result == data
    fake.compress_async.assert_not_called()
    fake.compress_batch_async.assert_not_called()


@pytest.mark.asyncio
async def test_compress_history_opt_in(user_api_key_dict, dual_cache):
    """Operators can opt in to compressing prior user history."""
    guardrail = _make_guardrail(compress_history=True)
    fake = Mock()
    fake.compress_async = AsyncMock(return_value=_single_response("[HIST]"))
    guardrail._client = fake

    data = {
        "messages": [
            {"role": "user", "content": LONG},  # prior history
            {"role": "user", "content": "final question?"},
        ],
    }

    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )

    assert result["messages"][0]["content"] == "[HIST]"
    assert result["messages"][1]["content"] == "final question?"
    _, kwargs = fake.compress_async.call_args
    assert kwargs["query"] == "final question?"


@pytest.mark.asyncio
async def test_compress_system_opt_in(user_api_key_dict, dual_cache):
    """Operators can opt in to compressing the system prompt."""
    guardrail = _make_guardrail(compress_system=True)
    fake = Mock()
    fake.compress_async = AsyncMock(return_value=_single_response("[SYS]"))
    guardrail._client = fake

    data = {
        "messages": [
            {"role": "system", "content": LONG},
            {"role": "user", "content": "Short question?"},
        ],
    }

    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )

    assert result["messages"][0]["content"] == "[SYS]"
    assert result["messages"][1]["content"] == "Short question?"
    _, kwargs = fake.compress_async.call_args
    assert kwargs["context"] == LONG
    assert kwargs["query"] == "Short question?"
    from compresr.integrations.litellm import DEFAULTS

    assert kwargs["compression_model_name"] == DEFAULTS.compression_model


@pytest.mark.asyncio
async def test_short_messages_skipped(user_api_key_dict, dual_cache):
    guardrail = _make_guardrail()
    fake = Mock()
    fake.compress_async = AsyncMock(return_value=_single_response())
    fake.compress_batch_async = AsyncMock(return_value=_batch_response(["x"]))
    guardrail._client = fake

    data = {
        "messages": [
            {"role": "system", "content": SHORT},
            {"role": "user", "content": SHORT},
        ],
    }

    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )

    assert result == data
    fake.compress_async.assert_not_called()
    fake.compress_batch_async.assert_not_called()


@pytest.mark.asyncio
async def test_multipart_text_compressed_image_preserved(user_api_key_dict, dual_cache):
    """Multimodal tool output: text parts are extracted + compressed, the
    compressed text is written back into the first text part, non-text parts
    pass through untouched. The query is still extracted from the multimodal
    last-user message's text parts."""
    guardrail = _make_guardrail()
    fake = Mock()
    fake.compress_batch_async = AsyncMock(return_value=_batch_response(["[MM]", "[H]"]))
    guardrail._client = fake

    long_text = "ignored non-text tool output " * 40
    multimodal_tool = {
        "role": "tool",
        "tool_call_id": "a",
        "content": [
            {"type": "text", "text": long_text},
            {"type": "image_url", "image_url": {"url": "https://x/h.jpg"}},
        ],
    }
    multimodal_query = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "https://x/y.jpg"}},
        ],
    }
    data = {
        "messages": [
            multimodal_tool,  # multipart with text + image -> text compressed
            {"role": "tool", "tool_call_id": "b", "content": LONG},  # plain text
            multimodal_query,  # query: text part extracted, not compressed
        ],
    }

    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )

    assert result["messages"][0]["content"] == [
        {"type": "text", "text": "[MM]"},
        {"type": "image_url", "image_url": {"url": "https://x/h.jpg"}},
    ]
    assert result["messages"][1]["content"] == "[H]"
    assert result["messages"][2] == multimodal_query  # query preserved verbatim

    _, kwargs = fake.compress_batch_async.call_args
    assert kwargs["contexts"] == [long_text, LONG]
    assert kwargs["queries"] == ["Describe this image", "Describe this image"]


@pytest.mark.asyncio
async def test_multipart_with_no_text_is_skipped(user_api_key_dict, dual_cache):
    """Tool output whose multipart content has no text parts is passed through
    untouched (nothing to compress)."""
    guardrail = _make_guardrail()
    fake = Mock()
    fake.compress_async = AsyncMock(return_value=_single_response("x"))
    fake.compress_batch_async = AsyncMock(return_value=_batch_response(["x"]))
    guardrail._client = fake

    image_only_tool = {
        "role": "tool",
        "tool_call_id": "a",
        "content": [{"type": "image_url", "image_url": {"url": "https://x/y.jpg"}}],
    }
    data = {
        "messages": [
            image_only_tool,
            {"role": "user", "content": "What's in the image?"},
        ],
    }

    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )

    assert result == data
    fake.compress_async.assert_not_called()
    fake.compress_batch_async.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_closed", [False, True])
async def test_missing_query_validation_error_always_raised(
    user_api_key_dict, dual_cache, fail_closed
):
    """No agnostic fallback: a validation error is surfaced as HTTPException
    regardless of fail_closed (operator/usage error must be visible)."""
    guardrail = _make_guardrail(fail_closed=fail_closed)
    fake = Mock()
    fake.compress_async = AsyncMock(
        side_effect=ValidationError("query: Field required (it is missing)")
    )
    guardrail._client = fake

    data = {"messages": [{"role": "tool", "tool_call_id": "a", "content": LONG}]}

    with pytest.raises(HTTPException) as exc:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=dual_cache,
            data=data,
            call_type="completion",
        )

    assert exc.value.status_code == 400
    assert "query" in str(exc.value.detail).lower()


@pytest.mark.asyncio
async def test_authentication_error_always_raised(user_api_key_dict, dual_cache):
    guardrail = _make_guardrail(fail_closed=False)
    fake = Mock()
    fake.compress_async = AsyncMock(
        side_effect=AuthenticationError("Authentication failed: bad key")
    )
    guardrail._client = fake

    data = {
        "messages": [
            {"role": "user", "content": "q?"},
            {"role": "tool", "tool_call_id": "a", "content": LONG},  # target
        ]
    }

    with pytest.raises(HTTPException) as exc:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=dual_cache,
            data=data,
            call_type="completion",
        )

    assert exc.value.status_code == 500
    assert "authentication" in str(exc.value.detail).lower()


@pytest.mark.asyncio
async def test_availability_error_fail_open(user_api_key_dict, dual_cache):
    guardrail = _make_guardrail(fail_closed=False)
    fake = Mock()
    fake.compress_async = AsyncMock(side_effect=CompresrError("upstream down"))
    guardrail._client = fake

    original = {
        "messages": [
            {"role": "user", "content": "q?"},
            {"role": "tool", "tool_call_id": "a", "content": LONG},  # target
        ]
    }

    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=original,
        call_type="completion",
    )

    # Original (uncompressed) request forwarded unchanged.
    assert result["messages"][1]["content"] == LONG
    assert "compresr_stats" not in result.get("metadata", {})


@pytest.mark.asyncio
async def test_availability_error_fail_closed_raises(user_api_key_dict, dual_cache):
    guardrail = _make_guardrail(fail_closed=True)
    fake = Mock()
    fake.compress_async = AsyncMock(side_effect=CompresrError("upstream down"))
    guardrail._client = fake

    data = {
        "messages": [
            {"role": "user", "content": "q?"},
            {"role": "tool", "tool_call_id": "a", "content": LONG},  # target
        ]
    }

    with pytest.raises(HTTPException) as exc:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=dual_cache,
            data=data,
            call_type="completion",
        )

    assert exc.value.status_code == 503
    assert "Unavailable" in str(exc.value.detail)


# ---------------------------------------------------------------------------
# Per-request override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_request_metadata_override(user_api_key_dict, dual_cache):
    """Per-request metadata can re-enable system compression and tune ratio,
    overriding the instance defaults (compress_system OFF)."""
    guardrail = _make_guardrail()  # compress_system defaults False
    fake = Mock()
    fake.compress_async = AsyncMock(return_value=_single_response("[SYS]"))
    guardrail._client = fake

    data = {
        "messages": [
            {"role": "system", "content": LONG},
            {"role": "user", "content": "final question?"},
        ],
        "metadata": {
            "guardrail_config": {
                "compress_system": True,
                "target_compression_ratio": 0.9,
            }
        },
    }

    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )

    assert result["messages"][0]["content"] == "[SYS]"
    assert result["messages"][1]["content"] == "final question?"
    _, kwargs = fake.compress_async.call_args
    assert kwargs["context"] == LONG
    assert kwargs["target_compression_ratio"] == 0.9


# ---------------------------------------------------------------------------
# DualCache integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_skips_backend_call(user_api_key_dict, dual_cache):
    """Pre-seeding the DualCache for a content/query/ratio bypasses the
    backend client entirely on the next request."""
    guardrail = _make_guardrail()
    fake = Mock()
    fake.compress_async = AsyncMock(return_value=_single_response("[FRESH]"))
    fake.compress_batch_async = AsyncMock(return_value=_batch_response(["[FRESH]"]))
    guardrail._client = fake

    data = {
        "messages": [
            {"role": "user", "content": "q?"},
            {"role": "tool", "tool_call_id": "a", "content": LONG},
        ]
    }

    # First call: cache miss -> backend hit + cache populated.
    await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data={
            "messages": [
                {"role": "user", "content": "q?"},
                {"role": "tool", "tool_call_id": "a", "content": LONG},
            ]
        },
        call_type="completion",
    )
    assert fake.compress_async.call_count == 1

    # Second call with identical content/query/ratio: cache hit, no extra call.
    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )
    assert fake.compress_async.call_count == 1  # unchanged
    assert result["messages"][1]["content"] == "[FRESH]"


@pytest.mark.asyncio
async def test_cache_keyed_by_query_not_just_content(user_api_key_dict, dual_cache):
    """Different queries against the same content produce different cache
    entries — a hit on one must not satisfy the other."""
    guardrail = _make_guardrail()
    fake = Mock()
    fake.compress_async = AsyncMock(
        side_effect=[_single_response("[Q1]"), _single_response("[Q2]")]
    )
    guardrail._client = fake

    base = lambda q: {  # noqa: E731
        "messages": [
            {"role": "user", "content": q},
            {"role": "tool", "tool_call_id": "a", "content": LONG},
        ]
    }

    r1 = await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=base("question one?"),
        call_type="completion",
    )
    r2 = await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=base("question two?"),
        call_type="completion",
    )
    assert r1["messages"][1]["content"] == "[Q1]"
    assert r2["messages"][1]["content"] == "[Q2]"
    assert fake.compress_async.call_count == 2


# ---------------------------------------------------------------------------
# Per-target ratios
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uniform_ratios_use_single_batch_call(user_api_key_dict, dual_cache):
    """Two tool outputs with the same ratio -> one batch call (one request)."""
    guardrail = _make_guardrail()
    fake = Mock()
    fake.compress_batch_async = AsyncMock(return_value=_batch_response(["[A]", "[B]"]))
    guardrail._client = fake

    data = {
        "messages": [
            {"role": "user", "content": "q?"},
            {"role": "tool", "tool_call_id": "a", "content": LONG},
            {"role": "tool", "tool_call_id": "b", "content": LONG},
        ]
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )
    assert fake.compress_batch_async.call_count == 1


@pytest.mark.asyncio
async def test_mixed_ratios_use_per_target_calls(user_api_key_dict, dual_cache):
    """Different roles with different ratios -> N parallel single calls."""
    guardrail = _make_guardrail(
        compress_system=True,
        target_ratio_by_role={"system": 0.3, "tool": 0.7},
    )
    fake = Mock()
    fake.compress_async = AsyncMock(
        side_effect=[_single_response("[SYS]"), _single_response("[TOOL]")]
    )
    guardrail._client = fake

    data = {
        "messages": [
            {"role": "system", "content": LONG},
            {"role": "user", "content": "q?"},
            {"role": "tool", "tool_call_id": "a", "content": LONG},
        ]
    }

    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )

    assert result["messages"][0]["content"] == "[SYS]"
    assert result["messages"][2]["content"] == "[TOOL]"
    assert fake.compress_async.call_count == 2
    ratios = [c.kwargs["target_compression_ratio"] for c in fake.compress_async.call_args_list]
    assert sorted(ratios) == [0.3, 0.7]


# ---------------------------------------------------------------------------
# Fail-open header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fail_open_adds_suffixed_header(user_api_key_dict, dual_cache):
    """When Compresr is down and fail_closed=False, the request is forwarded
    AND the applied-guardrails header is marked with `:fail_open` so callers
    can distinguish 'didn't fire' from 'fired but failed open'."""
    guardrail = _make_guardrail(fail_closed=False)
    fake = Mock()
    fake.compress_async = AsyncMock(side_effect=CompresrError("upstream down"))
    guardrail._client = fake

    data = {
        "messages": [
            {"role": "user", "content": "q?"},
            {"role": "tool", "tool_call_id": "a", "content": LONG},
        ]
    }

    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=dual_cache,
        data=data,
        call_type="completion",
    )

    applied = result.get("metadata", {}).get("applied_guardrails", [])
    assert applied == ["compresr-test:fail_open"]
    assert result["messages"][1]["content"] == LONG  # original forwarded


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_releases_client_idempotent():
    """`aclose` calls the underlying client's `aclose`/`close` once, then is
    a no-op on subsequent calls."""
    guardrail = _make_guardrail()
    client = Mock()
    client.aclose = AsyncMock()
    guardrail._client = client

    await guardrail.aclose()
    assert client.aclose.await_count == 1
    assert guardrail._client is None

    await guardrail.aclose()  # idempotent
    assert client.aclose.await_count == 1


@pytest.mark.asyncio
async def test_aclose_falls_back_to_sync_close():
    """If the client exposes only sync `close()`, `aclose` calls that."""
    guardrail = _make_guardrail()
    client = Mock(spec=["close"])
    client.close = Mock()
    guardrail._client = client

    await guardrail.aclose()
    assert client.close.call_count == 1
    assert guardrail._client is None


# ---------------------------------------------------------------------------
# Config model (Pydantic) — new fields
# ---------------------------------------------------------------------------


def test_config_model_exposes_new_optional_params():
    """timeout, target_ratio_by_role, cache_ttl are declared on the optional
    params model so they can come from YAML / admin-UI form."""
    from compresr.integrations.litellm.types import (
        CompresrGuardrailConfigModelOptionalParams,
    )

    fields = CompresrGuardrailConfigModelOptionalParams.model_fields
    assert "timeout" in fields
    assert "target_ratio_by_role" in fields
    assert "cache_ttl" in fields
