"""Proxy-startup initializer + registry wiring for the Compresr guardrail."""

from typing import TYPE_CHECKING, Any, Optional

from .guardrail import CompresrGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


COMPRESR_GUARDRAIL_ID = "compresr"


def _get_config_value(
    litellm_params: Any, optional_params: Optional[Any], attribute_name: str
) -> Any:
    if optional_params is not None:
        value = getattr(optional_params, attribute_name, None)
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


def initialize_guardrail(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
) -> "CompresrGuardrail":
    import litellm

    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Compresr guardrail name is required")

    optional_params = getattr(litellm_params, "optional_params", None)

    _compresr_callback = CompresrGuardrail(
        guardrail_name=guardrail_name,
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        compression_model_name=_get_config_value(
            litellm_params, optional_params, "compression_model_name"
        ),
        target_compression_ratio=_get_config_value(
            litellm_params, optional_params, "target_compression_ratio"
        ),
        coarse=_get_config_value(litellm_params, optional_params, "coarse"),
        min_chars_to_compress=_get_config_value(
            litellm_params, optional_params, "min_chars_to_compress"
        ),
        compress_tool_outputs=_get_config_value(
            litellm_params, optional_params, "compress_tool_outputs"
        ),
        compress_system=_get_config_value(litellm_params, optional_params, "compress_system"),
        compress_history=_get_config_value(litellm_params, optional_params, "compress_history"),
        compress_last_user=_get_config_value(litellm_params, optional_params, "compress_last_user"),
        fail_closed=_get_config_value(litellm_params, optional_params, "fail_closed"),
        timeout=_get_config_value(litellm_params, optional_params, "timeout"),
        target_ratio_by_role=_get_config_value(
            litellm_params, optional_params, "target_ratio_by_role"
        ),
        cache_ttl=_get_config_value(litellm_params, optional_params, "cache_ttl"),
    )
    litellm.logging_callback_manager.add_litellm_callback(_compresr_callback)

    return _compresr_callback


guardrail_initializer_registry = {
    COMPRESR_GUARDRAIL_ID: initialize_guardrail,
}


guardrail_class_registry = {
    COMPRESR_GUARDRAIL_ID: CompresrGuardrail,
}
