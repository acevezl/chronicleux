import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class LLMProvider:
    value: str
    label: str
    description: str
    api_key_env: str
    default_model_env: str
    default_model: str
    base_url: str | None = None
    response_format: dict[str, Any] | None = None


LLM_PROVIDERS = {
    "openai": LLMProvider(
        value="openai",
        label="OpenAI",
        description=(
            "GPT-4.1 mini is a compact general-purpose model optimized for "
            "instruction following, classification, and structured data extraction."
        ),
        api_key_env="OPENAI_API_KEY",
        default_model_env="OPENAI_MODEL",
        default_model="gpt-4.1-mini",
        response_format={"type": "json_object"},
    ),
    "gemini": LLMProvider(
        value="gemini",
        label="Google",
        description=(
            "Gemini 3.1 Flash-Lite is a low-latency, cost-efficient multimodal "
            "model designed for high-volume classification, data extraction, "
            "and lightweight language-analysis tasks."
        ),
        api_key_env="GEMINI_API_KEY",
        default_model_env="GEMINI_MODEL",
        default_model="gemini-3.1-flash-lite",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        response_format={"type": "json_object"},
    ),
    "anthropic": LLMProvider(
        value="anthropic",
        label="Anthropic",
        description=(
            "Claude Sonnet 4 is a general-purpose model designed for complex "
            "reasoning, nuanced text interpretation, instruction following, "
            "and structured analytical tasks."
        ),
        api_key_env="ANTHROPIC_API_KEY",
        default_model_env="ANTHROPIC_MODEL",
        default_model="claude-haiku-4-5-20251001",
        base_url="https://api.anthropic.com/v1",
        response_format=None,
    ),
    "meta": LLMProvider(
        value="meta",
        label="Meta",
        description=(
            "Llama 3.3 70B Instruct is a multilingual, instruction-tuned "
            "text model designed for dialogue, reasoning, classification, "
            "and general natural-language analysis."
        ),
        api_key_env="OPENROUTER_API_KEY",
        default_model_env="OPENROUTER_META_MODEL",
        default_model="meta-llama/llama-3.3-70b-instruct",
        base_url="https://openrouter.ai/api/v1",
        response_format={"type": "json_object"},
    ),
}


def get_available_llm_providers() -> list[dict[str, str]]:
    return [
        {
            "value": provider.value,
            "label": provider.label,
            "default_model": get_llm_model(provider.value),
            "description": provider.description,
        }
        for provider in LLM_PROVIDERS.values()
    ]


def get_llm_provider(provider: str) -> LLMProvider:
    try:
        return LLM_PROVIDERS[provider]
    except KeyError:
        available_providers = ", ".join(LLM_PROVIDERS.keys())
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Available providers: {available_providers}"
        )


def get_llm_api_key(provider: str) -> str:
    llm_provider = get_llm_provider(provider)
    api_key = os.getenv(llm_provider.api_key_env)

    if not api_key:
        raise RuntimeError(f"{llm_provider.api_key_env} is not set.")

    return api_key


def get_llm_model(provider: str, model: str | None = None) -> str:
    if model:
        return model

    llm_provider = get_llm_provider(provider)
    return os.getenv(llm_provider.default_model_env, llm_provider.default_model)


def get_llm_client(provider: str) -> OpenAI:
    llm_provider = get_llm_provider(provider)
    api_key = get_llm_api_key(provider)

    kwargs = {
        "api_key": api_key,
    }

    if llm_provider.base_url:
        kwargs["base_url"] = llm_provider.base_url

    if llm_provider.base_url == "https://models.github.ai/inference":
        kwargs["default_headers"] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    return OpenAI(**kwargs)


def connect_to_llm(provider: str, model: str | None = None) -> tuple[OpenAI, str]:
    client = get_llm_client(provider)
    selected_model = get_llm_model(provider, model)
    response_format = LLM_PROVIDERS[provider].response_format

    return client, selected_model, response_format