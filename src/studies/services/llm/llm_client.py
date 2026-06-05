# studies/services/nlp/llm_client.py

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class LLMProvider:
    value: str
    label: str
    api_key_env: str
    default_model_env: str
    default_model: str
    base_url: str | None = None


LLM_PROVIDERS = {
    "openai": LLMProvider(
        value="openai",
        label="OpenAI",
        api_key_env="OPENAI_API_KEY",
        default_model_env="OPENAI_MODEL",
        default_model="gpt-4.1-mini",
    ),
    "groq": LLMProvider(
        value="groq",
        label="Groq",
        api_key_env="GROQ_API_KEY",
        default_model_env="GROQ_MODEL",
        default_model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
    ),
    "gemini": LLMProvider(
        value="gemini",
        label="Gemini",
        api_key_env="GEMINI_API_KEY",
        default_model_env="GEMINI_MODEL",
        default_model="gemini-2.0-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "openrouter": LLMProvider(
        value="openrouter",
        label="OpenRouter",
        api_key_env="OPENROUTER_API_KEY",
        default_model_env="OPENROUTER_MODEL",
        default_model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
    ),
}


def get_available_llm_providers() -> list[dict[str, str]]:
    return [
        {
            "value": provider.value,
            "label": provider.label,
            "default_model": get_llm_model(provider.value),
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

    if llm_provider.base_url:
        return OpenAI(
            api_key=api_key,
            base_url=llm_provider.base_url,
        )

    return OpenAI(api_key=api_key)


def connect_to_llm(provider: str, model: str | None = None) -> tuple[OpenAI, str]:
    client = get_llm_client(provider)
    selected_model = get_llm_model(provider, model)

    return client, selected_model