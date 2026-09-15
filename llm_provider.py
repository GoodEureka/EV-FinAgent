"""Provider-specific compatibility helpers for OpenAI and DeepSeek APIs."""

from __future__ import annotations

import json
import os
from typing import Any


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


def is_deepseek(provider: str, base_url: str | None = None) -> bool:
    return provider.lower() == "deepseek" or "api.deepseek.com" in (base_url or "").lower()


def resolve_api_key(provider: str, api_key: str | None) -> str | None:
    if api_key:
        return api_key
    if is_deepseek(provider):
        return os.environ.get("DEEPSEEK_API_KEY")
    return os.environ.get("OPENAI_API_KEY")


def resolve_base_url(provider: str, base_url: str | None) -> str | None:
    if base_url:
        return base_url.rstrip("/")
    if is_deepseek(provider):
        return os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL).rstrip("/")
    value = os.environ.get("OPENAI_BASE_URL")
    return value.rstrip("/") if value else None


def resolve_model(provider: str, model: str | None) -> str:
    if model:
        return model
    if is_deepseek(provider):
        return os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_DEFAULT_MODEL)
    return os.environ.get("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)


def structured_response_format(
    provider: str, base_url: str | None, *, name: str, schema: dict[str, Any]
) -> dict[str, Any]:
    """DeepSeek supports JSON Object; OpenAI mode keeps strict JSON Schema."""
    if is_deepseek(provider, base_url):
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def parse_json_object(content: str | None, label: str) -> dict[str, Any]:
    if not content or not content.strip():
        raise ValueError(f"{label} returned empty content")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must return a JSON object")
    return parsed
