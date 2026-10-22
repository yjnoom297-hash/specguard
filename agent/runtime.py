"""Runtime policies shared by the SpecGuard graph.

The project runs without an external model, while an Ollama endpoint can be
enabled for the fast routing/planning tier and the quality validation tier.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

import httpx

T = TypeVar("T")


class ToolError(RuntimeError):
    def __init__(self, code: str, tool_name: str, message: str):
        self.code = code
        self.tool_name = tool_name
        self.user_message = f"{tool_name} 연결을 사용할 수 없어 제한된 분석을 진행합니다."
        super().__init__(message)


async def with_retry(
    call: Callable[[], Awaitable[T]], *, tool_name: str,
    max_retries: int = 2, initial_delay: float = 0.05,
) -> T:
    """Retry transient I/O failures only, using exponential backoff."""
    delay = initial_delay
    transient = (asyncio.TimeoutError, httpx.TimeoutException, httpx.NetworkError)
    for attempt in range(max_retries + 1):
        try:
            return await call()
        except transient as exc:
            if attempt >= max_retries:
                raise ToolError("TOOL_TIMEOUT_AFTER_RETRY", tool_name, str(exc)) from exc
            await asyncio.sleep(delay)
            delay *= 3
        except Exception as exc:
            raise ToolError("TOOL_EXECUTION_ERROR", tool_name, str(exc)) from exc
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class ModelDecision:
    value: dict[str, Any]
    model: str
    degraded: bool = False


class TieredModelClient:
    """Small Ollama JSON client with explicit fast/quality model tiers."""

    def __init__(self) -> None:
        self.enabled = os.getenv("SPECGUARD_USE_LLM", "false").lower() == "true"
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.fast_model = os.getenv("SPECGUARD_FAST_MODEL", "qwen3:8b")
        self.quality_model = os.getenv("SPECGUARD_QUALITY_MODEL", "qwen3:8b")

    async def json(self, *, tier: str, system: str, user: str) -> ModelDecision | None:
        if not self.enabled:
            return None
        model = self.fast_model if tier == "fast" else self.quality_model

        async def request() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={"model": model, "stream": False, "format": "json", "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]},
                )
                response.raise_for_status()
                return json.loads(response.json()["message"]["content"])

        try:
            value = await with_retry(request, tool_name=f"ollama:{model}")
            return ModelDecision(value=value, model=model)
        except (ToolError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
