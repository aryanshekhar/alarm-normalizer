import json
import logging
import threading
import time
from collections import deque

from config import settings

from .base import LLMProvider
from .claude_provider import ClaudeProvider

logger = logging.getLogger(__name__)

_call_timestamps: deque = deque()
_cb_lock = threading.Lock()

_FALLBACK_RESPONSE = json.dumps({
    "rca_text": "LLM rate limit reached — manual investigation required.",
    "recommended_action": (
        "Check affected cells and related alarms manually. "
        "AI diagnosis will resume when the rate limit resets (next hour window)."
    ),
    "confidence": "low",
})


class _FallbackProvider(LLMProvider):
    def complete(self, prompt: str, context: dict) -> str:
        return _FALLBACK_RESPONSE


def _rate_limit_check() -> bool:
    """Record a call attempt. Returns True if allowed, False if limit exceeded."""
    limit = settings.max_llm_calls_per_hour
    now = time.time()
    cutoff = now - 3600
    with _cb_lock:
        while _call_timestamps and _call_timestamps[0] < cutoff:
            _call_timestamps.popleft()
        current = len(_call_timestamps)
        if current >= limit:
            logger.warning(
                "LLM rate limit reached: %d/%d calls in last hour — returning fallback",
                current, limit,
            )
            return False
        _call_timestamps.append(now)
        remaining = limit - current - 1
        if remaining <= limit * 0.2:
            logger.warning(
                "LLM rate limit approaching: %d/%d calls used this hour",
                current + 1, limit,
            )
        return True


def get_llm_provider() -> LLMProvider:
    if not _rate_limit_check():
        return _FallbackProvider()
    provider = settings.llm_provider.lower()
    if provider == "claude":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=claude")
        return ClaudeProvider(api_key=settings.anthropic_api_key)
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=settings.openai_api_key)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Supported: claude, openai")


def warn_if_misconfigured() -> None:
    provider = settings.llm_provider.lower()
    if provider == "claude" and not settings.anthropic_api_key:
        logger.warning(
            "LLM misconfigured: LLM_PROVIDER=claude but ANTHROPIC_API_KEY is not set. "
            "AI features will return fallback responses."
        )
    elif provider == "openai" and not settings.openai_api_key:
        logger.warning(
            "LLM misconfigured: LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
            "AI features will return fallback responses."
        )
    elif provider not in ("claude", "openai"):
        logger.warning(
            "LLM misconfigured: unknown LLM_PROVIDER=%r. "
            "AI features will return fallback responses.",
            provider,
        )
    else:
        logger.info("LLM provider ready: %s", provider)
