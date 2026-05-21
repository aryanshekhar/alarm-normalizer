import logging
import threading
import time
from collections import deque

from config import settings

from .base import LLMProvider
from .claude_provider import ClaudeProvider

logger = logging.getLogger(__name__)

MAX_LLM_CALLS_PER_HOUR = 10

_call_timestamps: deque = deque()
_cb_lock = threading.Lock()


def _check_circuit_breaker() -> None:
    now = time.time()
    cutoff = now - 3600
    with _cb_lock:
        while _call_timestamps and _call_timestamps[0] < cutoff:
            _call_timestamps.popleft()
        if len(_call_timestamps) >= MAX_LLM_CALLS_PER_HOUR:
            raise RuntimeError(
                f"LLM circuit breaker open: exceeded {MAX_LLM_CALLS_PER_HOUR} "
                "calls in the last hour"
            )
        _call_timestamps.append(now)


def get_llm_provider() -> LLMProvider:
    _check_circuit_breaker()
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
