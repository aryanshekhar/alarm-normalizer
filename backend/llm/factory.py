from config import settings

from .base import LLMProvider
from .claude_provider import ClaudeProvider


def get_llm_provider() -> LLMProvider:
    provider = settings.llm_provider.lower()
    if provider == "claude":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=claude")
        return ClaudeProvider(api_key=settings.anthropic_api_key)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Supported: claude")
