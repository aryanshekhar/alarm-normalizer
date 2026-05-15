import json

import anthropic

from .base import LLMProvider

_MODEL = "claude-opus-4-7"


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt: str, context: dict) -> str:
        system = (
            "You are an AIOps assistant. Use the provided context to answer accurately.\n\n"
            f"Context:\n{json.dumps(context, indent=2)}"
        )
        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        return "\n".join(text_blocks)
