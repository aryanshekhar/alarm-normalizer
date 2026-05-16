import json

from .base import LLMProvider

_MODEL = "gpt-4o"


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)

    def complete(self, prompt: str, context: dict) -> str:
        system = (
            "You are an AIOps assistant. Use the provided context to answer accurately.\n\n"
            f"Context:\n{json.dumps(context, indent=2)}"
        )
        response = self._client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""
