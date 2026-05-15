from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, context: dict) -> str:
        """Send prompt + context to the LLM and return the text response."""
