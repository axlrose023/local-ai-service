"""LLM protocol for dependency injection."""
from typing import Protocol, AsyncIterator, runtime_checkable


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM client."""

    async def chat_stream(
        self,
        user_message: str,
        context: str | None = None,
        history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat response from LLM.

        Args:
            user_message: User's message.
            context: RAG context (optional). None means no docs.
            history: Chat history (optional).

        Yields:
            Response tokens.
        """
        ...
