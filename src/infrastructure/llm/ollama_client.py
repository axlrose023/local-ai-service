
from typing import AsyncIterator

from openai import AsyncOpenAI

SYSTEM_PROMPT = """Ти — корпоративний AI-помічник компанії УДО.
Відповідай ТІЛЬКИ українською мовою.
Відповідай на основі контексту — не вигадуй інформацію.
Формат відповіді залежить від питання:
- Прості питання (де, хто, коли) — коротко, 1-2 речення
- Процедурні питання (як зробити, що потрібно) — повна інструкція з усіма кроками"""

PROMPT_WITH_CONTEXT = """Контекст:
{context}

Питання: {question}

Дай відповідь на основі контексту. Якщо питання процедурне — включи всі важливі деталі (терміни, вимоги, кроки)."""


class OllamaClient:
    """LLM client for Ollama (OpenAI-compatible API)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        """Initialize Ollama client.

        Args:
            base_url: Ollama API URL.
            model: Model name.
            max_tokens: Max response tokens.
            temperature: Sampling temperature.
        """
        self._client = AsyncOpenAI(base_url=base_url, api_key="ollama")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def chat_stream(
        self, user_message: str, context: str = "", history: list[dict] | None = None
    ) -> AsyncIterator[str]:
        """Stream chat response.

        Args:
            user_message: User's message.
            context: RAG context.
            history: Chat history.

        Yields:
            Response tokens.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            messages.extend(history[-6:])

        if context:
            prompt = PROMPT_WITH_CONTEXT.format(context=context, question=user_message)
        else:
            prompt = user_message

        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        )

        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
