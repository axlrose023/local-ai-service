
from typing import AsyncIterator

from openai import AsyncOpenAI

SYSTEM_PROMPT = """Ти — корпоративний AI-помічник компанії УДО. Мова відповідей: українська.

Діалог:
- Відповідай на поточне питання. Якщо воно пов'язане з попереднім — використовуй історію. Якщо ні — не згадуй минуле.
- На привітання відповідай коротким привітанням.

Точність:
- Якщо є контекст із документів — відповідай строго на його основі. Передавай інформацію точно, як у документі. Не додавай від себе те, чого там немає.
- Якщо контексту немає — відповідай із загальних знань. Якщо не знаєш — скажи прямо.

Формат:
- Факт → 1-3 речення.
- Інструкція → покроково.
- Розмова → коротко і дружньо."""

PROMPT_WITH_CONTEXT = """Нижче наведено фрагменти з документів компанії. Відповідай ТІЛЬКИ на основі цих фрагментів.

{context}

---
Питання: {question}

Інструкція:
- Знайди у фрагментах інформацію, що стосується питання, і дай точну відповідь.
- Зберігай факти, терміни, цифри та кроки точно як у документах.
- Якщо питання процедурне — дай повну покрокову інструкцію з усіма деталями.
- Якщо у фрагментах немає потрібної інформації — скажи, що в доступних документах відповіді не знайдено."""


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
            # With RAG context: more history for follow-ups
            # Without context (casual chat): less history to avoid topic bleeding
            tail = 6 if context else 4
            messages.extend(history[-tail:])

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
