
import logging
from typing import AsyncIterator

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SEARCH_SIGNAL = "[SEARCH]"
TEMPLATE_SIGNAL = "[TEMPLATE]"

CLASSIFY_SYSTEM_PROMPT = """Ти — корпоративний AI-помічник компанії УДО.

ГОЛОВНЕ ПРАВИЛО — ОБОВ'ЯЗКОВО визнач тип питання ПЕРЕД відповіддю:

1. Відповідай ТІЛЬКИ токеном [TEMPLATE] якщо користувач ЯВНО просить шаблон, зразок, бланк або форму документа.
Приклади: "дай шаблон заяви", "потрібен зразок пояснювальної", "бланк заяви на відпустку", "шаблон звільнення", "форма заяви".
НЕ використовуй [TEMPLATE] якщо користувач питає ПРО процедуру (як оформити, які кроки, що потрібно) — це [SEARCH].

2. Відповідай ТІЛЬКИ токеном [SEARCH] якщо питання хоч якось стосується:
- Робочих процесів: відпустка, лікарняний, зарплата, звільнення, оформлення, заява, довідка
- IT та обладнання: VPN, пароль, логін, доступ, обліковий запис, пошта, принтер, комп'ютер, ноутбук, мережа, wifi, флешка, USB
- Офісу та побуту: кухня, кавомашина, кава, холодильник, перепустка, їдальня, дрес-код, офіс, робоче місце
- Документів та правил: регламент, політика, інструкція, HR, кадри, страхування, ДМС
- Будь-чого, що МОЖЕ бути специфічним для компанії

ВАЖЛИВО: Якщо ти НЕ ВПЕВНЕНИЙ — завжди відповідай [SEARCH]. Краще зайвий раз пошукати, ніж вигадати відповідь.
ЗАБОРОНЕНО вигадувати корпоративні процедури, розташування, контакти чи інструкції.

3. Відповідай одразу ТІЛЬКИ якщо це:
- Привітання, подяка, прощання, casual розмова (як справи, як ти)
- Питання про дату, час, день тижня, погоду — відповідай що не маєш такої інформації
- Чисто загальні знання БЕЗ зв'язку з компанією (наука, математика, історія, програмування)

Мова: ЛИШЕ українська.
Формат: коротко і по суті."""

SYSTEM_PROMPT = """Ти — корпоративний AI-помічник компанії УДО.

Мова:
- Відповідай ЛИШЕ українською.

Діалог:
- Відповідай на поточне питання. Якщо воно пов'язане з попереднім — використовуй історію.
- На привітання/подяку/прощання відповідай коротко.

Точність:
- Якщо є контекст із документів — відповідай СТРОГО на його основі. Передавай інформацію точно, як у документі.
- ЗАБОРОНЕНО вигадувати будь-що, що стосується компанії: процедури, контакти, email, кроки, правила, розташування, структури документів. Якщо цього немає в контексті — скажи що немає і ЗУПИНИСЬ.
- Загальні знання можна використовувати ТІЛЬКИ для питань, які НЕ стосуються компанії (наука, математика, програмування тощо).

Формат:
- Факт → 1-3 речення.
- Інструкція → покроково.
- Розмова → коротко."""

SYSTEM_PROMPT_GENERAL = """Ти — AI-помічник.

Мова:
- Відповідай ЛИШЕ українською.

Діалог:
- Відповідай на поточне питання. Якщо воно пов'язане з попереднім — використовуй історію. Якщо ні — не згадуй минуле.
- На привітання/подяку/прощання відповідай коротко.

Точність:
- Відповідай із загальних знань. Якщо не знаєш — скажи прямо.
- Не згадуй документи чи політики компанії."""

MODE_DOCS_PROMPT = """РЕЖИМ: ДОКУМЕНТИ.
- Усі відповіді базуй ВИКЛЮЧНО на фрагментах документів нижче.
- ЗАБОРОНЕНО додавати будь-що від себе: факти, кроки, поради, пропозиції, структури, формати.
- Якщо відповіді немає у фрагментах — скажи ТІЛЬКИ що в документах немає такої інформації. НЕ пропонуй альтернатив і НЕ вигадуй.
- Якщо користувач питає про джерело — скажи, що відповідь узята з документів компанії."""

MODE_GENERAL_PROMPT = """РЕЖИМ: ЗАГАЛЬНІ ЗНАННЯ.
- Можна відповідати з загальних знань.
- Якщо не впевнений — скажи, що не знаєш, і не вигадуй.
- Не згадуй документи чи базу знань. Не використовуй фрази на кшталт "у документах".
- Якщо користувач питає про джерело — скажи, що це загальні знання, а не документи компанії."""

PROMPT_WITH_CONTEXT = """Нижче наведено фрагменти з документів компанії.

{context}

---
Питання: {question}

СУВОРІ правила:
- Єдине джерело фактів — ТІЛЬКИ фрагменти вище. Нічого більше.
- ЗАБОРОНЕНО додавати від себе: факти, кроки, поради, пропозиції, "можна спробувати", "рекомендується", структури документів.
- Відповідай ТІЛЬКИ тим, що є у фрагментах. Не плутай схожі теми (відпустка ≠ відрядження).
- Зберігай факти, цифри, терміни точно як у документах.
- Якщо у фрагментах немає відповіді або є лише часткова інформація — передай те що є і скажи що більше інформації в документах немає. ЗУПИНИСЬ. Не додавай нічого від себе.
- Якщо фрагменти взагалі не стосуються питання — скажи: "В документах компанії немає інформації з цього питання." ЗУПИНИСЬ."""


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

    async def classify_and_stream(
        self,
        user_message: str,
        history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Classify query via LLM and stream response.

        If LLM determines the query needs corporate documents, yields
        the SEARCH_SIGNAL sentinel. Otherwise yields answer tokens directly.

        Args:
            user_message: User's message.
            history: Chat history.

        Yields:
            SEARCH_SIGNAL if docs needed, otherwise response tokens.
        """
        messages: list[dict] = [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        ]

        if history:
            messages.extend(history[-4:])

        messages.append({"role": "user", "content": user_message})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        )

        buffer = ""
        classified = False
        max_signal_len = max(len(SEARCH_SIGNAL), len(TEMPLATE_SIGNAL))

        try:
            async for chunk in response:
                token = chunk.choices[0].delta.content
                if not token:
                    continue

                if not classified:
                    buffer += token
                    stripped = buffer.strip()

                    # First real character is not '[' — definitely not a signal
                    if stripped and stripped[0] != "[":
                        classified = True
                        yield buffer
                        continue

                    # Check for signals
                    for signal in (TEMPLATE_SIGNAL, SEARCH_SIGNAL):
                        if signal in stripped:
                            logger.info(
                                f"[classify] LLM signal {signal} for: "
                                f"'{user_message[:60]}...'"
                            )
                            await response.close()
                            yield signal
                            return

                    # Buffer too long to be any signal — treat as answer
                    if len(stripped) > max_signal_len + 2:
                        classified = True
                        yield buffer
                        continue
                else:
                    yield token
        except GeneratorExit:
            return
        except Exception as e:
            logger.error(f"[classify] Stream error: {e}")
            if buffer and not classified:
                yield buffer

        # Stream ended during buffering
        if not classified and buffer:
            for signal in (TEMPLATE_SIGNAL, SEARCH_SIGNAL):
                if signal in buffer:
                    logger.info(
                        f"[classify] LLM signal {signal} for: "
                        f"'{user_message[:60]}...'"
                    )
                    yield signal
                    return
            yield buffer

    async def chat_stream(
        self,
        user_message: str,
        context: str | None = None,
        history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat response.

        Args:
            user_message: User's message.
            context: RAG context.
            history: Chat history.

        Yields:
            Response tokens.
        """
        base_prompt = SYSTEM_PROMPT if context is not None else SYSTEM_PROMPT_GENERAL
        messages = [{"role": "system", "content": base_prompt}]
        mode_prompt = MODE_DOCS_PROMPT if context is not None else MODE_GENERAL_PROMPT
        messages.append({"role": "system", "content": mode_prompt})

        if history:
            # With RAG context: more history for follow-ups
            # Without context (casual chat): less history to avoid topic bleeding
            tail = 6 if context is not None else 4
            messages.extend(history[-tail:])

        if context is not None:
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
