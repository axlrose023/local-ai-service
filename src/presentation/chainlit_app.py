
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chainlit as cl

from src.config.settings import settings
from src.container import configure_container, container
from src.core.models.chat import ChatHistory
from src.core.protocols.embedder import EmbedderProtocol
from src.core.services.chat_service import ChatService
from src.core.services.router_service import RouterService

configure_container(settings)


@cl.on_chat_start
async def start():
    """Initialize chat session."""
    cl.user_session.set("history", ChatHistory())
    cl.user_session.set("last_answer_source", None)
    cl.user_session.set("last_doc_sources", [])
    cl.user_session.set("last_doc_context", None)
    cl.user_session.set("last_doc_query", None)

    embedder = container.resolve(EmbedderProtocol)
    embedder.warmup()

    await cl.Message(
        content="Вітаю! Я корпоративний AI-помічник УДО.\n\n"
        "Задайте питання щодо документів компанії, і я постараюся допомогти."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user message."""
    user_input = message.content
    history: ChatHistory | None = cl.user_session.get("history")
    if history is None:
        # Defensive init in case on_chat_start hasn't run yet
        history = ChatHistory()
        cl.user_session.set("history", history)

    chat_service = container.resolve(ChatService)
    router = container.resolve(RouterService)

    def is_source_question(text: str) -> bool:
        t = (text or "").lower()
        patterns = [
            "ти вигадав",
            "ти вигадал",
            "ти выдумал",
            "ти придумал",
            "де це взяв",
            "звідки це",
            "звідки ти це",
            "на чому базується",
            "джерело",
            "з яких документів",
            "це з документів",
            "источник",
            "откуда это",
        ]
        return any(p in t for p in patterns)

    if is_source_question(user_input):
        last_source = cl.user_session.get("last_answer_source")
        last_sources = cl.user_session.get("last_doc_sources") or []
        if last_source == "docs":
            if last_sources:
                sources = ", ".join(last_sources)
                response = (
                    "Відповідь була сформована на основі документів компанії: "
                    f"{sources}."
                )
            else:
                response = (
                    "Я відповідав на основі документів компанії, але в них немає "
                    "прямої відповіді на це питання."
                )
        elif last_source == "general":
            response = (
                "Це була відповідь із загальних знань, а не з документів компанії."
            )
        else:
            response = (
                "Поки що немає попередньої відповіді, на яку можу послатися. "
                "Поставте, будь ласка, конкретне питання."
            )

        await cl.Message(content=response).send()
        history.add_pair(user_input, response)
        return

    if router.is_casual(user_input):
        text = user_input.lower()
        if any(k in text for k in ["дякую", "спасибо"]):
            response = "Будь ласка!"
        elif any(k in text for k in ["привіт", "вітаю", "добрий день", "hello", "hi"]):
            response = "Вітаю!"
        elif any(k in text for k in ["бувай", "до побачення", "пока"]):
            response = "До побачення!"
        else:
            response = "Гаразд."
        await cl.Message(content=response).send()
        history.add_pair(user_input, response)
        cl.user_session.set("last_answer_source", "general")
        return

    carryover_context = cl.user_session.get("last_doc_context")
    carryover_query = cl.user_session.get("last_doc_query")

    msg = cl.Message(content="")
    await msg.send()

    full_response = ""
    search_shown = False
    answer_mode = "general"
    last_sources: list[str] = []
    context_used: str | None = None

    try:
        async for token, search_response, mode, used_context in chat_service.process_message(
            user_input,
            history,
            carryover_context=carryover_context,
            carryover_query=carryover_query,
        ):
            answer_mode = mode
            context_used = used_context
            if search_response and not search_shown:
                async with cl.Step(name="Пошук у базі знань") as step:
                    step.input = user_input
                    if search_response.results:
                        step.output = f"Знайдено {len(search_response.results)} релевантних фрагментів"
                    else:
                        step.output = "Релевантні документи не знайдено"
                    last_sources = search_response.sources
                search_shown = True

            if token:
                full_response += token
                await msg.stream_token(token)

    except Exception as e:
        error_text = f"\n\nПомилка при генерації відповіді: {str(e)}"
        full_response += error_text
        await msg.stream_token(error_text)

    await msg.update()

    history.add_pair(user_input, full_response)
    cl.user_session.set("last_answer_source", answer_mode)
    if answer_mode == "docs":
        if last_sources:
            cl.user_session.set("last_doc_sources", last_sources)
        if context_used:
            cl.user_session.set("last_doc_context", context_used)
            cl.user_session.set("last_doc_query", user_input)
    else:
        cl.user_session.set("last_doc_sources", [])


@cl.on_stop
async def stop():
    """Handle stop button."""
    await cl.Message(content="Генерацію зупинено").send()
