
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chainlit as cl

from src.config.settings import settings
from src.container import configure_container, container
from src.core.models.chat import ChatHistory
from src.core.models.template import MatchConfidence
from src.core.protocols.embedder import EmbedderProtocol
from src.core.services.chat_service import ChatService
from src.core.services.router_service import RouterService
from src.core.services.template_service import TemplateService

configure_container(settings)


def _extract_meaningful_query(text: str, router: RouterService) -> tuple[str, bool]:
    """Extract meaningful query from a possibly multi-line message.

    Returns:
        (query, is_all_casual): If the message mixes casual lines with real
        questions, returns only the substantive parts. If everything is casual,
        returns the original text with is_all_casual=True.
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

    if len(lines) <= 1:
        return text.strip(), router.is_casual(text)

    substantive = [line for line in lines if not router.is_casual(line)]

    if not substantive:
        return text.strip(), True

    return "\n".join(substantive), False


def _is_source_question(text: str) -> bool:
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


async def _handle_template(user_input: str, history: ChatHistory) -> None:
    """Handle LLM-detected template request using semantic matching."""
    template_service = container.resolve(TemplateService)
    template_match = template_service.match(user_input)

    if template_match and template_match.path.exists():
        if template_match.confidence in (MatchConfidence.HIGH, MatchConfidence.MEDIUM):
            description = template_match.template.description
            response = (
                f"Ось шаблон **{template_match.display_name}**:\n\n"
                f"{description}"
            )
            await cl.Message(
                content=response,
                elements=[cl.File(name=template_match.file_name, path=str(template_match.path))]
            ).send()
            history.add_pair(user_input, response)
            cl.user_session.set("last_answer_source", "general")
            return

    # No matching template found — fall back to search flow
    await _run_search_flow(user_input, history)


async def _run_search_flow(user_input: str, history: ChatHistory) -> None:
    """Fallback: run RAG search flow (skips LLM classification)."""
    chat_service = container.resolve(ChatService)

    carryover_context = cl.user_session.get("last_doc_context")
    carryover_query = cl.user_session.get("last_doc_query")

    msg = cl.Message(content="")
    await msg.send()

    full_response = ""
    last_sources: list[str] = []
    context_used: str | None = None

    try:
        async for token, search_response, mode, used_context in chat_service.search_and_respond(
            user_input,
            history,
            carryover_context=carryover_context,
            carryover_query=carryover_query,
        ):
            context_used = used_context
            if search_response and search_response.results:
                async with cl.Step(name="Пошук у базі знань") as step:
                    step.input = user_input
                    step.output = f"Знайдено {len(search_response.results)} релевантних фрагментів"
                    last_sources = search_response.sources

            if token:
                full_response += token
                await msg.stream_token(token)

    except Exception as e:
        error_text = f"\n\nПомилка при генерації відповіді: {str(e)}"
        full_response += error_text
        await msg.stream_token(error_text)

    await msg.update()

    history.add_pair(user_input, full_response)
    cl.user_session.set("last_answer_source", "docs")
    if last_sources:
        cl.user_session.set("last_doc_sources", last_sources)
    if context_used:
        cl.user_session.set("last_doc_context", context_used)
        cl.user_session.set("last_doc_query", user_input)


async def _run_standard_flow(user_input: str, history: ChatHistory) -> None:
    chat_service = container.resolve(ChatService)

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

            if answer_mode == "template":
                # LLM detected template request — close this message and delegate
                await msg.remove()
                await _handle_template(user_input, history)
                return

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


@cl.on_chat_start
async def start():
    cl.user_session.set("history", ChatHistory())
    cl.user_session.set("last_answer_source", None)
    cl.user_session.set("last_doc_sources", [])
    cl.user_session.set("last_doc_context", None)
    cl.user_session.set("last_doc_query", None)

    embedder = container.resolve(EmbedderProtocol)
    embedder.warmup()
    container.resolve(TemplateService)

    await cl.Message(
        content="Вітаю! Я корпоративний AI-помічник УДО.\n\n"
        "Задайте питання щодо документів компанії, і я постараюся допомогти."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    user_input = message.content
    history: ChatHistory | None = cl.user_session.get("history")
    if history is None:
        history = ChatHistory()
        cl.user_session.set("history", history)

    # Handle source questions
    if _is_source_question(user_input):
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

    # Extract meaningful query from multi-line messages
    router = container.resolve(RouterService)
    query, _ = _extract_meaningful_query(user_input, router)

    # Standard flow — LLM decides search/template/general
    await _run_standard_flow(query, history)


@cl.on_stop
async def stop():
    await cl.Message(content="Генерацію зупинено").send()
