
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

    # Handle casual messages
    router = container.resolve(RouterService)
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

    # Check for template request
    template_service = container.resolve(TemplateService)
    template_match = template_service.match(user_input)

    if template_match:
        if template_match.confidence == MatchConfidence.HIGH:
            if template_match.path.exists():
                await cl.Message(
                    content=f"Ось шаблон **{template_match.display_name}**:",
                    elements=[cl.File(name=template_match.file_name, path=str(template_match.path))]
                ).send()
            else:
                await cl.Message(
                    content=f"Шаблон **{template_match.display_name}** наразі недоступний. Зверніться до адміністратора."
                ).send()
            return

        elif template_match.confidence == MatchConfidence.MEDIUM:
            cl.user_session.set("pending_template", template_match)
            cl.user_session.set("pending_template_query", user_input)
            await cl.Message(
                content=f"Можливо, вам потрібен шаблон **{template_match.display_name}**?",
                actions=[
                    cl.Action(name="download_template", value="yes", label="Так, завантажити"),
                    cl.Action(name="download_template", value="no", label="Ні, просто відповідь")
                ]
            ).send()
            return

    # Standard RAG flow
    await _run_standard_flow(user_input, history)


@cl.action_callback("download_template")
async def on_template_action(action: cl.Action):
    history: ChatHistory | None = cl.user_session.get("history")
    if history is None:
        history = ChatHistory()
        cl.user_session.set("history", history)

    if action.value == "yes":
        template_match = cl.user_session.get("pending_template")
        if template_match and template_match.path.exists():
            await cl.Message(
                content=f"Ось шаблон **{template_match.display_name}**:",
                elements=[cl.File(name=template_match.file_name, path=str(template_match.path))]
            ).send()
        else:
            await cl.Message(
                content="Шаблон наразі недоступний."
            ).send()
    else:
        original_query = cl.user_session.get("pending_template_query")
        if original_query:
            await _run_standard_flow(original_query, history)
        else:
            await cl.Message(
                content="Добре, продовжуємо без шаблону. Напишіть питання ще раз."
            ).send()

    cl.user_session.set("pending_template", None)
    cl.user_session.set("pending_template_query", None)


@cl.on_stop
async def stop():
    await cl.Message(content="Генерацію зупинено").send()
