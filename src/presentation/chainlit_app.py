
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
from src.core.services.template_service import TemplateService

configure_container(settings)


async def _run_standard_flow(user_input: str, history: ChatHistory) -> None:
    """Run standard chat flow (router -> search -> LLM streaming)."""
    chat_service = container.resolve(ChatService)

    msg = cl.Message(content="")
    await msg.send()

    full_response = ""
    search_shown = False

    try:
        async for token, search_response in chat_service.process_message(
            user_input, history
        ):
            if search_response and not search_shown:
                async with cl.Step(name="Пошук у базі знань") as step:
                    step.input = user_input
                    if search_response.results:
                        step.output = f"Знайдено {len(search_response.results)} релевантних фрагментів"
                    else:
                        step.output = "Релевантні документи не знайдено"
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


@cl.on_chat_start
async def start():
    """Initialize chat session."""
    cl.user_session.set("history", ChatHistory())

    embedder = container.resolve(EmbedderProtocol)
    embedder.warmup()
    container.resolve(TemplateService)

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

    # Check for template request first
    template_service = container.resolve(TemplateService)
    template_match = template_service.match(user_input)

    if template_match:
        if template_match.confidence == MatchConfidence.HIGH:
            # Deliver file immediately
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
            # Ask for confirmation
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
    """Handle template download confirmation."""
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
    """Handle stop button."""
    await cl.Message(content="Генерацію зупинено").send()
