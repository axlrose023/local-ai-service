
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chainlit as cl

from src.config.settings import settings
from src.container import configure_container, container
from src.core.models.chat import ChatHistory
from src.core.protocols.embedder import EmbedderProtocol
from src.core.services.chat_service import ChatService

configure_container(settings)


@cl.on_chat_start
async def start():
    """Initialize chat session."""
    cl.user_session.set("history", ChatHistory())

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


@cl.on_stop
async def stop():
    """Handle stop button."""
    await cl.Message(content="Генерацію зупинено").send()
