import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chainlit as cl

from src.config.settings import settings
from src.container import configure_container, container
from src.core.models.chat import ChatHistory, ChatMessage
from src.core.protocols.embedder import EmbedderProtocol
from src.core.protocols.llm import LLMProtocol
from src.core.services.chat_service import ChatService
from src.core.services.template_service import TemplateService

configure_container(settings)

# CJK characters + CJK punctuation
_CJK_RE = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff00-\uffef]"
)
_LATIN_RE = re.compile(r"[A-Za-z]")
_CYR_RE = re.compile(r"[А-Яа-яІіЇїЄєҐґ]")

_LATIN_TO_CYR = {
    "A": "А",
    "a": "а",
    "B": "Б",
    "b": "б",
    "C": "С",
    "c": "с",
    "D": "Д",
    "d": "д",
    "E": "Е",
    "e": "е",
    "F": "Ф",
    "f": "ф",
    "G": "Г",
    "g": "г",
    "H": "Х",
    "h": "х",
    "I": "І",
    "i": "і",
    "J": "Й",
    "j": "й",
    "K": "К",
    "k": "к",
    "L": "Л",
    "l": "л",
    "M": "М",
    "m": "м",
    "N": "Н",
    "n": "н",
    "O": "О",
    "o": "о",
    "P": "П",
    "p": "п",
    "Q": "К",
    "q": "к",
    "R": "Р",
    "r": "р",
    "S": "С",
    "s": "с",
    "T": "Т",
    "t": "т",
    "U": "У",
    "u": "у",
    "V": "В",
    "v": "в",
    "W": "В",
    "w": "в",
    "X": "Х",
    "x": "х",
    "Y": "И",
    "y": "и",
    "Z": "З",
    "z": "з",
}


def _normalize_output(text: str) -> str:
    """Normalize LLM output: strip CJK and fix mixed-script words."""
    if not text:
        return text

    cleaned = _CJK_RE.sub("", text)
    cleaned = re.sub(r"(?i)\bзупинись\b", "", cleaned)

    def fix_token(token: str) -> str:
        if _LATIN_RE.search(token) and _CYR_RE.search(token):
            return "".join(_LATIN_TO_CYR.get(ch, ch) for ch in token)
        return token

    parts = re.split(r"(\s+)", cleaned)
    parts = [fix_token(p) if not p.isspace() else p for p in parts]
    normalized = "".join(parts)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized.strip()


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


async def _handle_template(
    user_input: str, history: ChatHistory, template_id: str | None = None
) -> None:
    """Handle LLM-detected template request.

    If template_id is provided (from [TEMPLATE:id] signal), look up directly.
    Otherwise fall back to semantic matching, then to search flow.
    """
    template_service = container.resolve(TemplateService)
    llm = container.resolve(LLMProtocol)

    template_match = None

    # Try direct lookup by LLM-provided id
    if template_id:
        template_match = template_service.get_by_id(template_id)

    # Fallback: semantic matching on expanded query
    if not template_match:
        history_list = history.to_list()
        if history_list:
            expanded_query = await llm.generate_search_query(
                user_message=user_input,
                history=history_list,
            )
        else:
            expanded_query = user_input

        template_match = template_service.suggest_for_query(expanded_query)

    if template_match and template_match.path.exists():
        response = f"Ось шаблон **{template_match.display_name}**."
        await cl.Message(
            content=response,
            elements=[
                cl.File(name=template_match.file_name, path=str(template_match.path))
            ],
        ).send()
        history.add_pair(user_input, response)
        # Remember this template was already sent — don't suggest it again
        sent = list(cl.user_session.get("suggested_or_sent_templates", []))
        if template_match.template.id not in sent:
            sent.append(template_match.template.id)
        cl.user_session.set("suggested_or_sent_templates", sent)
        cl.user_session.set("last_answer_source", "general")
        return

    # No matching template found — fall back to search flow
    await _run_search_flow(user_input, history)


_MAX_TEMPLATE_QUERY_LENGTH = 800


def _query_for_template_suggestion(
    user_input: str,
    history: ChatHistory,
    context_query: str | None = None,
) -> str:
    """Build a rich query for template matching using conversation context.
    Uses context_query (e.g. search query) and/or recent messages so that
    follow-ups like 'А є більш детальна інформація?' still map to the right template.
    """
    parts: list[str] = []
    if context_query and context_query.strip():
        parts.append(context_query.strip())
    # Add recent user messages so "щодо цього" refers to the right topic (skip long assistant replies)
    recent = history.to_list()[-4:]
    for m in recent:
        if m.get("role") == "user" and m.get("content", "").strip():
            parts.append(m["content"].strip())
    parts.append(user_input.strip())
    combined = " ".join(p for p in parts if p)
    if len(combined) > _MAX_TEMPLATE_QUERY_LENGTH:
        combined = combined[:_MAX_TEMPLATE_QUERY_LENGTH].rsplit(" ", 1)[0]
    return combined


async def _suggest_template_if_relevant(
    user_input: str,
    history: ChatHistory,
    context_query: str | None = None,
) -> None:
    """After a docs-mode response, suggest a related template if relevant.
    Skips templates that were already suggested or sent in this conversation.
    Uses context_query and recent history for better mapping (e.g. mat. dopomoga -> рапорт).
    """
    already_used = set(cl.user_session.get("suggested_or_sent_templates", []))
    template_service = container.resolve(TemplateService)
    query_for_suggestion = _query_for_template_suggestion(
        user_input, history, context_query
    )
    suggestion = template_service.suggest_for_query(query_for_suggestion)

    if not suggestion or not suggestion.path.exists():
        return
    if suggestion.template.id in already_used:
        return

    response = (
        f"До речі, у мене є шаблон **{suggestion.display_name}**, "
        f"який може бути корисним."
    )
    await cl.Message(
        content=response,
        elements=[cl.File(name=suggestion.file_name, path=str(suggestion.path))],
    ).send()
    history.add(ChatMessage(role="assistant", content=response))
    # Remember so we don't suggest the same template again
    sent = list(already_used)
    sent.append(suggestion.template.id)
    cl.user_session.set("suggested_or_sent_templates", sent)


async def _run_search_flow(user_input: str, history: ChatHistory) -> None:
    """Fallback: run RAG search flow (skips LLM classification)."""
    chat_service = container.resolve(ChatService)

    msg = cl.Message(content="")
    await msg.send()

    full_response = ""
    last_sources: list[str] = []
    last_search_query: str | None = None

    try:
        async for token, search_response, _, _ in chat_service.search_and_respond(
            user_input,
            history,
        ):
            if search_response:
                search_query = (search_response.search_query or user_input).strip()
                last_search_query = search_query
                async with cl.Step(name="Пошук у базі знань") as step:
                    step.input = user_input
                    if search_response.results:
                        step.output = (
                            f"Пошуковий запит: {search_query}\n"
                            f"Знайдено {len(search_response.results)} релевантних фрагментів"
                        )
                    else:
                        step.output = (
                            f"Пошуковий запит: {search_query}\n"
                            "Релевантні документи не знайдено"
                        )
                    last_sources = search_response.sources

            if token:
                full_response += token
                await msg.stream_token(token)

    except Exception as e:
        error_text = f"\n\nПомилка при генерації відповіді: {str(e)}"
        full_response += error_text
        await msg.stream_token(error_text)

    normalized = _normalize_output(full_response)
    if normalized != full_response:
        full_response = normalized
        msg.content = full_response
    await msg.update()

    history.add_pair(user_input, full_response)
    cl.user_session.set("last_answer_source", "docs")
    if last_sources:
        cl.user_session.set("last_doc_sources", last_sources)

    # Suggest a related template using search query + history for better mapping
    await _suggest_template_if_relevant(
        user_input, history, context_query=last_search_query
    )


async def _run_standard_flow(user_input: str, history: ChatHistory) -> None:
    chat_service = container.resolve(ChatService)

    msg = cl.Message(content="")
    await msg.send()

    full_response = ""
    search_shown = False
    answer_mode = "general"
    last_sources: list[str] = []
    last_search_query: str | None = None

    try:
        async for (
            token,
            search_response,
            mode,
            context_or_id,
        ) in chat_service.process_message(
            user_input,
            history,
        ):
            answer_mode = mode

            if answer_mode == "template":
                # LLM detected template request — close this message and delegate
                await msg.remove()
                template_id = context_or_id  # chat_service passes template_id here
                await _handle_template(user_input, history, template_id)
                return

            if search_response and not search_shown:
                search_query = (search_response.search_query or user_input).strip()
                last_search_query = search_query
                async with cl.Step(name="Пошук у базі знань") as step:
                    step.input = user_input
                    if search_response.results:
                        step.output = (
                            f"Пошуковий запит: {search_query}\n"
                            f"Знайдено {len(search_response.results)} релевантних фрагментів"
                        )
                    else:
                        step.output = (
                            f"Пошуковий запит: {search_query}\n"
                            "Релевантні документи не знайдено"
                        )
                    last_sources = search_response.sources
                search_shown = True

            if token:
                full_response += token
                await msg.stream_token(token)

    except Exception as e:
        error_text = f"\n\nПомилка при генерації відповіді: {str(e)}"
        full_response += error_text
        await msg.stream_token(error_text)

    normalized = _normalize_output(full_response)
    if normalized != full_response:
        full_response = normalized
        msg.content = full_response
    await msg.update()

    history.add_pair(user_input, full_response)
    cl.user_session.set("last_answer_source", answer_mode)
    if answer_mode == "docs":
        if last_sources:
            cl.user_session.set("last_doc_sources", last_sources)
        # Suggest a related template using search query + history for better mapping
        await _suggest_template_if_relevant(
            user_input, history, context_query=last_search_query
        )
    else:
        cl.user_session.set("last_doc_sources", [])


@cl.on_chat_start
async def start():
    cl.user_session.set("history", ChatHistory())
    cl.user_session.set("last_answer_source", None)
    cl.user_session.set("last_doc_sources", [])
    cl.user_session.set("suggested_or_sent_templates", [])

    embedder = container.resolve(EmbedderProtocol)
    embedder.warmup()
    container.resolve(TemplateService)

    await cl.Message(
        content="Вітаю! Я AI-помічник Управління Державної охорони України.\n\n"
        "Задайте питання щодо службових документів, і я постараюся допомогти."
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
                    f"Відповідь була сформована на основі документів УДО: {sources}."
                )
            else:
                response = (
                    "Я відповідав на основі документів УДО, але в них немає "
                    "прямої відповіді на це питання."
                )
        elif last_source == "general":
            response = (
                "Це була відповідь із загальних знань, а не зі службових документів."
            )
        else:
            response = (
                "Поки що немає попередньої відповіді, на яку можу послатися. "
                "Поставте, будь ласка, конкретне питання."
            )

        await cl.Message(content=response).send()
        history.add_pair(user_input, response)
        return

    # Standard flow — LLM decides search/template/general
    await _run_standard_flow(user_input.strip(), history)


@cl.on_stop
async def stop():
    await cl.Message(content="Генерацію зупинено").send()
