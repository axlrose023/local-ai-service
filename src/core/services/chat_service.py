"""Chat service - coordinates search and LLM."""

import logging
from typing import AsyncIterator, Optional

from ..models.chat import ChatHistory
from ..models.document import SearchResponse
from ..protocols.llm import LLMProtocol
from .search_service import SearchService

logger = logging.getLogger(__name__)

# Sentinel values matching OllamaClient signals
_SEARCH_SIGNAL = "[SEARCH]"
_TEMPLATE_SIGNAL = "[TEMPLATE]"


class ChatService:
    """Chat service that coordinates LLM classification, search, and response."""

    def __init__(
        self,
        llm: LLMProtocol,
        search_service: SearchService,
    ):
        """Initialize chat service.

        Args:
            llm: LLM client.
            search_service: Search service.
        """
        self._llm = llm
        self._search = search_service

    def _resolve_context(
        self,
        search_response: SearchResponse,
        user_message: str,
    ) -> str | None:
        """Determine final context from search results."""
        context: str | None = None

        if search_response.results:
            context = search_response.context
        else:
            context = ""
            logger.info(
                "No results found. Search query: %r",
                user_message,
            )

        return context

    async def process_message(
        self,
        user_message: str,
        history: ChatHistory,
    ) -> AsyncIterator[tuple[str, Optional[SearchResponse], str, str | None]]:
        """Process user message using LLM-based classification.

        Flow:
            1. LLM classify_and_stream — either answers directly or signals [SEARCH]
            2. If [SEARCH]: search DB → second LLM call with context
            3. If direct answer: stream tokens from first call

        Args:
            user_message: User's message.
            history: Chat history.
        Yields:
            Tuples of (token, search_response, answer_mode, context_used).
            search_response is only set on first yield.
        """
        # For classify phase: always pass recent history so LLM knows
        # what was already said (avoids repeating greetings etc.)
        recent_history = history.to_list()[-4:] if history.messages else []

        # For docs phase: use full recent history
        history_list = history.to_list()

        # Phase 1: LLM classifies and either answers or signals [SEARCH]/[TEMPLATE]
        signal = None
        buffer_tokens: list[str] = []

        async for token in self._llm.classify_and_stream(
            user_message=user_message, history=recent_history
        ):
            if token == _SEARCH_SIGNAL:
                signal = _SEARCH_SIGNAL
                continue
            if token == _TEMPLATE_SIGNAL:
                signal = _TEMPLATE_SIGNAL
                continue
            if signal is not None:
                continue
            buffer_tokens.append(token)

        if signal == _TEMPLATE_SIGNAL:
            # Template request — let presentation layer handle delivery
            yield ("", None, "template", None)
            return

        if signal == _SEARCH_SIGNAL:
            rewrite_history = history_list if history_list else recent_history
            search_query = await self._llm.generate_search_query(
                user_message=user_message,
                history=rewrite_history,
            )
            # Phase 2: Search DB
            search_response = self._search.search(search_query)
            if not search_response.results:
                yield ("", search_response, "docs", None)
                yield ("В документах компанії немає інформації з цього питання.", None, "docs", None)
                return
            context = self._resolve_context(
                search_response, user_message,
            )
            answer_mode = "docs"
            context_used = context if context and context.strip() else None

            yield ("", search_response, answer_mode, context_used)

            # Phase 3: Second LLM call with document context
            async for token in self._llm.chat_stream(
                user_message=user_message, context=context, history=history_list
            ):
                yield (token, None, answer_mode, context_used)
        else:
            # Direct answer from first call — general mode
            answer_mode = "general"
            yield ("", None, answer_mode, None)

            for token in buffer_tokens:
                yield (token, None, answer_mode, None)

    async def search_and_respond(
        self,
        user_message: str,
        history: ChatHistory,
    ) -> AsyncIterator[tuple[str, Optional[SearchResponse], str, str | None]]:
        """Search DB and respond — skips LLM classification.

        Used as fallback when [TEMPLATE] was detected but no template matched.
        """
        history_list = history.to_list()

        search_query = await self._llm.generate_search_query(
            user_message=user_message,
            history=history_list,
        )
        search_response = self._search.search(search_query)
        if not search_response.results:
            yield ("", search_response, "docs", None)
            yield ("В документах компанії немає інформації з цього питання.", None, "docs", None)
            return
        context = self._resolve_context(
            search_response, user_message,
        )
        context_used = context if context and context.strip() else None

        yield ("", search_response, "docs", context_used)

        async for token in self._llm.chat_stream(
            user_message=user_message, context=context, history=history_list
        ):
            yield (token, None, "docs", context_used)
