"""Chat service - coordinates search and LLM."""

import logging
from typing import AsyncIterator, Optional

from ..models.chat import ChatHistory
from ..models.document import SearchResponse
from ..protocols.llm import LLMProtocol
from .router_service import RouterService
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
        router: RouterService,
        fallback_min_score: float = 0.0,
    ):
        """Initialize chat service.

        Args:
            llm: LLM client.
            search_service: Search service.
            router: Router service.
            fallback_min_score: Min best-result score to use RAG context.
                If all results score below this, fall back to general knowledge.
        """
        self._llm = llm
        self._search = search_service
        self._router = router
        self._fallback_min_score = fallback_min_score

    def _resolve_context(
        self,
        search_response: SearchResponse,
        user_message: str,
        carryover_context: str | None,
        carryover_query: str | None,
    ) -> str | None:
        """Determine final context from search results and carryover."""
        context: str | None = None

        if search_response.results:
            best_score = max(r.score for r in search_response.results)
            if best_score >= self._fallback_min_score:
                context = search_response.context
            else:
                if self._router.has_keyword(user_message):
                    context = search_response.context
                    logger.info(
                        f"Docs kept due to keyword despite low score: {best_score:.2f} "
                        f"< {self._fallback_min_score} for '{user_message[:50]}...'"
                    )
                else:
                    context = ""
                    logger.info(
                        f"Docs skipped (weak score): best_score={best_score:.2f} "
                        f"< {self._fallback_min_score} for '{user_message[:50]}...'"
                    )
        else:
            context = ""
            logger.info(
                f"No results after filtering for '{user_message[:50]}...', "
                "replying with 'no info in docs'"
            )

        # Carry over previous docs context for short related follow-ups.
        if (context is None or not context.strip()) and carryover_context and carryover_query:
            if self._router.is_related(user_message, carryover_query):
                context = carryover_context
                logger.info(
                    f"Using carryover context for '{user_message[:50]}...' "
                    f"related to '{carryover_query[:50]}...'"
                )

        return context

    async def process_message(
        self,
        user_message: str,
        history: ChatHistory,
        carryover_context: str | None = None,
        carryover_query: str | None = None,
    ) -> AsyncIterator[tuple[str, Optional[SearchResponse], str, str | None]]:
        """Process user message using LLM-based classification.

        Flow:
            1. LLM classify_and_stream — either answers directly or signals [SEARCH]
            2. If [SEARCH]: search DB → second LLM call with context
            3. If direct answer: stream tokens from first call

        Args:
            user_message: User's message.
            history: Chat history.
            carryover_context: Previous docs context for follow-ups.
            carryover_query: Previous query for follow-up detection.

        Yields:
            Tuples of (token, search_response, answer_mode, context_used).
            search_response is only set on first yield.
        """
        # For classify phase: always pass recent history so LLM knows
        # what was already said (avoids repeating greetings etc.)
        recent_history = history.to_list()[-4:] if history.messages else []

        # For docs phase: use semantically filtered history
        history_list = self._router.select_history(user_message, history)

        # Phase 1: LLM classifies and either answers or signals [SEARCH]/[TEMPLATE]
        signal = None
        buffer_tokens: list[str] = []

        async for token in self._llm.classify_and_stream(
            user_message=user_message, history=recent_history
        ):
            if token == _SEARCH_SIGNAL:
                signal = _SEARCH_SIGNAL
                break
            if token == _TEMPLATE_SIGNAL:
                signal = _TEMPLATE_SIGNAL
                break
            buffer_tokens.append(token)

        if signal == _TEMPLATE_SIGNAL:
            # Template request — let presentation layer handle delivery
            yield ("", None, "template", None)
            return

        if signal == _SEARCH_SIGNAL:
            # Phase 2: Search DB
            search_response = self._search.search(user_message)
            context = self._resolve_context(
                search_response, user_message,
                carryover_context, carryover_query,
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
        carryover_context: str | None = None,
        carryover_query: str | None = None,
    ) -> AsyncIterator[tuple[str, Optional[SearchResponse], str, str | None]]:
        """Search DB and respond — skips LLM classification.

        Used as fallback when [TEMPLATE] was detected but no template matched.
        """
        history_list = self._router.select_history(user_message, history)

        search_response = self._search.search(user_message)
        context = self._resolve_context(
            search_response, user_message,
            carryover_context, carryover_query,
        )
        context_used = context if context and context.strip() else None

        yield ("", search_response, "docs", context_used)

        async for token in self._llm.chat_stream(
            user_message=user_message, context=context, history=history_list
        ):
            yield (token, None, "docs", context_used)
