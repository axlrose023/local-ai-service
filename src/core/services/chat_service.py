"""Chat service - coordinates search and LLM."""

import logging
from typing import AsyncIterator, Optional

from ..models.chat import ChatHistory
from ..models.document import SearchResponse
from ..protocols.llm import LLMProtocol
from .router_service import RouterService
from .search_service import SearchService

logger = logging.getLogger(__name__)


class ChatService:
    """Chat service that coordinates router, search, and LLM."""

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

    async def process_message(
        self,
        user_message: str,
        history: ChatHistory,
        carryover_context: str | None = None,
        carryover_query: str | None = None,
    ) -> AsyncIterator[tuple[str, Optional[SearchResponse], str, str | None]]:
        """Process user message and stream response.

        Args:
            user_message: User's message.
            history: Chat history.

        Yields:
            Tuples of (token, search_response, answer_mode, context_used).
            search_response is only set on first yield.
        """
        search_response: Optional[SearchResponse] = None
        context: str | None = None
        context_used: str | None = None
        answer_mode = "general"

        if self._router.should_search(user_message):
            answer_mode = "docs"
            search_response = self._search.search(user_message)

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
                # No results survived filtering — doc-only mode with empty context.
                context = ""
                logger.info(
                    f"No results after filtering for '{user_message[:50]}...', "
                    "replying with 'no info in docs'"
                )

        # Carry over previous docs context for short related follow-ups.
        if (context is None or not context.strip()) and carryover_context and carryover_query:
            if self._router.is_related(user_message, carryover_query):
                context = carryover_context
                answer_mode = "docs"
                logger.info(
                    f"Using carryover context for '{user_message[:50]}...' "
                    f"related to '{carryover_query[:50]}...'"
                )

        if context is not None and context.strip():
            context_used = context

        yield ("", search_response, answer_mode, context_used)

        history_list = self._router.select_history(user_message, history)

        async for token in self._llm.chat_stream(
            user_message=user_message, context=context, history=history_list
        ):
            yield (token, None, answer_mode, context_used)
