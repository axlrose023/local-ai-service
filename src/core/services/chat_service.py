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
        self, user_message: str, history: ChatHistory
    ) -> AsyncIterator[tuple[str, Optional[SearchResponse]]]:
        """Process user message and stream response.

        Args:
            user_message: User's message.
            history: Chat history.

        Yields:
            Tuples of (token, search_response).
            search_response is only set on first yield.
        """
        search_response: Optional[SearchResponse] = None
        context: str | None = None

        if self._router.should_search(user_message):
            search_response = self._search.search(user_message)

            if search_response.results:
                best_score = max(r.score for r in search_response.results)
                if best_score >= self._fallback_min_score:
                    context = search_response.context
                else:
                    context = None
                    logger.info(
                        f"Fallback to general knowledge: best_score={best_score:.2f} "
                        f"< {self._fallback_min_score} for '{user_message[:50]}...'"
                    )
            else:
                # No results survived filtering — fall back to general knowledge.
                context = None
                logger.info(
                    f"No results after filtering for '{user_message[:50]}...', "
                    "falling back to general knowledge"
                )

        yield ("", search_response)

        history_list = self._router.select_history(user_message, history)

        async for token in self._llm.chat_stream(
            user_message=user_message, context=context, history=history_list
        ):
            yield (token, None)
