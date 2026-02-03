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
        self, llm: LLMProtocol, search_service: SearchService, router: RouterService
    ):
        """Initialize chat service.

        Args:
            llm: LLM client.
            search_service: Search service.
            router: Router service.
        """
        self._llm = llm
        self._search = search_service
        self._router = router

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

        if self._router.should_search(user_message):
            search_response = self._search.search(user_message)
            context = search_response.context
        else:
            context = ""

        yield ("", search_response)

        async for token in self._llm.chat_stream(
            user_message=user_message, context=context, history=history.to_list()
        ):
            yield (token, None)
