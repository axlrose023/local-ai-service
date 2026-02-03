import logging
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from .config.settings import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class Container:
    _factories: dict[type, Callable[[], Any]] = field(default_factory=dict)
    _singletons: dict[type, Any] = field(default_factory=dict)
    _singleton_flags: set[type] = field(default_factory=set)

    def register(
        self, interface: type[T], factory: Callable[[], T], singleton: bool = False
    ) -> None:
        """Register factory for interface.

        Args:
            interface: Interface type.
            factory: Factory function.
            singleton: Whether to cache instance.
        """
        self._factories[interface] = factory
        if singleton:
            self._singleton_flags.add(interface)

    def resolve(self, interface: type[T]) -> T:
        if interface in self._singletons:
            return self._singletons[interface]

        if interface not in self._factories:
            raise KeyError(f"No factory registered for {interface}")

        instance = self._factories[interface]()

        if interface in self._singleton_flags:
            self._singletons[interface] = instance

        return instance

    def reset(self) -> None:
        """Reset singletons (for testing)."""
        self._singletons.clear()


container = Container()


def configure_container(settings: Settings) -> Container:
    """Configure container with all dependencies.

    Args:
        settings: Application settings.

    Returns:
        Configured container.
    """
    from .core.protocols.embedder import EmbedderProtocol
    from .core.protocols.llm import LLMProtocol
    from .core.protocols.reranker import RerankerProtocol
    from .core.protocols.vector_store import VectorStoreProtocol
    from .core.services.chat_service import ChatService
    from .core.services.ingest_service import IngestService
    from .core.services.router_service import RouterService
    from .core.services.search_service import SearchService
    from .infrastructure.embeddings.sentence_transformer import (
        SentenceTransformerEmbedder,
    )
    from .infrastructure.llm.ollama_client import OllamaClient
    from .infrastructure.rerankers.cross_encoder import CrossEncoderReranker
    from .infrastructure.vector_stores.chroma_store import ChromaVectorStore

    container.register(
        EmbedderProtocol,
        lambda: SentenceTransformerEmbedder(settings.embedding_model),
        singleton=True,
    )

    container.register(
        VectorStoreProtocol,
        lambda: ChromaVectorStore(
            host=settings.chroma_host,
            port=settings.chroma_port,
            collection_name=settings.chroma_collection,
        ),
        singleton=True,
    )

    container.register(
        RerankerProtocol,
        lambda: CrossEncoderReranker(settings.reranker_model),
        singleton=True,
    )

    container.register(
        LLMProtocol,
        lambda: OllamaClient(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        ),
        singleton=True,
    )

    container.register(
        SearchService,
        lambda: SearchService(
            embedder=container.resolve(EmbedderProtocol),
            vector_store=container.resolve(VectorStoreProtocol),
            reranker=container.resolve(RerankerProtocol),
            top_k=settings.rag_top_k,
            fetch_k=settings.rag_fetch_k,
            vector_threshold=settings.rag_vector_threshold,
            reranker_threshold=settings.rag_reranker_threshold,
            score_ratio=settings.rag_score_ratio,
        ),
        singleton=True,
    )

    container.register(
        RouterService,
        lambda: RouterService(
            embedder=container.resolve(EmbedderProtocol),
            config_path=settings.router_config_path,
            threshold=settings.router_threshold,
        ),
        singleton=True,
    )

    container.register(
        ChatService,
        lambda: ChatService(
            llm=container.resolve(LLMProtocol),
            search_service=container.resolve(SearchService),
            router=container.resolve(RouterService),
        ),
        singleton=True,
    )

    container.register(
        IngestService,
        lambda: IngestService(
            embedder=container.resolve(EmbedderProtocol),
            vector_store=container.resolve(VectorStoreProtocol),
            docs_path=settings.docs_path,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        singleton=True,
    )

    logger.info("Container configured")
    return container
