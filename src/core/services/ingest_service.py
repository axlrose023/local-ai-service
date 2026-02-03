"""Ingest service - document indexing."""

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

from ..models.document import Chunk
from ..protocols.embedder import EmbedderProtocol
from ..protocols.vector_store import VectorStoreProtocol

logger = logging.getLogger(__name__)


class IngestService:
    """Service for indexing documents into vector store."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
        docs_path: str = "./docs",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 50,
    ):
        """Initialize ingest service.

        Args:
            embedder: Embedding service.
            vector_store: Vector store.
            docs_path: Path to documents folder.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks.
            batch_size: Batch size for indexing.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._docs_path = Path(docs_path)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._batch_size = batch_size

        self._loader: Optional["CompositeLoader"] = None

    @property
    def loader(self):
        """Lazy load document loader."""
        if self._loader is None:
            from src.infrastructure.document_loaders import CompositeLoader

            self._loader = CompositeLoader()
        return self._loader

    def _compute_hash(self, content: str) -> str:
        """Compute content hash."""
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks preserving sentences.

        Args:
            text: Text to chunk.

        Returns:
            List of chunks.
        """
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= self._chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                if len(para) > self._chunk_size:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= self._chunk_size:
                            current_chunk = (current_chunk + " " + sent).strip()
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sent
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def run(self, force: bool = False) -> int:
        """Index documents.

        Args:
            force: Force re-indexing of all documents.

        Returns:
            Number of new chunks indexed.
        """
        if not self._docs_path.exists():
            logger.error(f"Docs path not found: {self._docs_path}")
            return 0

        existing_hashes = set()
        if not force:
            for meta in self._vector_store.get_all_metadatas():
                if "file_hash" in meta:
                    existing_hashes.add(meta["file_hash"])

        all_chunks: list[Chunk] = []

        for file_path in self._docs_path.iterdir():
            if not self.loader.supports(file_path):
                continue

            content = self.loader.load(file_path)
            if not content:
                continue

            file_hash = self._compute_hash(content)
            if file_hash in existing_hashes:
                logger.debug(f"Skip unchanged: {file_path.name}")
                continue

            text_chunks = self._chunk_text(content)
            for i, chunk_text in enumerate(text_chunks):
                all_chunks.append(
                    Chunk(
                        content=chunk_text,
                        source=file_path.name,
                        file_path=str(file_path),
                        file_hash=file_hash,
                        chunk_index=i,
                    )
                )

        if not all_chunks:
            logger.info("No new documents to index")
            return 0

        total_indexed = 0
        for i in range(0, len(all_chunks), self._batch_size):
            batch = all_chunks[i : i + self._batch_size]

            ids = [f"{c.file_hash}_{c.chunk_index}" for c in batch]
            documents = [c.content for c in batch]
            metadatas = [
                {
                    "source": c.source,
                    "file_hash": c.file_hash,
                    "chunk_index": c.chunk_index,
                }
                for c in batch
            ]

            texts_with_prefix = [f"passage: {c.content}" for c in batch]
            embeddings = self._embedder.encode(texts_with_prefix).tolist()

            self._vector_store.add(
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )

            total_indexed += len(batch)
            logger.info(f"Indexed batch: {total_indexed}/{len(all_chunks)}")

        logger.info(
            f"Indexing complete: {total_indexed} chunks from {len(set(c.source for c in all_chunks))} files"
        )
        return total_indexed
