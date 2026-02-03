import logging
from pathlib import Path
from typing import Optional

from .pdf_loader import PDFLoader
from .docx_loader import DocxLoader
from .text_loader import TextLoader

logger = logging.getLogger(__name__)


class CompositeLoader:

    def __init__(self):
        self._loaders = [
            PDFLoader(),
            DocxLoader(),
            TextLoader(),
        ]

    def supports(self, file_path: Path) -> bool:
        return any(loader.supports(file_path) for loader in self._loaders)

    def load(self, file_path: Path) -> Optional[str]:
        
        for loader in self._loaders:
            if loader.supports(file_path):
                try:
                    return loader.load(file_path)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    return None
        return None
