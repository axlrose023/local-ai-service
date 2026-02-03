from pathlib import Path


class TextLoader:

    EXTENSIONS = {".txt", ".md", ".markdown"}

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.EXTENSIONS

    def load(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8")
