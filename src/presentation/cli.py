
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx

from src.config.settings import settings
from src.container import configure_container, container
from src.core.services.ingest_service import IngestService

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def ensure_ollama_model() -> bool:
    """Ensure Ollama model is available.

    Returns:
        True if model ready, False otherwise.
    """
    model = settings.llm_model
    base_url = settings.llm_base_url.replace("/v1", "")

    logger.info(f"Checking Ollama model: {model}")

    for attempt in range(30):
        try:
            resp = httpx.get(f"{base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if any(model in m for m in models):
                    logger.info(f"Model {model} is ready")
                    return True

                logger.info(f"Pulling model {model}...")
                # Use API instead of CLI (ollama not installed in this container)
                pull_resp = httpx.post(
                    f"{base_url}/api/pull",
                    json={"name": model},
                    timeout=600,  # Model pull can take a while
                )
                if pull_resp.status_code == 200:
                    logger.info(f"Model {model} pulled successfully")
                    return True
                else:
                    logger.error(f"Failed to pull model: {pull_resp.text}")
                    continue
        except Exception:
            logger.info(f"Waiting for Ollama... ({attempt + 1}/30)")
            time.sleep(2)

    logger.error("Ollama not available")
    return False


def cmd_startup():
    """Startup command - init, index, run."""
    logger.info("Starting RAG Bot...")

    if not ensure_ollama_model():
        sys.exit(1)

    configure_container(settings)

    ingest_service = container.resolve(IngestService)
    ingest_service.run()

    logger.info("Starting Chainlit...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "chainlit",
            "run",
            "src/presentation/chainlit_app.py",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    )


def cmd_ingest():
    """Ingest command - index documents only."""
    configure_container(settings)
    ingest_service = container.resolve(IngestService)
    count = ingest_service.run()
    logger.info(f"Indexed {count} chunks")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.presentation.cli <command>")
        print("Commands: startup, ingest")
        sys.exit(1)

    command = sys.argv[1]

    if command == "startup":
        cmd_startup()
    elif command == "ingest":
        cmd_ingest()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
