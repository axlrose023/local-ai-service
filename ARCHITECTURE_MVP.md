# Local Corporate AI Assistant — MVP Architecture

> **Философия**: Запустить работающий продукт за 3 дня, а не идеальную систему за 3 месяца.

---

## Обзор

**3 контейнера** вместо 10:

```
┌─────────────────────────────────────────────────────────────┐
│                     LOCAL NETWORK                            │
│                                                              │
│    ┌──────────────────────────────────────────────────┐     │
│    │                  APP (Monolith)                   │     │
│    │         Chainlit + FastAPI + Embeddings           │     │
│    │                    :8000                          │     │
│    └──────────────────┬───────────────┬───────────────┘     │
│                       │               │                      │
│              ┌────────▼──────┐ ┌──────▼────────┐            │
│              │   CHROMADB    │ │   INFERENCE   │            │
│              │    :8001      │ │  vLLM/Ollama  │            │
│              │               │ │    :8002      │            │
│              └───────────────┘ └───────────────┘            │
│                                                              │
│    ┌─────────────────────────────────────────────────┐      │
│    │                ./docs (Volume)                   │      │
│    │         PDF/DOCX файлы для индексации           │      │
│    └─────────────────────────────────────────────────┘      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Что вырезано (и почему)

| Компонент | Статус | Причина |
|-----------|--------|---------|
| Nginx Gateway | ❌ | Chainlit имеет встроенный auth |
| OIDC/SSO | ❌ | Простой пароль на старте |
| WAF/Rate Limiting | ❌ | Локальная сеть, доверенные юзеры |
| Hybrid Search (BM25) | ❌ | Vector search достаточен для <50k документов |
| Reranker | ❌ | top_k=5 работает нормально |
| ACL фильтрация | ❌ | Все документы доступны всем |
| TEI (Embedder service) | ❌ | sentence-transformers в процессе app |
| Redis | ❌ | Сессии в памяти |
| Prometheus/Grafana/Loki | ❌ | docker logs достаточно |
| Kubernetes | ❌ | Docker Compose |

---

## Структура проекта

```
/local-ai-assistant
├── docker-compose.yml
├── Dockerfile
├── .env
├── .env.example
│
├── app.py                    # Точка входа (Chainlit + FastAPI)
├── config.py                 # Настройки (Pydantic)
├── ingest.py                 # Индексация документов
├── rag.py                    # Поиск в базе знаний
├── llm.py                    # Общение с vLLM
│
├── requirements.txt
│
├── /docs                     # Сюда кидаем PDF/DOCX
│   └── example.pdf
│
├── /chroma_data              # Данные ChromaDB (git-ignored)
│
└── /.chainlit
    └── config.toml           # Настройки UI
```

**Всего 5 Python-файлов** вместо 30+.

---

## Docker Compose

```yaml
# docker-compose.yml
version: "3.9"

services:
  # ========== APP (Monolith) ==========
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
      - LLM_BASE_URL=http://inference:8000/v1
      - LLM_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
      - EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
      - DOCS_PATH=/app/docs
    volumes:
      - ./docs:/app/docs:ro
      - ./chroma_data:/app/chroma_data
    depends_on:
      chromadb:
        condition: service_healthy
      inference:
        condition: service_healthy
    restart: unless-stopped

  # ========== VECTOR DB ==========
  chromadb:
    image: chromadb/chroma:0.4.24
    volumes:
      - ./chroma_data:/chroma/chroma
    environment:
      - ANONYMIZED_TELEMETRY=false
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # ========== LLM ==========
  inference:
    image: vllm/vllm-openai:latest
    command:
      - "--model"
      - "Qwen/Qwen2.5-32B-Instruct-AWQ"
      - "--quantization"
      - "awq"
      - "--max-model-len"
      - "8192"
      - "--gpu-memory-utilization"
      - "0.90"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
    restart: unless-stopped
```

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Зависимости для PDF
RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Порт Chainlit
EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
```

---

## requirements.txt

```
# Core
chainlit>=1.0.0
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# LLM
openai>=1.10.0

# RAG
chromadb>=0.4.24
sentence-transformers>=2.3.0

# Document parsing
pypdf>=4.0.0
python-docx>=1.1.0
python-magic>=0.4.27

# Utils
python-dotenv>=1.0.0
```

---

## Код

### config.py

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "corporate_docs"

    # LLM
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.7

    # Embeddings
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"

    # Documents
    docs_path: str = "./docs"

    # RAG
    rag_top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"


settings = Settings()
```

### ingest.py

```python
"""
Индексация документов в ChromaDB.
Запуск: python ingest.py
"""
import hashlib
from pathlib import Path

import chromadb
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer

from config import settings


def get_file_hash(file_path: Path) -> str:
    """MD5 хэш файла для отслеживания изменений."""
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def extract_text(file_path: Path) -> str:
    """Извлекает текст из PDF или DOCX."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif suffix == ".docx":
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)

    elif suffix in [".txt", ".md"]:
        return file_path.read_text(encoding="utf-8")

    return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Разбивает текст на чанки с перекрытием."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Ищем конец предложения
        if end < len(text):
            for sep in [". ", ".\n", "\n\n", "\n"]:
                pos = text.rfind(sep, start, end)
                if pos != -1:
                    end = pos + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def ingest_folder(folder_path: str = None):
    """Индексирует все документы из папки."""
    folder = Path(folder_path or settings.docs_path)

    if not folder.exists():
        print(f"Папка {folder} не существует")
        return

    # Подключаемся к ChromaDB
    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port
    )

    # Получаем или создаём коллекцию
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"}
    )

    # Загружаем модель эмбеддингов
    print(f"Загрузка модели {settings.embedding_model}...")
    embedder = SentenceTransformer(settings.embedding_model, trust_remote_code=True)

    # Получаем существующие хэши (для инкрементального обновления)
    existing = collection.get(include=["metadatas"])
    existing_hashes = {
        m.get("file_hash")
        for m in (existing.get("metadatas") or [])
        if m
    }

    # Обрабатываем файлы
    extensions = ["*.pdf", "*.docx", "*.txt", "*.md"]
    files = []
    for ext in extensions:
        files.extend(folder.rglob(ext))

    print(f"Найдено {len(files)} файлов")

    for file_path in files:
        file_hash = get_file_hash(file_path)

        # Пропускаем если не изменился
        if file_hash in existing_hashes:
            print(f"⏭ {file_path.name} — без изменений")
            continue

        print(f"📄 Обработка {file_path.name}...")

        # Извлекаем текст
        text = extract_text(file_path)
        if not text.strip():
            print(f"  ⚠ Пустой файл, пропуск")
            continue

        # Разбиваем на чанки
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        print(f"  → {len(chunks)} чанков")

        # Создаём эмбеддинги
        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

        # Сохраняем в ChromaDB
        ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": file_path.name,
                "file_path": str(file_path),
                "file_hash": file_hash,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        print(f"  ✓ Добавлено в базу")

    print(f"\n✅ Индексация завершена. Всего документов в базе: {collection.count()}")


if __name__ == "__main__":
    ingest_folder()
```

### rag.py

```python
"""
RAG: поиск релевантных документов.
"""
import chromadb
from sentence_transformers import SentenceTransformer

from config import settings

# Глобальные объекты (инициализируются один раз)
_embedder: SentenceTransformer = None
_collection = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(
            settings.embedding_model,
            trust_remote_code=True
        )
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port
        )
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def search_documents(query: str, top_k: int = None) -> list[dict]:
    """
    Ищет релевантные документы по запросу.

    Returns:
        Список словарей с ключами: content, source, score
    """
    top_k = top_k or settings.rag_top_k

    # Получаем эмбеддинг запроса
    embedder = get_embedder()
    query_embedding = embedder.encode(query).tolist()

    # Ищем в ChromaDB
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Форматируем результаты
    documents = []
    for i in range(len(results["ids"][0])):
        documents.append({
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "Unknown"),
            "score": 1 - results["distances"][0][i]  # cosine similarity
        })

    return documents


def format_context(documents: list[dict]) -> str:
    """Форматирует документы в контекст для LLM."""
    if not documents:
        return "Релевантные документы не найдены."

    parts = []
    for i, doc in enumerate(documents, 1):
        parts.append(f"[Документ {i}: {doc['source']}]\n{doc['content']}")

    return "\n\n".join(parts)


def get_sources(documents: list[dict]) -> list[str]:
    """Возвращает список уникальных источников."""
    seen = set()
    sources = []
    for doc in documents:
        if doc["source"] not in seen:
            seen.add(doc["source"])
            sources.append(doc["source"])
    return sources
```

### llm.py

```python
"""
Клиент для общения с LLM (vLLM).
"""
from typing import AsyncIterator

from openai import AsyncOpenAI

from config import settings

# Клиент OpenAI (vLLM совместим с API)
client = AsyncOpenAI(
    base_url=settings.llm_base_url,
    api_key="not-needed"  # vLLM не требует ключ
)

SYSTEM_PROMPT = """Ты — корпоративный AI-помощник. Твоя задача — помогать сотрудникам находить информацию в базе знаний компании.

Правила:
1. Отвечай только на основе предоставленного контекста
2. Если информации нет в контексте — честно скажи об этом
3. Указывай источники информации
4. Отвечай на русском языке
5. Будь кратким и по делу"""


async def chat_stream(
    user_message: str,
    context: str = "",
    history: list[dict] = None
) -> AsyncIterator[str]:
    """
    Стриминг ответа от LLM.

    Args:
        user_message: Вопрос пользователя
        context: Контекст из RAG (документы)
        history: История диалога

    Yields:
        Токены ответа
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Добавляем историю
    if history:
        messages.extend(history)

    # Формируем промпт с контекстом
    if context:
        prompt = f"""Контекст из базы знаний:
{context}

Вопрос пользователя: {user_message}

Дай ответ на основе контекста выше."""
    else:
        prompt = user_message

    messages.append({"role": "user", "content": prompt})

    # Запрос к LLM
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        stream=True
    )

    async for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def should_search(user_message: str) -> bool:
    """
    Простая эвристика: нужен ли поиск в базе знаний.

    Для MVP используем простые правила вместо отдельного LLM вызова.
    """
    # Приветствия и small talk — не ищем
    greetings = ["привет", "здравствуй", "добрый день", "hi", "hello", "как дела"]
    message_lower = user_message.lower().strip()

    if any(g in message_lower for g in greetings):
        return False

    # Короткие сообщения без вопросительных слов — не ищем
    if len(message_lower) < 10 and "?" not in message_lower:
        return False

    # Всё остальное — ищем
    return True
```

### app.py

```python
"""
Главный файл приложения.
Chainlit UI + RAG + LLM.
"""
import chainlit as cl

from config import settings
from rag import search_documents, format_context, get_sources
from llm import chat_stream, should_search


@cl.on_chat_start
async def start():
    """Инициализация сессии."""
    cl.user_session.set("history", [])

    await cl.Message(
        content="Здравствуйте! Я корпоративный AI-помощник. "
                "Задайте вопрос по документам компании, и я постараюсь помочь."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Обработка сообщения пользователя."""
    user_input = message.content
    history = cl.user_session.get("history", [])

    # Проверяем, нужен ли поиск
    need_search = await should_search(user_input)

    context = ""
    sources = []

    if need_search:
        # Показываем что ищем
        async with cl.Step(name="Поиск в базе знаний") as step:
            step.input = user_input

            # Ищем документы
            documents = search_documents(user_input)

            if documents:
                context = format_context(documents)
                sources = get_sources(documents)
                step.output = f"Найдено {len(documents)} релевантных фрагментов"
            else:
                step.output = "Документы не найдены"

    # Создаём сообщение для стриминга
    msg = cl.Message(content="")
    await msg.send()

    # Генерируем ответ
    full_response = ""
    async for token in chat_stream(user_input, context, history):
        full_response += token
        await msg.stream_token(token)

    # Добавляем источники
    if sources:
        sources_text = "\n\n---\n**Источники:** " + ", ".join(sources)
        full_response += sources_text
        await msg.stream_token(sources_text)

    await msg.update()

    # Сохраняем историю (последние 10 сообщений)
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("history", history[-10:])


@cl.on_stop
async def stop():
    """Пользователь нажал Stop."""
    pass
```

### .chainlit/config.toml

```toml
[project]
name = "Corporate AI Assistant"
enable_telemetry = false

[UI]
name = "Корпоративный AI-помощник"
description = "Задайте вопрос по документам компании"
default_theme = "light"
show_readme_as_default = false

[UI.theme]
primary = "#1976D2"
background = "#FFFFFF"
paper = "#F5F5F5"

[features]
spontaneous_file_upload = false
audio = false

[session]
timeout = 3600
```

### .env.example

```bash
# ChromaDB
CHROMA_HOST=chromadb
CHROMA_PORT=8000

# LLM (vLLM)
LLM_BASE_URL=http://inference:8000/v1
LLM_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.7

# Embeddings
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5

# Documents
DOCS_PATH=./docs

# RAG
RAG_TOP_K=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Optional: Chainlit auth
# CHAINLIT_AUTH_SECRET=your-secret-key
```

---

## Запуск

### 1. Подготовка

```bash
# Клонируем/создаём проект
mkdir local-ai-assistant && cd local-ai-assistant

# Копируем файлы (или создаём по шаблонам выше)

# Создаём .env
cp .env.example .env

# Создаём папку для документов
mkdir -p docs
# Копируем туда PDF/DOCX файлы
```

### 2. Запуск инфраструктуры

```bash
# Поднимаем все контейнеры
docker-compose up -d

# Проверяем статус
docker-compose ps

# Смотрим логи (особенно inference — модель грузится долго)
docker-compose logs -f inference
```

### 3. Индексация документов

```bash
# Заходим в контейнер app
docker-compose exec app bash

# Запускаем индексацию
python ingest.py
```

### 4. Использование

Открываем в браузере: **http://localhost:8000**

---

## Альтернатива: Ollama вместо vLLM

Если хочется ещё проще (без возни с CUDA/vLLM):

```yaml
# Заменяем inference в docker-compose.yml
inference:
  image: ollama/ollama:latest
  volumes:
    - ollama_data:/root/.ollama
  ports:
    - "11434:11434"
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]

volumes:
  ollama_data:
```

И в `.env`:

```bash
LLM_BASE_URL=http://inference:11434/v1
LLM_MODEL=qwen2.5:32b
```

После запуска:

```bash
# Скачиваем модель
docker-compose exec inference ollama pull qwen2.5:32b
```

---

## Roadmap: MVP → Production

После запуска MVP, добавляем по одной фиче за раз:

### Неделя 2: Улучшение качества
- [ ] Увеличить chunk_size до 1000 с overlap 200
- [ ] Добавить фильтрацию по типу документа
- [ ] Улучшить промпт на основе фидбека

### Неделя 3-4: Безопасность
- [ ] Добавить Chainlit OAuth (Google/GitHub)
- [ ] Или простой Nginx с Basic Auth перед app

### Месяц 2: Надёжность
- [ ] Добавить Redis для сессий
- [ ] Health checks на /health endpoint
- [ ] Простой backup скрипт для chroma_data

### Месяц 3+: Enterprise (см. ARCHITECTURE.md)
- [ ] ACL на документах
- [ ] Hybrid search
- [ ] Мониторинг
- [ ] И т.д.

---

## Troubleshooting

### vLLM не стартует / OOM

```bash
# Проверяем GPU память
nvidia-smi

# Уменьшаем контекст в docker-compose.yml
--max-model-len 4096  # вместо 8192

# Или берём модель поменьше
--model Qwen/Qwen2.5-14B-Instruct-AWQ
```

### ChromaDB connection refused

```bash
# Проверяем что контейнер запущен
docker-compose ps chromadb

# Проверяем health
curl http://localhost:8001/api/v1/heartbeat
```

### Медленная индексация

Первый запуск sentence-transformers скачивает модель (~500MB).
Последующие запуски быстрее (модель кэшируется).

### Chainlit не видит изменения в коде

```bash
# Перезапускаем контейнер
docker-compose restart app

# Или пересобираем
docker-compose up -d --build app
```

---

## Файлы для копирования

Все файлы выше готовы к использованию. Просто:

1. Создай папку проекта
2. Скопируй каждый файл
3. `docker-compose up -d`
4. `docker-compose exec app python ingest.py`
5. Открой http://localhost:8000

**Время до первого работающего чата: ~30 минут** (+ время загрузки модели).
