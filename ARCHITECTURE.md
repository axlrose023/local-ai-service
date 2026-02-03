# Local Corporate AI Assistant — Enterprise Architecture v2.0

## Обзор системы

Система состоит из **6 изолированных слоёв**, объединённых во внутреннюю Docker-сеть с единой точкой входа.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL NETWORK                                │
│                                    │                                         │
│                              ┌─────▼─────┐                                   │
│                              │  GATEWAY  │ :443 (TLS)                        │
│                              │  nginx    │ Auth, Rate Limit, WAF             │
│                              └─────┬─────┘                                   │
│                                    │                                         │
├────────────────────────────────────┼────────────────────────────────────────┤
│                           INTERNAL NETWORK (isolated)                        │
│                                    │                                         │
│    ┌───────────────────────────────┼───────────────────────────────────┐    │
│    │                               │                                    │    │
│    │  ┌────────────┐    ┌─────────▼─────────┐    ┌─────────────────┐  │    │
│    │  │  FRONTEND  │◄───│     BACKEND       │───►│   INFERENCE     │  │    │
│    │  │  Chainlit  │    │  FastAPI + MCP    │    │   vLLM/Ollama   │  │    │
│    │  │   :8080    │    │      :8000        │    │     :8001       │  │    │
│    │  └────────────┘    └────────┬──────────┘    └─────────────────┘  │    │
│    │                             │                                     │    │
│    │         ┌───────────────────┼───────────────────┐                │    │
│    │         │                   │                   │                │    │
│    │         ▼                   ▼                   ▼                │    │
│    │  ┌────────────┐    ┌──────────────┐    ┌──────────────┐         │    │
│    │  │  EMBEDDER  │    │   CHROMA DB  │    │    REDIS     │         │    │
│    │  │    TEI     │    │    :8002     │    │    :6379     │         │    │
│    │  │   :8003    │    └──────────────┘    └──────────────┘         │    │
│    │  └────────────┘                                                  │    │
│    │                                                                   │    │
│    │  ┌─────────────────────────────────────────────────────────┐    │    │
│    │  │                    MONITORING                            │    │    │
│    │  │  Prometheus :9090  │  Grafana :3000  │  Loki :3100      │    │    │
│    │  └─────────────────────────────────────────────────────────┘    │    │
│    │                                                                   │    │
│    └───────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │                         PERSISTENT STORAGE                          │  │
│    │  ./volumes/chroma_data  │  ./volumes/docs  │  ./volumes/redis      │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Слой 1: Gateway (Точка входа)

### Назначение
Единственный компонент, доступный из внешней сети. Терминирует TLS, аутентифицирует пользователей, ограничивает нагрузку.

### Технологии
- **nginx** (или Traefik)
- **lua-resty-openidc** для SSO/OIDC
- **fail2ban** для защиты от brute-force

### Конфигурация

```nginx
# /gateway/nginx.conf

upstream backend {
    server backend:8000;
    keepalive 32;
}

upstream frontend {
    server frontend:8080;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=chat_limit:10m rate=2r/s;

server {
    listen 443 ssl http2;
    server_name ai-assistant.corp.local;

    # TLS
    ssl_certificate /etc/nginx/certs/server.crt;
    ssl_certificate_key /etc/nginx/certs/server.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;

    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Content-Security-Policy "default-src 'self'";

    # Health check (no auth)
    location /health {
        access_log off;
        return 200 "OK";
    }

    # API endpoints
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;

        # Auth check (JWT validation)
        auth_request /auth/validate;
        auth_request_set $user_id $upstream_http_x_user_id;
        auth_request_set $user_department $upstream_http_x_user_department;
        auth_request_set $user_access_level $upstream_http_x_user_access_level;

        # Pass user context to backend
        proxy_set_header X-User-ID $user_id;
        proxy_set_header X-User-Department $user_department;
        proxy_set_header X-User-Access-Level $user_access_level;

        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }

    # Chat streaming endpoint
    location /api/v1/chat/stream {
        limit_req zone=chat_limit burst=5 nodelay;

        auth_request /auth/validate;
        # ... same headers ...

        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
    }

    # Frontend
    location / {
        auth_request /auth/validate;
        proxy_pass http://frontend;
    }

    # Internal auth validation
    location = /auth/validate {
        internal;
        proxy_pass http://backend:8000/auth/validate;
        proxy_pass_request_body off;
        proxy_set_header Content-Length "";
        proxy_set_header X-Original-URI $request_uri;
        proxy_set_header Authorization $http_authorization;
    }
}
```

### Безопасность
| Аспект | Решение |
|--------|---------|
| Аутентификация | JWT токены от корпоративного SSO (Keycloak/AD FS) |
| Rate Limiting | 10 req/s общий, 2 req/s на chat endpoint |
| WAF | ModSecurity с OWASP CRS |
| TLS | Только 1.2+, сертификаты от корпоративного CA |
| Egress | Запрещён (no internet access) |

---

## Слой 2: Frontend (UI)

### Назначение
Чат-интерфейс для пользователей с отображением процесса "мышления" AI.

### Технологии
- **Chainlit** 1.x
- Python 3.11+

### Ключевые функции
- Streaming ответов (SSE)
- Отображение Steps (какие инструменты вызывались)
- Кнопка отмены генерации
- Отображение источников (citations)
- Темная/светлая тема
- История сессий

### Структура

```
/frontend
├── Dockerfile
├── requirements.txt
├── app.py                    # Основной код Chainlit
├── auth.py                   # Проверка JWT, получение user context
├── api_client.py             # Async клиент к backend
├── .chainlit/
│   ├── config.toml           # Настройки UI
│   └── translations/         # Локализация (ru/en)
└── public/
    ├── logo.png              # Логотип компании
    └── favicon.ico
```

### Код: app.py

```python
import chainlit as cl
from api_client import BackendClient
from auth import get_user_from_headers

client = BackendClient(base_url="http://backend:8000")

@cl.on_chat_start
async def start():
    """Инициализация сессии."""
    user = get_user_from_headers(cl.user_session.get("headers"))
    cl.user_session.set("user", user)
    cl.user_session.set("conversation_id", None)

    await cl.Message(
        content=f"Здравствуйте, {user.display_name}! Я корпоративный AI-помощник. Чем могу помочь?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Обработка сообщения пользователя."""
    user = cl.user_session.get("user")
    conversation_id = cl.user_session.get("conversation_id")

    # Создаём сообщение для streaming
    msg = cl.Message(content="")
    await msg.send()

    sources = []

    try:
        async for event in client.chat_stream(
            message=message.content,
            user_id=user.id,
            department=user.department,
            access_level=user.access_level,
            conversation_id=conversation_id
        ):
            if event.type == "token":
                await msg.stream_token(event.data)

            elif event.type == "tool_start":
                async with cl.Step(name=event.tool_name) as step:
                    step.input = event.tool_input

            elif event.type == "tool_end":
                step.output = event.tool_output

            elif event.type == "source":
                sources.append(event.data)

            elif event.type == "conversation_id":
                cl.user_session.set("conversation_id", event.data)

            elif event.type == "error":
                msg.content = f"Произошла ошибка: {event.data}"
                break

    except Exception as e:
        msg.content = "Сервис временно недоступен. Попробуйте позже."
        # Log error

    # Добавляем источники
    if sources:
        sources_text = "\n\n---\n**Источники:**\n"
        for src in sources:
            sources_text += f"- {src['title']} (стр. {src.get('page', 'N/A')})\n"
        msg.content += sources_text

    await msg.update()

@cl.on_stop
async def on_stop():
    """Отмена генерации."""
    # Backend автоматически прервёт stream при disconnect
    pass
```

---

## Слой 3: Backend Orchestrator (Мозг)

### Назначение
Центральный компонент: принимает запросы, управляет контекстом, вызывает LLM и инструменты, контролирует доступ к данным.

### Технологии
- **FastAPI** с async
- **Pydantic v2** для валидации
- **openai** SDK (для vLLM)
- **chromadb** client
- **redis** для кэша и сессий
- **tenacity** для retry logic
- **structlog** для логирования

### Структура

```
/backend
├── Dockerfile
├── requirements.txt
├── pyproject.toml
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app, startup/shutdown
│   ├── config.py                  # Pydantic Settings
│   ├── dependencies.py            # DI: clients, services
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── router.py          # API router
│   │   │   ├── chat.py            # POST /chat, GET /chat/stream
│   │   │   ├── documents.py       # Управление документами
│   │   │   └── health.py          # Health checks
│   │   └── auth.py                # JWT validation endpoint
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py            # JWT decode, user context
│   │   ├── exceptions.py          # Custom exceptions
│   │   └── middleware.py          # Logging, error handling
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py         # Общение с vLLM
│   │   ├── rag_service.py         # RAG pipeline
│   │   ├── embedding_service.py   # Клиент к embedder
│   │   └── session_service.py     # Память разговора (Redis)
│   │
│   ├── tools/                     # MCP Tools
│   │   ├── __init__.py
│   │   ├── base.py                # Базовый класс Tool
│   │   ├── registry.py            # Реестр инструментов
│   │   ├── search_docs.py         # Поиск в базе знаний
│   │   ├── search_confluence.py   # (опционально)
│   │   └── system_status.py       # Статус систем
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py           # Hybrid search: vector + BM25
│   │   ├── reranker.py            # BGE-reranker
│   │   ├── acl_filter.py          # Фильтрация по правам
│   │   └── citation.py            # Извлечение источников
│   │
│   └── models/
│       ├── __init__.py
│       ├── schemas.py             # Pydantic модели API
│       ├── user.py                # User context model
│       └── document.py            # Document metadata model
│
├── ingest/                        # Отдельный модуль для индексации
│   ├── __init__.py
│   ├── cli.py                     # CLI для запуска ingest
│   ├── pipeline.py                # Основной pipeline
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── pdf.py                 # PDF парсер (pdfplumber + OCR)
│   │   ├── docx.py                # DOCX парсер
│   │   └── xlsx.py                # Excel парсер
│   ├── chunkers/
│   │   ├── __init__.py
│   │   ├── semantic.py            # Semantic chunking
│   │   └── recursive.py           # Recursive text splitter
│   └── processors/
│       ├── __init__.py
│       ├── metadata.py            # Извлечение метаданных
│       └── dedup.py               # Дедупликация
│
└── tests/
    ├── conftest.py
    ├── test_api/
    ├── test_services/
    ├── test_tools/
    └── test_rag/
```

### Ключевые компоненты

#### config.py

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache

class Settings(BaseSettings):
    # App
    app_name: str = "Corporate AI Assistant"
    api_version: str = "v1"
    debug: bool = False

    # LLM
    llm_base_url: str = "http://inference:8001/v1"
    llm_model: str = "qwen2.5-32b-instruct"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.7
    llm_timeout: int = 120
    llm_max_context: int = 8192  # Безопасный лимит для 24GB

    # Embeddings
    embedder_url: str = "http://embedder:8003"
    embedding_model: str = "nomic-embed-text-v1.5"
    embedding_dim: int = 768

    # ChromaDB
    chroma_host: str = "chromadb"
    chroma_port: int = 8002
    chroma_collection: str = "corporate_docs"

    # Redis
    redis_url: str = "redis://redis:6379/0"
    session_ttl: int = 3600  # 1 hour
    cache_ttl: int = 86400   # 24 hours

    # RAG
    rag_top_k: int = 10           # Retrieve top K
    rag_rerank_top_k: int = 3     # After reranking
    rag_min_score: float = 0.5    # Minimum relevance score

    # Security
    jwt_secret: str = Field(..., env="JWT_SECRET")
    jwt_algorithm: str = "HS256"

    # Retry
    retry_attempts: int = 3
    retry_delay: float = 1.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

#### llm_service.py (с Function Calling)

```python
import json
from typing import AsyncIterator
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from app.config import get_settings
from app.tools.registry import ToolRegistry
from app.models.schemas import ChatMessage, ToolCall

logger = structlog.get_logger()
settings = get_settings()

class LLMService:
    def __init__(self, tool_registry: ToolRegistry):
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key="not-needed",  # vLLM не требует ключ
            timeout=settings.llm_timeout
        )
        self.tool_registry = tool_registry

    def _build_tools_schema(self) -> list[dict]:
        """Генерирует JSON Schema для function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema
                }
            }
            for tool in self.tool_registry.get_all()
        ]

    @retry(
        stop=stop_after_attempt(settings.retry_attempts),
        wait=wait_exponential(multiplier=settings.retry_delay)
    )
    async def chat_with_tools(
        self,
        messages: list[ChatMessage],
        user_context: dict
    ) -> AsyncIterator[dict]:
        """
        Основной метод: отправляет запрос в LLM с поддержкой tools.
        Yields события: token, tool_start, tool_end, source, done.
        """
        tools = self._build_tools_schema()

        response = await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[m.model_dump() for m in messages],
            tools=tools,
            tool_choice="auto",
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            stream=True
        )

        collected_content = ""
        tool_calls_buffer = {}

        async for chunk in response:
            delta = chunk.choices[0].delta

            # Streaming текста
            if delta.content:
                collected_content += delta.content
                yield {"type": "token", "data": delta.content}

            # Tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name or "",
                            "arguments": ""
                        }
                    if tc.function.arguments:
                        tool_calls_buffer[idx]["arguments"] += tc.function.arguments

            # Finish reason
            if chunk.choices[0].finish_reason == "tool_calls":
                # Выполняем инструменты
                for idx, tc in tool_calls_buffer.items():
                    yield {"type": "tool_start", "tool_name": tc["name"], "tool_input": tc["arguments"]}

                    try:
                        args = json.loads(tc["arguments"])
                        result = await self.tool_registry.execute(
                            tc["name"],
                            args,
                            user_context
                        )
                        yield {"type": "tool_end", "tool_name": tc["name"], "tool_output": result.output}

                        # Добавляем источники если есть
                        if result.sources:
                            for source in result.sources:
                                yield {"type": "source", "data": source}

                        # Добавляем результат в контекст и продолжаем
                        messages.append(ChatMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[{
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": tc["arguments"]}
                            }]
                        ))
                        messages.append(ChatMessage(
                            role="tool",
                            content=result.output,
                            tool_call_id=tc["id"]
                        ))
                    except Exception as e:
                        logger.error("tool_execution_failed", tool=tc["name"], error=str(e))
                        yield {"type": "tool_end", "tool_name": tc["name"], "tool_output": f"Ошибка: {e}"}

                # Рекурсивный вызов для финального ответа
                async for event in self.chat_with_tools(messages, user_context):
                    yield event
                return

        yield {"type": "done", "data": collected_content}
```

#### rag/retriever.py (Hybrid Search)

```python
from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.services.embedding_service import EmbeddingService
from app.rag.acl_filter import ACLFilter

settings = get_settings()

class HybridRetriever:
    """Гибридный поиск: Vector + BM25."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        acl_filter: ACLFilter
    ):
        self.embedding_service = embedding_service
        self.acl_filter = acl_filter

        self.chroma_client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_collection(settings.chroma_collection)

        # BM25 индекс (загружается при старте)
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_docs: list[dict] = []

    async def search(
        self,
        query: str,
        user_department: str,
        user_access_level: int,
        top_k: int = 10
    ) -> list[dict]:
        """
        Гибридный поиск с фильтрацией по ACL.

        1. Vector search в ChromaDB с ACL фильтром
        2. BM25 search по тем же документам
        3. Reciprocal Rank Fusion для объединения
        """
        # ACL where clause
        acl_filter = self.acl_filter.build_where_clause(
            user_department=user_department,
            user_access_level=user_access_level
        )

        # 1. Vector search
        query_embedding = await self.embedding_service.embed(query)

        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # Берём больше для fusion
            where=acl_filter,
            include=["documents", "metadatas", "distances"]
        )

        # 2. BM25 search (фильтруем только доступные документы)
        bm25_results = self._bm25_search(
            query=query,
            allowed_ids=set(vector_results["ids"][0]),
            top_k=top_k * 2
        )

        # 3. Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            k=60  # RRF параметр
        )

        return fused[:top_k]

    def _bm25_search(self, query: str, allowed_ids: set, top_k: int) -> list[dict]:
        """BM25 поиск по разрешённым документам."""
        if not self._bm25_index:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)

        results = []
        for idx, score in enumerate(scores):
            doc = self._bm25_docs[idx]
            if doc["id"] in allowed_ids:
                results.append({"id": doc["id"], "score": score, **doc})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: dict,
        bm25_results: list[dict],
        k: int = 60
    ) -> list[dict]:
        """
        RRF: score = sum(1 / (k + rank))
        Объединяет результаты из разных источников.
        """
        rrf_scores = {}
        doc_data = {}

        # Vector results
        for rank, (doc_id, doc, metadata, distance) in enumerate(zip(
            vector_results["ids"][0],
            vector_results["documents"][0],
            vector_results["metadatas"][0],
            vector_results["distances"][0]
        )):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
            doc_data[doc_id] = {
                "id": doc_id,
                "content": doc,
                "metadata": metadata,
                "vector_distance": distance
            }

        # BM25 results
        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
            if doc_id not in doc_data:
                doc_data[doc_id] = result

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        return [
            {**doc_data[doc_id], "rrf_score": rrf_scores[doc_id]}
            for doc_id in sorted_ids
        ]
```

#### rag/acl_filter.py

```python
class ACLFilter:
    """Фильтрация документов по правам доступа."""

    # Уровни доступа
    ACCESS_LEVELS = {
        "public": 0,        # Все сотрудники
        "internal": 1,      # Внутренние документы
        "confidential": 2,  # Конфиденциально
        "restricted": 3     # Ограниченный доступ
    }

    def build_where_clause(
        self,
        user_department: str,
        user_access_level: int
    ) -> dict:
        """
        Строит ChromaDB where clause для фильтрации.

        Логика:
        - Документы с access_level <= user_access_level
        - ИЛИ документы из того же department
        - ИЛИ документы с owner = "all"
        """
        return {
            "$or": [
                # Публичные документы
                {"access_level": {"$lte": user_access_level}},
                # Документы своего отдела
                {"department": user_department},
                # Общедоступные
                {"owner": "all"}
            ]
        }

    def validate_access(
        self,
        document_metadata: dict,
        user_department: str,
        user_access_level: int
    ) -> bool:
        """Проверка доступа к конкретному документу."""
        doc_level = document_metadata.get("access_level", 0)
        doc_dept = document_metadata.get("department", "all")
        doc_owner = document_metadata.get("owner", "all")

        # Проверка уровня
        if doc_level <= user_access_level:
            return True

        # Проверка отдела
        if doc_dept == user_department:
            return True

        # Общедоступный
        if doc_owner == "all":
            return True

        return False
```

#### tools/search_docs.py

```python
from typing import Optional
from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolResult
from app.rag.retriever import HybridRetriever
from app.rag.reranker import Reranker

class SearchDocsInput(BaseModel):
    query: str = Field(..., description="Поисковый запрос")
    doc_type: Optional[str] = Field(None, description="Тип документа: policy, instruction, faq")

class SearchDocsTool(Tool):
    name = "search_docs"
    description = "Поиск в корпоративной базе знаний. Используй для вопросов о политиках, инструкциях, процедурах компании."
    parameters_schema = SearchDocsInput.model_json_schema()

    def __init__(self, retriever: HybridRetriever, reranker: Reranker):
        self.retriever = retriever
        self.reranker = reranker

    async def execute(self, args: dict, user_context: dict) -> ToolResult:
        query = args["query"]
        doc_type = args.get("doc_type")

        # Hybrid search с ACL
        results = await self.retriever.search(
            query=query,
            user_department=user_context["department"],
            user_access_level=user_context["access_level"],
            top_k=10
        )

        if not results:
            return ToolResult(
                output="Документы по запросу не найдены.",
                sources=[]
            )

        # Reranking
        reranked = await self.reranker.rerank(
            query=query,
            documents=results,
            top_k=3
        )

        # Форматируем контекст для LLM
        context_parts = []
        sources = []

        for i, doc in enumerate(reranked):
            context_parts.append(f"[Документ {i+1}]\n{doc['content']}")
            sources.append({
                "id": doc["id"],
                "title": doc["metadata"].get("title", "Без названия"),
                "page": doc["metadata"].get("page"),
                "file": doc["metadata"].get("source_file")
            })

        return ToolResult(
            output="\n\n".join(context_parts),
            sources=sources
        )
```

---

## Слой 4: Inference Engine (LLM)

### Назначение
Держит модель в VRAM, предоставляет OpenAI-совместимый API.

### Технологии
- **vLLM** (production) или **Ollama** (dev)
- CUDA 12.x

### Конфигурация vLLM

```yaml
# В docker-compose
inference:
  image: vllm/vllm-openai:latest
  command:
    - "--model"
    - "Qwen/Qwen2.5-32B-Instruct-AWQ"
    - "--quantization"
    - "awq"
    - "--max-model-len"
    - "8192"              # Безопасный лимит для 24GB
    - "--gpu-memory-utilization"
    - "0.85"              # Оставляем запас
    - "--enable-prefix-caching"
    - "--disable-log-requests"
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Мониторинг GPU

```python
# /backend/app/services/gpu_monitor.py
import subprocess
import json

async def get_gpu_stats() -> dict:
    """Получает статистику GPU для мониторинга."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        parts = result.stdout.strip().split(", ")
        return {
            "memory_used_mb": int(parts[0]),
            "memory_total_mb": int(parts[1]),
            "gpu_utilization": int(parts[2]),
            "memory_utilization": round(int(parts[0]) / int(parts[1]) * 100, 1)
        }
    return {}
```

### Fallback стратегия

```python
# При OOM или timeout — fallback на меньшую модель
class LLMServiceWithFallback:
    def __init__(self):
        self.primary = AsyncOpenAI(base_url="http://inference:8001/v1")
        self.fallback = AsyncOpenAI(base_url="http://ollama:11434/v1")  # Ollama с 7B

    async def chat(self, messages, **kwargs):
        try:
            return await asyncio.wait_for(
                self.primary.chat.completions.create(messages=messages, **kwargs),
                timeout=120
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning("primary_llm_failed", error=str(e))
            # Fallback с меньшим контекстом
            truncated = self._truncate_context(messages, max_tokens=4096)
            return await self.fallback.chat.completions.create(
                model="qwen2.5:7b",
                messages=truncated,
                **kwargs
            )
```

---

## Слой 5: Data Layer

### Компоненты

#### ChromaDB (Vector Store)

```yaml
chromadb:
  image: chromadb/chroma:latest
  volumes:
    - ./volumes/chroma_data:/chroma/chroma
  environment:
    - CHROMA_SERVER_AUTH_CREDENTIALS=${CHROMA_AUTH_TOKEN}
    - CHROMA_SERVER_AUTH_PROVIDER=chromadb.auth.token.TokenAuthServerProvider
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
    interval: 30s
    timeout: 10s
    retries: 3
```

#### Redis (Cache + Sessions)

```yaml
redis:
  image: redis:7-alpine
  command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
  volumes:
    - ./volumes/redis:/data
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 10s
    timeout: 5s
    retries: 3
```

#### Embedder Service (TEI)

```yaml
embedder:
  image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.2
  command:
    - "--model-id"
    - "nomic-ai/nomic-embed-text-v1.5"
    - "--port"
    - "8003"
  # Или GPU версия:
  # image: ghcr.io/huggingface/text-embeddings-inference:1.2
```

### Schema документа в ChromaDB

```python
# Метаданные документа
{
    "id": "doc_abc123",
    "content": "Текст чанка...",
    "metadata": {
        # Источник
        "source_file": "Инструкция_VPN.pdf",
        "source_path": "/docs/IT/Инструкция_VPN.pdf",
        "file_hash": "sha256:...",

        # Структура
        "title": "Инструкция по настройке VPN",
        "section": "2.1 Установка клиента",
        "page": 5,
        "chunk_index": 12,

        # ACL
        "department": "IT",           # или "all"
        "owner": "admin@corp.local",
        "access_level": 1,            # 0=public, 1=internal, 2=confidential, 3=restricted

        # Timestamps
        "created_at": "2024-01-15T10:30:00Z",
        "indexed_at": "2024-01-20T14:00:00Z",

        # Versioning
        "version": 2,
        "previous_version_id": "doc_abc122"
    }
}
```

---

## Слой 6: Monitoring & Observability

### Stack
- **Prometheus** — метрики
- **Grafana** — дашборды
- **Loki** — логи
- **Alertmanager** — алерты

### Ключевые метрики

```python
# /backend/app/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Запросы
REQUEST_COUNT = Counter(
    "chat_requests_total",
    "Total chat requests",
    ["endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "chat_request_duration_seconds",
    "Request latency",
    ["endpoint"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120]
)

# LLM
LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Tokens processed",
    ["type"]  # input, output
)

LLM_LATENCY = Histogram(
    "llm_inference_seconds",
    "LLM inference time",
    buckets=[0.5, 1, 2, 5, 10, 30, 60]
)

# RAG
RAG_RETRIEVAL_COUNT = Counter(
    "rag_retrievals_total",
    "RAG retrievals",
    ["status"]  # success, no_results, error
)

RAG_DOCUMENTS_RETURNED = Histogram(
    "rag_documents_returned",
    "Documents returned per query",
    buckets=[0, 1, 2, 3, 5, 10]
)

# GPU
GPU_MEMORY_USED = Gauge(
    "gpu_memory_used_bytes",
    "GPU memory used"
)

GPU_UTILIZATION = Gauge(
    "gpu_utilization_percent",
    "GPU utilization"
)
```

### Grafana Dashboard

```json
{
  "panels": [
    {
      "title": "Request Rate",
      "expr": "rate(chat_requests_total[5m])"
    },
    {
      "title": "P95 Latency",
      "expr": "histogram_quantile(0.95, rate(chat_request_duration_seconds_bucket[5m]))"
    },
    {
      "title": "Error Rate",
      "expr": "rate(chat_requests_total{status='error'}[5m]) / rate(chat_requests_total[5m])"
    },
    {
      "title": "GPU Memory",
      "expr": "gpu_memory_used_bytes / 1024 / 1024 / 1024"
    },
    {
      "title": "Tokens/sec",
      "expr": "rate(llm_tokens_total{type='output'}[1m])"
    }
  ]
}
```

### Алерты

```yaml
# /monitoring/alerts.yml
groups:
  - name: ai-assistant
    rules:
      - alert: HighErrorRate
        expr: rate(chat_requests_total{status="error"}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in AI Assistant"

      - alert: GPUMemoryCritical
        expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.95
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory near limit - OOM risk"

      - alert: LLMLatencyHigh
        expr: histogram_quantile(0.95, rate(llm_inference_seconds_bucket[5m])) > 30
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "LLM inference taking too long"

      - alert: ChromaDBDown
        expr: up{job="chromadb"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "ChromaDB is down"
```

---

## Ingest Pipeline

### Архитектура

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  File Watch │───►│   Parser    │───►│   Chunker   │───►│  Embedder   │
│   (inotify) │    │ PDF/DOCX/.. │    │  Semantic   │    │    TEI      │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                 │
                                                                 ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Notify    │◄───│   Index     │◄───│   Dedup     │◄───│  Metadata   │
│   Admin     │    │  ChromaDB   │    │   Check     │    │  Extract    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Chunking Strategy

```python
# /backend/ingest/chunkers/semantic.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import re

class SemanticChunker:
    """
    Семантическое разбиение с учётом структуры документа.
    """

    def __init__(
        self,
        chunk_size: int = 512,      # В токенах
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Разделители в порядке приоритета
        self.separators = [
            "\n\n\n",           # Разделы
            "\n\n",             # Параграфы
            "\n",               # Строки
            ". ",               # Предложения
            ", ",               # Части предложений
            " ",                # Слова
        ]

    def chunk(self, text: str, metadata: dict) -> List[dict]:
        """
        Разбивает текст на чанки с сохранением контекста.
        """
        # Предобработка
        text = self._normalize(text)

        # Извлекаем структуру (заголовки)
        sections = self._extract_sections(text)

        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section)
            for i, chunk_text in enumerate(section_chunks):
                if len(chunk_text) < self.min_chunk_size:
                    continue

                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "section": section["title"],
                        "chunk_index": len(chunks),
                    }
                })

        return chunks

    def _normalize(self, text: str) -> str:
        """Нормализация текста."""
        # Убираем множественные пробелы
        text = re.sub(r' +', ' ', text)
        # Нормализуем переносы строк
        text = re.sub(r'\n{3,}', '\n\n\n', text)
        return text.strip()

    def _extract_sections(self, text: str) -> List[dict]:
        """Извлекает секции по заголовкам."""
        # Паттерны заголовков
        header_pattern = r'^(?:\d+\.?\s+)?[A-ZА-ЯЁ][^.!?\n]{0,100}$'

        lines = text.split('\n')
        sections = []
        current_section = {"title": "Введение", "content": ""}

        for line in lines:
            if re.match(header_pattern, line.strip()) and len(line.strip()) < 100:
                if current_section["content"].strip():
                    sections.append(current_section)
                current_section = {"title": line.strip(), "content": ""}
            else:
                current_section["content"] += line + "\n"

        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def _chunk_section(self, section: dict) -> List[str]:
        """Разбивает секцию на чанки."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,  # ~4 chars per token
            chunk_overlap=self.chunk_overlap * 4,
            separators=self.separators,
            length_function=len
        )

        chunks = splitter.split_text(section["content"])

        # Добавляем заголовок секции к каждому чанку для контекста
        return [f"[{section['title']}]\n{chunk}" for chunk in chunks]
```

### CLI для Ingest

```python
# /backend/ingest/cli.py
import asyncio
import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress

from ingest.pipeline import IngestPipeline

console = Console()

@click.group()
def cli():
    """Инструменты для индексации документов."""
    pass

@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--department", "-d", default="all", help="Отдел-владелец")
@click.option("--access-level", "-a", default=0, type=int, help="Уровень доступа (0-3)")
@click.option("--recursive", "-r", is_flag=True, help="Рекурсивно обработать папку")
@click.option("--force", "-f", is_flag=True, help="Переиндексировать даже если не изменился")
def index(path: str, department: str, access_level: int, recursive: bool, force: bool):
    """Индексировать документ или папку."""

    pipeline = IngestPipeline()
    path = Path(path)

    if path.is_file():
        files = [path]
    else:
        pattern = "**/*" if recursive else "*"
        files = [f for f in path.glob(pattern) if f.suffix.lower() in [".pdf", ".docx", ".txt", ".md"]]

    console.print(f"[blue]Найдено {len(files)} файлов для индексации[/blue]")

    with Progress() as progress:
        task = progress.add_task("Индексация...", total=len(files))

        for file in files:
            try:
                result = asyncio.run(pipeline.process_file(
                    file_path=file,
                    department=department,
                    access_level=access_level,
                    force=force
                ))

                if result.skipped:
                    console.print(f"[yellow]⏭ {file.name} — не изменён[/yellow]")
                else:
                    console.print(f"[green]✓ {file.name} — {result.chunks_count} чанков[/green]")

            except Exception as e:
                console.print(f"[red]✗ {file.name} — {e}[/red]")

            progress.advance(task)

@cli.command()
@click.option("--older-than", "-o", default=30, type=int, help="Удалить документы старше N дней")
def cleanup(older_than: int):
    """Удалить устаревшие документы из индекса."""
    # Implementation...
    pass

@cli.command()
def stats():
    """Показать статистику индекса."""
    # Implementation...
    pass

if __name__ == "__main__":
    cli()
```

---

## Docker Compose

```yaml
# docker-compose.yml
version: "3.9"

x-common-env: &common-env
  TZ: Europe/Moscow

x-healthcheck-defaults: &healthcheck-defaults
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 10s

services:
  # ============ GATEWAY ============
  gateway:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"  # Redirect to HTTPS
    volumes:
      - ./gateway/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./gateway/certs:/etc/nginx/certs:ro
    depends_on:
      backend:
        condition: service_healthy
      frontend:
        condition: service_healthy
    networks:
      - frontend-net
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "nginx", "-t"]
    restart: unless-stopped

  # ============ FRONTEND ============
  frontend:
    build: ./frontend
    environment:
      <<: *common-env
      BACKEND_URL: http://backend:8000
    depends_on:
      backend:
        condition: service_healthy
    networks:
      - frontend-net
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
    restart: unless-stopped

  # ============ BACKEND ============
  backend:
    build: ./backend
    environment:
      <<: *common-env
      # LLM
      LLM_BASE_URL: http://inference:8001/v1
      LLM_MODEL: qwen2.5-32b-instruct
      # Embeddings
      EMBEDDER_URL: http://embedder:8003
      # ChromaDB
      CHROMA_HOST: chromadb
      CHROMA_PORT: 8002
      # Redis
      REDIS_URL: redis://redis:6379/0
      # Security
      JWT_SECRET: ${JWT_SECRET}
    depends_on:
      chromadb:
        condition: service_healthy
      redis:
        condition: service_healthy
      inference:
        condition: service_healthy
      embedder:
        condition: service_healthy
    networks:
      - frontend-net
      - backend-net
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    restart: unless-stopped

  # ============ INFERENCE (LLM) ============
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
      - "0.85"
      - "--enable-prefix-caching"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8001"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./volumes/models:/root/.cache/huggingface
    networks:
      - backend-net
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      start_period: 120s  # Модель долго грузится
    restart: unless-stopped

  # ============ EMBEDDER ============
  embedder:
    image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.2
    command:
      - "--model-id"
      - "nomic-ai/nomic-embed-text-v1.5"
      - "--port"
      - "8003"
    volumes:
      - ./volumes/embeddings_cache:/data
    networks:
      - backend-net
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8003/health"]
    restart: unless-stopped

  # ============ CHROMADB ============
  chromadb:
    image: chromadb/chroma:latest
    environment:
      <<: *common-env
      CHROMA_SERVER_AUTH_CREDENTIALS: ${CHROMA_AUTH_TOKEN}
      CHROMA_SERVER_AUTH_PROVIDER: chromadb.auth.token.TokenAuthServerProvider
      ANONYMIZED_TELEMETRY: "false"
    volumes:
      - ./volumes/chroma_data:/chroma/chroma
    networks:
      - backend-net
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
    restart: unless-stopped

  # ============ REDIS ============
  redis:
    image: redis:7-alpine
    command: >
      redis-server
      --appendonly yes
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
    volumes:
      - ./volumes/redis:/data
    networks:
      - backend-net
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "redis-cli", "ping"]
    restart: unless-stopped

  # ============ MONITORING ============
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/alerts.yml:/etc/prometheus/alerts.yml:ro
      - ./volumes/prometheus:/prometheus
    networks:
      - backend-net
      - monitoring-net
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./volumes/grafana:/var/lib/grafana
    networks:
      - monitoring-net
    restart: unless-stopped

  loki:
    image: grafana/loki:latest
    volumes:
      - ./monitoring/loki.yml:/etc/loki/local-config.yaml:ro
      - ./volumes/loki:/loki
    networks:
      - backend-net
      - monitoring-net
    restart: unless-stopped

networks:
  frontend-net:
    driver: bridge
  backend-net:
    driver: bridge
    internal: true  # Нет доступа в интернет
  monitoring-net:
    driver: bridge
    internal: true

volumes:
  chroma_data:
  redis:
  models:
  prometheus:
  grafana:
  loki:
```

---

## Структура проекта (финальная)

```
/local-ai-assistant
├── docker-compose.yml
├── docker-compose.override.yml     # Dev overrides
├── docker-compose.prod.yml         # Production overrides
├── .env.example
├── Makefile                        # Команды: make up, make ingest, etc.
├── README.md
│
├── /gateway
│   ├── nginx.conf
│   ├── certs/
│   │   ├── server.crt
│   │   └── server.key
│   └── modsecurity/                # WAF rules (optional)
│
├── /frontend
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   ├── auth.py
│   ├── api_client.py
│   ├── .chainlit/
│   │   └── config.toml
│   └── public/
│       └── logo.png
│
├── /backend
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── router.py
│   │   │   │   ├── chat.py
│   │   │   │   ├── documents.py
│   │   │   │   └── health.py
│   │   │   └── auth.py
│   │   ├── core/
│   │   │   ├── security.py
│   │   │   ├── exceptions.py
│   │   │   ├── middleware.py
│   │   │   └── metrics.py
│   │   ├── services/
│   │   │   ├── llm_service.py
│   │   │   ├── rag_service.py
│   │   │   ├── embedding_service.py
│   │   │   └── session_service.py
│   │   ├── tools/
│   │   │   ├── base.py
│   │   │   ├── registry.py
│   │   │   ├── search_docs.py
│   │   │   └── system_status.py
│   │   ├── rag/
│   │   │   ├── retriever.py
│   │   │   ├── reranker.py
│   │   │   ├── acl_filter.py
│   │   │   └── citation.py
│   │   └── models/
│   │       ├── schemas.py
│   │       ├── user.py
│   │       └── document.py
│   ├── ingest/
│   │   ├── cli.py
│   │   ├── pipeline.py
│   │   ├── parsers/
│   │   ├── chunkers/
│   │   └── processors/
│   └── tests/
│
├── /monitoring
│   ├── prometheus.yml
│   ├── alerts.yml
│   ├── loki.yml
│   └── grafana/
│       └── provisioning/
│           ├── dashboards/
│           └── datasources/
│
├── /volumes                        # Persistent data (git-ignored)
│   ├── chroma_data/
│   ├── redis/
│   ├── models/
│   ├── prometheus/
│   ├── grafana/
│   └── loki/
│
├── /docs                           # Документы для индексации
│   ├── IT/
│   ├── HR/
│   └── Finance/
│
└── /scripts
    ├── init.sh                     # Первоначальная настройка
    ├── backup.sh                   # Бэкап данных
    └── generate_certs.sh           # Генерация TLS сертификатов
```

---

## Roadmap разработки

### Фаза 1: Infrastructure (1 неделя)
- [ ] docker-compose.yml с healthchecks
- [ ] Базовый nginx gateway (без auth)
- [ ] Поднять Ollama (проще для dev)
- [ ] Поднять ChromaDB + Redis
- [ ] Health endpoints

### Фаза 2: Core RAG (1-2 недели)
- [ ] Ingest pipeline (PDF → chunks → ChromaDB)
- [ ] Basic retriever (vector only)
- [ ] Backend skeleton с /chat endpoint
- [ ] Function calling (search_docs tool)

### Фаза 3: UI + Basic Flow (1 неделя)
- [ ] Chainlit frontend
- [ ] Streaming ответов
- [ ] Отображение источников

### Фаза 4: Production Hardening (2 недели)
- [ ] ACL фильтрация
- [ ] Hybrid search (BM25 + vector)
- [ ] Reranker
- [ ] JWT auth
- [ ] Rate limiting

### Фаза 5: Monitoring & Ops (1 неделя)
- [ ] Prometheus metrics
- [ ] Grafana dashboards
- [ ] Loki logging
- [ ] Alerting

### Фаза 6: Enterprise Features (ongoing)
- [ ] SSO/LDAP интеграция
- [ ] Audit logging
- [ ] Backup automation
- [ ] Multi-GPU scaling

---

## Чеклист для Production

### Безопасность
- [ ] TLS везде (даже внутри сети)
- [ ] JWT токены с коротким TTL
- [ ] Секреты через environment / Vault
- [ ] ACL на документах
- [ ] Rate limiting
- [ ] WAF (ModSecurity)
- [ ] Audit log всех запросов

### Надёжность
- [ ] Healthchecks на всех сервисах
- [ ] Graceful shutdown
- [ ] Retry с exponential backoff
- [ ] Circuit breaker для LLM
- [ ] Fallback модель

### Наблюдаемость
- [ ] Structured logging (JSON)
- [ ] Метрики (latency, error rate, tokens/s)
- [ ] Трейсинг (OpenTelemetry)
- [ ] GPU мониторинг
- [ ] Алерты

### Данные
- [ ] Регулярные бэкапы ChromaDB
- [ ] Версионирование индекса
- [ ] Инкрементальный ingest
- [ ] Дедупликация

---

## Контакты и поддержка

- **Документация**: `/docs/admin-guide.md`
- **Логи**: `docker-compose logs -f backend`
- **Метрики**: `http://localhost:3000` (Grafana)
- **Health**: `curl http://localhost:8000/health`
