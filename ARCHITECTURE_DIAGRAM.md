# Архитектура RAG-системы

## Общая схема компонентов

```mermaid
graph TD
    subgraph "👤 User Layer"
        User((Пользователь))
    end

    subgraph "🖥️ Presentation Layer"
        Chainlit[app.py<br/>Chainlit UI]
    end

    subgraph "🧠 Logic Layer"
        Router[router.py<br/>Гибридный роутер]
        RAG[rag.py<br/>RAG Service]
        LLM[llm.py<br/>LLM Client]
    end

    subgraph "🔧 Shared Services"
        Embedder[embeddings.py<br/>Shared Embedder]
        Config[config.py<br/>Settings]
        ChromaClient[chroma_client.py<br/>HTTP Client]
    end

    subgraph "💾 Data Layer"
        ChromaDB[(ChromaDB<br/>Vector Store)]
        Ollama[(Ollama<br/>LLM Server)]
        Docs[docs/<br/>PDF, DOCX, TXT]
    end

    subgraph "🔄 Offline Pipeline"
        Ingest[ingest.py<br/>Document Indexer]
    end

    User --> Chainlit
    Chainlit --> Router
    Chainlit --> RAG
    Chainlit --> LLM

    Router --> Embedder
    RAG --> Embedder
    RAG --> ChromaClient
    Ingest --> Embedder
    Ingest --> ChromaClient

    ChromaClient --> ChromaDB
    LLM --> Ollama
    Ingest --> Docs

    Config -.-> Router
    Config -.-> RAG
    Config -.-> LLM
    Config -.-> Ingest
```

## Поток обработки запроса (Request Flow)

```mermaid
sequenceDiagram
    autonumber
    participant U as 👤 User
    participant A as 🖥️ app.py
    participant R as 🔀 router.py
    participant E as 🧠 embeddings.py
    participant RAG as 🔍 rag.py
    participant C as 🗄️ ChromaDB
    participant L as 🤖 llm.py
    participant O as 🦙 Ollama

    U->>A: "Как настроить VPN?"
    A->>R: should_search(query)

    Note over R: 1. Skip patterns? ❌<br/>2. Keywords? ✅ "vpn"
    R-->>A: True (искать)

    A->>RAG: search(query)
    RAG->>E: encode(query)
    E-->>RAG: vector [768]
    RAG->>C: query(vector, top_k=5)
    C-->>RAG: 5 chunks + scores

    Note over RAG: Фильтр: score > 0.45
    RAG-->>A: 3 релевантных документа

    A->>L: chat_stream(query, context, history)
    L->>O: POST /v1/chat/completions

    loop Streaming
        O-->>L: token
        L-->>A: token
        A-->>U: token
    end

    A-->>U: Ответ + Источники
```

## Fast Flow (без RAG)

```mermaid
sequenceDiagram
    autonumber
    participant U as 👤 User
    participant A as 🖥️ app.py
    participant R as 🔀 router.py
    participant L as 🤖 llm.py
    participant O as 🦙 Ollama

    U->>A: "Привет!"
    A->>R: should_search("Привет!")

    Note over R: Skip pattern: "привет" ✅
    R-->>A: False (не искать)

    Note over A: RAG пропущен<br/>context = ""

    A->>L: chat_stream(query, "", history)
    L->>O: POST /v1/chat/completions
    O-->>L: stream
    L-->>A: stream
    A-->>U: "Здравствуйте! Чем могу помочь?"
```

## Логика роутера (3-Stage Decision)

```mermaid
flowchart TD
    Start([Запрос пользователя]) --> Len{len < 3?}
    Len -->|Да| NoSearch[❌ НЕ ИСКАТЬ]
    Len -->|Нет| Skip

    subgraph "Stage 1: Skip Patterns (0ms)"
        Skip{Содержит<br/>'привет', 'спасибо'...?}
    end
    Skip -->|Да| NoSearch
    Skip -->|Нет| Keywords

    subgraph "Stage 2: Keywords (0ms)"
        Keywords{Содержит<br/>'vpn', 'отпуск'...?}
    end
    Keywords -->|Да| Search[✅ ИСКАТЬ]
    Keywords -->|Нет| Semantic

    subgraph "Stage 3: Semantic (~20ms)"
        Semantic[Cosine similarity<br/>с эталонами]
        Semantic --> Threshold{score > 0.35?}
    end
    Threshold -->|Да| Search
    Threshold -->|Нет| NoSearch

    Search --> RAG[Запуск RAG Pipeline]
    NoSearch --> LLM[Прямой ответ LLM]
```

## Pipeline индексации документов

```mermaid
flowchart LR
    subgraph "📁 Input"
        PDF[PDF]
        DOCX[DOCX]
        TXT[TXT/MD]
    end

    subgraph "🔄 ingest.py"
        Extract[extract_text]
        Chunk[chunk_text<br/>500 chars, 50 overlap]
        Hash[MD5 hash<br/>дедупликация]
        Embed[embedder.encode<br/>batch=50]
    end

    subgraph "💾 Output"
        Chroma[(ChromaDB)]
    end

    PDF --> Extract
    DOCX --> Extract
    TXT --> Extract
    Extract --> Chunk
    Chunk --> Hash
    Hash -->|новый/изменён| Embed
    Hash -->|без изменений| Skip[⏭️ Skip]
    Embed --> Chroma
```

## Структура данных в ChromaDB

```mermaid
erDiagram
    COLLECTION ||--o{ DOCUMENT : contains

    COLLECTION {
        string id PK
        string name "corporate_docs"
        json metadata "hnsw:space=cosine"
    }

    DOCUMENT {
        string id PK "filename_hash_index"
        float[] embedding "768 dimensions"
        string content "chunk text"
        string source "filename.pdf"
        string file_path "/path/to/file"
        string file_hash "md5"
        int chunk_index "0, 1, 2..."
    }
```

## Зависимости между модулями

```mermaid
graph BT
    subgraph "Core"
        config[config.py]
        embeddings[embeddings.py]
        chroma[chroma_client.py]
    end

    subgraph "Services"
        router[router.py]
        rag[rag.py]
        llm[llm.py]
        ingest[ingest.py]
    end

    subgraph "App"
        app[app.py]
    end

    config --> embeddings
    config --> chroma
    config --> router
    config --> rag
    config --> llm
    config --> ingest

    embeddings --> router
    embeddings --> rag
    embeddings --> ingest

    chroma --> rag
    chroma --> ingest

    router --> app
    rag --> app
    llm --> app
```

## Порты и сервисы

| Сервис | Порт | Назначение |
|--------|------|------------|
| Chainlit | 8000 | Web UI |
| ChromaDB | 8001 | Vector Database |
| Ollama | 11434 | LLM Inference |

## Переменные окружения (.env)

```
CHROMA_HOST=localhost
CHROMA_PORT=8001
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b
```
