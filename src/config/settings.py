
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "ukr_docs_v2"

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1

    embedding_model: str = "intfloat/multilingual-e5-base"

    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    docs_path: str = "./docs"

    rag_top_k: int = 5
    rag_fetch_k: int = 20
    rag_vector_threshold: float = 0.3
    rag_reranker_threshold: float = 0.0
    rag_score_ratio: float = 0.3
    rag_fallback_min_score: float = 0.0
    chunk_size: int = 1000
    chunk_overlap: int = 200

    router_config_path: str = "router_config.json"
    router_threshold: float = 0.78

    # Templates
    templates_path: str = "./templates"
    templates_config_path: str = "templates_config.json"
    templates_high_threshold: float = 0.85
    templates_low_threshold: float = 0.82

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
