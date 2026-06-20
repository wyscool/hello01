# ============================================================
# rag_kb/config.py — RAG 知识库配置
# ============================================================

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """RAG 知识库全局配置, from_env() 从环境变量加载。"""

    # --- Service ---
    host: str = "0.0.0.0"
    port: int = 8002

    # --- LLM (与 deploy/ 共用相同的 API key) ---
    llm_model: str = "claude-sonnet-4-6"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_max_retries: int = 2
    llm_timeout_seconds: float = 90.0

    # --- Rate Limiting ---
    rate_limit_per_minute: int = 30

    # --- RAG: Embedding & Chunking ---
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- RAG: ChromaDB ---
    chroma_persist_dir: str = "./rag_kb/data/chroma_db"
    chroma_collection_name: str = "rag_kb_main"

    # --- RAG: Retrieval ---
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.3
    use_mmr: bool = True
    mmr_lambda: float = 0.7

    # --- Caching ---
    cache_enabled: bool = True
    cache_ttl_seconds: float = 300.0
    cache_max_size: int = 1000

    # --- Ingest ---
    max_ingest_file_mb: float = 10.0
    allowed_suffixes: tuple = (".txt", ".md", ".json", ".py", ".java",
                               ".yml", ".yaml", ".xml", ".sql", ".html",
                               ".css", ".sh", ".c", ".h", ".cpp", ".hpp")

    # --- Project ---
    project_root: str = "."

    @classmethod
    def from_env(cls) -> "AppConfig":
        overrides: dict[str, str | int | float | bool | tuple] = {}

        for name, type_fn in [
            ("HOST", str), ("PORT", int),
            ("LLM_MODEL", str), ("LLM_API_KEY", str),
            ("LLM_BASE_URL", str), ("LLM_MAX_RETRIES", int),
            ("LLM_TIMEOUT_SECONDS", float), ("RATE_LIMIT_PER_MINUTE", int),
            ("EMBEDDING_MODEL", str), ("CHUNK_SIZE", int),
            ("CHUNK_OVERLAP", int), ("CHROMA_PERSIST_DIR", str),
            ("CHROMA_COLLECTION_NAME", str), ("RETRIEVAL_TOP_K", int),
            ("RETRIEVAL_MIN_SCORE", float), ("USE_MMR", bool),
            ("MMR_LAMBDA", float), ("CACHE_ENABLED", bool),
            ("CACHE_TTL_SECONDS", float), ("CACHE_MAX_SIZE", int),
            ("MAX_INGEST_FILE_MB", float), ("PROJECT_ROOT", str),
        ]:
            env_val = os.getenv(name)
            if env_val is not None:
                if type_fn is bool:
                    overrides[name.lower()] = env_val.strip().lower() in (
                        "1", "true", "yes", "on"
                    )
                else:
                    overrides[name.lower()] = type_fn(env_val)

        # allowed_suffixes 特殊处理
        suffixes_str = os.getenv("ALLOWED_SUFFIXES", "")
        if suffixes_str:
            overrides["allowed_suffixes"] = tuple(
                s.strip() for s in suffixes_str.split(",")
            )

        # RAG 特定 env 映射
        env_to_field: dict[str, str] = {
            "ANTHROPIC_API_KEY": "llm_api_key",
            "ANTHROPIC_BASE_URL": "llm_base_url",
        }
        for env_name, field_name in env_to_field.items():
            if field_name not in overrides:
                val = os.getenv(env_name)
                if val is not None:
                    overrides[field_name] = val

        return cls(**overrides)
