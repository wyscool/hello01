# ============================================================
# codebase_qa/config.py — AppConfig 数据类
# ============================================================
# 22 个配置字段，from_env() 工厂方法从环境变量读取。
# 参考 rag_kb/config.py 的模式。
# ============================================================

import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Codebase Q&A 全局配置。"""

    # --- Service ---
    host: str = "0.0.0.0"
    port: int = 8003

    # --- LLM ---
    llm_model: str = "claude-sonnet-4-6"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_max_retries: int = 2
    llm_timeout_seconds: float = 90.0

    # --- Rate Limiting ---
    rate_limit_per_minute: int = 30

    # --- Embedding ---
    embedding_model: str = "BAAI/bge-m3"

    # --- ChromaDB ---
    chroma_persist_dir: str = "./codebase_qa/chroma_db"
    collection_name: str = "codebase_main"

    # --- Index ---
    project_dirs: str = "."           # 逗号分隔的索引目录列表
    exclude_patterns: str = "tests,venv,.git,__pycache__,node_modules,build,dist"

    # --- Retrieval ---
    top_k: int = 5
    min_score: float = 0.3
    use_mmr: bool = True
    mmr_lambda: float = 0.7

    # --- Cache ---
    cache_enabled: bool = True
    cache_ttl_seconds: float = 300.0
    cache_max_size: int = 1000

    @classmethod
    def from_env(cls) -> "AppConfig":
        """从环境变量读取全部配置，未设置的字段使用默认值。"""
        overrides: dict = {}

        env_map: list[tuple[str, str, type]] = [
            ("HOST", "host", str),
            ("PORT", "port", int),
            ("LLM_MODEL", "llm_model", str),
            ("LLM_MAX_RETRIES", "llm_max_retries", int),
            ("LLM_TIMEOUT", "llm_timeout_seconds", float),
            ("RATE_LIMIT", "rate_limit_per_minute", int),
            ("EMBEDDING_MODEL", "embedding_model", str),
            ("CHROMA_PERSIST_DIR", "chroma_persist_dir", str),
            ("COLLECTION_NAME", "collection_name", str),
            ("PROJECT_DIRS", "project_dirs", str),
            ("EXCLUDE_PATTERNS", "exclude_patterns", str),
            ("TOP_K", "top_k", int),
            ("MIN_SCORE", "min_score", float),
            ("USE_MMR", "use_mmr", cls._parse_bool),
            ("MMR_LAMBDA", "mmr_lambda", float),
            ("CACHE_ENABLED", "cache_enabled", cls._parse_bool),
            ("CACHE_TTL", "cache_ttl_seconds", float),
            ("CACHE_MAX_SIZE", "cache_max_size", int),
        ]

        for env_name, field_name, type_fn in env_map:
            value = os.getenv(env_name)
            if value is not None:
                overrides[field_name] = type_fn(value)

        # Aliases: Anthropic 变量名 → 内部字段名
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            overrides["llm_api_key"] = api_key

        base_url = os.getenv("ANTHROPIC_BASE_URL")
        if base_url:
            overrides["llm_base_url"] = base_url

        return cls(**overrides)

    @staticmethod
    def _parse_bool(value: str) -> bool:
        return value.lower() in ("1", "true", "yes", "on")

    @property
    def exclude_set(self) -> set[str]:
        """排除目录名 → set，用于 walk_directory 快速查找。"""
        return {d.strip() for d in self.exclude_patterns.split(",") if d.strip()}

    @property
    def project_dir_list(self) -> list[str]:
        """索引目录列表。"""
        return [d.strip() for d in self.project_dirs.split(",") if d.strip()]
