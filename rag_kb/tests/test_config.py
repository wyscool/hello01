# ============================================================
# rag_kb/tests/test_config.py — AppConfig 测试
# ============================================================

import os
import pytest
from rag_kb.config import AppConfig


class TestAppConfig:
    def test_defaults(self):
        config = AppConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8002
        assert config.llm_model == "claude-sonnet-4-6"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.retrieval_top_k == 5
        assert config.retrieval_min_score == 0.3
        assert config.use_mmr is True
        assert config.mmr_lambda == 0.7
        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 300
        assert config.cache_max_size == 1000
        assert config.max_ingest_file_mb == 10
        assert config.rate_limit_per_minute == 30

    def test_allowed_suffixes(self):
        config = AppConfig()
        assert ".txt" in config.allowed_suffixes
        assert ".md" in config.allowed_suffixes
        assert ".py" in config.allowed_suffixes
        assert ".java" in config.allowed_suffixes

    def test_from_env_override(self):
        os.environ["PORT"] = "9999"
        os.environ["LLM_MODEL"] = "test-model"
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        os.environ["EMBEDDING_MODEL"] = "test-embed"
        os.environ["CHUNK_SIZE"] = "1000"
        os.environ["RETRIEVAL_TOP_K"] = "10"
        os.environ["CACHE_ENABLED"] = "false"
        os.environ["CACHE_TTL_SECONDS"] = "600"
        try:
            config = AppConfig.from_env()
            assert config.port == 9999
            assert config.llm_model == "test-model"
            assert config.llm_api_key == "sk-test"
            assert config.embedding_model == "test-embed"
            assert config.chunk_size == 1000
            assert config.retrieval_top_k == 10
            assert config.cache_enabled is False
            assert config.cache_ttl_seconds == 600.0
        finally:
            for k in ["PORT", "LLM_MODEL", "ANTHROPIC_API_KEY",
                      "EMBEDDING_MODEL", "CHUNK_SIZE", "RETRIEVAL_TOP_K",
                      "CACHE_ENABLED", "CACHE_TTL_SECONDS"]:
                os.environ.pop(k, None)
