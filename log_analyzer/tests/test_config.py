# ============================================================
# log_analyzer/tests/test_config.py — AppConfig 测试
# ============================================================

import os
import pytest
from log_analyzer.config import AppConfig


class TestAppConfigDefaults:
    def test_defaults(self):
        config = AppConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8003
        assert config.llm_model == "claude-sonnet-4-6"
        assert config.agent_max_iterations == 10
        assert config.max_file_size_mb == 50
        assert config.rate_limit_per_minute == 30
        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 600
        assert config.max_context_lines == 10
        assert config.max_sample_size == 20

    def test_from_env_empty(self):
        """无环境变量时使用默认值。"""
        config = AppConfig.from_env()
        assert config.port == 8003
        assert config.llm_model == "claude-sonnet-4-6"

    def test_from_env_override(self):
        """环境变量覆盖默认值。"""
        os.environ["PORT"] = "9999"
        os.environ["LLM_MODEL"] = "test-model"
        os.environ["ANTHROPIC_API_KEY"] = "sk-test-key"
        os.environ["CACHE_ENABLED"] = "false"
        os.environ["CACHE_TTL"] = "1200"

        try:
            config = AppConfig.from_env()
            assert config.port == 9999
            assert config.llm_model == "test-model"
            assert config.llm_api_key == "sk-test-key"
            assert config.cache_enabled is False
            assert config.cache_ttl_seconds == 1200
        finally:
            del os.environ["PORT"]
            del os.environ["LLM_MODEL"]
            del os.environ["ANTHROPIC_API_KEY"]
            del os.environ["CACHE_ENABLED"]
            del os.environ["CACHE_TTL"]

    def test_llm_api_key_fallback(self):
        """ANTHROPIC_API_KEY 未设置时回退到 LLM_API_KEY。"""
        os.environ["LLM_API_KEY"] = "sk-fallback"
        try:
            config = AppConfig.from_env()
            assert config.llm_api_key == "sk-fallback"
        finally:
            del os.environ["LLM_API_KEY"]

    def test_anthropic_api_key_priority(self):
        """ANTHROPIC_API_KEY 优先于 LLM_API_KEY。"""
        os.environ["ANTHROPIC_API_KEY"] = "sk-primary"
        os.environ["LLM_API_KEY"] = "sk-secondary"
        try:
            config = AppConfig.from_env()
            assert config.llm_api_key == "sk-primary"
        finally:
            del os.environ["ANTHROPIC_API_KEY"]
            del os.environ["LLM_API_KEY"]
