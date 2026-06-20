# ============================================================
# deploy/tests/test_config.py — AppConfig 测试
# ============================================================

import os
import pytest
from deploy.config import AppConfig


class TestAppConfig:
    def test_defaults(self):
        config = AppConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.llm_model == "claude-sonnet-4-6"
        assert config.llm_api_key == ""
        assert config.llm_base_url == ""
        assert config.llm_max_retries == 3
        assert config.llm_timeout_seconds == 60.0
        assert config.max_concurrent_llm == 5
        assert config.rate_limit_per_minute == 30
        assert config.token_daily_budget == 1_000_000
        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 300.0
        assert config.semantic_cache_threshold == 0.85
        assert config.project_root == ""

    def test_from_env_override(self):
        os.environ["PORT"] = "9999"
        os.environ["LLM_MODEL"] = "test-model"
        os.environ["ANTHROPIC_API_KEY"] = "sk-test-key"
        os.environ["ANTHROPIC_BASE_URL"] = "https://test.api.com"
        os.environ["LLM_MAX_RETRIES"] = "5"
        os.environ["LLM_TIMEOUT"] = "120"
        os.environ["MAX_CONCURRENT_LLM"] = "10"
        os.environ["RATE_LIMIT"] = "60"
        os.environ["TOKEN_DAILY_BUDGET"] = "500000"
        os.environ["CACHE_ENABLED"] = "false"
        os.environ["CACHE_TTL"] = "600"
        os.environ["SEMANTIC_CACHE_THRESHOLD"] = "0.9"
        try:
            config = AppConfig.from_env()
            assert config.port == 9999
            assert config.llm_model == "test-model"
            assert config.llm_api_key == "sk-test-key"
            assert config.llm_base_url == "https://test.api.com"
            assert config.llm_max_retries == 5
            assert config.llm_timeout_seconds == 120.0
            assert config.max_concurrent_llm == 10
            assert config.rate_limit_per_minute == 60
            assert config.token_daily_budget == 500000
            assert config.cache_enabled is False
            assert config.cache_ttl_seconds == 600.0
            assert config.semantic_cache_threshold == 0.9
        finally:
            for k in ["PORT", "LLM_MODEL", "ANTHROPIC_API_KEY",
                      "ANTHROPIC_BASE_URL", "LLM_MAX_RETRIES", "LLM_TIMEOUT",
                      "MAX_CONCURRENT_LLM", "RATE_LIMIT", "TOKEN_DAILY_BUDGET",
                      "CACHE_ENABLED", "CACHE_TTL", "SEMANTIC_CACHE_THRESHOLD"]:
                os.environ.pop(k, None)

    def test_bool_parsing(self):
        """cache_enabled 的 bool 解析 (lower + == 'true')。"""
        for true_val in ["true", "True", "TRUE"]:
            os.environ["CACHE_ENABLED"] = true_val
            try:
                c = AppConfig.from_env()
                assert c.cache_enabled is True
            finally:
                os.environ.pop("CACHE_ENABLED", None)

        for false_val in ["false", "False", "no", "0", ""]:
            os.environ["CACHE_ENABLED"] = false_val
            try:
                c = AppConfig.from_env()
                assert c.cache_enabled is False
            finally:
                os.environ.pop("CACHE_ENABLED", None)
