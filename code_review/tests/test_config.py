# ============================================================
# code_review/tests/test_config.py — AppConfig 测试
# ============================================================

import os
import pytest
from code_review.config import AppConfig


class TestAppConfig:
    def test_defaults(self):
        config = AppConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8001
        assert config.llm_model == "claude-sonnet-4-6"
        assert config.llm_max_retries == 2
        assert config.llm_timeout_seconds == 90.0

    def test_from_env_override(self):
        os.environ["PORT"] = "9999"
        os.environ["LLM_MODEL"] = "test-model"
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        os.environ["ANTHROPIC_BASE_URL"] = "https://test.api"
        os.environ["LLM_MAX_RETRIES"] = "5"
        os.environ["LLM_TIMEOUT"] = "120"
        try:
            config = AppConfig.from_env()
            assert config.port == 9999
            assert config.llm_model == "test-model"
            assert config.llm_api_key == "sk-test"
            assert config.llm_base_url == "https://test.api"
            assert config.llm_max_retries == 5
            assert config.llm_timeout_seconds == 120.0
        finally:
            for k in ["PORT", "LLM_MODEL", "ANTHROPIC_API_KEY",
                      "ANTHROPIC_BASE_URL", "LLM_MAX_RETRIES", "LLM_TIMEOUT"]:
                os.environ.pop(k, None)

    def test_project_root_default(self):
        config = AppConfig.from_env()
        assert config.project_root != ""
        assert "hello01" in config.project_root
