# ============================================================
# deploy/config.py — 配置中心 (12-Factor App 风格)
# ============================================================
# 所有配置从环境变量读取，无硬编码。
# 类比 Java: Spring @ConfigurationProperties + application.yml

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AppConfig:
    """应用配置 — 统一管理所有参数。"""

    # --- 服务 ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- LLM ---
    llm_model: str = "claude-sonnet-4-6"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_max_retries: int = 3
    llm_timeout_seconds: float = 60.0

    # --- 并发控制 ---
    max_concurrent_llm: int = 5
    rate_limit_per_minute: int = 30

    # --- Token 预算 ---
    token_daily_budget: int = 1_000_000

    # --- 缓存 ---
    cache_enabled: bool = True
    cache_ttl_seconds: float = 300.0
    semantic_cache_threshold: float = 0.85

    # --- 项目 ---
    project_root: str = ""  # 文件系统工具的安全边界

    @classmethod
    def from_env(cls) -> "AppConfig":
        """从环境变量加载配置。"""
        project_root = os.getenv(
            "PROJECT_ROOT",
            str(Path(__file__).parent.parent.resolve()),
        )
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            llm_model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
            llm_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            llm_base_url=os.getenv("ANTHROPIC_BASE_URL", ""),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT", "60")),
            max_concurrent_llm=int(os.getenv("MAX_CONCURRENT_LLM", "5")),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT", "30")),
            token_daily_budget=int(os.getenv("TOKEN_DAILY_BUDGET", "1000000")),
            cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            cache_ttl_seconds=float(os.getenv("CACHE_TTL", "300")),
            semantic_cache_threshold=float(
                os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.85")
            ),
            project_root=project_root,
        )
