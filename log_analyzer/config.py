# ============================================================
# log_analyzer/config.py — AppConfig
# ============================================================
# 配置统一入口，from_env() 从环境变量/.env 读取。
# ============================================================

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    """日志分析 Agent 配置。"""

    # --- 服务 ---
    host: str = "0.0.0.0"
    port: int = 8003

    # --- LLM ---
    llm_model: str = "claude-sonnet-4-6"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_max_retries: int = 3
    llm_timeout_seconds: float = 60.0

    # --- Agent ---
    agent_max_iterations: int = 10        # ReAct 最大轮次
    agent_max_tokens: int = 4096          # LLM 输出 token
    agent_temperature: float = 0.0

    # --- 日志解析 ---
    max_file_size_mb: int = 50           # 单文件上限
    max_context_lines: int = 10           # 搜索上下文行数
    max_sample_size: int = 20            # 采样最大条数

    # --- 速率 ---
    rate_limit_per_minute: int = 30

    # --- 缓存 ---
    cache_enabled: bool = True
    cache_ttl_seconds: int = 600         # 分析结果缓存 10 分钟
    cache_max_size: int = 200

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8003")),

            llm_model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
            llm_api_key=os.getenv("ANTHROPIC_API_KEY", os.getenv("LLM_API_KEY", "")),
            llm_base_url=os.getenv("LLM_BASE_URL", ""),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT", "60")),

            agent_max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "10")),
            agent_max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "4096")),
            agent_temperature=float(os.getenv("AGENT_TEMPERATURE", "0.0")),

            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "50")),
            max_context_lines=int(os.getenv("MAX_CONTEXT_LINES", "10")),
            max_sample_size=int(os.getenv("MAX_SAMPLE_SIZE", "20")),

            rate_limit_per_minute=int(os.getenv("RATE_LIMIT", "30")),

            cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            cache_ttl_seconds=int(os.getenv("CACHE_TTL", "600")),
            cache_max_size=int(os.getenv("CACHE_MAX_SIZE", "200")),
        )
