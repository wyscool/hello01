# ============================================================
# code-review/config.py — 配置中心
# ============================================================

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8001
    llm_model: str = "claude-sonnet-4-6"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_max_retries: int = 2
    llm_timeout_seconds: float = 90.0
    project_root: str = ""

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8001")),
            llm_model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
            llm_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            llm_base_url=os.getenv("ANTHROPIC_BASE_URL", ""),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT", "90")),
            project_root=os.getenv(
                "PROJECT_ROOT",
                str(Path(__file__).parent.parent.resolve()),
            ),
        )
