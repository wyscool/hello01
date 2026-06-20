# ============================================================
# mcp_server/config.py — McpServerConfig 数据类
# ============================================================
# 包装 codebase_qa 的 AppConfig，新增 MCP 专用字段。
# 参考 codebase_qa/config.py 的 from_env() 模式。
# ============================================================

import os
from dataclasses import dataclass, field

from codebase_qa.config import AppConfig


@dataclass
class McpServerConfig:
    """MCP Server 配置 — 在 AppConfig 基础上增加 MCP 专用属性。

    类比 Java: 相当于 @ConfigurationProperties 的扩展子类，
    在通用配置基础上增加 MCP server 特有的字段。
    """

    # --- MCP Server identity ---
    server_name: str = "codebase-qa"
    server_version: str = "0.1.0"

    # --- Transport ---
    transport: str = "stdio"

    # --- 继承 codebase_qa 的全部配置 ---
    qa: AppConfig = field(default_factory=AppConfig)

    @classmethod
    def from_env(cls) -> "McpServerConfig":
        import os
        qa = AppConfig.from_env()
        return cls(
            server_name=os.getenv("MCP_SERVER_NAME", "codebase-qa"),
            server_version=os.getenv("MCP_SERVER_VERSION", "0.1.0"),
            transport=os.getenv("MCP_TRANSPORT", "stdio"),
            qa=qa,
        )
