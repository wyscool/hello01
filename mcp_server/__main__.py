# ============================================================
# mcp_server/__main__.py — MCP Server 入口点
# ============================================================
# 用法: python -m mcp_server
#
# 在 stdio 模式下启动 MCP Server:
#   stdin  → 接收 MCP Host 的 JSON-RPC 请求
#   stdout → 发送 JSON-RPC 响应回 MCP Host
#   stderr → 服务端日志 (不影响协议通信)
#
# 类比 Java: 相当于 SpringApplication.run()。
# ============================================================

import sys
from pathlib import Path

from dotenv import load_dotenv

# 加载 .env 文件
# 注意: print 必须写 stderr, stdin/stdout 被 MCP 协议占用
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"  Loaded: {env_path}", file=sys.stderr)

from mcp_server.server import mcp  # noqa: E402

if __name__ == "__main__":
    print("  MCP Server starting on stdio...", file=sys.stderr)
    print("  Tools: codebase_search, codebase_index, codebase_status",
          file=sys.stderr)
    mcp.run(transport="stdio")
