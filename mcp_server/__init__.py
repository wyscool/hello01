# ============================================================
# mcp_server — Codebase QA MCP Server
# ============================================================
# 使用官方 mcp Python SDK (FastMCP) 把 codebase_qa 的能力
# 包装成标准 MCP Server，通过 stdio 与 MCP Host 通信。
#
# 三个 MCP 工具:
#   codebase_search — 自然语言搜索代码库
#   codebase_index  — 索引代码目录
#   codebase_status — 查询索引状态
#
# 类比 Java: 把 Spring Boot REST 服务重新包装成 gRPC 服务
#            — 同样的业务逻辑, 不同的通信协议。
#
# 用法:
#   python -m mcp_server
# ============================================================
