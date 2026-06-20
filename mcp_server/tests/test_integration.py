# ============================================================
# mcp_server/tests/test_integration.py — 端到端 MCP 集成测试
# ============================================================
# 使用官方 mcp SDK 的 ClientSession + stdio_client
# 模拟真实 MCP Host: 连接 → 列出工具 → 调用工具。
#
# 前提: 需要 codebase_qa/chroma_db/ 已经有索引数据。
#
# 类比 Java:
#   @SpringBootTest(webEnvironment=RANDOM_PORT) + TestRestTemplate
# ============================================================

import json
import pytest
import sys


@pytest.fixture
def server_params():
    from mcp import StdioServerParameters
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server"],
    )


class TestMCPIntegration:
    pytestmark = pytest.mark.asyncio
    """MCP 客户端集成测试 — 需完整的 lifespan（含 embedding 加载）。"""

    @pytest.mark.asyncio
    async def test_list_tools(self, server_params):
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                tool_names = {t.name for t in result.tools}
                assert "codebase_search" in tool_names
                assert "codebase_index" in tool_names
                assert "codebase_status" in tool_names

    @pytest.mark.asyncio
    async def test_codebase_status(self, server_params):
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("codebase_status", {})
                text = result.content[0].text
                data = json.loads(text)
                assert "service" in data
                assert "index" in data
                assert "config" in data

    @pytest.mark.asyncio
    async def test_codebase_search(self, server_params):
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "codebase_search",
                    {"query": "retry decorator", "top_k": 3},
                )
                text = result.content[0].text
                data = json.loads(text)
                # 即使没有索引数据，也不应返回 error
                assert "question" in data
