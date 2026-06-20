# ============================================================
# mcp_server/tests/test_server.py — FastMCP 初始化测试
# ============================================================
# 测试:
#   - Server 创建成功
#   - 三个工具正确注册
#   - 工具 schema 完整 (含参数定义)
#
# 类比 Java:
#   @SpringBootTest 验证 Bean 注册和配置完整性。
# ============================================================

import json
import pytest
import asyncio

from mcp_server.server import create_server


@pytest.fixture(scope="module")
def server():
    """Create the server once for all tests in this module."""
    return create_server()


class TestServerCreation:
    pytestmark = pytest.mark.asyncio

    async def test_server_name(self, server):
        assert server.name == "codebase-qa"

    async def test_tools_registered(self, server):
        tools = await server.list_tools()
        tool_names = {t.name for t in tools}
        assert "codebase_search" in tool_names
        assert "codebase_index" in tool_names
        assert "codebase_status" in tool_names
        assert len(tools) == 3

    async def test_codebase_search_schema(self, server):
        tools = await server.list_tools()
        tool = None
        for t in tools:
            if t.name == "codebase_search":
                tool = t
                break
        assert tool is not None, "codebase_search tool not found"

        properties = tool.inputSchema.get("properties", {})
        required = tool.inputSchema.get("required", [])

        assert "query" in properties
        assert "query" in required
        assert properties["query"]["type"] == "string"

        assert "top_k" in properties
        assert properties["top_k"]["type"] == "integer"

        assert "filter_type" in properties
        assert properties["filter_type"]["type"] == "string"

    async def test_codebase_index_schema(self, server):
        tools = await server.list_tools()
        tool = None
        for t in tools:
            if t.name == "codebase_index":
                tool = t
                break
        assert tool is not None, "codebase_index tool not found"

        properties = tool.inputSchema.get("properties", {})
        required = tool.inputSchema.get("required", [])

        assert "dirs" in properties
        assert "dirs" in required
        assert properties["dirs"]["type"] == "array"

    async def test_codebase_status_schema(self, server):
        tools = await server.list_tools()
        tool = None
        for t in tools:
            if t.name == "codebase_status":
                tool = t
                break
        assert tool is not None, "codebase_status tool not found"

        # status 工具没有必需参数
        properties = tool.inputSchema.get("properties", {})
        # 应该没有 properties 或为空
        assert isinstance(properties, dict)
