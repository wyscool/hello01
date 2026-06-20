# ============================================================
# code_review/tests/test_agent.py — ReviewAgent + MCP 协议测试
# ============================================================

import json
import pytest
from unittest.mock import Mock, patch, PropertyMock

from code_review.agent import (
    LlmClient, MCPServer, MCPClient, MCPResponse,
    ReviewAgent, create_mcp_server, create_review_agent,
)
from code_review.tools import (
    tool_check_style, tool_detect_patterns, tool_read_file,
)


# ============================================================
# 测试用样本
# ============================================================

JAVA_SAMPLE = """\
public class UserService {
    private Connection conn;
    public String GetUser(String id) {
        Statement stmt = conn.createStatement();
        return stmt.executeQuery("SELECT * FROM users WHERE id=" + id).toString();
    }
}
"""


# ============================================================
# MCPServer
# ============================================================

class TestMCPServer:
    @pytest.fixture
    def server(self):
        server = MCPServer("test")
        server.register(
            "greet", "打招呼",
            {"name": {"type": "string", "description": "名字"}},
            ["name"],
            lambda name: {"msg": f"Hello, {name}"},
        )
        return server

    def test_register_and_count(self, server):
        assert server.tool_count == 1

    def test_handle_tools_list(self, server):
        resp = server.handle(json.dumps({
            "method": "tools/list", "params": {}, "id": 1,
        }))
        data = json.loads(resp)
        assert data["result"]["tools"][0]["name"] == "greet"

    def test_handle_tools_call_success(self, server):
        resp = server.handle(json.dumps({
            "method": "tools/call",
            "params": {"name": "greet", "arguments": {"name": "World"}},
            "id": 1,
        }))
        data = json.loads(resp)
        text = data["result"]["content"][0]["text"]
        assert "Hello, World" in text

    def test_handle_tools_call_unknown(self, server):
        resp = server.handle(json.dumps({
            "method": "tools/call",
            "params": {"name": "unknown", "arguments": {}},
            "id": 1,
        }))
        data = json.loads(resp)
        assert "error" in data
        assert "未知工具" in data["error"]["message"]

    def test_handle_initialize(self, server):
        resp = server.handle(json.dumps({
            "method": "initialize", "params": {}, "id": 1,
        }))
        data = json.loads(resp)
        assert data["result"]["protocolVersion"] == "0.2"
        assert data["result"]["serverInfo"]["name"] == "test"

    def test_handle_unknown_method(self, server):
        resp = server.handle(json.dumps({
            "method": "unknown", "params": {}, "id": 1,
        }))
        data = json.loads(resp)
        assert "error" in data

    def test_handle_invalid_json(self, server):
        resp = server.handle("not json")
        data = json.loads(resp)
        assert data["error"]["message"] == "Invalid JSON"

    def test_handle_tool_execution_error(self, server):
        server.register(
            "failing", "会失败的工具",
            {}, [],
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        resp = server.handle(json.dumps({
            "method": "tools/call",
            "params": {"name": "failing", "arguments": {}},
            "id": 1,
        }))
        data = json.loads(resp)
        assert "error" in data
        assert "执行失败" in data["error"]["message"]


# ============================================================
# MCPClient
# ============================================================

class TestMCPClient:
    @pytest.fixture
    def server(self):
        server = MCPServer("test")
        server.register(
            "check", "检查工具",
            {"input": {"type": "string"}},
            ["input"],
            lambda input: {"status": "ok", "input": input},
        )
        return server

    def test_unconnected_client_has_error(self):
        client = MCPClient()
        resp = client._send("tools/list")
        assert resp.error is not None

    def test_connect_and_list_tools(self, server):
        client = MCPClient()
        client.connect(server)
        tools = client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "check"

    def test_call_tool(self, server):
        client = MCPClient()
        client.connect(server)
        result = client.call_tool("check", {"input": "test"})
        assert result == {"status": "ok", "input": "test"}


# ============================================================
# MCPResponse
# ============================================================

class TestMCPResponse:
    def test_success_to_json(self):
        resp = MCPResponse(result={"key": "value"}, id=1)
        data = json.loads(resp.to_json())
        assert data["result"]["key"] == "value"

    def test_error_to_json(self):
        resp = MCPResponse(error="something wrong", id=2)
        data = json.loads(resp.to_json())
        assert data["error"]["message"] == "something wrong"

    def test_from_json_success(self):
        raw = json.dumps({"result": {"x": 1}, "id": 3})
        resp = MCPResponse.from_json(raw)
        assert resp.result == {"x": 1}
        assert resp.id == 3

    def test_from_json_error(self):
        raw = json.dumps({"error": {"message": "fail"}, "id": 4})
        resp = MCPResponse.from_json(raw)
        assert resp.error == "fail"
        assert resp.id == 4


# ============================================================
# create_mcp_server
# ============================================================

class TestCreateMcpServer:
    def test_registers_3_tools(self):
        server = create_mcp_server()
        assert server.tool_count == 3
        tools = json.loads(server.handle(
            json.dumps({"method": "tools/list", "params": {}, "id": 1})
        ))["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {"check_style", "detect_patterns", "read_file"}


# ============================================================
# ReviewAgent offline
# ============================================================

class TestReviewAgentOffline:
    @pytest.fixture
    def agent(self):
        llm = LlmClient(api_key="sk-fake", max_retries=1, timeout=1.0)
        llm._healthy = False
        server = create_mcp_server()
        client = MCPClient()
        client.connect(server)
        return ReviewAgent(client, llm)

    def test_offline_review_returns_structured(self, agent):
        result = agent.review(JAVA_SAMPLE, language="java")
        assert "summary" in result
        assert result["summary"] == "[离线模式] 无法进行 LLM 审查"
        assert result["findings"] == []
        assert result["score"] == 0


# ============================================================
# ReviewAgent._parse_json
# ============================================================

class TestParseJson:
    def test_clean_json(self):
        agent = create_review_agent.__wrapped__ if hasattr(
            create_review_agent, '__wrapped__'
        ) else None
        # 直接测试方法
        from code_review.agent import ReviewAgent
        agent = ReviewAgent(Mock(), Mock())
        result = agent._parse_json('{"goal": "test", "steps": []}')
        assert result == {"goal": "test", "steps": []}

    def test_json_with_code_fence(self):
        from code_review.agent import ReviewAgent
        agent = ReviewAgent(Mock(), Mock())
        result = agent._parse_json(
            'Some text\n```json\n{"goal": "review", "steps": [{"id": 1}]}\n```\nMore text'
        )
        assert result == {"goal": "review", "steps": [{"id": 1}]}

    def test_invalid_json_returns_plan(self):
        from code_review.agent import ReviewAgent
        agent = ReviewAgent(Mock(), Mock())
        result = agent._parse_json("this is not json at all")
        assert isinstance(result, dict)
        assert "goal" in result
        assert len(result["steps"]) >= 1

    def test_non_json_text_returns_summary(self):
        from code_review.agent import ReviewAgent
        agent = ReviewAgent(Mock(), Mock())
        result = agent._parse_json(
            '{"summary": "all good", "findings": []}'
        )
        assert result["summary"] == "all good"
        assert result["findings"] == []


# ============================================================
# LlmClient
# ============================================================

class TestLlmClient:
    def test_creates_with_defaults(self):
        llm = LlmClient()
        assert llm.model == "claude-sonnet-4-6"

    def test_is_healthy_caches(self):
        llm = LlmClient(api_key="sk-fake")
        llm._healthy = True
        assert llm.is_healthy is True

    def test_get_text_from_message(self):
        msg = Mock()
        msg.content = [
            Mock(type="text", text="Hello"),
            Mock(type="text", text="World"),
        ]
        result = LlmClient.get_text(msg)
        assert result == "Hello\nWorld"

    def test_get_tool_uses(self):
        msg = Mock()
        tool_use = Mock(type="tool_use")
        msg.content = [
            Mock(type="text", text="ok"),
            tool_use,
        ]
        result = LlmClient.get_tool_uses(msg)
        assert len(result) == 1
        assert result[0] is tool_use


# ============================================================
# create_review_agent factory
# ============================================================

class TestCreateReviewAgent:
    def test_returns_agent(self):
        config = None
        from code_review.config import AppConfig
        config = AppConfig()
        agent = create_review_agent(config)
        assert isinstance(agent, ReviewAgent)
        assert agent.mcp.server.tool_count == 3
