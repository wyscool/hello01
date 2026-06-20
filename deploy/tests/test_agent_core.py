# ============================================================
# deploy/tests/test_agent_core.py — Agent 核心组件测试
# ============================================================

import json
import time
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from deploy.agent_core import (
    MCPResponse, MCPServer, MCPClient, TokenBudget,
    DevAssistant, LlmClient,
    tool_calculate, tool_get_current_time, tool_text_stats,
    tool_read_file, tool_list_files,
    create_mcp_server, create_agent,
)


# ============================================================
# MCPResponse
# ============================================================

class TestMCPResponse:
    def test_result_to_json(self):
        resp = MCPResponse(result={"tools": []}, id=1)
        data = json.loads(resp.to_json())
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert data["result"] == {"tools": []}
        assert "error" not in data

    def test_error_to_json(self):
        resp = MCPResponse(error="something wrong", id=2)
        data = json.loads(resp.to_json())
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 2
        assert data["error"]["code"] == -1
        assert data["error"]["message"] == "something wrong"

    def test_from_json_result(self):
        raw = json.dumps({"jsonrpc": "2.0", "id": 3, "result": {"ok": True}})
        resp = MCPResponse.from_json(raw)
        assert resp.result == {"ok": True}
        assert resp.error is None
        assert resp.id == 3

    def test_from_json_error(self):
        raw = json.dumps({"jsonrpc": "2.0", "id": 4,
                         "error": {"code": -1, "message": "fail"}})
        resp = MCPResponse.from_json(raw)
        assert resp.error == "fail"
        assert resp.result is None


# ============================================================
# MCPServer
# ============================================================

class TestMCPServer:
    @pytest.fixture
    def server(self):
        s = MCPServer("test-server")
        s.register(
            "greet", "Say hello",
            {"name": {"type": "string"}}, ["name"],
            lambda name: f"Hello, {name}",
        )
        return s

    def test_tool_count(self, server):
        assert server.tool_count == 1

    def test_tool_names(self, server):
        assert server.tool_names == ["greet"]

    def test_fluent_register(self, server):
        result = server.register(
            "extra", "desc",
            {}, [], lambda: "ok",
        )
        assert result is server
        assert server.tool_count == 2

    def test_handle_initialize(self, server):
        req = json.dumps({
            "jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1,
        })
        resp = json.loads(server.handle(req))
        assert resp["result"]["protocolVersion"] == "0.2"
        assert resp["result"]["serverInfo"]["name"] == "test-server"

    def test_handle_tools_list(self, server):
        req = json.dumps({
            "jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 2,
        })
        resp = json.loads(server.handle(req))
        tools = resp["result"]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "greet"

    def test_handle_tools_call_success(self, server):
        req = json.dumps({
            "jsonrpc": "2.0", "method": "tools/call",
            "params": {"name": "greet", "arguments": {"name": "World"}},
            "id": 3,
        })
        resp = json.loads(server.handle(req))
        content = resp["result"]["content"][0]["text"]
        result = json.loads(content)
        assert result == "Hello, World"

    def test_handle_tools_call_unknown(self, server):
        req = json.dumps({
            "jsonrpc": "2.0", "method": "tools/call",
            "params": {"name": "nonexistent"}, "id": 4,
        })
        resp = json.loads(server.handle(req))
        assert "未知工具" in resp["error"]["message"]

    def test_handle_tools_call_handler_exception(self, server):
        server.register("bad", "throws", {}, [], lambda: (_ for _ in ()).throw(ValueError("boom")))
        req = json.dumps({
            "jsonrpc": "2.0", "method": "tools/call",
            "params": {"name": "bad", "arguments": {}}, "id": 5,
        })
        resp = json.loads(server.handle(req))
        assert "执行失败" in resp["error"]["message"]

    def test_handle_unknown_method(self, server):
        req = json.dumps({
            "jsonrpc": "2.0", "method": "unknown", "params": {}, "id": 6,
        })
        resp = json.loads(server.handle(req))
        assert "未知方法" in resp["error"]["message"]

    def test_handle_invalid_json(self, server):
        resp = json.loads(server.handle("not json"))
        assert resp["error"]["message"] == "Invalid JSON"


# ============================================================
# MCPClient
# ============================================================

class TestMCPClient:
    @pytest.fixture
    def server(self):
        s = MCPServer("test")
        s.register("add", "Add numbers",
                   {"a": {"type": "integer"}, "b": {"type": "integer"}},
                   ["a", "b"], lambda a, b: a + b)
        return s

    @pytest.fixture
    def client(self, server):
        c = MCPClient()
        c.connect(server)
        return c

    def test_connect_sets_connected(self, server):
        c = MCPClient()
        c.connect(server)
        assert c.connected is True

    def test_connect_returns_server_info(self, server):
        c = MCPClient()
        result = c.connect(server)
        assert result["serverInfo"]["name"] == "test"

    def test_not_connected_error(self):
        c = MCPClient()
        resp = c._send("tools/list")
        assert resp.error == "未连接"

    def test_list_tools(self, client):
        tools = client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "add"
        assert "input_schema" in tools[0]

    def test_call_tool(self, client):
        result = client.call_tool("add", {"a": 3, "b": 4})
        assert result == 7

    def test_call_tool_unknown(self, client):
        result = client.call_tool("no", {})
        assert "error" in result

    def test_call_tool_text_fallback(self):
        """handle 返回 text 但不是 JSON 时应作为 text 返回。"""
        s = MCPServer("t")
        s.register("text", "d", {}, [],
                   lambda: "plain text not json")
        c = MCPClient()
        c.connect(s)
        result = c.call_tool("text", {})
        # handler 返回的 "plain text not json" 被 MCPResponse 序列化成
        # {"result": {"content": [{"type": "text", "text": "\"plain text not json\""}]}}
        # 然后 call_tool 用 json.loads 解析 text → "plain text not json"
        assert result == "plain text not json"

    def test_list_tools_not_connected(self):
        c = MCPClient()
        assert c.list_tools() == []


# ============================================================
# TokenBudget
# ============================================================

class TestTokenBudget:
    def test_estimate(self):
        tb = TokenBudget(max_tokens=10000)
        msgs = [{"role": "user", "content": "hello" * 100}]
        est = tb.estimate(msgs)
        assert est > 0

    def test_check_normal(self):
        tb = TokenBudget(max_tokens=10000)
        msgs = [{"role": "user", "content": "hi"}]
        ok, info = tb.check(msgs)
        assert ok is True
        assert "正常" in info

    def test_check_warning(self):
        tb = TokenBudget(max_tokens=1000)
        msgs = [{"role": "user", "content": "x" * 3000}]  # ~750 tokens
        ok, info = tb.check(msgs)
        assert ok is True
        assert "高" in info

    def test_check_exceeded(self):
        tb = TokenBudget(max_tokens=100)
        msgs = [{"role": "user", "content": "x" * 500}]  # ~125 tokens
        ok, info = tb.check(msgs)
        assert ok is False
        assert "超出" in info

    def test_used_updated(self):
        tb = TokenBudget()
        msgs = [{"role": "user", "content": "x" * 400}]
        tb.check(msgs)
        assert tb.used > 0


# ============================================================
# Tool Functions
# ============================================================

class TestToolCalculate:
    def test_basic_arithmetic(self):
        r = tool_calculate("2 + 3 * 4")
        assert r["result"] == 14

    def test_sqrt(self):
        r = tool_calculate("sqrt(144)")
        assert r["result"] == 12.0

    def test_trig(self):
        r = tool_calculate("sin(0)")
        assert r["result"] == 0.0

    def test_constants(self):
        r = tool_calculate("pi")
        assert r["result"] == pytest.approx(3.14159, abs=0.001)

    def test_dangerous_eval_blocked(self):
        with pytest.raises(Exception):
            tool_calculate("__import__('os').system('ls')")


class TestToolGetCurrentTime:
    def test_returns_fields(self):
        r = tool_get_current_time()
        assert "datetime" in r
        assert "weekday" in r
        assert "timestamp" in r

    def test_weekday_is_string(self):
        r = tool_get_current_time()
        assert r["weekday"] in ["一", "二", "三", "四", "五", "六", "日"]


class TestToolTextStats:
    def test_counts(self):
        r = tool_text_stats("hello world\nfoo")
        assert r["chars"] == 15
        assert r["words"] == 3
        assert r["lines"] == 2


class TestToolReadFile:
    def test_reads_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = str(Path(d))
            p = Path(d) / "test.py"
            p.write_text("print('hello')\nprint('world')\n", encoding="utf-8")
            result = tool_read_file(str(p), project_root=root)
            assert "error" not in result
            assert result["total_lines"] == 3
            assert "print" in result["content"]

    def test_max_lines_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            root = str(Path(d))
            p = Path(d) / "big.txt"
            p.write_text("\n".join([f"line {i}" for i in range(100)]))
            result = tool_read_file(str(p), max_lines=30, project_root=root)
            assert result["preview_lines"] == 30
            assert result["total_lines"] == 100

    def test_file_not_found(self):
        with tempfile.TemporaryDirectory() as d:
            result = tool_read_file(str(Path(d) / "no.txt"), project_root=str(d))
            assert "不存在" in result["error"]

    def test_outside_project(self):
        with tempfile.TemporaryDirectory() as d:
            result = tool_read_file("/etc/passwd", project_root=str(d))
            assert "安全限制" in result["error"]

    def test_directory_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            result = tool_read_file(d, project_root=d)
            assert "目录" in result["error"]

    def test_unsupported_type(self):
        with tempfile.TemporaryDirectory() as d:
            root = str(Path(d))
            p = Path(d) / "file.exe"
            p.write_text("binary", encoding="utf-8")
            result = tool_read_file(str(p), project_root=root)
            assert "不支持" in result["error"]


class TestToolListFiles:
    def test_lists_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root = str(Path(d))
            (Path(d) / "a.py").write_text("")
            (Path(d) / "b.txt").write_text("")
            (Path(d) / "sub").mkdir()
            result = tool_list_files(d, project_root=root)
            assert result["item_count"] == 3
            names = [i["name"] for i in result["items"]]
            assert "a.py" in names
            assert "b.txt" in names
            assert "sub" in names

    def test_path_alias(self):
        with tempfile.TemporaryDirectory() as d:
            root = str(Path(d))
            (Path(d) / "x.py").write_text("")
            result = tool_list_files(path=d, project_root=root)
            assert result["item_count"] == 1

    def test_not_a_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root = str(Path(d))
            p = Path(d) / "file.txt"
            p.write_text("")
            result = tool_list_files(str(p), project_root=root)
            assert "不是目录" in result["error"]

    def test_outside_project(self):
        with tempfile.TemporaryDirectory() as d:
            result = tool_list_files("/etc", project_root=str(d))
            assert "安全限制" in result["error"]

    def test_nonexistent(self):
        with tempfile.TemporaryDirectory() as d:
            inner = str(Path(d) / "no_dir")
            result = tool_list_files(inner, project_root=str(d))
            assert "不存在" in result["error"]


# ============================================================
# Factory Functions
# ============================================================

class TestCreateMcpServer:
    def test_creates_with_all_tools(self):
        server = create_mcp_server(name="test", project_root=".")
        assert server.tool_count == 5
        assert "calculate" in server.tool_names
        assert "get_current_time" in server.tool_names
        assert "text_stats" in server.tool_names
        assert "read_file" in server.tool_names
        assert "list_files" in server.tool_names

    def test_read_file_bound_with_project_root(self):
        """project_root 通过 partial 绑定到 tool_read_file。"""
        # 调用 handle 确认工具能正常执行
        server = create_mcp_server(project_root="/tmp")
        req = json.dumps({
            "jsonrpc": "2.0", "method": "tools/call",
            "params": {"name": "read_file",
                      "arguments": {"path": "/etc/passwd"}},
            "id": 1,
        })
        resp = json.loads(server.handle(req))
        content = resp["result"]["content"][0]["text"]
        result = json.loads(content)
        assert "安全限制" in result.get("error", "")


# ============================================================
# DevAssistant (offline / unit-testable parts)
# ============================================================

class TestDevAssistantOffline:
    @pytest.fixture
    def mock_llm(self):
        llm = Mock(spec=LlmClient)
        llm.is_healthy = False  # 离线模式
        return llm

    @pytest.fixture
    def mock_mcp(self):
        mcp = Mock(spec=MCPClient)
        mcp.list_tools.return_value = []
        return mcp

    @pytest.fixture
    def agent(self, mock_mcp, mock_llm):
        return DevAssistant(mock_mcp, mock_llm)

    def test_quick_mode_offline(self, agent):
        result = agent.ask("测试任务", mode="quick")
        assert "[离线模式]" in result["answer"]
        assert result["iterations"] == 0
        assert result["mode"] == "quick"

    def test_plan_mode_offline(self, agent):
        result = agent.ask("复杂任务", mode="plan")
        assert "[离线模式]" in result["answer"]
        assert result["iterations"] == 0
        assert result["mode"] == "plan"

    def test_parse_plan_extracts_json(self, agent):
        goal, steps = agent._parse_plan(
            '{"goal": "分析", "steps": [{"id": 1, "description": "步骤1",'
            '"tool": "none", "params": {}, "depends_on": []}]}'
        )
        assert goal == "分析"
        assert len(steps) == 1
        assert steps[0]["id"] == 1

    def test_parse_plan_with_code_fence(self, agent):
        goal, steps = agent._parse_plan(
            '```json\n{"goal": "测试", "steps": []}\n```'
        )
        assert goal == "测试"

    def test_parse_plan_invalid_json(self, agent):
        goal, steps = agent._parse_plan("not json at all")
        assert len(steps) == 1  # fallback step
        assert steps[0]["tool"] == "none"

    def test_build_quick_prompt_no_history(self, agent):
        result = agent._build_quick_prompt("新任务")
        assert result == "新任务"

    def test_build_quick_prompt_with_history(self, agent):
        agent.history = [
            {"role": "user", "content": "上一步"},
            {"role": "assistant", "content": "上一步结果"},
        ]
        prompt = agent._build_quick_prompt("当前任务")
        assert "上一步" in prompt
        assert "当前任务" in prompt
        assert "对话历史" in prompt

    def test_default_mode(self, agent):
        assert agent.default_mode == "quick"
