# ============================================================
# log_analyzer/tests/test_agent.py — LogAnalysisAgent 测试
# ============================================================

import tempfile
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

from log_analyzer.parser import LogEntry
from log_analyzer.agent import LogAnalysisAgent, AnalysisResult, SYSTEM_PROMPT
from log_analyzer.config import AppConfig
from log_analyzer.tools import create_log_server

TZ = timezone(timedelta(hours=8))


def _make_entry(line_no: int, level: str, message: str, ts: str = "",
                source: str = "test.log") -> LogEntry:
    timestamp = None
    if ts:
        timestamp = datetime.fromisoformat(ts)
    return LogEntry(
        line_number=line_no, timestamp=timestamp, level=level,
        message=message, raw=f"{ts} {level} {message}".strip(),
        source=source,
    )


@pytest.fixture
def sample_entries():
    return [
        _make_entry(1, "INFO",  "Started", "2024-06-17T08:00:00", "test.log"),
        _make_entry(2, "ERROR", "Connection timeout",
                    "2024-06-17T08:03:00", "test.log"),
        _make_entry(3, "WARN",  "Disk 85%", "2024-06-17T08:03:01", "test.log"),
        _make_entry(4, "ERROR", "OOM", "2024-06-17T08:03:25", "test.log"),
        _make_entry(5, "INFO",  "Shutdown", "2024-06-17T08:10:00", "test.log"),
    ]


# ============================================================
# Agent 离线模式
# ============================================================

class TestOfflineMode:
    def test_offline_answer_structure(self, sample_entries, monkeypatch):
        """LLM 不可用时返回离线分析。"""
        from deploy.agent_core import LlmClient

        # 创建一个健康的 LlmClient (触发健康检查)
        # 但 monkeypatch 掉 is_healthy 为 False
        llm = LlmClient(api_key="sk-fake", max_retries=1, timeout=1.0)
        monkeypatch.setattr(llm, "_healthy", False)

        config = AppConfig()
        config.agent_max_iterations = 2  # 加速测试

        agent = LogAnalysisAgent(llm, sample_entries, config)
        result = agent.analyze("分析错误")

        assert isinstance(result, AnalysisResult)
        assert result.iterations == 1
        assert "[离线模式]" in result.answer
        assert result.total_entries == 5


# ============================================================
# AnalysisResult
# ============================================================

class TestAnalysisResult:
    def test_result_fields(self):
        result = AnalysisResult(
            question="test question",
            answer="test answer",
            iterations=3,
            latency_ms=1500.0,
            file_name="test.log",
            total_entries=100,
        )
        assert result.question == "test question"
        assert result.answer == "test answer"
        assert result.iterations == 3
        assert result.latency_ms == 1500.0
        assert result.file_name == "test.log"
        assert result.total_entries == 100


# ============================================================
# System prompt
# ============================================================

class TestSystemPrompt:
    def test_contains_key_instructions(self):
        assert "SRE" in SYSTEM_PROMPT or "日志分析" in SYSTEM_PROMPT
        assert "stats" in SYSTEM_PROMPT
        assert "top_errors" in SYSTEM_PROMPT
        assert "search" in SYSTEM_PROMPT
        assert "根因" in SYSTEM_PROMPT


# ============================================================
# _build_user_prompt
# ============================================================

class TestBuildUserPrompt:
    def test_includes_summary(self, sample_entries):
        from deploy.agent_core import LlmClient
        llm = LlmClient(api_key="sk-fake", max_retries=1, timeout=1.0)
        agent = LogAnalysisAgent(llm, sample_entries, AppConfig())

        prompt = agent._build_user_prompt("分析日志", "test.log")
        assert "test.log" in prompt
        assert "总行数: 5" in prompt
        assert "ERROR" in prompt
        assert "分析任务: 分析日志" in prompt

    def test_level_distribution_in_prompt(self, sample_entries):
        from deploy.agent_core import LlmClient
        llm = LlmClient(api_key="sk-fake", max_retries=1, timeout=1.0)
        agent = LogAnalysisAgent(llm, sample_entries, AppConfig())

        prompt = agent._build_user_prompt("test", "f.log")
        assert "ERROR: 2" in prompt
        assert "WARN: 1" in prompt
        assert "INFO: 2" in prompt


# ============================================================
# tool_names
# ============================================================

class TestToolNames:
    def test_registers_all_7_tools(self, sample_entries):
        from deploy.agent_core import LlmClient
        llm = LlmClient(api_key="sk-fake", max_retries=1, timeout=1.0)
        agent = LogAnalysisAgent(llm, sample_entries, AppConfig())

        names = agent.tool_names
        assert len(names) == 7
        assert "stats" in names
        assert "search" in names
        assert "timeline" in names
        assert "top_errors" in names


# ============================================================
# create_log_server (via agent)
# ============================================================

class TestAgentServer:
    def test_agent_mcp_works(self, sample_entries):
        from deploy.agent_core import LlmClient
        llm = LlmClient(api_key="sk-fake", max_retries=1, timeout=1.0)
        agent = LogAnalysisAgent(llm, sample_entries, AppConfig())

        # 通过 agent._mcp_client 直接调用工具
        result = agent._mcp_client.call_tool("stats", {})
        assert result["total_lines"] == 5
        assert result["by_level"]["ERROR"] == 2


# ============================================================
# create_log_agent factory
# ============================================================

class TestFactory:
    def test_creates_agent_from_file(self):
        content = (
            "2024-06-17 08:00:00 INFO  Started\n"
            "2024-06-17 08:03:00 ERROR Timeout\n"
            "2024-06-17 08:03:01 WARN  Disk 85%\n"
        )
        import tempfile
        p = Path(tempfile.gettempdir()) / "test_factory.log"
        p.write_text(content, encoding="utf-8")
        try:
            config = AppConfig()
            config.agent_max_iterations = 1
            from log_analyzer.agent import create_log_agent
            agent = create_log_agent(p, config)
            assert len(agent.entries) == 3
            assert agent.tool_names == [
                "search", "count", "sample", "timeline",
                "top_errors", "get_section", "stats",
            ]
        finally:
            p.unlink(missing_ok=True)
