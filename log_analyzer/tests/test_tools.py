# ============================================================
# log_analyzer/tests/test_tools.py — 日志分析工具测试
# ============================================================

import pytest
from datetime import datetime, timezone, timedelta

from log_analyzer.parser import LogEntry
from log_analyzer.tools import (
    tool_search, tool_count, tool_sample, tool_timeline,
    tool_top_errors, tool_get_section, tool_stats,
    create_log_server,
)

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


# 共享测试数据
@pytest.fixture
def sample_entries():
    return [
        _make_entry(1, "INFO",  "Application started",
                    "2024-06-17T08:00:01", "app.log"),
        _make_entry(2, "INFO",  "Database connected to db01",
                    "2024-06-17T08:00:02", "app.log"),
        _make_entry(3, "ERROR", "Connection timeout after 5000ms",
                    "2024-06-17T08:03:10", "app.log"),
        _make_entry(4, "ERROR", "OutOfMemoryError: Java heap space",
                    "2024-06-17T08:03:25", "app.log"),
        _make_entry(5, "WARN",  "Disk usage at 85%",
                    "2024-06-17T08:03:26", "app.log"),
        _make_entry(6, "ERROR", "Connection timeout after 5000ms",
                    "2024-06-17T08:03:30", "app.log"),
        _make_entry(7, "ERROR", "disk space full",
                    "2024-06-17T08:14:00", "app.log"),
        _make_entry(8, "ERROR", "disk space full",
                    "2024-06-17T08:14:01", "app.log"),
        _make_entry(9, "ERROR", "disk space full",
                    "2024-06-17T08:14:02", "app.log"),
        _make_entry(10, "INFO", "Health check: OK",
                    "2024-06-17T08:15:00", "app.log"),
    ]


# ============================================================
# tool_search
# ============================================================

class TestSearch:
    def test_finds_keyword(self, sample_entries):
        result = tool_search(sample_entries, "timeout")
        assert result["total_matches"] == 2
        assert all("timeout" in m["text"].lower() for m in result["matches"])

    def test_case_insensitive(self, sample_entries):
        result = tool_search(sample_entries, "TIMEOUT")
        assert result["total_matches"] == 2

    def test_no_match(self, sample_entries):
        result = tool_search(sample_entries, "nonexistent")
        assert result["total_matches"] == 0
        assert result["matches"] == []

    def test_context_lines(self, sample_entries):
        result = tool_search(sample_entries, "OutOfMemory", context_lines=2)
        assert result["total_matches"] == 1
        m = result["matches"][0]
        # 上下文应包含前后各 2 行
        assert " | " in m["context"]

    def test_truncation(self, sample_entries):
        # 超过 30 条匹配时截断
        result = tool_search(sample_entries, "d", context_lines=0)
        if result["total_matches"] > 30:
            assert result["truncated"]


# ============================================================
# tool_count
# ============================================================

class TestCount:
    def test_total_count(self, sample_entries):
        result = tool_count(sample_entries)
        assert result["total"] == 10
        assert result["total_all"] == 10

    def test_filter_by_level(self, sample_entries):
        result = tool_count(sample_entries, level="ERROR")
        assert result["total"] == 6
        assert result["by_level"].get("ERROR", 0) == 6

    def test_filter_by_keyword(self, sample_entries):
        result = tool_count(sample_entries, keyword="disk")
        assert result["total"] == 4  # "Disk usage" + 3x "disk space full"

    def test_filter_level_and_keyword(self, sample_entries):
        result = tool_count(sample_entries, level="ERROR", keyword="disk")
        assert result["total"] == 3  # 3x "disk space full"

    def test_error_rate(self, sample_entries):
        result = tool_count(sample_entries)
        assert "error_rate" in result
        assert "%" in result["error_rate"]

    def test_filter_no_match(self, sample_entries):
        result = tool_count(sample_entries, level="DEBUG")
        assert result["total"] == 0


# ============================================================
# tool_sample
# ============================================================

class TestSample:
    def test_returns_last_n(self, sample_entries):
        result = tool_sample(sample_entries, n=3)
        assert result["sample_count"] == 3
        # 返回最近 3 条 (行号最大)
        lines = [s["line"] for s in result["samples"]]
        assert lines == [10, 9, 8]

    def test_filter_by_level(self, sample_entries):
        result = tool_sample(sample_entries, n=10, level="ERROR")
        assert result["sample_count"] == 6

    def test_n_larger_than_total(self, sample_entries):
        result = tool_sample(sample_entries, n=100)
        assert result["sample_count"] == 10

    def test_empty_entries(self):
        result = tool_sample([], n=10)
        assert result["sample_count"] == 0
        assert result["samples"] == []


# ============================================================
# tool_timeline
# ============================================================

class TestTimeline:
    def test_hour_interval(self, sample_entries):
        result = tool_timeline(sample_entries, interval="hour")
        assert result["interval"] == "hour"
        assert result["bucket_count"] == 1  # 都在同一小时

    def test_minute_interval(self, sample_entries):
        result = tool_timeline(sample_entries, interval="minute")
        assert result["bucket_count"] >= 1
        for b in result["buckets"]:
            assert "total" in b
            assert "error" in b

    def test_entries_without_timestamps(self):
        entries = [
            _make_entry(1, "ERROR", "no timestamp"),
        ]
        result = tool_timeline(entries, interval="minute")
        assert result["entries_with_ts"] == 0

    def test_empty(self):
        result = tool_timeline([], interval="hour")
        assert result["bucket_count"] == 0


# ============================================================
# tool_top_errors
# ============================================================

class TestTopErrors:
    def test_clusters_same_pattern(self, sample_entries):
        result = tool_top_errors(sample_entries, n=5)
        assert result["total_errors"] == 6
        # "disk space full" x3 应该聚合在一起
        disk_item = next(
            (e for e in result["top_errors"] if "disk" in e["pattern"]), None
        )
        assert disk_item is not None
        assert disk_item["count"] == 3

    def test_normalizes_numbers(self, sample_entries):
        result = tool_top_errors(sample_entries, n=10)
        # "Connection timeout after Nms" — 数字 N 被归一化
        conn_item = next(
            (e for e in result["top_errors"] if "timeout" in e["pattern"]), None
        )
        assert conn_item is not None
        assert conn_item["count"] == 2
        assert "5000" not in conn_item["pattern"]  # 数字被替换为 N

    def test_no_errors(self):
        entries = [
            _make_entry(1, "INFO", "all good"),
        ]
        result = tool_top_errors(entries, n=5)
        assert result["total_errors"] == 0
        assert result["top_errors"] == []

    def test_empty_entries(self):
        result = tool_top_errors([], n=5)
        assert result["total_errors"] == 0


# ============================================================
# tool_get_section
# ============================================================

class TestGetSection:
    def test_returns_specified_range(self, sample_entries):
        result = tool_get_section(sample_entries, start_line=3, end_line=5)
        assert result["total_lines"] == 3

    def test_start_equals_end(self, sample_entries):
        result = tool_get_section(sample_entries, start_line=5, end_line=5)
        assert result["total_lines"] == 1

    def test_out_of_range(self, sample_entries):
        result = tool_get_section(sample_entries, start_line=100, end_line=200)
        assert result["total_lines"] == 0

    def test_start_greater_than_end(self, sample_entries):
        result = tool_get_section(sample_entries, start_line=10, end_line=1)
        assert result["total_lines"] == 0


# ============================================================
# tool_stats
# ============================================================

class TestStats:
    def test_stats_overview(self, sample_entries):
        result = tool_stats(sample_entries)
        assert result["total_lines"] == 10
        assert result["by_level"]["ERROR"] == 6
        assert result["by_level"]["INFO"] == 3
        assert result["by_level"]["WARN"] == 1
        assert result["time_range"]["start"] != ""
        assert result["time_range"]["end"] != ""
        assert result["sources"]["app.log"] == 10

    def test_error_rate(self, sample_entries):
        result = tool_stats(sample_entries)
        assert "error_rate" in result
        assert result["error_rate"] == "60.0%"

    def test_empty_entries(self):
        result = tool_stats([])
        assert result["total_lines"] == 0
        assert result["by_level"] == {}
        assert result["time_range"]["start"] == ""
        assert result["entries_with_ts"] == 0


# ============================================================
# create_log_server
# ============================================================

class TestCreateLogServer:
    def test_creates_server_with_all_tools(self, sample_entries):
        server = create_log_server(sample_entries)
        assert server.tool_count == 7
        expected = {"search", "count", "sample", "timeline",
                    "top_errors", "get_section", "stats"}
        assert set(server.tool_names) == expected

    def test_tools_are_callable(self, sample_entries):
        server = create_log_server(sample_entries)
        from deploy.agent_core import MCPClient
        client = MCPClient()
        client.connect(server)

        result = client.call_tool("stats", {})
        assert result["total_lines"] == 10

        result = client.call_tool("search", {"keyword": "timeout"})
        assert result["total_matches"] == 2

    def test_custom_max_params(self, sample_entries):
        server = create_log_server(
            sample_entries, max_context=5, max_sample=3
        )
        from deploy.agent_core import MCPClient
        client = MCPClient()
        client.connect(server)

        result = client.call_tool("sample", {"n": 3})
        assert result["sample_count"] == 3
