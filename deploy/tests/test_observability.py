# ============================================================
# deploy/tests/test_observability.py — 可观测性组件测试
# ============================================================

import io
import json
import time
import pytest
from unittest.mock import Mock

from deploy.observability import (
    JsonLogger, Span, Trace, TokenMonitor,
)


# ============================================================
# JsonLogger
# ============================================================

class TestJsonLogger:
    @pytest.fixture
    def buf(self):
        return io.StringIO()

    @pytest.fixture
    def logger(self, buf):
        return JsonLogger("test-logger", min_level="DEBUG", output=buf)

    def _read_records(self, buf):
        buf.seek(0)
        return [json.loads(line) for line in buf.getvalue().strip().split("\n") if line]

    def test_logs_json_structure(self, logger, buf):
        logger.info("hello", user="test")
        records = self._read_records(buf)
        assert len(records) == 1
        r = records[0]
        assert r["level"] == "INFO"
        assert r["logger"] == "test-logger"
        assert r["message"] == "hello"
        assert r["user"] == "test"
        assert "timestamp" in r

    def test_level_filtering(self, buf):
        logger = JsonLogger("test", min_level="WARN", output=buf)
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warn("warn msg")
        logger.error("error msg")
        records = self._read_records(buf)
        levels = [r["level"] for r in records]
        assert "DEBUG" not in levels
        assert "INFO" not in levels
        assert "WARN" in levels
        assert "ERROR" in levels

    def test_all_levels(self, logger, buf):
        logger.debug("d")
        logger.info("i")
        logger.warn("w")
        logger.error("e")
        records = self._read_records(buf)
        assert len(records) == 4
        assert [r["level"] for r in records] == ["DEBUG", "INFO", "WARN", "ERROR"]

    def test_context_merging(self, logger, buf):
        logger.info("task done", duration_ms=150, status="ok")
        records = self._read_records(buf)
        r = records[0]
        assert r["duration_ms"] == 150
        assert r["status"] == "ok"

    def test_min_level_case_insensitive(self, buf):
        logger = JsonLogger("t", min_level="info", output=buf)
        logger.debug("no")
        logger.info("yes")
        records = self._read_records(buf)
        assert len(records) == 1

    def test_invalid_level_defaults_to_debug(self, buf):
        logger = JsonLogger("t", min_level="INVALID", output=buf)
        logger.debug("msg")
        records = self._read_records(buf)
        assert len(records) == 1


# ============================================================
# Span
# ============================================================

class TestSpan:
    def test_initial_state(self):
        s = Span(name="test", span_id="s1")
        assert s.name == "test"
        assert s.span_id == "s1"
        assert s.parent_id == ""
        assert s.status == "ok"

    def test_duration_ms(self):
        s = Span(name="t", span_id="1", start_time=100.0, end_time=100.5)
        assert s.duration_ms == 500.0

    def test_add_event(self):
        s = Span(name="t", span_id="1")
        s.add_event("step1", elapsed=10)
        s.add_event("step2", elapsed=20)
        assert len(s.events) == 2
        assert s.events[0]["name"] == "step1"
        assert s.events[0]["elapsed"] == 10


# ============================================================
# Trace
# ============================================================

class TestTrace:
    def test_creates_with_auto_request_id(self):
        t = Trace("my-trace")
        assert t.name == "my-trace"
        assert t.request_id.startswith("req-")
        assert len(t.spans) == 0

    def test_uses_given_request_id(self):
        t = Trace("t", request_id="abc123")
        assert t.request_id == "abc123"

    def test_start_span(self):
        t = Trace("t")
        s = t.start_span("sub-task", key="val")
        assert s.name == "sub-task"
        assert s.span_id.startswith(t.request_id + "-")
        assert s.metadata["key"] == "val"
        assert len(t.spans) == 1

    def test_start_span_with_parent(self):
        t = Trace("t")
        parent = t.start_span("parent")
        child = t.start_span("child", parent=parent)
        assert child.parent_id == parent.span_id

    def test_end_span(self):
        t = Trace("t")
        s = t.start_span("work")
        time.sleep(0.01)
        t.end_span(s, status="ok", result="done")
        assert s.end_time > s.start_time
        assert s.status == "ok"
        assert s.metadata["result"] == "done"

    def test_span_id_increments(self):
        t = Trace("t", request_id="r1")
        s1 = t.start_span("a")
        s2 = t.start_span("b")
        assert s1.span_id == "r1-1"
        assert s2.span_id == "r1-2"

    def test_duration_ms(self):
        t = Trace("t")
        time.sleep(0.01)
        assert t.duration_ms > 0

    def test_to_dict(self):
        t = Trace("test-trace", request_id="xyz")
        s = t.start_span("compute")
        t.end_span(s)
        d = t.to_dict()
        assert d["trace_name"] == "test-trace"
        assert d["request_id"] == "xyz"
        assert d["span_count"] == 1
        assert len(d["spans"]) == 1
        assert d["spans"][0]["name"] == "compute"
        assert "duration_ms" in d["spans"][0]
        assert "status" in d["spans"][0]

    def test_empty_trace_to_dict(self):
        t = Trace("empty")
        d = t.to_dict()
        assert d["span_count"] == 0
        assert d["spans"] == []


# ============================================================
# TokenMonitor
# ============================================================

class TestTokenMonitor:
    @pytest.fixture
    def tm(self):
        return TokenMonitor(daily_budget=100000)

    def test_record_and_total(self, tm):
        tm.record("claude-sonnet", 500, 200)
        tm.record("claude-sonnet", 300, 100)
        assert tm.total_used == 1100
        assert tm.call_count == 2

    def test_budget_remaining(self, tm):
        tm.record("m", 10000, 5000)
        assert tm.budget_remaining == 85000

    def test_budget_exhausted(self, tm):
        tm.record("m", 100000, 0)
        assert tm.budget_remaining == 0

    def test_budget_pct(self, tm):
        tm.record("m", 25000, 0)
        assert tm.budget_pct == pytest.approx(25.0)

    def test_by_model(self, tm):
        tm.record("sonnet", 100, 50)
        tm.record("haiku", 200, 100)
        tm.record("sonnet", 300, 150)
        by_model = tm.by_model()
        assert by_model["sonnet"] == 600
        assert by_model["haiku"] == 300

    def test_snapshot(self, tm):
        tm.record("sonnet", 1000, 500)
        snap = tm.snapshot()
        assert snap["total_used"] == 1500
        assert snap["call_count"] == 1
        assert snap["budget_pct"] == 1.5
        assert "by_model" in snap

    def test_record_truncates_task(self, tm):
        tm.record("m", 10, 10, task="x" * 100)
        assert len(tm._usage[0]["task"]) == 50

    def test_record_optional_fields(self, tm):
        tm.record("m", 10, 10, agent="DevAssistant", task="test task")
        u = tm._usage[0]
        assert u["agent"] == "DevAssistant"
        assert u["task"] == "test task"
