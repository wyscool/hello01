# ============================================================
# log_analyzer/tests/test_parser.py — LogParser 测试
# ============================================================

import tempfile
import pytest
from pathlib import Path

from log_analyzer.parser import LogParser, _RE_STD_LOG


def _write_temp(name: str, content: str) -> Path:
    p = Path(tempfile.gettempdir()) / f"test_log_{name}"
    p.write_text(content, encoding="utf-8")
    return p


# ============================================================
# detect_format
# ============================================================

class TestDetectFormat:
    def test_detect_json_lines(self):
        parser = LogParser()
        fmt = parser.detect_format([
            '{"timestamp": "2024-01-01T00:00:00Z", "level": "INFO", "message": "hello"}',
        ])
        assert fmt == "json"

    def test_detect_structured(self):
        parser = LogParser()
        fmt = parser.detect_format([
            "2024-01-15 10:30:45 ERROR [main] com.example.App: something wrong",
        ])
        assert fmt == "structured"

    def test_detect_plain_text(self):
        parser = LogParser()
        fmt = parser.detect_format(["some random log message"])
        assert fmt == "plain"

    def test_detect_empty_lines(self):
        parser = LogParser()
        assert parser.detect_format([]) == "plain"


# ============================================================
# parse_file — structured logs
# ============================================================

class TestParseStructured:
    @pytest.fixture
    def parser(self):
        return LogParser()

    def test_basic_parsing(self, parser):
        content = (
            "2024-01-15 10:30:45 ERROR [main] com.example.App: something broke\n"
            "2024-01-15 10:30:46 INFO  [pool-1] com.example.DB: connected\n"
            "2024-01-15 10:30:47 WARN  Disk usage at 85%\n"
        )
        p = _write_temp("structured.log", content)
        entries = parser.parse_file(p)

        assert len(entries) == 3
        assert entries[0].level == "ERROR"
        assert entries[0].message == "something broke"
        assert entries[0].metadata["thread"] == "main"
        assert entries[0].metadata["logger"] == "com.example.App"
        assert entries[0].source == p.name

    def test_timestamps_parsed_correctly(self, parser):
        content = "2024-01-15 10:30:45 ERROR something\n"
        p = _write_temp("ts.log", content)
        entries = parser.parse_file(p)
        ts = entries[0].timestamp
        assert ts is not None
        assert ts.year == 2024
        assert ts.month == 1
        assert ts.day == 15
        assert ts.hour == 10
        assert ts.minute == 30

    def test_milliseconds_dot_format(self, parser):
        content = "2024-06-17 08:03:10.123 ERROR timeout\n"
        p = _write_temp("ms.log", content)
        entries = parser.parse_file(p)
        assert entries[0].timestamp is not None
        assert entries[0].timestamp.microsecond == 123000

    def test_milliseconds_comma_format(self, parser):
        content = "2024-06-17 08:03:10,456 ERROR timeout\n"
        p = _write_temp("comma.log", content)
        entries = parser.parse_file(p)
        assert entries[0].timestamp is not None
        assert entries[0].timestamp.microsecond == 456000

    def test_minimal_format_no_thread_no_logger(self, parser):
        content = "2024-01-15 10:30:45 DEBUG just a debug message\n"
        p = _write_temp("minimal.log", content)
        entries = parser.parse_file(p)
        assert entries[0].level == "DEBUG"
        assert entries[0].message == "just a debug message"
        assert entries[0].metadata["thread"] == ""

    def test_logger_with_colon(self, parser):
        content = "2024-01-15 10:30:45 ERROR com.example.App: message\n"
        p = _write_temp("colon.log", content)
        entries = parser.parse_file(p)
        assert entries[0].metadata["logger"] == "com.example.App"

    def test_syslog_like_timestamp(self, parser):
        content = "Jan 15 10:30:45 ERROR something happened\n"
        p = _write_temp("syslog.log", content)
        entries = parser.parse_file(p)
        # syslog 时间戳不匹配 STD_LOG 开头，会走回退逻辑
        assert len(entries) == 1
        assert entries[0].level == "ERROR"

    def test_fatal_level(self, parser):
        content = "2024-01-15 10:30:45 FATAL system crash\n"
        p = _write_temp("fatal.log", content)
        entries = parser.parse_file(p)
        assert entries[0].level == "FATAL"

    def test_severe_level(self, parser):
        content = "2024-01-15 10:30:45 SEVERE critical\n"
        p = _write_temp("severe.log", content)
        entries = parser.parse_file(p)
        assert entries[0].level == "SEVERE"

    def test_warning_level(self, parser):
        content = "2024-01-15 10:30:45 WARNING caution\n"
        p = _write_temp("warning.log", content)
        entries = parser.parse_file(p)
        assert entries[0].level == "WARNING"

    def test_blank_lines_skipped(self, parser):
        content = "2024-01-15 10:30:45 ERROR a\n\n\n2024-01-15 10:30:46 ERROR b\n"
        p = _write_temp("blanks.log", content)
        entries = parser.parse_file(p)
        assert len(entries) == 2
        assert entries[0].line_number == 1
        assert entries[1].line_number == 4

    def test_file_not_found(self, parser):
        with pytest.raises(FileNotFoundError):
            parser.parse_file("/nonexistent/path/xyz.log")


# ============================================================
# parse_file — JSON lines
# ============================================================

class TestParseJsonLines:
    @pytest.fixture
    def parser(self):
        return LogParser()

    def test_basic_json_parsing(self, parser):
        content = (
            '{"timestamp": "2024-01-15T10:30:45Z", "level": "ERROR", "message": "disk full"}\n'
            '{"timestamp": "2024-01-15T10:30:46Z", "level": "INFO", "message": "ok"}\n'
        )
        p = _write_temp("json.log", content)
        entries = parser.parse_file(p)
        assert len(entries) == 2
        assert entries[0].level == "ERROR"
        assert entries[0].message == "disk full"
        assert entries[1].level == "INFO"

    def test_alt_field_names(self, parser):
        content = (
            '{"time": "2024-01-15T10:30:45", "severity": "WARN", "msg": "slow query"}\n'
        )
        p = _write_temp("alt.log", content)
        entries = parser.parse_file(p)
        assert entries[0].level == "WARN"
        assert entries[0].message == "slow query"

    def test_at_timestamp_field(self, parser):
        content = (
            '{"@timestamp": "2024-01-15T10:30:45Z", "level": "ERROR", "message": "fail"}\n'
        )
        p = _write_temp("at_ts.log", content)
        entries = parser.parse_file(p)
        assert entries[0].timestamp is not None

    def test_log_level_field(self, parser):
        content = (
            '{"timestamp": "2024-01-15T10:30:45Z", "log_level": "INFO", "text": "started"}\n'
        )
        p = _write_temp("log_level.log", content)
        entries = parser.parse_file(p)
        assert entries[0].level == "INFO"
        assert entries[0].message == "started"

    def test_extra_fields_in_metadata(self, parser):
        content = (
            '{"timestamp": "2024-01-15T10:30:45Z", "level": "ERROR", "message": "fail", "trace_id": "abc123", "user_id": 42}\n'
        )
        p = _write_temp("extra.log", content)
        entries = parser.parse_file(p)
        assert "trace_id" in entries[0].metadata
        assert entries[0].metadata["trace_id"] == "abc123"
        assert entries[0].metadata["user_id"] == 42

    def test_invalid_json_line_fallback(self, parser):
        content = (
            '{"valid": "json"}\n'
            'this is not json at all\n'
        )
        p = _write_temp("mixed.log", content)
        entries = parser.parse_file(p)
        assert len(entries) == 2
        assert entries[1].raw == "this is not json at all"


# ============================================================
# parse_file — plain text
# ============================================================

class TestParsePlainText:
    @pytest.fixture
    def parser(self):
        return LogParser()

    def test_all_levels_are_none(self, parser):
        content = "line one\nline two\nline three\n"
        p = _write_temp("plain.log", content)
        entries = parser.parse_file(p)
        assert len(entries) == 3
        assert all(e.level is None for e in entries)

    def test_level_detected_from_text(self, parser):
        content = "ERROR: something failed\nINFO: all good\njust a note\n"
        p = _write_temp("levels.log", content)
        entries = parser.parse_file(p)
        assert entries[0].level == "ERROR"
        assert entries[1].level == "INFO"
        assert entries[2].level is None


# ============================================================
# stack trace merging
# ============================================================

class TestMergeStackTraces:
    @pytest.fixture
    def parser(self):
        return LogParser()

    def test_java_stack_merged(self, parser):
        content = (
            "2024-01-15 10:30:45 ERROR [main] com.example.App: Connection timeout\n"
            "java.sql.SQLException: Timeout\n"
            "\tat com.example.db.connect(DB.java:42)\n"
            "\tat com.example.App.run(App.java:15)\n"
            "\t... 3 more\n"
            "2024-01-15 10:30:46 INFO  App started\n"
        )
        p = _write_temp("stack.log", content)
        entries = parser.parse_file(p)
        assert len(entries) == 2
        assert "java.sql.SQLException" in entries[0].raw
        assert "\tat " in entries[0].raw
        assert "Connection timeout" in entries[0].message
        assert entries[1].level == "INFO"

    def test_caused_by_merged(self, parser):
        content = (
            "2024-01-15 10:30:45 ERROR fail\n"
            "Caused by: java.io.IOException: disk full\n"
            "\tat com.example.Write.write(Write.java:10)\n"
        )
        p = _write_temp("caused.log", content)
        entries = parser.parse_file(p)
        assert len(entries) == 1
        assert "Caused by" in entries[0].raw

    def test_suppressed_merged(self, parser):
        content = (
            "2024-01-15 10:30:45 ERROR [main] fail\n"
            "Suppressed: java.io.IOException\n"
        )
        p = _write_temp("suppressed.log", content)
        entries = parser.parse_file(p)
        assert len(entries) == 1
        assert "Suppressed" in entries[0].raw

    def test_info_not_merged(self, parser):
        content = (
            "2024-01-15 10:30:45 INFO  Started\n"
            "   some indented detail\n"
            "2024-01-15 10:30:46 ERROR boom\n"
        )
        p = _write_temp("nomerge.log", content)
        entries = parser.parse_file(p)
        # INFO 后的缩进行不合并 (只合并 ERROR/FATAL/SEVERE/WARN 后的续行)
        assert len(entries) == 3
        assert entries[0].level == "INFO"
        assert entries[2].level == "ERROR"

    def test_debug_not_merged(self, parser):
        content = (
            "2024-01-15 10:30:45 DEBUG trace\n"
            "\tat some.code(S.java:1)\n"
        )
        p = _write_temp("dbg.log", content)
        entries = parser.parse_file(p)
        # DEBUG 后的堆栈行不合并
        assert len(entries) == 2


# ============================================================
# timestamp parsing
# ============================================================

class TestParseTimestamp:
    def test_iso_utc_z(self):
        ts = LogParser._parse_ts("2024-01-15T10:30:45Z")
        assert ts is not None
        assert ts.year == 2024
        assert ts.month == 1
        assert ts.day == 15
        assert ts.hour == 10

    def test_iso_with_microseconds(self):
        ts = LogParser._parse_ts("2024-01-15T10:30:45.123456Z")
        assert ts is not None
        assert ts.microsecond == 123456

    def test_iso_without_tz(self):
        ts = LogParser._parse_ts("2024-01-15T10:30:45")
        assert ts is not None
        assert ts.tzinfo is not None  # 默认 +08:00

    def test_standard_format(self):
        ts = LogParser._parse_ts("2024-01-15 10:30:45")
        assert ts is not None
        assert ts.hour == 10

    def test_slash_date_format(self):
        ts = LogParser._parse_ts("2024/01/15 10:30:45")
        assert ts is not None
        assert ts.day == 15

    def test_syslog_format(self):
        ts = LogParser._parse_ts("Jan 15 10:30:45")
        assert ts is not None
        assert ts.month == 1
        assert ts.day == 15

    def test_empty_string(self):
        assert LogParser._parse_ts("") is None

    def test_invalid_string(self):
        assert LogParser._parse_ts("not a timestamp at all") is None

    def test_none_input(self):
        assert LogParser._parse_ts(None) is None


# ============================================================
# _RE_STD_LOG regex
# ============================================================

class TestStdLogRegex:
    def test_full_match_all_fields(self):
        m = _RE_STD_LOG.match(
            "2024-01-15 10:30:45 ERROR [main] com.example.App: message text"
        )
        assert m is not None
        assert m.group("ts") == "2024-01-15 10:30:45"
        assert m.group("level") == "ERROR"
        assert m.group("thread") == "main"
        assert m.group("logger") == "com.example.App"
        assert m.group("msg") == "message text"

    def test_no_thread(self):
        m = _RE_STD_LOG.match("2024-01-15 10:30:45 WARN something here")
        assert m is not None
        assert m.group("level") == "WARN"

    def test_logger_hyphen_not_colon(self):
        m = _RE_STD_LOG.match("2024-01-15 10:30:45 INFO com.example.DB - connected to db")
        assert m is not None
        assert m.group("logger") == "com.example.DB"
        assert m.group("msg") == "connected to db"

    def test_no_match_for_plain_line(self):
        assert _RE_STD_LOG.match("just a random line") is None

    def test_no_match_for_json(self):
        assert _RE_STD_LOG.match(
            '{"timestamp": "2024-01-15T10:30:45Z", "level": "INFO"}'
        ) is None


# ============================================================
# encoding
# ============================================================

class TestEncoding:
    def test_gbk_encoding(self):
        content = "2024-01-15 10:30:45 ERROR gbk test\n"
        p = Path(tempfile.gettempdir()) / "test_gbk.log"
        p.write_text(content, encoding="gbk")
        try:
            parser = LogParser()
            entries = parser.parse_file(p)
            assert len(entries) == 1
        finally:
            p.unlink(missing_ok=True)


# ============================================================
# edge cases
# ============================================================

class TestEdgeCases:
    @pytest.fixture
    def parser(self):
        return LogParser()

    def test_empty_file(self, parser):
        p = _write_temp("empty.log", "")
        entries = parser.parse_file(p)
        assert entries == []

    def test_max_file_size_exceeded(self):
        parser = LogParser(max_file_size_mb=0)
        p = _write_temp("big.log", "x\n" * 100)
        with pytest.raises(ValueError, match="文件过大"):
            parser.parse_file(p)
