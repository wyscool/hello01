# ============================================================
# log_analyzer/parser.py — 多格式日志解析器
# ============================================================
# 支持格式:
#   1. JSON lines:  {"timestamp": "...", "level": "ERROR", ...}
#   2. 标准日志:    2024-01-15 10:30:45 ERROR [thread] msg
#   3. Java 异常栈: 多行堆栈跟踪自动合并
#   4. 纯文本:      每行作为一条消息
#
# 类比 Java:
#   LogEntry  ≈ POJO
#   LogParser ≈ Log4j PatternLayout 解析器 + 多行 StackTrace 合并
# ============================================================

import re
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

TZ = timezone(timedelta(hours=8))

# 时间戳正则 (常见格式)
_RE_TS = re.compile(
    r"(?P<ts>"
    r"\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[,.]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    r"|"
    r"[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"  # syslog: Jan 15 10:30:45
    r")"
)

# 日志级别正则
_RE_LEVEL = re.compile(
    r"\b(?P<level>"
    r"FATAL|CRITICAL|ERROR|SEVERE|WARN(?:ING)?|INFO|DEBUG|TRACE|FINE[R]?|FINEST"
    r")\b",
    re.IGNORECASE,
)

# 标准日志格式: <ts> <level> [<thread>] [<logger>] <msg>
_RE_STD_LOG = re.compile(
    r"^"
    r"(?P<ts>\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[,.]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s*"
    r"(?P<level>FATAL|CRITICAL|ERROR|SEVERE|WARN(?:ING)?|INFO|DEBUG|TRACE|FINE[R]?|FINEST)\s*"
    r"(?:\[(?P<thread>[^\]]*)\]\s*)?"
    r"(?:(?P<logger>[\w.]+)\s*[:-]\s*)?"
    r"(?P<msg>.*)",
    re.IGNORECASE,
)

# 堆栈跟踪续行: 空白行 / at ... / Caused by / Suppressed / ... N more / 异常类名
_RE_STACK_CONT = re.compile(
    r"^(\s+|"
    r"at\s+|"
    r"\.{3}\s+\d+\s+(?:more|common)|"
    r"Caused\s+by:|"
    r"Suppressed:|"
    r"\w+(?:\.\w+)+\w*Exception:|"
    r"\w+(?:\.\w+)+\w*Error:)"
)


@dataclass
class LogEntry:
    """单条日志记录。"""
    line_number: int
    timestamp: datetime | None
    level: str | None           # ERROR / WARN / INFO / DEBUG / None (未知)
    message: str
    raw: str                    # 原始行内容
    source: str = ""            # 文件名
    metadata: dict = field(default_factory=dict)


class LogParser:
    """多格式日志文件解析器。

    自动检测格式 + 合并多行异常栈，返回 LogEntry 列表。

    用法:
        parser = LogParser()
        entries = parser.parse_file("app.log")
        print(f"解析 {len(entries)} 条日志")
    """

    def __init__(self, max_file_size_mb: int = 50):
        self.max_bytes = max_file_size_mb * 1024 * 1024

    # --- 公开 API ---

    def parse_file(self, file_path: Path | str) -> list[LogEntry]:
        """解析日志文件 → LogEntry 列表。"""
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if p.stat().st_size > self.max_bytes:
            raise ValueError(
                f"文件过大: {p.stat().st_size / 1024 / 1024:.1f}MB > "
                f"{self.max_bytes / 1024 / 1024:.0f}MB"
            )

        lines = self._read_lines(p)
        fmt = self.detect_format(lines)
        entries = self._parse_lines(lines, p.name, fmt)
        return self._merge_stack_traces(entries)

    def detect_format(self, lines: list[str]) -> str:
        """检测日志格式: json / structured / plain。"""
        if not lines:
            return "plain"
        sample = "".join(lines[:20])

        # JSON lines: 第一行是 JSON 对象
        if sample.strip().startswith("{"):
            try:
                json.loads(lines[0].strip())
                return "json"
            except (json.JSONDecodeError, IndexError):
                pass

        # 标准日志: 有时间戳 + 级别
        if _RE_STD_LOG.match(sample.split("\n")[0]):
            return "structured"

        return "plain"

    # --- 内部 ---

    def _read_lines(self, file_path: Path) -> list[str]:
        """读取文件行，自动检测编码。"""
        for enc in ("utf-8", "gbk", "latin-1"):
            try:
                return file_path.read_text(encoding=enc).split("\n")
            except (UnicodeDecodeError, UnicodeError):
                continue
        return []

    def _parse_lines(self, lines: list[str], source: str,
                     fmt: str) -> list[LogEntry]:
        """逐行解析。"""
        if fmt == "json":
            return self._parse_json_lines(lines, source)
        if fmt == "structured":
            return self._parse_structured_lines(lines, source)
        return self._parse_plain_lines(lines, source)

    def _parse_json_lines(self, lines: list[str], source: str) -> list[LogEntry]:
        entries: list[LogEntry] = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                entries.append(LogEntry(
                    line_number=i, timestamp=None, level=None,
                    message=line, raw=line, source=source,
                ))
                continue

            ts = self._parse_ts(obj.get("timestamp", obj.get("time", obj.get("@timestamp", ""))))
            level = obj.get("level", obj.get("severity", obj.get("log_level")))
            msg = obj.get("message", obj.get("msg", obj.get("text", line)))
            entries.append(LogEntry(
                line_number=i, timestamp=ts,
                level=level.upper() if isinstance(level, str) else None,
                message=str(msg), raw=line, source=source,
                metadata={k: v for k, v in obj.items()
                          if k not in ("timestamp", "time", "@timestamp",
                                       "level", "severity", "message", "msg", "text")},
            ))
        return entries

    def _parse_structured_lines(self, lines: list[str],
                                 source: str) -> list[LogEntry]:
        entries: list[LogEntry] = []
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            m = _RE_STD_LOG.match(line)
            if m:
                ts = self._parse_ts(m.group("ts"))
                level = m.group("level").upper()
                msg = m.group("msg") or ""
                entries.append(LogEntry(
                    line_number=i, timestamp=ts, level=level,
                    message=msg, raw=line, source=source,
                    metadata={
                        "thread": m.group("thread") or "",
                        "logger": m.group("logger") or "",
                    },
                ))
            else:
                # 回退: 尝试提取级别
                entries.append(self._fallback_parse(line, i, source))
        return entries

    def _parse_plain_lines(self, lines: list[str], source: str) -> list[LogEntry]:
        entries: list[LogEntry] = []
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            entries.append(self._fallback_parse(line, i, source))
        return entries

    def _fallback_parse(self, line: str, line_no: int,
                        source: str) -> LogEntry:
        """宽松解析: 尝试提取时间戳和级别。"""
        ts = None
        level = None
        message = line

        ts_m = _RE_TS.search(line)
        if ts_m:
            ts = self._parse_ts(ts_m.group("ts"))

        level_m = _RE_LEVEL.search(line)
        if level_m:
            level = level_m.group("level").upper()

        return LogEntry(
            line_number=line_no, timestamp=ts, level=level,
            message=message, raw=line, source=source,
        )

    def _merge_stack_traces(self, entries: list[LogEntry]) -> list[LogEntry]:
        """合并多行异常栈到前一条 ERROR/FATAL 日志。"""
        if not entries:
            return entries

        merged: list[LogEntry] = []
        for entry in entries:
            if (merged and _RE_STACK_CONT.match(entry.raw)
                    and merged[-1].level in ("ERROR", "FATAL", "SEVERE", "WARN", "WARNING")):
                # 合并到前一条
                prev = merged[-1]
                prev.message += "\n" + entry.raw
                prev.raw += "\n" + entry.raw
            else:
                merged.append(entry)
        return merged

    @staticmethod
    def _parse_ts(raw: str) -> datetime | None:
        """解析时间戳字符串。"""
        if not raw or not isinstance(raw, str):
            return None
        raw = raw.strip()
        if not raw:
            return None

        patterns = [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S,%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%b %d %H:%M:%S",          # syslog
            "%b %d %Y %H:%M:%S",
        ]

        for fmt in patterns:
            try:
                dt = datetime.strptime(raw, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=TZ)
                return dt
            except ValueError:
                continue
        return None
