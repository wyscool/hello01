# ============================================================
# deploy/observability.py — 可观测性
# ============================================================
# 提取自 phase5/42_observability.py
#
# 类比 Java (OpenTelemetry):
#   JsonLogger  ≈ SLF4J + MDC
#   Trace/Span  ≈ OpenTelemetry Tracing
#   TokenMonitor ≈ Micrometer Gauge
# ============================================================

import sys
import json
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

TZ = timezone(timedelta(hours=8))


# ============================================================
# 一、结构化日志
# ============================================================

class JsonLogger:
    """结构化 JSON 日志器。

    每条日志是一个 JSON 行，包含 timestamp / level / message / 上下文。
    """

    LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

    def __init__(self, name: str = "dev-assistant",
                 min_level: str = "DEBUG", output=sys.stderr):
        self.name = name
        self.min_level = self.LEVELS.get(min_level.upper(), 10)
        self.output = output
        self._counts: dict[str, int] = defaultdict(int)

    def _log(self, level: str, message: str, **context):
        if self.LEVELS.get(level, 0) < self.min_level:
            return
        self._counts[level] += 1
        record = {
            "timestamp": datetime.now(TZ).isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **context,
        }
        print(json.dumps(record, ensure_ascii=False), file=self.output)

    def debug(self, message: str, **ctx):
        self._log("DEBUG", message, **ctx)

    def info(self, message: str, **ctx):
        self._log("INFO", message, **ctx)

    def warn(self, message: str, **ctx):
        self._log("WARN", message, **ctx)

    def error(self, message: str, **ctx):
        self._log("ERROR", message, **ctx)


# ============================================================
# 二、Trace / Span — 请求链路追踪
# ============================================================

@dataclass
class Span:
    name: str
    span_id: str
    parent_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "ok"
    metadata: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def add_event(self, name: str, **attrs):
        self.events.append({
            "time": datetime.now(TZ).isoformat(),
            "name": name, **attrs,
        })


class Trace:
    """一次完整请求的追踪。"""

    def __init__(self, name: str, request_id: str = ""):
        self.name = name
        self.request_id = request_id or f"req-{int(time.time() * 1000)}"
        self.spans: list[Span] = []
        self.start_time = time.time()
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"{self.request_id}-{self._counter}"

    def start_span(self, name: str, parent: Span | None = None,
                   **metadata) -> Span:
        span = Span(
            name=name, span_id=self._next_id(),
            parent_id=parent.span_id if parent else "",
            start_time=time.time(), metadata=metadata,
        )
        self.spans.append(span)
        return span

    def end_span(self, span: Span, status: str = "ok", **attrs):
        span.end_time = time.time()
        span.status = status
        if attrs:
            span.metadata.update(attrs)

    @property
    def duration_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "trace_name": self.name,
            "request_id": self.request_id,
            "duration_ms": round(self.duration_ms, 1),
            "span_count": len(self.spans),
            "spans": [
                {
                    "name": s.name, "span_id": s.span_id,
                    "parent_id": s.parent_id,
                    "duration_ms": round(s.duration_ms, 1),
                    "status": s.status,
                    "metadata": s.metadata,
                }
                for s in self.spans
            ],
        }


# ============================================================
# 三、Token 用量监控
# ============================================================

class TokenMonitor:
    """Token 用量实时监控。"""

    def __init__(self, daily_budget: int = 1_000_000):
        self.daily_budget = daily_budget
        self._usage: list[dict] = []

    def record(self, model: str, input_tokens: int, output_tokens: int,
               agent: str = "", task: str = ""):
        self._usage.append({
            "time": datetime.now(TZ).isoformat(),
            "model": model,
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
            "agent": agent,
            "task": task[:50] if task else "",
        })

    @property
    def total_used(self) -> int:
        return sum(u["total"] for u in self._usage)

    @property
    def call_count(self) -> int:
        return len(self._usage)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.daily_budget - self.total_used)

    @property
    def budget_pct(self) -> float:
        return self.total_used / self.daily_budget * 100

    def by_model(self) -> dict[str, int]:
        result: dict[str, int] = defaultdict(int)
        for u in self._usage:
            result[u["model"]] += u["total"]
        return dict(result)

    def snapshot(self) -> dict:
        return {
            "total_used": self.total_used,
            "call_count": self.call_count,
            "budget_remaining": self.budget_remaining,
            "budget_pct": round(self.budget_pct, 1),
            "by_model": self.by_model(),
        }
