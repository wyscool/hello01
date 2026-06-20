# ============================================================
# Phase 5, Lesson 42: 可观测性 — 日志、追踪、监控
# ============================================================
#
# 本课目标:
#   学会观察 AI 系统在生产环境中的行为 — 结构化日志、调用链追踪、
#   Token 用量监控。
#
#   Java 背景映射:
#     StructuredLogger ≈ SLF4J + MDC (结构化 + 上下文)
#     LlmTracer        ≈ OpenTelemetry Span (调用链追踪)
#     TokenMonitor     ≈ Micrometer Gauge (指标采集)
#
#   核心概念:
#     1. print() 为什么不够 → 结构化日志
#     2. LLM 调用追踪 — 记录每次 API 调用的完整信息
#     3. Trace/Span 模型 — 追踪多步 Agent 的完整路径
#     4. Token 用量监控 — 成本控制的前置条件
#
#   为什么 AI 系统的可观测性更难?
#     传统系统: 日志 → 找到哪行代码出错
#     AI 系统:  日志 → 找到哪个 Prompt/工具/模型环节出问题
#     问题可能不在代码, 而在模型行为、Prompt 设计、工具选择...
#
# 预计阅读 + 实操时间: 50-60 分钟
#
# 前置: Phase 1-4, Lesson 41
# ============================================================

import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Callable, Any
from collections import defaultdict

# ============================================================
# 〇、print() 为什么不够
# ============================================================
# 开发阶段用 print() 看输出很直观, 但生产环境不行:
#   - 格式不一致, 机器难以解析
#   - 没有时间戳, 不知道何时发生
#   - 没有上下文, 不知道关联哪个请求
#   - 无法聚合、搜索、告警
#
# 解决方法: 结构化日志 (Structured Logging)

print("=" * 60)
print("  Phase 5, Lesson 42: 可观测性")
print("=" * 60)
print()


# ============================================================
# 一、结构化日志
# ============================================================

# 北京时区 (方便阅读)
TZ = timezone(timedelta(hours=8))


class JsonLogger:
    """结构化 JSON 日志器。

    每条日志是一个 JSON 对象, 包含固定的字段:
      timestamp, level, message, context

    类比 Java:
      log.info("user login", kv("userId", 123))
      → {"timestamp": "...", "level": "INFO", "message": "user login", "user_id": 123}
    """

    LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

    def __init__(self, name: str = "app", min_level: str = "DEBUG",
                 output=sys.stderr):
        self.name = name
        self.min_level = self.LEVELS.get(min_level.upper(), 10)
        self.output = output
        self._counts: dict[str, int] = defaultdict(int)

    def _log(self, level: str, message: str, **context):
        if self.LEVELS[level] < self.min_level:
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

    @property
    def stats(self) -> dict:
        return dict(self._counts)


# 演示
print("--- 结构化日志演示 ---")
log = JsonLogger("demo")
log.info("应用启动", phase=5, lesson=42)
log.debug("加载配置", env_file=".env", keys_found=2)
log.warn("API 响应延迟", latency_ms=1250, threshold_ms=1000)
log.error("工具调用失败", tool="calculator", error="division by zero")

print(f"\n  日志统计: {log.stats}")
print()


# ============================================================
# 二、LLM 调用追踪
# ============================================================
# AI 系统的核心操作是调用 LLM。每次调用记录:
#   - 哪个模型、什么时间、耗时多久
#   - 输入 token 数、输出 token 数
#   - 是否成功、错误信息
#   - 调用上下文 (哪个 Agent、哪个 Step)

@dataclass
class LlmCall:
    """单次 LLM 调用记录。

    类比 Java: 一个 POJO 记录 API 调用的完整快照。
    """
    timestamp: str = ""
    model: str = ""
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error: str = ""
    context: dict = field(default_factory=dict)  # agent, step, task...

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


class LlmTracer:
    """LLM 调用追踪器。

    记录每次 LLM 调用的详细信息, 支持聚合统计。

    用法:
        tracer = LlmTracer()
        with tracer.track(model="claude-sonnet-4-6", context={"agent": "DevAssistant"}):
            response = llm.messages.create(...)
        # 自动记录耗时、token 用量等
    """

    def __init__(self, logger: JsonLogger | None = None):
        self.calls: list[LlmCall] = []
        self.log = logger or JsonLogger("llm-tracer")

    def record(self, call: LlmCall):
        self.calls.append(call)
        self.log.info(
            "llm_call",
            model=call.model,
            duration_ms=round(call.duration_ms),
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            total_tokens=call.total_tokens,
            success=call.success,
            **call.context,
        )

    def stats(self) -> dict:
        """聚合统计: 总调用次数、Token 总量、平均延迟。"""
        if not self.calls:
            return {}
        total = len(self.calls)
        success = sum(1 for c in self.calls if c.success)
        total_input = sum(c.input_tokens for c in self.calls)
        total_output = sum(c.output_tokens for c in self.calls)
        avg_duration = sum(c.duration_ms for c in self.calls) / total

        # 按模型分组
        by_model: dict = defaultdict(lambda: {"calls": 0, "tokens": 0, "errors": 0})
        for c in self.calls:
            m = by_model[c.model]
            m["calls"] += 1
            m["tokens"] += c.total_tokens
            if not c.success:
                m["errors"] += 1

        return {
            "total_calls": total,
            "success_rate": round(success / total, 3),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "avg_duration_ms": round(avg_duration, 1),
            "by_model": {k: dict(v) for k, v in by_model.items()},
        }

    def report(self) -> str:
        """生成可读的追踪报告。"""
        s = self.stats()
        if not s:
            return "无 LLM 调用记录。"

        lines = [
            f"\n  ╔{'═' * 40}╗",
            f"  ║  LLM 调用追踪报告{' ' * 22}║",
            f"  ╠{'═' * 40}╣",
            f"  ║  总调用: {s['total_calls']:<5}  成功率: {s['success_rate']:.1%}{' ' * 12}║",
            f"  ║  输入 Token: {s['total_input_tokens']:<8}  输出: {s['total_output_tokens']:<8}{' ' * 4}║",
            f"  ║  总 Token: {s['total_tokens']:<8}  平均延迟: {s['avg_duration_ms']:.0f}ms{' ' * 8}║",
        ]

        for model, ms in s["by_model"].items():
            lines.append(
                f"  ║  [{model}] {ms['calls']}次 {ms['tokens']}tk "
                f"{'⚠' if ms['errors'] else '✓'}{' ' * 8}║"
            )
        lines.append(f"  ╚{'═' * 40}╝")
        return "\n".join(lines)


# 演示: 模拟一个 Agent 的 LLM 调用
print("--- LLM 调用追踪演示 ---")
tracer = LlmTracer(log)

# 模拟几次调用
simulated_calls = [
    {"model": "claude-sonnet-4-6", "input": 520, "output": 180, "ms": 850,
     "ctx": {"agent": "DevAssistant", "mode": "quick", "step": "think"}},
    {"model": "claude-sonnet-4-6", "input": 850, "output": 95, "ms": 620,
     "ctx": {"agent": "DevAssistant", "mode": "quick", "step": "tools", "tool": "calculate"}},
    {"model": "claude-sonnet-4-6", "input": 980, "output": 230, "ms": 1020,
     "ctx": {"agent": "DevAssistant", "mode": "quick", "step": "answer"}},
    {"model": "claude-sonnet-4-6", "input": 340, "output": 400, "ms": 2400,
     "ctx": {"agent": "DevAssistant", "mode": "plan", "step": "generate_plan"}},
]

for sc in simulated_calls:
    call = LlmCall(
        timestamp=datetime.now(TZ).isoformat(),
        model=sc["model"],
        duration_ms=sc["ms"],
        input_tokens=sc["input"],
        output_tokens=sc["output"],
        context=sc["ctx"],
    )
    tracer.record(call)

print(tracer.report())
print()


# ============================================================
# 三、Trace/Span 模型 — 多步追踪
# ============================================================
# Agent 的一个请求可能涉及多次 LLM 调用、多次工具调用。
# Trace = 一个完整请求的追踪 (比如一次 /ask)
# Span  = 追踪中的一个步骤 (比如一次 LLM 调用)
#
# 类比 Java:
#   Trace ≈ OpenTelemetry Trace (一次 HTTP 请求的完整链路)
#   Span  ≈ Span (链路中的一个操作: DB 查询、RPC 调用)

@dataclass
class Span:
    """追踪中的一个操作步骤。"""
    name: str
    span_id: str
    parent_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "ok"  # ok | error
    metadata: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def add_event(self, name: str, **attrs):
        self.events.append({
            "time": datetime.now(TZ).isoformat(),
            "name": name,
            **attrs,
        })


class Trace:
    """一次完整请求的追踪。

    包含多个 Span, 形成一棵调用树。

    用法:
        trace = Trace("ask", request_id="req-001")
        span1 = trace.start_span("llm_call", metadata={"model": "..."})
        # ... do work ...
        trace.end_span(span1)
        span2 = trace.start_span("tool_call", parent=span1)
        # ... do work ...
        trace.end_span(span2)
    """

    def __init__(self, name: str, request_id: str = ""):
        self.name = name
        self.request_id = request_id or f"req-{int(time.time() * 1000)}"
        self.spans: list[Span] = []
        self.start_time = time.time()
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"{self.request_id}-{self._counter}"

    def start_span(self, name: str, parent: Span | None = None, **metadata) -> Span:
        span = Span(
            name=name,
            span_id=self._next_id(),
            parent_id=parent.span_id if parent else "",
            start_time=time.time(),
            metadata=metadata,
        )
        self.spans.append(span)
        return span

    def end_span(self, span: Span, status: str = "ok", **attrs):
        span.end_time = time.time()
        span.status = status
        if attrs:
            span.metadata.update(attrs)

    def to_dict(self) -> dict:
        """导出为字典 (可序列化为 JSON)。"""
        return {
            "trace_name": self.name,
            "request_id": self.request_id,
            "duration_ms": round(
                (time.time() - self.start_time) * 1000, 1
            ),
            "span_count": len(self.spans),
            "spans": [
                {
                    "name": s.name,
                    "span_id": s.span_id,
                    "parent_id": s.parent_id,
                    "duration_ms": round(s.duration_ms, 1),
                    "status": s.status,
                    "metadata": s.metadata,
                    "events": s.events,
                }
                for s in self.spans
            ],
        }

    def report(self) -> str:
        """可视化追踪树。"""
        d = self.to_dict()
        lines = [
            f"\n  Trace: {d['trace_name']} ({d['request_id']})",
            f"  总耗时: {d['duration_ms']}ms | Spans: {d['span_count']}",
        ]
        # 构建父子关系
        by_parent: dict[str, list] = defaultdict(list)
        for s in self.spans:
            by_parent[s.parent_id].append(s)

        def _print_tree(parent_id: str = "", indent: int = 2):
            for s in by_parent.get(parent_id, []):
                prefix = "  " * indent
                icon = "✓" if s.status == "ok" else "✗"
                lines.append(
                    f"{prefix}{icon} {s.name} [{s.span_id}] "
                    f"({s.duration_ms:.0f}ms)"
                )
                if s.metadata:
                    meta = " ".join(
                        f"{k}={v}" for k, v in s.metadata.items()
                    )
                    lines.append(f"{prefix}  {meta}")
                _print_tree(s.span_id, indent + 1)

        _print_tree()
        return "\n".join(lines)


# 演示: 追踪一次 Agent 请求
print("--- Trace/Span 演示 ---")
trace = Trace("quick-mode-ask", request_id="demo-001")

# Step 1: LLM 思考
span_think = trace.start_span("llm_call",
                              model="claude-sonnet-4-6",
                              step="think")
time.sleep(0.01)  # 模拟耗时
span_think.add_event("tokens_used", input=520, output=180)
trace.end_span(span_think, status="ok", tokens=700)

# Step 2: 工具调用 (作为 think 的子 Span)
span_tool = trace.start_span("tool_call",
                             parent=span_think,
                             tool="calculate",
                             args={"expression": "sqrt(256)"})
time.sleep(0.01)
trace.end_span(span_tool, status="ok", result="16")

# Step 3: 第二次 LLM 调用 (回答)
span_answer = trace.start_span("llm_call",
                               model="claude-sonnet-4-6",
                               step="answer")
time.sleep(0.005)
trace.end_span(span_answer, status="ok", tokens=420)

print(trace.report())
print(f"\n  导出 JSON 长度: {len(json.dumps(trace.to_dict(), ensure_ascii=False))} 字符")
print()


# ============================================================
# 四、Token 用量监控
# ============================================================
# 成本控制的核心: 先知道用了多少 Token, 才能控制成本。
# Lesson 43 会讲成本优化, 这里先建立监控基础。

class TokenMonitor:
    """Token 用量实时监控。

    按模型、按时间段、按 Agent 维度统计 Token 用量。
    """

    def __init__(self, daily_budget: int = 1_000_000):
        self.daily_budget = daily_budget
        self._usage: list[dict] = []  # 每次用量的记录
        self._start_time = time.time()

    def record(self, model: str, input_tokens: int, output_tokens: int,
               agent: str = "", task: str = ""):
        total = input_tokens + output_tokens
        self._usage.append({
            "time": datetime.now(TZ).isoformat(),
            "model": model,
            "input": input_tokens,
            "output": output_tokens,
            "total": total,
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

    def dashboard(self) -> str:
        """打印简洁的用量仪表盘。"""
        bar_len = 20
        filled = int(bar_len * self.budget_pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        lines = [
            f"\n  ╔{'═' * 45}╗",
            f"  ║  Token 用量仪表盘{' ' * 27}║",
            f"  ╠{'═' * 45}╣",
            f"  ║  预算: {self.daily_budget:,} tk  [{bar}] {self.budget_pct:.1f}%{' ' * 3}║",
            f"  ║  已用: {self.total_used:,} tk  剩余: {self.budget_remaining:,} tk{' ' * 3}║",
            f"  ║  调用: {self.call_count} 次{' ' * 36}║",
        ]
        for model, tokens in self.by_model().items():
            lines.append(f"  ║    [{model}]: {tokens:,} tk{' ' * (25 - len(model))}║")

        # 预警
        if self.budget_pct > 80:
            lines.append(f"  ║  ⚠ 预算使用 > 80%!{' ' * 27}║")
        lines.append(f"  ╚{'═' * 45}╝")
        return "\n".join(lines)


# 演示
print("--- Token 用量监控演示 ---")
monitor = TokenMonitor(daily_budget=100_000)

# 模拟一段时间的使用
for _ in range(15):
    import random
    monitor.record(
        model="claude-sonnet-4-6",
        input_tokens=random.randint(200, 1200),
        output_tokens=random.randint(50, 600),
        agent="DevAssistant",
        task=random.choice(["代码审查", "翻译任务", "问答", "文件分析"]),
    )

print(monitor.dashboard())
print()


# ============================================================
# 五、组合使用 — 可观测性中间件
# ============================================================
# 在实际项目中, 日志、追踪、监控是组合使用的。
# 下面是一个简单的 "可观测性中间件" 示例:

@dataclass
class Observability:
    """可观测性组合 — 日志 + 追踪 + Token 监控。

    用法:
        obs = Observability()
        obs.log.info("Agent 启动")

        trace = obs.start_trace("ask", req_id="...")
        span = obs.start_span(trace, "llm_call")
        # ... call LLM ...
        obs.record_llm(span, tracer, input_tk, output_tk)
    """

    log: JsonLogger = field(default_factory=lambda: JsonLogger("ai-app"))
    token_monitor: TokenMonitor = field(default_factory=TokenMonitor)

    def start_trace(self, name: str, request_id: str = "") -> Trace:
        trace = Trace(name, request_id)
        self.log.info("trace_start", name=name, request_id=trace.request_id)
        return trace

    def start_span(self, trace: Trace, name: str,
                   parent: Span | None = None, **meta) -> Span:
        return trace.start_span(name, parent, **meta)

    def end_span(self, trace: Trace, span: Span,
                 status: str = "ok", **attrs):
        trace.end_span(span, status, **attrs)
        self.log.debug("span_end",
                       span=span.name,
                       duration_ms=round(span.duration_ms, 1),
                       status=status)

    def record_llm(self, model: str, input_tk: int, output_tk: int,
                   duration_ms: float, agent: str = "",
                   task: str = "", success: bool = True):
        self.token_monitor.record(model, input_tk, output_tk, agent, task)
        return {
            "model": model,
            "input_tokens": input_tk,
            "output_tokens": output_tk,
            "total_tokens": input_tk + output_tk,
            "duration_ms": duration_ms,
        }


# ============================================================
# 六、入口
# ============================================================

if __name__ == "__main__":
    # 所有演示已在各节中运行

    print("=" * 60)
    print("  Lesson 42 完成!")
    print("=" * 60)
    print(f"""
  你学到了:
    1. JsonLogger    — 结构化日志 (告别 print 调试)
    2. LlmTracer     — LLM 调用记录 + 聚合统计
    3. Trace / Span  — 多步请求的完整链路追踪
    4. TokenMonitor  — 用量仪表盘 + 预算预警

  可观测性三支柱 (类比 OpenTelemetry):
    Logs   → JsonLogger    (事件记录)
    Traces → Trace + Span  (请求链路)
    Metrics → TokenMonitor (聚合指标)

  下一课: Lesson 43 — 成本控制: 缓存、批处理、模型降级。
    Token 监控已经搭好, 接下来学习如何省钱!
  """)


# ============================================================
# 试试看 (Try This) — 解答
# ============================================================

print("\n" + "=" * 60)
print("  试试看 (Try This) — Lesson 42 练习")
print("=" * 60)
print()


# ============================================================
# 练习 1: 日志持久化 — 同时输出到控制台和文件
# ============================================================
# 修改 JsonLogger, 增加文件输出支持。

class FileJsonLogger(JsonLogger):
    """扩展 JsonLogger — 同时输出到控制台和文件。

    类比 Java:
      类似 Logback 的 FileAppender + ConsoleAppender
      可以同时输出到多个目标。
    """

    def __init__(self, name: str = "app", min_level: str = "DEBUG",
                 file_path: str | None = None):
        # 控制台输出 (父类行为)
        super().__init__(name, min_level, output=sys.stderr)
        # 文件输出
        self.file_path = file_path
        self._file_handle = None
        if file_path:
            self._file_handle = open(file_path, "a", encoding="utf-8")

    def _log(self, level: str, message: str, **context):
        """重写 _log: 同时写到控制台和文件。"""
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
        line = json.dumps(record, ensure_ascii=False)

        # 控制台输出
        print(line, file=self.output)
        # 文件输出
        if self._file_handle:
            self._file_handle.write(line + "\n")
            self._file_handle.flush()  # 立即写入, 防止丢失

    def close(self):
        """关闭文件句柄。"""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


def ex1_log_persistence():
    """演示日志持久化。"""
    print("--- 练习 1: 日志持久化 ---")

    # 持久化到文件 (存放在项目根目录)
    log_file = Path(__file__).parent.parent / "phase5_app.log"
    flog = FileJsonLogger("persistent-app", "DEBUG",
                         file_path=str(log_file))
    flog.info("持久化测试启动", phase=5, lesson=42)
    flog.debug("调试信息", data={"key": "value"})
    flog.warn("警告: 磁盘使用率", usage_pct=85)
    flog.error("错误: 连接超时", target="llm-api", timeout_s=30)
    flog.close()

    # 验证写入
    with open(log_file, "r") as f:
        lines = f.readlines()
    print(f"  日志文件: {log_file}")
    print(f"  写入行数: {len(lines)}")
    for line in lines:
        record = json.loads(line)
        print(f"    [{record['level']}] {record['message']}")
    print(f"  控制台 + 文件双写模式运行成功")

    # 清理测试文件
    log_file.unlink(missing_ok=True)

ex1_log_persistence()


# ============================================================
# 练习 2: 追踪 DevAssistant
# ============================================================
# 参考: deploy/observability.py 中的 Trace/Span 已在 deploy/app.py
# 中集成。每次 /ask 请求创建一个 Trace, 每个 Span 记录一次操作。
#
# 这里展示如何将 Trace 集成到模拟的 DevAssistant 流程中。

def ex2_trace_devassistant():
    """模拟将 Trace/Span 集成到 DevAssistant 的 /ask 流程。"""
    print("--- 练习 2: 追踪 DevAssistant ---")

    class TracedDevAssistant:
        """带追踪的 DevAssistant 模拟。

        deploy/app.py 中已经完整实现了:
          - 每个 POST /ask 创建 Trace (line 306)
          - agent 调用创建 Span (line 315-317)
          - 追踪数据随响应返回 (line 343)

        这里展示其核心模式。
        """

        def __init__(self):
            self.tracer = LlmTracer()
            self.tool_results: list[dict] = []

        def ask(self, task: str, mode: str = "quick") -> dict:
            """处理一次 /ask 请求, 全程追踪。"""
            trace = Trace(f"ask-{mode}",
                         request_id=f"req-{int(time.time() * 1000) % 100000:05d}")

            # Step 1: Think (LLM 调用)
            span_think = trace.start_span("llm_think",
                                         model="claude-sonnet-4-6")
            time.sleep(0.005)  # 模拟延迟
            trace.end_span(span_think, status="ok", tokens=500)

            # Step 2: 可能需要工具调用
            if "计算" in task:
                span_tool = trace.start_span("tool_calculate",
                                            parent=span_think,
                                            tool="calculator")
                time.sleep(0.003)
                trace.end_span(span_tool, status="ok", result="42")
                self.tool_results.append(
                    {"tool": "calculator", "result": "42"}
                )

            # Step 3: Answer (LLM 调用)
            span_answer = trace.start_span("llm_answer",
                                          model="claude-sonnet-4-6")
            time.sleep(0.005)
            trace.end_span(span_answer, status="ok", tokens=300)

            # 记录 LLM 调用 (可选)
            for sp in trace.spans:
                if "llm" in sp.name:
                    self.tracer.record(LlmCall(
                        timestamp=datetime.now(TZ).isoformat(),
                        model=sp.metadata.get("model", "unknown"),
                        duration_ms=sp.duration_ms,
                        input_tokens=sp.metadata.get("tokens", 0) // 2,
                        output_tokens=sp.metadata.get("tokens", 0) // 2,
                        context={"span": sp.name, "request_id": trace.request_id},
                    ))

            return {
                "request_id": trace.request_id,
                "answer": f"[模拟] 关于 '{task[:30]}...' 的回答",
                "mode": mode,
                "trace": trace.to_dict(),
                "trace_tree": trace.report(),
            }

    assistant = TracedDevAssistant()

    # 测试两个请求
    for task in ["计算 123 + 456", "解释什么是可观测性"]:
        result = assistant.ask(task)
        print(result["trace_tree"])
        print(f"  回答: {result['answer'][:50]}...")

    print(f"\n  LLM 调用追踪统计: {assistant.tracer.stats()}")
    print(f"\n  deploy/app.py 已完整实现此模式, 参见:")
    print(f"    - POST /ask 中创建 Trace (L306)")
    print(f"    - Span 追踪 agent 调用 (L315-317)")
    print(f"    - trace.to_dict() 随响应返回 (L343)")

ex2_trace_devassistant()


# ============================================================
# 练习 3: 错误追踪 — 扩展 LlmTracer
# ============================================================
# 扩展 LlmTracer, 记录错误详情。

class ErrorTrackingTracer(LlmTracer):
    """扩展 LlmTracer — 记录详细的错误信息。

    新增字段:
      - error_type: rate_limit / context_overflow / tool_error / api_error / timeout
      - retry_count: 重试次数
      - fallback_strategy: 降级策略 (如果有)
    """

    ERROR_TYPES = [
        "rate_limit",       # 429 速率限制
        "context_overflow",  # 上下文超出窗口
        "tool_error",       # 工具执行失败
        "api_error",        # 其他 API 错误
        "timeout",          # 超时
        "unknown",          # 未知错误
    ]

    def __init__(self, logger: JsonLogger | None = None, max_retries: int = 3):
        super().__init__(logger)
        self.max_retries = max_retries
        self.errors: list[dict] = []  # 详细错误记录

    def record_error(self, model: str, error_type: str,
                     retry_count: int = 0, fallback: str = "",
                     detail: str = "", context: dict | None = None):
        """记录一次错误及其上下文。"""
        error_record = {
            "timestamp": datetime.now(TZ).isoformat(),
            "model": model,
            "error_type": error_type,
            "retry_count": retry_count,
            "fallback_strategy": fallback,
            "detail": detail,
            "context": context or {},
        }
        self.errors.append(error_record)

        # 同时记录到日志
        self.log.error(
            "llm_error",
            model=model,
            error_type=error_type,
            retry_count=retry_count,
            fallback=fallback,
            detail=detail[:100],
        )

    def error_stats(self) -> dict:
        """错误统计: 按类型聚合。"""
        if not self.errors:
            return {"total_errors": 0, "by_type": {}}
        by_type: dict[str, int] = defaultdict(int)
        by_model: dict[str, int] = defaultdict(int)
        total_retries = 0
        for e in self.errors:
            by_type[e["error_type"]] += 1
            by_model[e["model"]] += 1
            total_retries += e["retry_count"]

        return {
            "total_errors": len(self.errors),
            "by_type": dict(by_type),
            "by_model": dict(by_model),
            "total_retries": total_retries,
            "avg_retries": round(total_retries / len(self.errors), 1),
        }


def ex3_error_tracking():
    """演示错误追踪。"""
    print("--- 练习 3: 错误追踪 ---")

    et = ErrorTrackingTracer(max_retries=3)

    # 模拟各种错误场景
    simulated_errors = [
        ("claude-sonnet-4-6", "rate_limit", 2, "切换到 deepseek-v3",
         "429 Too Many Requests"),
        ("claude-sonnet-4-6", "context_overflow", 1, "截断上下文到 50k",
         "输入 token 超过 200k 限制"),
        ("claude-haiku-4-5", "tool_error", 0, "",
         "calculator 工具返回 NaN"),
        ("claude-opus-4", "timeout", 1, "",
         "请求在 60s 内未响应"),
        ("claude-sonnet-4-6", "rate_limit", 3, "队列等待",
         "连续 3 次 429, 进入退避队列"),
    ]

    for model, err_type, retries, fallback, detail in simulated_errors:
        et.record_error(model, err_type, retries, fallback, detail)

    stats = et.error_stats()
    print(f"  总错误数: {stats['total_errors']}")
    print(f"  按类型:")
    for etype, count in stats["by_type"].items():
        print(f"    {etype}: {count} 次")
    print(f"  按模型:")
    for model, count in stats["by_model"].items():
        print(f"    {model}: {count} 次")
    print(f"  总重试次数: {stats['total_retries']}")
    print(f"  平均重试: {stats['avg_retries']}")

    print(f"""
  关键模式:
    deploy/ 中的 LlmClient (agent_core.py) 已经内置了重试逻辑。
    这里的 ErrorTrackingTracer 是额外的一层: 记录每次错误的上下文,
    帮助事后分析:
    - 哪个模型最容易触发限流? → 调整并发策略
    - 哪种错误最多? → 优化 Prompt/工具
    - 降级策略有效吗? → 追踪 fallback 的成功率
  """)

ex3_error_tracking()


# ============================================================
# 练习 4: Token 用量告警 — 增强版
# ============================================================
# 添加按小时统计、速率预测、预算耗尽预估。

class AlertingTokenMonitor(TokenMonitor):
    """增强版 TokenMonitor — 按小时统计 + 速率告警。"""

    def __init__(self, daily_budget: int = 1_000_000,
                 alert_threshold: float = 0.8):
        super().__init__(daily_budget)
        self.alert_threshold = alert_threshold

    def hourly_usage(self) -> dict[int, int]:
        """按小时统计 Token 用量。"""
        hourly: dict[int, int] = defaultdict(int)
        for u in self._usage:
            dt = datetime.fromisoformat(u["time"])
            hour_key = dt.hour
            hourly[hour_key] += u["total"]
        return dict(sorted(hourly.items()))

    def current_rate_per_hour(self) -> float:
        """当前消耗速率 (Token/小时), 基于最近 1 小时的数据。"""
        if not self._usage:
            return 0.0
        now = datetime.now(TZ)
        recent_tokens = 0
        for u in self._usage:
            dt = datetime.fromisoformat(u["time"])
            if (now - dt).total_seconds() <= 3600:
                recent_tokens += u["total"]
        # 如果最近只有一小段时间, 按实际时长折算
        if self._usage:
            first_time = datetime.fromisoformat(self._usage[0]["time"])
            elapsed_hours = max(0.1, (now - first_time).total_seconds() / 3600)
            if recent_tokens == 0:
                # 使用全部数据估算
                return self.total_used / elapsed_hours
        return recent_tokens  # 最近一小时的实际用量

    def budget_exhaustion_estimate(self) -> float | None:
        """预估预算何时耗尽 (小时数)。

        Returns:
            耗尽还需多少小时, 若速率 <= 0 则返回 None
        """
        rate = self.current_rate_per_hour()
        if rate <= 0:
            return None
        remaining = self.budget_remaining
        return remaining / rate

    def alerts(self) -> list[str]:
        """生成告警列表。"""
        alerts_list = []
        pct = self.budget_pct
        rate = self.current_rate_per_hour()
        exhaustion = self.budget_exhaustion_estimate()
        expected_hourly = self.daily_budget / 24

        if pct >= 90:
            alerts_list.append(
                f"严重: 预算已使用 {pct:.0f}%, 仅剩 {self.budget_remaining:,} Token!"
            )
        elif pct >= 80:
            alerts_list.append(
                f"警告: 预算已使用 {pct:.0f}%"
            )

        if rate > expected_hourly * 1.5:
            alerts_list.append(
                f"注意: 当前速率 {rate:,.0f} tk/h > "
                f"预算均速 {expected_hourly:,.0f} tk/h"
            )

        if exhaustion is not None and exhaustion < 4:
            alerts_list.append(
                f"紧急: 按当前速率, 预算将在 {exhaustion:.1f} 小时后耗尽!"
            )
        elif exhaustion is not None:
            alerts_list.append(
                f"按当前速率, 预算预计在 {exhaustion:.1f} 小时后耗尽"
            )

        return alerts_list


def ex4_token_alerts():
    """演示增强版 Token 监控。"""
    print("--- 练习 4: Token 用量告警 ---")

    monitor = AlertingTokenMonitor(daily_budget=50_000)

    # 模拟快速消耗 (模拟高负载场景)
    import random

    # 模拟 3 小时的用量 (用随机时间戳)
    base_time = datetime.now(TZ)
    for i in range(40):
        # 模拟时间递增
        sim_time = base_time - timedelta(minutes=random.randint(0, 180))
        monitor._usage.append({
            "time": sim_time.isoformat(),
            "model": "claude-sonnet-4-6",
            "input": random.randint(200, 1500),
            "output": random.randint(50, 500),
            "total": 0,  # 下面计算
            "agent": "DevAssistant",
            "task": f"task-{i}",
        })
        monitor._usage[-1]["total"] = (
            monitor._usage[-1]["input"] + monitor._usage[-1]["output"]
        )

    # 显示统计
    print(f"  日预算: {monitor.daily_budget:,} Token")
    print(f"  总使用: {monitor.total_used:,} Token ({monitor.budget_pct:.1f}%)")
    print(f"  当前速率: {monitor.current_rate_per_hour():,.0f} tk/h")
    print(f"  预算均速: {monitor.daily_budget / 24:,.0f} tk/h")

    # 按小时统计
    hour_usage = monitor.hourly_usage()
    print(f"\n  按小时统计:")
    for hour, tokens in sorted(hour_usage.items()):
        bar_len = int(20 * tokens / max(hour_usage.values()))
        bar = "#" * bar_len
        print(f"    {hour:02d}:00 | {bar:<20s} {tokens:>6,} tk")

    # 告警
    alerts_list = monitor.alerts()
    if alerts_list:
        print(f"\n  告警:")
        for a in alerts_list:
            print(f"    ! {a}")
    else:
        print(f"\n  无告警 (用量正常)")

ex4_token_alerts()


# ============================================================
# 练习 5 (挑战): Span 的 context manager
# ============================================================
# 让 Span 支持 with 语句, 自动记录 start/end, 异常时自动标记 error。

class TracedSpan:
    """Span 的 context manager 包装器。

    用法:
        trace = Trace("test")
        with trace.span("llm_call", model="claude") as span:
            response = call_llm(...)
            span.set_attribute("tokens", 500)
        # 自动 end_span, 异常时自动标记 error
    """

    def __init__(self, trace: "Trace", name: str,
                 parent: Span | None = None, **metadata):
        self.trace = trace
        self.name = name
        self.parent = parent
        self.metadata = metadata
        self._span: Span | None = None
        self._attributes: dict = {}

    def __enter__(self) -> "TracedSpan":
        self._span = self.trace.start_span(
            self.name, parent=self.parent, **self.metadata
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span is None:
            return
        if exc_type is not None:
            # 异常 → 标记 error
            self.trace.end_span(
                self._span,
                status="error",
                error_type=exc_type.__name__,
                error_message=str(exc_val)[:200],
                **self._attributes,
            )
        else:
            # 正常 → 标记 ok
            self.trace.end_span(self._span, status="ok", **self._attributes)
        # 返回 False 让异常继续传播
        return False

    def set_attribute(self, key: str, value):
        """设置 Span 属性。"""
        self._attributes[key] = value

    def add_event(self, name: str, **attrs):
        """添加 Span 事件。"""
        if self._span:
            self._span.add_event(name, **attrs)


# 为 Trace 类添加 span() 方法 (monkey-patch 风格扩展)
def _trace_span_method(self, name: str, parent: Span | None = None,
                       **metadata) -> TracedSpan:
    """Trace 的 context manager span 方法。"""
    return TracedSpan(self, name, parent, **metadata)


Trace.span = _trace_span_method  # type: ignore


def ex5_span_context_manager():
    """演示 Span context manager。"""
    print("--- 练习 5: Span context manager ---")

    trace = Trace("context-mgr-demo", request_id="demo-cm")

    # 正常流程
    with trace.span("llm_think", model="claude-sonnet-4-6") as span:
        time.sleep(0.005)
        span.set_attribute("tokens", 520)
        span.add_event("thinking_started", phase="analyze")

    # 带异常的流程
    try:
        with trace.span("tool_call", tool="risky_operation") as span:
            span.set_attribute("args", {"x": 1, "y": 0})
            if True:  # 模拟工具失败
                raise ValueError("division by zero")
    except ValueError:
        pass  # 捕获异常, 但 Span 已标记 error

    # 嵌套 Span
    with trace.span("llm_answer", model="claude-sonnet-4-6") as parent_span:
        time.sleep(0.003)
        parent_span.set_attribute("tokens", 300)
        with trace.span("formatting", parent=parent_span._span) as child_span:
            time.sleep(0.001)
            child_span.set_attribute("format", "markdown")

    print(trace.report())
    print(f"\n  Span 状态检查:")
    for s in trace.spans:
        print(f"    {s.name}: status={s.status}, "
              f"duration={s.duration_ms:.1f}ms, "
              f"metadata={s.metadata}")

    print(f"""
  关键优势:
    1. with 语句自动调用 start/end, 不会忘记 end_span
    2. 异常自动捕获并标记 error, 无需手动处理
    3. 链式调用: trace.span("a").span("b") 表达父子关系
    4. 与 OpenTelemetry 的 tracer.start_as_current_span() 风格一致
    5. 生产环境建议直接用 OpenTelemetry SDK,
       但理解其原理有助于排查追踪问题。
  """)

ex5_span_context_manager()


# ============================================================
# 练习 6 (思考): 可观测性的成本
# ============================================================

print("--- 练习 6: 可观测性的成本 (思考) ---")
print("""
  Q: 在生产环境中, 你会记录 "所有 LLM 调用" 还是 "采样记录"?

  A: 取决于调用量和场景。

  全部记录 (100%):
    - 适合: 低流量 (<1000 次/天)、调试阶段、合规要求 (审计)
    - 优点: 完整的调用历史, 可追溯任何问题
    - 缺点: I/O 开销, 存储成本 (每个 Span ~500 bytes, 百万次 = 500MB)

  采样记录 (1-10%):
    - 适合: 高流量 (>10000 次/天)、稳定运行阶段
    - 优点: 大幅降低存储和 I/O 开销, 足够用于统计
    - 缺点: 可能错过偶发的异常

  混合策略 (推荐):
    - 错误 100% 记录 (错误本身就少, 但信息价值极高)
    - 正常调用 10% 采样 (足够统计延迟/Token 分布)
    - 慢请求 100% 记录 (P99 以上延迟的请求全量记录)
    - 关键用户/业务 100% 记录 (付费用户、核心功能)

  类比 Java:
    OpenTelemetry 的 Sampler 接口:
    - AlwaysOnSampler  → 100%
    - TraceIdRatioBased → 10%
    - ParentBased       → 如果父 Span 被采样, 子 Span 也被采样

  Q: 如果生产环境突然 Token 用量翻三倍, 你怎么排查?

  A: 排查步骤 (按可观测性工具):
    1. TokenMonitor.by_model() → 哪个模型的用量激增?
       - 如果是 Opus 激增 → 可能 ModelRouter 降级失效
       - 如果是所有模型 → 可能是流量激增

    2. TokenMonitor.hourly_usage() → 什么时间段开始的?
       - 固定时间段 → 可能是定时任务/批处理
       - 持续增长 → 可能是新功能上线

    3. LlmTracer.stats() → 单次调用 Token 数是否变大了?
       - 输入 Token 变大 → Prompt 膨胀 or 上下文变长
       - 输出 Token 变大 → max_tokens 配置改了

    4. Trace 链路 → 具体哪个 Agent / 工具在消耗?
       - 按 agent 维度看 by_model
       - 某个 Agent 的调用量是否异常

    5. 关键指标:
       - avg input tokens per call (是否异常)
       - call count per minute (是否陡增)
       - cache hit rate (缓存是否失效)

    根本原因可能是:
    - Prompts 被意外修改 (加了更多 system prompt)
    - 代码 bug 导致无限循环调用
    - 缓存失效 (缓存服务器挂了, 全部穿透)
    - DDoS 攻击
    - 某个 Agent 的 plan 策略变得非常激进
""")


# ============================================================
# 课后反思
# ============================================================

print("--- 课后反思 ---")
print("""
  Q: Trace 和 LlmTracer 分别能回答什么问题?

  Trace (链路):
    - "一个请求经历了哪些步骤?"  → 看 Span 树
    - "哪个步骤最慢?"            → 看 duration_ms
    - "工具调用是否成功?"         → 看 status
    - "为什么会触发那个工具?"     → 看 Span 的父子关系和 metadata

  LlmTracer (聚合):
    - "今天调了多少次 LLM?"       → total_calls
    - "哪个模型用得最多?"         → by_model
    - "平均延迟是多少?"           → avg_duration_ms
    - "有没有错误?"              → success_rate

  简单说:
    Trace 回答 "这个请求发生了什么" (纵向)
    LlmTracer 回答 "整体趋势是什么" (横向)
    两者互补, 缺一不可。
""")
