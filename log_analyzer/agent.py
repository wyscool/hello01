# ============================================================
# log_analyzer/agent.py — 日志分析 Agent (ReAct 循环)
# ============================================================
# LLM 驱动的日志分析引擎。
# Agent 自主决定: 先用 stats 了解全貌 → 用 top_errors 找模式
#                 → 用 search 深入定位 → 用 context 看细节 → 给出根因分析
#
# 类比 Java:
#   LogAnalysisAgent ≈ @Service (业务编排层)
#   组合 LlmClient (数据访问) + MCPServer (工具注册)
# ============================================================

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from deploy.agent_core import LlmClient, MCPServer, MCPClient, TokenBudget
from deploy.observability import Trace, JsonLogger
from log_analyzer.parser import LogParser, LogEntry
from log_analyzer.tools import create_log_server
from log_analyzer.config import AppConfig


# ============================================================
# 一、Result 类型
# ============================================================

@dataclass
class AnalysisResult:
    """日志分析结果。"""
    question: str
    answer: str
    iterations: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    file_name: str = ""
    total_entries: int = 0
    trace: dict | None = None


# ============================================================
# 二、系统提示词
# ============================================================

SYSTEM_PROMPT = """你是一个资深 SRE 日志分析专家 Agent。你可以调用工具来分析日志文件。

## 分析策略
1. **先全局后局部**: 先用 stats 了解整体情况，再看 timeline 了解时间分布
2. **聚焦异常**: 用 top_errors 发现高频错误模式，用 search 深入定位
3. **根因推理**: 根据时间序列、错误关联、上下文，推理出根因
4. **行号引用**: 每个发现都要标注行号，如 "第 42 行"

## 常见分析场景
- "分析这个错误": 先用 search 定位错误 → 用 get_section 看上下文 → 分析根因
- "有没有性能问题": 用 timeline 看时间分布 → search 找 timeout/slow/lag 关键词
- "总结日志情况": stats → top_errors → sample 各看几条 → 总结

## 工具使用建议
- 每次调用 1-2 个工具，不要一次调用太多
- 如果 search 结果太多，缩小范围或用更精确的关键词
- 最终分析报告包含: 日志概览、发现的异常、根因分析、处理建议

## 回答格式
用中文回答，按以下结构:
1. **日志概览** — 总量、级别分布、时间范围
2. **发现的问题** — 每条带行号引用
3. **根因分析** — 推断根本原因
4. **处理建议** — 可操作的修复建议"""


# ============================================================
# 三、LogAnalysisAgent
# ============================================================

class LogAnalysisAgent:
    """日志分析 Agent — ReAct 循环。

    用法:
        agent = LogAnalysisAgent(llm_client, log_entries, config)
        result = agent.analyze("分析错误日志")
    """

    def __init__(self, llm: LlmClient, entries: list[LogEntry],
                 config: AppConfig | None = None,
                 logger: JsonLogger | None = None):
        self.llm = llm
        self.entries = entries
        self.config = config or AppConfig.from_env()
        self.logger = logger or JsonLogger("log-agent", min_level="INFO")

        # 组装 MCP
        self._mcp_server = create_log_server(
            entries,
            max_context=self.config.max_context_lines,
            max_sample=self.config.max_sample_size,
        )
        self._mcp_client = MCPClient()
        self._mcp_client.connect(self._mcp_server)
        self._tools = self._mcp_client.list_tools()

        # Token 预算
        self._budget = TokenBudget(max_tokens=50000)

    @property
    def tool_names(self) -> list[str]:
        return self._mcp_server.tool_names

    def analyze(self, question: str) -> AnalysisResult:
        """执行日志分析 (ReAct 循环)。

        Args:
            question: 分析问题，如 "分析错误日志" / "有没有异常" / "总结"

        Returns:
            AnalysisResult with answer, iterations, tool_calls, trace
        """
        start = time.time()
        trace = Trace("log-analysis")
        tool_calls_record: list[dict] = []

        # 构建初始消息
        file_name = self.entries[0].source if self.entries else "unknown"
        user_msg = self._build_user_prompt(question, file_name)

        messages: list[dict] = [
            {"role": "user", "content": user_msg},
        ]

        iterations = 0
        final_answer = ""

        span = trace.start_span("react-loop", question=question[:80])

        for i in range(self.config.agent_max_iterations):
            iterations = i + 1

            # Token 预算检查
            ok, info = self._budget.check(messages)
            if not ok:
                messages = messages[:1] + messages[-6:]

            if not self.llm.is_healthy:
                final_answer = self._offline_answer(question)
                break

            try:
                response = self.llm.create(
                    messages=messages,
                    tools=self._tools,
                    system=SYSTEM_PROMPT,
                    max_tokens=self.config.agent_max_tokens,
                    temperature=self.config.agent_temperature,
                )
            except Exception as e:
                self.logger.error("llm_call_failed", error=str(e))
                final_answer = f"[LLM 调用失败: {e}]"
                break

            tool_uses = self.llm.get_tool_uses(response)
            text = self.llm.get_text(response)

            # 无工具调用 → 最终答案
            if not tool_uses:
                final_answer = text
                break

            # 追加 assistant 消息
            assistant_content = []
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    assistant_content.append({
                        "type": "text", "text": block.text,
                    })
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use", "id": block.id,
                        "name": block.name, "input": block.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            # 执行工具 + 追加结果
            tool_results = []
            for tu in tool_uses:
                tool_name = tu.name
                tool_input = tu.input or {}

                step_span = trace.start_span(
                    f"tool:{tool_name}", **tool_input,
                )
                try:
                    result = self._mcp_client.call_tool(tool_name, tool_input)
                    status = "ok" if "error" not in result else "error"
                except Exception as e:
                    result = {"error": str(e)}
                    status = "error"

                trace.end_span(step_span, status)
                tool_calls_record.append({
                    "iter": i + 1, "tool": tool_name,
                    "input": {k: str(v)[:80] for k, v in tool_input.items()},
                    "status": status,
                })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

            messages.append({"role": "user", "content": tool_results})

        # 超出最大迭代
        if not final_answer and iterations >= self.config.agent_max_iterations:
            final_answer = self._force_summary(messages)

        trace.end_span(span, "ok" if final_answer else "max_iter",
                       iterations=iterations)

        elapsed = (time.time() - start) * 1000
        return AnalysisResult(
            question=question,
            answer=final_answer,
            iterations=iterations,
            tool_calls=tool_calls_record,
            latency_ms=round(elapsed, 1),
            file_name=file_name,
            total_entries=len(self.entries),
            trace=trace.to_dict(),
        )

    def _build_user_prompt(self, question: str, file_name: str) -> str:
        """构建首条用户消息，注入日志概要。"""
        n = len(self.entries)
        levels = {}
        for e in self.entries:
            if e.level:
                levels[e.level] = levels.get(e.level, 0) + 1

        summary = f"文件: {file_name}, 总行数: {n}"
        if levels:
            lv_str = ", ".join(f"{k}: {v}" for k, v in sorted(levels.items()))
            summary += f", 级别分布: {lv_str}"

        return f"[日志概要] {summary}\n\n分析任务: {question}"

    def _offline_answer(self, question: str) -> str:
        """LLM 不可用时的离线回答。"""
        n = len(self.entries)
        errors = sum(1 for e in self.entries
                     if e.level in ("ERROR", "FATAL", "SEVERE"))
        warns = sum(1 for e in self.entries
                    if e.level in ("WARN", "WARNING"))

        lines: list[str] = [
            f"[离线模式] 日志分析: {question}",
            f"",
            f"## 日志概览",
            f"- 总行数: {n}",
            f"- 错误: {errors} ({errors / max(n, 1) * 100:.1f}%)",
            f"- 警告: {warns} ({warns / max(n, 1) * 100:.1f}%)",
        ]

        # 最近 5 条错误
        recent_errors = [e for e in self.entries
                         if e.level in ("ERROR", "FATAL", "SEVERE")][-5:]
        if recent_errors:
            lines.append(f"\n## 最近错误 (后 5 条)")
            for e in recent_errors:
                lines.append(f"- 行 {e.line_number}: {e.message[:120]}")

        return "\n".join(lines)

    def _force_summary(self, messages: list[dict]) -> str:
        """超出最大迭代时强制总结。"""
        if not self.llm.is_healthy:
            return "[超出最大迭代次数]"

        try:
            response = self.llm.create(
                messages=messages + [{"role": "user", "content": "基于已收集的工具结果，给出最终分析报告。用中文回答。"}],
                max_tokens=self.config.agent_max_tokens,
                temperature=0.0,
            )
            return self.llm.get_text(response)
        except Exception:
            return "[超出最大迭代次数，且 LLM 调用失败]"


# ============================================================
# 四、工厂函数
# ============================================================

def create_log_agent(file_path: Path | str,
                     config: AppConfig | None = None,
                     logger: JsonLogger | None = None) -> LogAnalysisAgent:
    """从文件路径创建完整配置的 LogAnalysisAgent。

    用法:
        agent = create_log_agent("app.log")
        result = agent.analyze("分析错误")
    """
    if config is None:
        config = AppConfig.from_env()

    if logger is None:
        logger = JsonLogger("log-agent", min_level="INFO")

    # 1. 解析日志
    parser = LogParser(max_file_size_mb=config.max_file_size_mb)
    entries = parser.parse_file(file_path)
    logger.info("log_parsed", file=str(file_path), entries=len(entries))

    # 2. LlmClient
    llm = LlmClient(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
        max_retries=config.llm_max_retries,
        timeout=config.llm_timeout_seconds,
    )

    # 3. Agent
    agent = LogAnalysisAgent(llm, entries, config, logger)
    logger.info("agent_ready", tools=agent.tool_names)

    return agent
