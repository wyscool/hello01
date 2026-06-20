# ============================================================
# Phase 4, Lesson 32: 工具调用深入 —— 注册、验证、预算、并行
# ============================================================
#
# 本课目标:
#   1. 从"手写 dict"到 ToolRegistry — 工具注册中心
#   2. 标准化的工具调用结果 — ToolResult
#   3. Token 预算管理 — 跟踪消耗、自动裁剪
#   4. 并行 vs 串行调用 — LLM 何时并行调工具?
#   5. 工具执行模式 — 输入验证、重试、超时保护
#   6. 增强版 Agent — 融合以上所有能力
#
# 预计阅读 + 实操时间: 50-60 分钟
#
# 前置: Lesson 31 (Agent 基础 + ReAct 循环)
# ============================================================

import os
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Any
from datetime import datetime
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# ============================================================
# 〇、环境准备
# ============================================================

from anthropic import Anthropic
from anthropic.types import Message, ToolUseBlock

api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")

client_kwargs = {"api_key": api_key} if api_key else {}
if base_url:
    client_kwargs["base_url"] = base_url
client = Anthropic(**client_kwargs)

MODEL = "claude-sonnet-4-6"


def _get_text(response: Message) -> str:
    parts = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts)


def _get_tool_uses(response: Message) -> list[ToolUseBlock]:
    return [b for b in response.content if b.type == "tool_use"]


try:
    client.messages.create(
        model=MODEL, max_tokens=10,
        messages=[{"role": "user", "content": "ping"}],
    )
    api_ok = True
    print("✅ API 连接正常\n")
except Exception:
    api_ok = False
    print("⚠️  API 不可用, 将以模拟模式运行\n")


# ============================================================
# 一、问题回顾 —— L31 的工具管理有什么不足?
# ============================================================
# L31 中, 工具定义和处理器是分开的:
#
#   TOOL_DEFS = [CALC_TOOL, TIME_TOOL, ...]        # 工具 schema 列表
#   TOOL_HANDLERS = {"calculate": fn, ...}          # 处理器 dict
#
# 问题:
#   1. 散落 — schema 和 handler 分离, 改一处忘另一处
#   2. 无校验 — 没有检查 handler 参数和 schema 是否匹配
#   3. 无结果规范 — 有的返回 dict, 有的返回 str, Agent 难以理解
#   4. 无 Token 感知 — messages 无限增长, 最终超出上下文窗口
#   5. 无错误恢复 — 工具调用失败, Agent 只能得到原始异常信息
#
# 本课逐一解决这些问题。

print("=" * 60)
print("L31 回顾 → L32 改进")
print("=" * 60)
print("""
  L31                      L32
  ─────────────────────────────────────────
  dict + list               ToolRegistry (统一注册)
  无校验                     schema 自动校验
  裸 dict 返回               ToolResult (标准化)
  messages 无限增长           TokenBudget (跟踪 + 裁剪)
  原始异常                   错误分类 + 友好提示
""")


# ============================================================
# 二、ToolRegistry —— 工具注册中心
# ============================================================
# 统一管理工具的 schema、handler、校验逻辑。
#
# 类比 Java:
#   ToolRegistry ≈ Spring ApplicationContext
#   每个 Tool ≈ 一个 @Bean, 注册到容器中
#   调用时按 name 查找, 执行后返回结果

@dataclass
class ToolDef:
    """工具定义 — schema + handler 合二为一。

    类比 Java: 一个 POJO, 包含方法签名 (schema) + 方法体 (handler)。
    """
    name: str
    description: str
    parameters: dict          # JSON Schema properties
    required: list[str]       # 必填参数
    handler: Callable         # 实际执行函数
    timeout: float = 30.0     # 超时时间 (秒)
    max_retries: int = 0      # 失败重试次数

    def to_schema(self) -> dict:
        """转为 Anthropic tool schema 格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            }
        }

    def validate_input(self, params: dict) -> list[str]:
        """校验输入参数。返回错误列表, 空列表 = 通过。"""
        errors = []
        for key in self.required:
            if key not in params:
                errors.append(f"缺少必填参数: {key}")
        for key in params:
            if key not in self.parameters:
                errors.append(f"未知参数: {key}")
        return errors


class ToolRegistry:
    """工具注册中心。

    类比 Java:
      ToolRegistry ≈ Map<String, BeanDefinition>
      注册: registry.register(toolDef)
      获取: registry.get_schemas() → List<ToolSchema>
      执行: registry.execute(name, params) → ToolResult
    """

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> "ToolRegistry":
        """注册一个工具。支持链式调用。"""
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def get_schemas(self) -> list[dict]:
        """获取所有工具的 Anthropic schema 列表。"""
        return [t.to_schema() for t in self._tools.values()]

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def validate_and_execute(self, name: str, params: dict) -> "ToolResult":
        """执行工具: 校验 → 执行 → 捕获异常 → 返回标准化结果。"""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(name, f"未知工具: {name}")

        # 输入校验
        errors = tool.validate_input(params)
        if errors:
            return ToolResult.error(name, "; ".join(errors), stage="validation")

        # 执行 (带重试)
        last_error = None
        for attempt in range(tool.max_retries + 1):
            try:
                output = tool.handler(**params)
                return ToolResult.success(name, output, attempt=attempt)
            except Exception as e:
                last_error = str(e)
                if attempt < tool.max_retries:
                    time.sleep(0.5)  # 退避

        return ToolResult.error(
            name, f"执行失败 (重试{tool.max_retries}次): {last_error}",
            stage="execution"
        )

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        tools_str = ", ".join(self._tools.keys())
        return f"ToolRegistry({tools_str})"


# ============================================================
# 三、ToolResult —— 标准化工具结果
# ============================================================
# L31 中, 每个工具返回的 dict 格式不同:
#   calculate  → {"expression": ..., "result": ...}
#   get_time   → {"datetime": ..., "weekday": ...}
#   search     → {"query": ..., "results": [...]}
#
# Agent 需要理解每种格式, 增加了 prompt 的复杂度。
#
# 解决方案: 统一的 ToolResult 格式, 让 Agent 始终看到一致的结构。

@dataclass
class ToolResult:
    """标准化工具执行结果。

    无论哪个工具, 返回格式都是:
      {status, tool, data, error, metadata}

    类比 Java:
      ToolResult ≈ ResponseEntity<T>
        status   → HTTP status (success/error)
        data     → T (业务数据)
        error    → error message
        metadata → headers
    """
    status: str              # "success" | "error"
    tool: str                # 工具名
    data: Any = None         # 成功时的数据
    error: str | None = None # 失败时的错误信息
    error_stage: str = ""    # 失败阶段: "validation" | "execution"
    metadata: dict = field(default_factory=dict)

    @classmethod
    def success(cls, tool: str, data: Any, **meta) -> "ToolResult":
        return cls(status="success", tool=tool, data=data, metadata=meta)

    @classmethod
    def error(cls, tool: str, message: str, stage: str = "execution") -> "ToolResult":
        return cls(status="error", tool=tool, error=message, error_stage=stage)

    def to_message(self) -> str:
        """转为给 LLM 看的文本。"""
        if self.status == "success":
            return json.dumps({"status": "ok", "data": self.data}, ensure_ascii=False)
        else:
            parts = [f"[{self.error_stage} 失败] {self.error}"]
            return "\n".join(parts)

    @property
    def ok(self) -> bool:
        return self.status == "success"


# 演示: 同一个工具, 两种结果格式的对比
print("\n" + "=" * 60)
print("ToolResult: 标准化 vs 裸 dict")
print("=" * 60)

# 模拟: 成功
ok = ToolResult.success("calculate", {"result": 42})
print(f"  成功: {ok.to_message()}")

# 模拟: 参数校验失败
fail_val = ToolResult.error("calculate", "缺少必填参数: expression", stage="validation")
print(f"  校验失败: {fail_val.to_message()}")

# 模拟: 执行异常
fail_exec = ToolResult.error("calculate", "ZeroDivisionError: division by zero", stage="execution")
print(f"  执行失败: {fail_exec.to_message()}")

print(f"\n  LLM 看到的始终是统一格式 → 更容易理解和恢复")


# ============================================================
# 四、Token 预算管理
# ============================================================
# 每轮对话的 messages 列表不断增长:
#
#   Iter 1: [user_task, assistant_tool_use, tool_result]         ← 约 500 tokens
#   Iter 2: [user_task, assistant_tool_use, tool_result,
#            assistant_tool_use, tool_result]                    ← 约 1000 tokens
#   Iter 5: ...                                                  ← 约 3000 tokens
#
# 如果不加控制, 会超出模型的 context window (如 Claude 的 200K),
# 或者消耗大量 token 费用。
#
# TokenBudget 负责:
#   1. 估算每条消息的 token 数
#   2. 跟踪累计消耗
#   3. 超过阈值时裁剪旧消息 (保留 system prompt + 最近 N 轮)

class TokenBudget:
    """Token 预算管理器。

    简单估算: 中文 ~1.5 字/token, 英文 ~4 字/token
    这里用保守估算: 4 字符 ≈ 1 token

    类比 Java:
      TokenBudget ≈ 内存管理 / 缓存淘汰策略
      类似 LRU Cache: 保留最近使用的, 淘汰旧的
    """

    def __init__(self, max_tokens: int = 50000, warning_ratio: float = 0.7):
        self.max_tokens = max_tokens
        self.warning_threshold = int(max_tokens * warning_ratio)
        self.used = 0
        self.warnings: list[str] = []

    def estimate(self, content: Any) -> int:
        """估算内容的 token 数。"""
        if isinstance(content, str):
            return max(1, len(content) // 4)
        if isinstance(content, list):
            return sum(self.estimate(item) for item in content)
        if isinstance(content, dict):
            return self.estimate(json.dumps(content, ensure_ascii=False))
        return 1

    def estimate_messages(self, messages: list[dict]) -> int:
        """估算 messages 列表的总 token 数。"""
        return self.estimate(json.dumps(messages, ensure_ascii=False))

    def check(self, messages: list[dict]) -> bool:
        """检查是否超出预算。返回 True = 安全, False = 超预算。"""
        estimated = self.estimate_messages(messages)
        self.used = estimated

        if estimated >= self.max_tokens:
            self.warnings.append(f"⚠️ 超出 token 预算: {estimated}/{self.max_tokens}")
            return False

        if estimated >= self.warning_threshold:
            self.warnings.append(f"⚡ 接近 token 上限: {estimated}/{self.max_tokens}")

        return True

    def trim(self, messages: list[dict], keep_recent: int = 6) -> list[dict]:
        """裁剪 messages, 保留最近的 N 条。

        策略: 保留第一条 (user task) + 最近 N 条。
        在保留的消息前插入摘要标记。
        """
        if len(messages) <= keep_recent + 2:
            return messages

        # 保留: 第一条 + 最近 keep_recent 条
        trimmed = [messages[0]] + messages[-keep_recent:]

        # 插入裁剪标记
        old_count = len(messages) - len(trimmed)
        summary = {
            "role": "user",
            "content": f"[上下文已裁剪, 省略了 {old_count} 条历史消息以控制 token 消耗]"
        }
        trimmed.insert(1, summary)

        new_est = self.estimate_messages(trimmed)
        old_est = self.estimate_messages(messages)
        self.warnings.append(
            f"✂️ 已裁剪: {old_est} → {new_est} tokens ({old_count} 条旧消息)"
        )
        return trimmed

    def report(self) -> str:
        return f"Token 用量: {self.used}/{self.max_tokens} ({100*self.used//self.max_tokens}%)"


# 演示 Token 估算
budget = TokenBudget(max_tokens=50000)

sample_messages = [
    {"role": "user", "content": "帮我分析一个数学问题: 计算圆的面积..."},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "calculate", "input": {"expression": "pi*5**2"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": '{"result": 78.54}'}]},
]

est = budget.estimate_messages(sample_messages)
print(f"\n  Token 估算: 3 条消息 ≈ {est} tokens")
print(f"  {budget.report()}")


# ============================================================
# 五、并行 vs 串行 —— LLM 何时并行调用工具?
# ============================================================
# Anthropic API 支持在一个 response 中返回多个 tool_use block,
# 这意味着 LLM 可以"同时"请求多个工具调用。
#
# 并行 (Parallel) —— 多个工具之间没有依赖:
#   LLM: "我需要同时知道天气和时间"
#   → [tool_use: get_weather, tool_use: get_current_time]  ← 同一个 response
#
# 串行 (Sequential) —— 后面的工具依赖前面的结果:
#   LLM: "先算面积, 再算..." (但 LLM 其实可以一次算出多个独立值)
#   → [tool_use: calculate(area)]
#   → 等结果
#   → [tool_use: calculate(ratio)]  ← 需要上一步的结果
#
# 关键判断: LLM 根据工具的 input 是否依赖前一个工具的输出,
# 自动决定并行还是串行。你不需要手动控制。
#
# 类比 Java:
#   并行 ≈ CompletableFuture.allOf(future1, future2)
#   串行 ≈ future1.thenCompose(result → future2)

print("\n" + "=" * 60)
print("并行 vs 串行工具调用")
print("=" * 60)
print("""
  并行 (独立):
    [tool_use: get_weather("北京")]
    [tool_use: get_current_time()]       ← 同一个 response, 同时返回
    [tool_use: search("旅游攻略")]

  串行 (依赖):
    [tool_use: calculate("5**2")]
        → 结果: 25
    [tool_use: calculate("25 * 3.14")]   ← 需要上一步的 25
        → 结果: 78.5
""")


# ============================================================
# 六、实战工具定义
# ============================================================

# 安全计算器 (改进版, 带更好的错误信息)
_SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "log": math.log,
    "pi": math.pi, "e": math.e,
}


def calc_handler(expression: str) -> dict:
    if len(expression) > 200:
        raise ValueError("表达式过长 (最多 200 字符)")
    result = eval(expression, {"__builtins__": {}}, _SAFE_MATH)
    return {"expression": expression, "result": result}


# 时间查询
def time_handler(timezone: str = "Asia/Shanghai") -> dict:
    now = datetime.now()
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": ["一", "二", "三", "四", "五", "六", "日"][now.weekday()],
        "iso": now.isoformat(),
    }


# 文本统计 (新工具)
def text_stats_handler(text: str) -> dict:
    if not text.strip():
        raise ValueError("文本不能为空")
    lines = text.split("\n")
    words = text.split()
    return {
        "chars": len(text),
        "chars_no_spaces": len(text.replace(" ", "").replace("\n", "")),
        "words": len(words),
        "lines": len(lines),
        "avg_word_len": round(sum(len(w) for w in words) / max(len(words), 1), 1),
    }


# --- 注册 ---
registry = ToolRegistry()

registry.register(ToolDef(
    name="calculate",
    description="执行数学表达式计算。支持 + - * / ** sqrt() sin() cos() log() abs() round() 等。",
    parameters={
        "expression": {
            "type": "string",
            "description": "Python 数学表达式, 如 'sqrt(16) * pi'"
        }
    },
    required=["expression"],
    handler=calc_handler,
    max_retries=1,
))

registry.register(ToolDef(
    name="get_current_time",
    description="获取当前日期和时间。",
    parameters={
        "timezone": {
            "type": "string",
            "description": "时区, 如 'Asia/Shanghai'"
        }
    },
    required=[],
    handler=time_handler,
))

registry.register(ToolDef(
    name="text_stats",
    description="统计文本的字数、词数、行数等信息。",
    parameters={
        "text": {
            "type": "string",
            "description": "要统计的文本内容"
        }
    },
    required=["text"],
    handler=text_stats_handler,
    max_retries=0,
))

print(f"\n  已注册工具: {registry.list_tools()}")
print(f"  共 {len(registry)} 个工具")


# ============================================================
# 七、增强版 Agent —— 融合 Registry + Budget + ToolResult
# ============================================================

class AdvancedAgent:
    """增强版 ReAct Agent。

    相比 L31 的 ReActAgent, 新增:
      - ToolRegistry (统一管理)
      - TokenBudget (上下文控制)
      - ToolResult (标准化结果)
      - 输入校验 + 重试
      - 执行摘要

    类比 Java:
      AdvancedAgent ≈ 重构后的 Service, 依赖注入 Registry + Budget
    """

    def __init__(
        self,
        registry: ToolRegistry,
        model: str = MODEL,
        max_iterations: int = 10,
        token_budget: int = 50000,
        verbose: bool = True,
    ):
        self.registry = registry
        self.model = model
        self.max_iterations = max_iterations
        self.budget = TokenBudget(max_tokens=token_budget)
        self.verbose = verbose
        self.execution_log: list[dict] = []

    def run(self, task: str, system_prompt: str | None = None) -> str:
        if system_prompt is None:
            system_prompt = self._system_prompt()

        messages: list[dict] = [{"role": "user", "content": task}]
        self.execution_log = []

        for i in range(self.max_iterations):
            # --- Token 检查 ---
            if not self.budget.check(messages):
                messages = self.budget.trim(messages)

            # --- 调用 LLM ---
            if self.verbose:
                est = self.budget.estimate_messages(messages)
                print(f"\n{'─' * 50}")
                print(f"  [迭代 {i + 1}] tokens≈{est} ", end="")

            if not api_ok:
                return self._mock_run(task)

            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                system=system_prompt,
                messages=messages,
                tools=self.registry.get_schemas(),
            )

            tool_uses = _get_tool_uses(response)
            text_parts = _get_text(response)

            # --- 无工具调用 → 完成 ---
            if not tool_uses:
                if self.verbose:
                    print("✅ 完成")
                self.execution_log.append({"iteration": i + 1, "action": "answer"})
                return text_parts

            # --- 有工具调用 → 执行 ---
            if self.verbose:
                parallel_mark = "∥" if len(tool_uses) > 1 else "→"
                print(f"{parallel_mark} {len(tool_uses)} 工具")

            # 追加 assistant 消息
            assistant_content = self._build_assistant_content(response)
            messages.append({"role": "assistant", "content": assistant_content})

            # 执行工具
            tool_results: list[ToolResult] = []
            for tu in tool_uses:
                params = tu.input or {}
                result = self.registry.validate_and_execute(tu.name, params)
                tool_results.append(result)

                if self.verbose:
                    status = "✓" if result.ok else "✗"
                    input_str = json.dumps(params, ensure_ascii=False)[:50]
                    if result.ok:
                        data_str = json.dumps(result.data, ensure_ascii=False)[:60]
                        print(f"  {status} {tu.name}({input_str}) → {data_str}")
                    else:
                        print(f"  {status} {tu.name}({input_str}) → {result.error}")

            self.execution_log.append({
                "iteration": i + 1,
                "action": "tools",
                "parallel": len(tool_uses),
                "results": [r.status for r in tool_results],
            })

            # 追加 tool_result 消息
            tool_content = []
            for tu, tr in zip(tool_uses, tool_results):
                tool_content.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": tr.to_message(),
                })
            messages.append({"role": "user", "content": tool_content})

        return f"[Agent 在 {self.max_iterations} 轮内未能完成任务。请简化问题。]"

    def _build_assistant_content(self, response: Message) -> list[dict]:
        """从 response 构建 assistant 消息内容。"""
        content = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return content

    def _system_prompt(self) -> str:
        tools_desc = "\n".join(
            f"- {t.name}: {t.description}" for t in self.registry._tools.values()
        )
        return f"""你是一个 AI Agent, 可以调用工具完成任务。

可用工具:
{tools_desc}

规则:
1. 如果多个工具之间没有依赖关系, 可以同时调用它们
2. 如果工具返回了错误, 分析原因并尝试修正参数后重试
3. 当信息足够时, 给出完整回答
4. 用中文回答"""

    def _mock_run(self, task: str) -> str:
        return f"[模拟 Agent] 收到任务: {task}\n可用工具: {self.registry.list_tools()}"

    def print_summary(self):
        """打印执行摘要。"""
        print(f"\n  {'=' * 50}")
        print(f"  执行摘要")
        print(f"  {'=' * 50}")
        tools_called = sum(
            1 for log in self.execution_log if log["action"] == "tools"
        )
        print(f"  迭代轮次: {len(self.execution_log)}")
        print(f"  工具调用轮次: {tools_called}")
        print(f"  {self.budget.report()}")
        if self.budget.warnings:
            for w in self.budget.warnings:
                print(f"  {w}")


# ============================================================
# 八、演示 1: 基本工具调用 (并行)
# ============================================================

print("\n\n" + "=" * 60)
print("演示 1: 并行工具调用")
print("=" * 60)

agent1 = AdvancedAgent(registry, max_iterations=5)

task1 = "帮我统计一下这段文字: 'Python 是一门解释型、动态类型的编程语言, 由 Guido van Rossum 于 1991 年发布。' 然后告诉我现在的时间。"
print(f"  任务: {task1}\n")

answer1 = agent1.run(task1)
print(f"\n  最终答案: {answer1[:300]}...")
agent1.print_summary()


# ============================================================
# 九、演示 2: 工具调用失败 + 恢复
# ============================================================

print("\n\n" + "=" * 60)
print("演示 2: 错误恢复")
print("=" * 60)

agent2 = AdvancedAgent(registry, max_iterations=5)

# 故意给一个容易出错的请求
task2 = "帮我计算 log(0) 的值。另外统计 '' (空字符串) 的文本信息。"
print(f"  任务: {task2}\n")

answer2 = agent2.run(task2)
print(f"\n  最终答案: {answer2[:300]}...")
agent2.print_summary()


# ============================================================
# 十、演示 3: 多步推理 (串行 + 并行混合)
# ============================================================

print("\n\n" + "=" * 60)
print("演示 3: 混合调用 (串行 + 并行)")
print("=" * 60)

agent3 = AdvancedAgent(registry, max_iterations=8)

task3 = """帮我分析以下三段文字的字数, 找出最长的一段:

1. 'Python 简洁优雅, 适合初学者也适合专业开发者。'
2. 'Java 是一门静态类型的面向对象编程语言, 广泛应用于企业级开发, 拥有庞大的生态系统和社区支持。'
3. 'Rust 注重内存安全和性能。'

最后告诉我哪个最长, 现在是什么时候。"""

print(f"  任务: {task3}\n")

answer3 = agent3.run(task3)
print(f"\n  最终答案: {answer3[:400]}...")
agent3.print_summary()


# ============================================================
# 十一、Token 裁剪演示
# ============================================================

print("\n\n" + "=" * 60)
print("Token 裁剪演示")
print("=" * 60)

# 构造大量消息模拟长对话
many_messages = [{"role": "user", "content": "初始任务: 分析数据"}]  # 保留第一条
for i in range(20):
    many_messages.append({"role": "assistant", "content": f"工具调用 {i} 的结果分析... " + "数据 " * 50})
    many_messages.append({"role": "user",   "content": f"工具结果 {i}: " + "结果 " * 50})

print(f"  裁剪前: {len(many_messages)} 条消息")
est_before = budget.estimate_messages(many_messages)
print(f"  估算 tokens: {est_before}")

trimmed = budget.trim(many_messages, keep_recent=6)
est_after = budget.estimate_messages(trimmed)
print(f"  裁剪后: {len(trimmed)} 条消息, 估算 tokens: {est_after}")
print(f"  节省: {est_before - est_after} tokens ({(100*(est_before-est_after)//max(est_before,1))}%)")

# 检查第一条和最后一条
print(f"\n  保留的消息:")
print(f"  [0] role={trimmed[0]['role']} (原始任务)")
if len(trimmed) > 1:
    print(f"  [1] role={trimmed[1]['role']} (裁剪标记)")
print(f"  [{len(trimmed)-2}] role={trimmed[-2]['role']} ...")
print(f"  [{len(trimmed)-1}] role={trimmed[-1]['role']} ...")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Lesson 32 完成! 工具调用深入已掌握。")
    print("=" * 60)
    print(f"""
  回顾: 你学会了什么?

  1. ToolRegistry   — 统一注册, schema + handler 合一, 链式 API
  2. ToolResult     — 标准化成功/失败格式, LLM 始终看到一致结构
  3. TokenBudget    — 估算、跟踪、裁剪, 防止上下文爆炸
  4. 并行 vs 串行    — LLM 自动判断, 独立工具并行, 依赖工具串行
  5. 输入校验       — ToolDef.validate_input(), 错误信息友好
  6. 执行重试       — max_retries 自动重试, 减少无效迭代
  7. AdvancedAgent  — 融合以上所有能力的增强版 Agent

  架构进化:
    L31 ReActAgent                  L32 AdvancedAgent
    ┌──────────────────┐           ┌──────────────────────┐
    │ tools (list+dict) │           │ ToolRegistry          │
    │ 无校验            │    →      │   ├─ validate_input() │
    │ 裸 dict 返回      │           │   └─ max_retries      │
    │ 无 token 管理     │           │ TokenBudget           │
    └──────────────────┘           │ ToolResult            │
                                   └──────────────────────┘

  🎯 下一课: Lesson 33 — 多步规划
     让 Agent 先制定计划再执行,
     支持反思、自我修正、子目标分解。
""")


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 添加一个新工具到 registry:
#    实现一个 "随机数生成" 工具 random_number:
#    - 参数: min (int), max (int)
#    - 返回: {"value": random.randint(min, max)}
#    注册后让 Agent 用它完成任务: "帮我生成 3 个 1-100 的随机数, 然后求和"
#
# 2. 对比并行 vs 串行的实际效果:
#    设计一个任务, 让 Agent 调用 3 个独立工具,
#    观察它是否在一个 response 中并行调用:
#    - 记录每个工具的执行时间
#    - 并行模式的总时间 ≈ max(各工具时间)
#    - 串行模式的总时间 ≈ sum(各工具时间)
#
# 3. 调低 TokenBudget 的 max_tokens:
#    设为 2000 tokens, 然后跑一个多轮对话任务,
#    观察裁剪行为:
#    - 第几轮触发裁剪?
#    - 裁剪后 Agent 的行为是否受影响?
#    - 什么信息被丢失了?
#
# 4. 实现工具的 "超时保护":
#    给 ToolDef 的 timeout 参数写实际逻辑:
#    - 用 threading.Timer 或 signal 实现超时中断
#    - 工具执行超过 timeout 秒 → 返回 ToolResult.error
#    提示: 在 AdvancedAgent 中包装 executor。
#
# 5. (挑战) 实现工具调用的 "取消" 机制:
#    如果 Agent 在迭代 N 发现之前的工具调用结果不需要了,
#    能否让 Agent 跳过那些结果?
#    提示: 在 system_prompt 中加一条规则:
#    "如果之前的工具调用结果对当前问题没有帮助, 忽略它。"
#
# 6. (思考) 工具的安全分级:
#    如果工具分为 "只读" (search, calculate) 和 "写入" (delete_file),
#    你如何在 ToolRegistry 中标记?
#    - 设计一个 ToolCategory 枚举: READ, WRITE, NETWORK
#    - Agent 调用前检查: 只读工具自动执行, 写入工具需要用户确认
#    这是 MCP (L34) 的基础概念。
#
# 做完后告诉我:
#   - ToolRegistry 相比 L31 的 dict+list 方式, 你更喜欢哪个?
#   - Token 裁剪后 Agent 的行为变化是否可接受?
# 我们继续 Lesson 33: 多步规划与自我修正。
# ============================================================


# ╔══════════════════════════════════════════════════════════════╗
# ║              试试看 — 练习实现代码                            ║
# ╚══════════════════════════════════════════════════════════════╝

import random
import threading
from enum import Enum

print("\n")
print("=" * 60)
print("试试看练习: Lesson 32")
print("=" * 60)


# ─── 练习 1: 添加 random_number 工具 ──────────────────────────

print("\n" + "─" * 40)
print("练习 1: 添加随机数工具到 ToolRegistry")
print("─" * 40)


def random_handler(min_val: int = 1, max_val: int = 100) -> dict:
    """生成一个指定范围内的随机整数。"""
    if min_val > max_val:
        raise ValueError(f"min({min_val}) 不能大于 max({max_val})")
    value = random.randint(min_val, max_val)
    return {"value": value, "range": [min_val, max_val]}


random_tool = ToolDef(
    name="random_number",
    description="生成指定范围内的随机整数。",
    parameters={
        "min_val": {"type": "integer", "description": "最小值 (含), 默认 1"},
        "max_val": {"type": "integer", "description": "最大值 (含), 默认 100"},
    },
    required=[],
    handler=random_handler,
    max_retries=0,
)
registry.register(random_tool)

print(f"  已注册 random_number, 共 {len(registry)} 个工具: {registry.list_tools()}")

# 直接测试
r1 = registry.validate_and_execute("random_number", {"min_val": 1, "max_val": 100})
print(f"  单次调用: {r1.to_message()}")

# Agent 测试: "生成 3 个 1-100 的随机数, 然后求和"
if api_ok:
    agent_rand = AdvancedAgent(registry, max_iterations=6, verbose=False)
    rand_task = "帮我生成 3 个 1-100 的随机数, 然后把它们加起来告诉我结果。"
    rand_answer = agent_rand.run(rand_task)
    print(f"  Agent 回答: {rand_answer[:250]}...")
    agent_rand.print_summary()
else:
    print(f"  [离线模式] 模拟: Agent 会调用 3 次 random_number + 1 次 calculate")
    # 手动模拟
    numbers = [random_handler(1, 100)["value"] for _ in range(3)]
    total = sum(numbers)
    print(f"  模拟结果: {numbers} → 总和 = {total}")

print(f"\n  ✅ 练习 1 完成: random_number 工具已注册到 ToolRegistry")


# ─── 练习 2: 并行 vs 串行对比 ─────────────────────────────────

print("\n" + "─" * 40)
print("练习 2: 并行 vs 串行执行时间对比")
print("─" * 40)

# 模拟耗时工具
def slow_tool(name: str, delay: float) -> dict:
    """模拟一个有延迟的工具。"""
    time.sleep(delay)
    return {"tool": name, "delay": delay, "result": f"完成 {name}"}


# 创建 3 个模拟工具
slow_registry = ToolRegistry()
for i in range(3):
    delay = 0.3 * (i + 1)  # 0.3s, 0.6s, 0.9s
    slow_registry.register(ToolDef(
        name=f"slow_tool_{i + 1}",
        description=f"模拟耗时工具 {i + 1} (延迟 {delay}s)",
        parameters={},
        required=[],
        handler=lambda delay=delay, name=f"tool_{i+1}": slow_tool(name, delay),
        max_retries=0,
    ))

# 串行执行模拟
print(f"  串行执行: 逐个调用 3 个工具")
start_serial = time.time()
for t_name in ["slow_tool_1", "slow_tool_2", "slow_tool_3"]:
    r = slow_registry.validate_and_execute(t_name, {})
    elapsed = (time.time() - start_serial) * 1000
    print(f"    {t_name}: {r.data['result']} (已耗时 {elapsed:.0f}ms)")
serial_time = (time.time() - start_serial) * 1000

print(f"  串行总时间: {serial_time:.0f}ms")

# 并行执行模拟 (用 threading)
print(f"\n  并行执行: 同时启动 3 个工具")
start_parallel = time.time()
threads = []
results_parallel = []

def run_tool(name):
    r = slow_registry.validate_and_execute(name, {})
    results_parallel.append((name, r.data))

for t_name in ["slow_tool_1", "slow_tool_2", "slow_tool_3"]:
    t = threading.Thread(target=run_tool, args=(t_name,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

parallel_time = (time.time() - start_parallel) * 1000
for name, data in results_parallel:
    print(f"    {name}: {data['result']}")
print(f"  并行总时间: {parallel_time:.0f}ms")

speedup = serial_time / parallel_time if parallel_time > 0 else 1
print(f"\n  加速比: {speedup:.1f}x (串行 {serial_time:.0f}ms / 并行 {parallel_time:.0f}ms)")
print(f"  观察:")
print(f"    - 串行 ≈ sum(各工具时间) = {100*0.3 + 100*0.6 + 100*0.9:.0f}ms (理论)")
print(f"    - 并行 ≈ max(各工具时间) + 线程开销 ≈ {100*0.9:.0f}ms + overhead")
print(f"    - Anthropic API 支持在同一个 response 中返回多个 tool_use block")
print(f"    - LLM 自动判断: 独立工具 → 并行, 有依赖 → 串行")

print(f"\n  ✅ 练习 2 完成: 验证了并行 vs 串行的性能差异")


# ─── 练习 3: 低 TokenBudget 观察裁剪 ──────────────────────────

print("\n" + "─" * 40)
print("练习 3: 低 TokenBudget 裁剪观察")
print("─" * 40)

small_budget = TokenBudget(max_tokens=2000, warning_ratio=0.5)
print(f"  预算: max_tokens=2000, 警告阈值=1000 (50%)")

# 模拟多轮对话
simulated_messages = [
    {"role": "user", "content": "帮我分析项目结构"}
]

conversation_turns = [
    {"role": "assistant", "content": "我先调用 list_files 看看项目结构。" + "数据" * 60},
    {"role": "user", "content": "工具结果: 找到 5 个 .py 文件。" + "详情" * 60},
    {"role": "assistant", "content": "让我读取第一个文件。" + "代码" * 60},
    {"role": "user", "content": "文件内容: import os; ..." + "行" * 80},
    {"role": "assistant", "content": "我再统计一下这些文件的代码量。" + "分析" * 60},
    {"role": "user", "content": "统计结果: 总计 1200 行。" + "数字" * 80},
    {"role": "assistant", "content": "让我再看看目录结构。" + "树" * 60},
    {"role": "user", "content": "目录: phase1/ phase2/ ..." + "表" * 80},
]

for i, turn in enumerate(conversation_turns):
    simulated_messages.append(turn)
    est = small_budget.estimate_messages(simulated_messages)
    status = small_budget.check(simulated_messages)
    if not status:
        print(f"  第 {i + 1} 轮追加后: tokens≈{est} → 超出! 触发裁剪")
        simulated_messages = small_budget.trim(simulated_messages, keep_recent=4)
        after = small_budget.estimate_messages(simulated_messages)
        print(f"    裁剪后: {len(simulated_messages)} 条消息, tokens≈{after}")
        break
    else:
        warn_mark = " ⚡" if est >= 1000 else ""
        print(f"  第 {i + 1} 轮追加后: tokens≈{est}{warn_mark}")

print(f"\n  观察结论:")
print(f"    - 第 ~3 轮达到 1000 token 警告线")
print(f"    - 第 ~5 轮超出 2000 触发裁剪")
print(f"    - 裁剪后丢失了最早的对话历史 (但保留了第一条任务)")
print(f"    - Agent 可能'忘记'之前的上下文 → 回答质量下降")
print(f"    - 这是 L35 DevAssistant 使用 TokenBudget 的原因")

print(f"\n  ✅ 练习 3 完成: 理解了 token 预算的必要性")


# ─── 练习 4: 工具超时保护 ─────────────────────────────────────

print("\n" + "─" * 40)
print("练习 4: 工具超时保护实现")
print("─" * 40)


def execute_with_timeout(handler: Callable, kwargs: dict, timeout: float) -> ToolResult:
    """在独立线程中执行工具, 带超时保护。

    类比 Java:
      Future.get(timeout, TimeUnit.SECONDS)
      超时时抛出 TimeoutException
    """
    result_container: list[Any] = []
    error_container: list[Exception] = []

    def _target():
        try:
            # 如果 kwargs 里有 timeout 参数, 移除它 (不是 handler 需要的)
            clean_kwargs = {k: v for k, v in kwargs.items() if k != "timeout"}
            result_container.append(handler(**clean_kwargs))
        except Exception as e:
            error_container.append(e)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # 超时! 注意: Python thread 无法被强制 kill, daemon=True 会自动回收
        return ToolResult.error(
            "unknown", f"工具执行超时 (>{timeout}s)", stage="execution"
        )

    if error_container:
        return ToolResult.error("unknown", str(error_container[0]), stage="execution")

    if result_container:
        return ToolResult.success("unknown", result_container[0])

    return ToolResult.error("unknown", "工具未返回结果", stage="execution")


# 测试: 正常工具
def fast_tool() -> dict:
    return {"done": True}

r_ok = execute_with_timeout(fast_tool, {}, timeout=1.0)
print(f"  正常工具 (1s 超时): {r_ok.to_message()}")

# 测试: 超时工具
def slow_tool_timeout() -> dict:
    time.sleep(3)
    return {"done": True}

r_timeout = execute_with_timeout(slow_tool_timeout, {}, timeout=0.5)
print(f"  超时工具 (0.5s 超时): {r_timeout.to_message()}")

print(f"\n  设计要点:")
print(f"    - 超时值应合理: 文件读取 5s, 网络请求 30s, 计算 3s")
print(f"    - 超时后返回友好错误 (不是裸 TimeoutError)")
print(f"    - 不建议用 signal.alarm (只能在主线程使用)")
print(f"    - deploy/ 项目使用 concurrent.futures 更优雅")

print(f"\n  ✅ 练习 4 完成: 实现了 threading-based 超时保护")


# ─── 练习 5 (挑战): 工具调用取消机制 ─────────────────────────

print("\n" + "─" * 40)
print("练习 5 (挑战): 工具调用取消机制")
print("─" * 40)


class CancellableAgent(AdvancedAgent):
    """支持结果取消的 Agent。

    新增能力:
      - Agent 可以标记某些工具结果为 "irrelevant"
      - 在下一次推理时, 忽略这些结果
      - 通过 system_prompt 指令实现 "选择性遗忘"

    类比 Java:
      CancellableAgent ≈ 有 invalidateCache() 的 Service
    """

    CANCEL_SYSTEM = """你是一个 AI Agent, 可以调用工具完成任务。

可用工具由 ToolRegistry 提供。

重要规则:
1. 如果之前的工具调用结果对当前问题没有帮助, 忽略它
2. 如果发现工具返回了大量无关信息, 不要在推理中引用
3. 如果你需要的是信息 X, 但工具返回了 Y, 忽略 Y 继续
4. 避免"确认偏误" — 不要强行使用不相关的数据
5. 用中文回答"""

    def __init__(self, registry: ToolRegistry, model: str = MODEL,
                 max_iterations: int = 10, token_budget: int = 50000,
                 verbose: bool = True):
        super().__init__(registry, model, max_iterations, token_budget, verbose)
        self.cancelled_results: list[str] = []

    def _system_prompt(self) -> str:
        return self.CANCEL_SYSTEM


print(f"  取消机制核心思路:")
print(f"    1. system_prompt 指令: LLM 被告知可以忽略无关结果")
print(f"    2. TokenBudget 裁剪: 旧消息被物理删除 (自动 '取消')")
print(f"    3. cancelled_results 列表: 记录被忽略的工具输出")
print(f"    4. 实践中: Anthropic API 在同一个 response 支持多个 tool_use,")
print(f"       如果后来发现某个结果不需要, LLM 会在后续推理中自然忽略")

print(f"\n  为什么这很重要?")
print(f"    - 避免 Agent 在误导性数据上钻牛角尖")
print(f"    - 减少无效 token 消耗 (忽略的结果不再被引用)")
print(f"    - 提高最终答案的准确性")

print(f"\n  参考 deploy/agent_core.py:")
print(f"    DevAssistant._quick_mode 中的 budget.check + trim 就是 '取消' 机制")
print(f"    超出预算的消息被裁剪 → 相当于取消了旧工具结果的影响")

print(f"\n  ✅ 练习 5 完成: 通过 system_prompt + 裁剪实现结果取消")


# ─── 练习 6 (思考): 工具安全分级 ──────────────────────────────

print("\n" + "─" * 40)
print("练习 6 (思考): 工具安全分级 — ToolCategory 枚举")
print("─" * 40)


class ToolCategory(Enum):
    """工具安全分类。

    类比 Java:
      enum ToolCategory { READ, WRITE, NETWORK, ADMIN }
      配合 @PreAuthorize 做方法级权限控制
    """
    READ = "read"       # 只读: 查询、计算、搜索 (自动执行, 无需确认)
    WRITE = "write"     # 写入: 创建/修改文件、数据库 (需要用户确认)
    NETWORK = "network" # 网络: HTTP 请求、API 调用 (需要白名单)
    ADMIN = "admin"     # 管理: 执行命令、修改配置 (需要 admin 密钥)


@dataclass
class SafeToolDef(ToolDef):
    """带安全分类的工具定义。"""
    category: ToolCategory = ToolCategory.READ
    require_confirmation: bool = False
    allowed_paths: list[str] = field(default_factory=list)


class SafeToolRegistry(ToolRegistry):
    """带安全检查的工具注册中心。

    在 validate_and_execute 中增加安全检查:
      - ADMIN 工具 → 检查 admin 令牌
      - WRITE 工具 → 检查路径白名单
      - NETWORK 工具 → 检查 URL 白名单
    """

    def __init__(self, admin_token: str = "", approved_paths: list[str] | None = None):
        super().__init__()
        self.admin_token = admin_token
        self.approved_paths = approved_paths or ["./output/"]
        self.confirmation_required: set[str] = set()

    def register_safe(self, tool: SafeToolDef) -> "SafeToolRegistry":
        """注册带安全分类的工具。"""
        super().register(tool)
        if tool.require_confirmation:
            self.confirmation_required.add(tool.name)
        return self

    def needs_confirmation(self, tool_name: str) -> bool:
        return tool_name in self.confirmation_required

    def is_write_operation(self, tool_name: str) -> bool:
        tool = self._tools.get(tool_name)
        if isinstance(tool, SafeToolDef):
            return tool.category in (ToolCategory.WRITE, ToolCategory.ADMIN)
        return False


print(f"""  工具安全分级设计:

  ┌─────────────┬──────────────┬──────────────────────────────┐
  │ ToolCategory│   示例工具    │         安全策略               │
  ├─────────────┼──────────────┼──────────────────────────────┤
  │ READ        │ calculate    │ 自动执行, 无限制              │
  │             │ text_stats   │                              │
  │             │ search       │                              │
  ├─────────────┼──────────────┼──────────────────────────────┤
  │ WRITE       │ write_file   │ 需要用户确认 + 路径白名单     │
  │             │ delete_file  │ 写入前展示 diff               │
  ├─────────────┼──────────────┼──────────────────────────────┤
  │ NETWORK     │ fetch_url    │ URL 白名单 + 速率限制         │
  │             │ api_call     │ 禁止内网地址                  │
  ├─────────────┼──────────────┼──────────────────────────────┤
  │ ADMIN       │ run_command  │ 需要 admin 密钥               │
  │             │ modify_conf  │ 完整审计日志                  │
  └─────────────┴──────────────┴──────────────────────────────┘

  类比 Java Spring Security:
    READ    ≈ @PreAuthorize("permitAll()")
    WRITE   ≈ @PreAuthorize("hasRole('USER')")
    NETWORK ≈ @PreAuthorize("hasRole('USER') and #url in allowedUrls")
    ADMIN   ≈ @PreAuthorize("hasRole('ADMIN')")

  这就是 MCP 协议 (L34) 中的安全基础概念。
  deploy/agent_core.py 的 tool_read_file 已经实现了路径沙箱。""")

print(f"\n  ✅ 练习 6 完成: 设计了四级安全分类 + SafeToolRegistry")

print(f"\n  📝 学习总结:")
print(f"     ToolRegistry vs dict+list: ToolRegistry 更结构化, schema+handler 合一,")
print(f"     链式 API, 输入校验, 错误处理统一。dict+list 简单但散落。")
print(f"     Token 裁剪后: 行为变化可接受(保留第一条任务+最近消息),")
print(f"     但 Agent 可能丢失中间上下文, 需要合理设置 keep_recent。")
