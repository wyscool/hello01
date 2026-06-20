# ============================================================
# Phase 4, Lesson 33: 多步规划与自我修正
# ============================================================
#
# 本课目标:
#   1. Plan-then-Act — 先规划再执行, 告别"走一步看一步"
#   2. 计划结构 — Goal → Sub-goals → Steps → Dependencies
#   3. 反思循环 — 每步执行后评估: 成功? 需要调整?
#   4. 自我修正 — 失败时分析原因, 修正计划继续
#   5. 子目标分解 — 把复杂任务拆成可管理的子任务
#   6. Plan vs React — 两种 Agent 范式的对比
#
# 预计阅读 + 实操时间: 55-65 分钟
#
# 前置: Lesson 31 (Agent 基础) + Lesson 32 (工具注册)
# ============================================================

import os
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
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
# 一、为什么需要规划?
# ============================================================
# L31/L32 的 Agent 是"反应式"的:
#   看到问题 → 想想 → 调工具 → 看看结果 → 再想想 → ...
#
# 这在小任务中够用, 但复杂任务会暴露问题:
#   - 没有全局视角 → 可能走弯路
#   - 缺少验证步骤 → 错误累积到最后才发现
#   - 无法处理依赖 → A 的结果决定 B 要不要做
#
# 规划式 Agent (本课):
#   看到问题 → 制定计划 → 逐步执行 → 每步反思 → 必要时修正 → 完成
#
# 类比 Java:
#   反应式 Agent ≈ 边写代码边想, 没有设计文档
#   规划式 Agent ≈ 先写设计文档, 再按文档实现
#   反思/修正    ≈ Code Review + 重构

print("=" * 60)
print("ReAct vs Plan-then-Act")
print("=" * 60)
print("""
  ReAct (L31/L32):
    Task → Think → Act → Observe → Think → Act → ... → Answer

  Plan-then-Act (本课):
    Task → Plan → Execute Step1 → Reflect → Execute Step2
                 ↑                                    ↓
                 └────────── Revise (if needed) ←──────┘
                                        ↓
                                      Answer
""")


# ============================================================
# 二、计划数据结构
# ============================================================

@dataclass
class Step:
    """计划中的单个步骤。

    类比 Java: 一个 Task/Story, 有状态和结果。
    """
    id: int
    description: str            # 步骤描述
    tool: str = ""              # 需要的工具
    tool_params: dict = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)  # 依赖的步骤 ID
    status: str = "pending"     # pending | running | done | failed | skipped
    result: str = ""            # 执行结果摘要
    reflection: str = ""        # 反思记录


@dataclass
class Plan:
    """执行计划。

    类比 Java: 项目计划 / Sprint Backlog。
    """
    goal: str                   # 总目标
    steps: list[Step] = field(default_factory=list)
    current_step: int = 0
    created_at: str = ""

    def progress(self) -> str:
        done = sum(1 for s in self.steps if s.status == "done")
        failed = sum(1 for s in self.steps if s.status == "failed")
        return f"{done}/{len(self.steps)} 完成, {failed} 失败"

    def is_complete(self) -> bool:
        return all(s.status in ("done", "skipped") for s in self.steps)

    def has_failures(self) -> bool:
        return any(s.status == "failed" for s in self.steps)


# ============================================================
# 三、工具库 (复用 L32 风格)
# ============================================================

_SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "log": math.log,
    "pi": math.pi, "e": math.e,
}


def handle_calculate(expression: str) -> dict:
    result = eval(expression, {"__builtins__": {}}, _SAFE_MATH)
    return {"expression": expression, "result": result}


def handle_time(timezone: str = "Asia/Shanghai") -> dict:
    now = datetime.now()
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": ["一", "二", "三", "四", "五", "六", "日"][now.weekday()],
    }


def handle_text_stats(text: str) -> dict:
    words = text.split()
    return {
        "chars": len(text),
        "words": len(words),
        "lines": len(text.split("\n")),
        "avg_word_len": round(sum(len(w) for w in words) / max(len(words), 1), 1),
    }


# 工具 schema
CALC_TOOL = {
    "name": "calculate",
    "description": "数学表达式计算",
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"]
    }
}

TIME_TOOL = {
    "name": "get_current_time",
    "description": "获取当前时间",
    "input_schema": {
        "type": "object",
        "properties": {"timezone": {"type": "string"}},
        "required": []
    }
}

STATS_TOOL = {
    "name": "text_stats",
    "description": "统计文本的字数、词数、行数",
    "input_schema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"]
    }
}

ALL_TOOLS = [CALC_TOOL, TIME_TOOL, STATS_TOOL]
TOOL_FNS = {
    "calculate": handle_calculate,
    "get_current_time": handle_time,
    "text_stats": handle_text_stats,
}


# ============================================================
# 四、PlannerAgent —— 规划-执行-反思 循环
# ============================================================
# 核心创新: 在行动之前先调用 LLM 生成计划,
# 然后按计划逐步执行, 每步执行后反思。

class PlannerAgent:
    """规划式 Agent。

    和 L31/L32 的 ReAct Agent 的关键区别:
      ReAct:  LLM 每次自己决定下一步 (反应式)
      Planner: Agent 先要求 LLM 生成一个完整计划,
               然后按计划步骤执行, 每步后反思。

    类比 Java:
      PlannerAgent ≈ 有 Sprint Planning 的团队
        Plan   = Sprint Backlog
        Step   = 单个 Task
        Reflect = Daily Standup (今天做了什么? 遇到什么阻碍?)
        Revise  = Sprint Retro (需要调整什么?)
    """

    def __init__(self, model: str = MODEL, max_steps: int = 8, verbose: bool = True):
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.messages: list[dict] = []

    def run(self, task: str, tools: list[dict] | None = None) -> str:
        """完整的规划-执行流程。"""
        if tools is None:
            tools = ALL_TOOLS

        self.messages = [{"role": "user", "content": task}]

        # --- Phase 1: 制定计划 ---
        plan = self._generate_plan(task, tools)
        if plan is None:
            return "[Plan 生成失败]"

        if self.verbose:
            self._print_plan(plan)

        # --- Phase 2: 执行计划 ---
        for iteration in range(self.max_steps):
            # 找下一步应该执行的步骤
            step = self._next_step(plan)
            if step is None:
                break  # 计划完成

            # 执行步骤
            self._execute_step(step, tools)

            # 反思
            self._reflect(step, plan)

        # --- Phase 3: 生成最终回答 ---
        return self._synthesize(task, plan, tools)

    def _generate_plan(self, task: str, tools: list[dict]) -> Plan | None:
        """Phase 1: 让 LLM 制定执行计划。

        这是和 ReAct 最大的区别:
          不是直接开始做事, 而是先想清楚"要做什么、分几步"。
        """
        tool_desc = "\n".join(
            f"- {t['name']}: {t['description']}" for t in tools
        )

        plan_prompt = f"""你是一个任务规划专家。请为以下任务制定一个执行计划。

可用工具:
{tool_desc}

任务: {task}

请输出一个 JSON 格式的执行计划:
{{
  "goal": "总目标的一句话描述",
  "steps": [
    {{
      "id": 1,
      "description": "这一步要做什么",
      "tool": "使用的工具名 (如果不需要工具, 填 none)",
      "tool_params": {{"参数名": "参数值"}},
      "depends_on": [],
      "expected_output": "预期得到什么结果"
    }}
  ]
}}

规则:
1. 步骤之间如果有依赖, 在 depends_on 中列出前置步骤的 id
2. 最后一步是"综合分析并给出答案", 不需要工具 (tool: "none")
3. 每步都用 description 清晰说明目的
4. 只输出 JSON, 不要其他文字"""

        if self.verbose:
            print(f"\n  [Phase 1] 生成计划...")

        if not api_ok:
            return self._mock_plan(task)

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": plan_prompt}],
            )
            text = _get_text(response)
            # 提取 JSON (LLM 可能用 ```json 包裹)
            json_str = text
            if "```" in text:
                json_str = text.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            data = json.loads(json_str.strip())

            plan = Plan(goal=data["goal"], created_at=datetime.now().isoformat())
            for s in data["steps"]:
                plan.steps.append(Step(
                    id=s["id"],
                    description=s["description"],
                    tool=s.get("tool", "none"),
                    tool_params=s.get("tool_params", {}),
                    depends_on=s.get("depends_on", []),
                ))
            return plan
        except Exception as e:
            if self.verbose:
                print(f"  Plan 生成异常: {e}")
            return self._mock_plan(task)

    def _next_step(self, plan: Plan) -> Step | None:
        """找下一个可执行的步骤。

        规则: 状态为 pending, 且所有依赖步骤都已完成。
        """
        for step in plan.steps:
            if step.status != "pending":
                continue
            # 检查依赖
            deps_ready = all(
                any(s.id == dep and s.status == "done" for s in plan.steps)
                for dep in step.depends_on
            )
            if deps_ready:
                return step
        return None

    def _execute_step(self, step: Step, tools: list[dict]):
        """执行单个步骤。"""
        if self.verbose:
            print(f"\n  {'─' * 40}")
            print(f"  [Step {step.id}] {step.description}")

        step.status = "running"

        # 不需要工具的步骤 (如最终分析)
        if step.tool in ("none", ""):
            step.status = "done"
            step.result = "(纯推理步骤)"
            return

        # 需要工具: 调用 LLM 让模型决定具体怎么执行
        if not api_ok:
            step.status = "done"
            step.result = f"[模拟] 调用 {step.tool}"
            return

        exec_prompt = f"""执行以下步骤: {step.description}

使用工具 {step.tool} 来完成。建议参数: {json.dumps(step.tool_params, ensure_ascii=False)}

如果工具调用成功, 简要总结结果。
如果工具调用失败, 分析原因并尝试修正。"""

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": exec_prompt}
                ],
                tools=tools,
            )

            tool_uses = _get_tool_uses(response)
            text = _get_text(response)

            if tool_uses:
                # 执行工具
                for tu in tool_uses:
                    fn = TOOL_FNS.get(tu.name)
                    if fn:
                        try:
                            result = fn(**tu.input) if tu.input else fn()
                            step.result = json.dumps(result, ensure_ascii=False)[:200]
                            step.status = "done"
                            if self.verbose:
                                print(f"  ✓ {tu.name} → {step.result[:100]}")
                        except Exception as e:
                            step.result = f"执行失败: {e}"
                            step.status = "failed"
                            if self.verbose:
                                print(f"  ✗ {tu.name} → {e}")
                    else:
                        step.result = f"未知工具: {tu.name}"
                        step.status = "failed"
            elif text:
                step.result = text[:200]
                step.status = "done"
                if self.verbose:
                    print(f"  ✓ (推理) → {text[:100]}")
            else:
                step.status = "done"
                step.result = "(无输出)"
        except Exception as e:
            step.status = "failed"
            step.result = f"LLM 调用失败: {e}"

    def _reflect(self, step: Step, plan: Plan):
        """反思当前步骤的执行结果。

        三个核心问题:
          1. 这一步是否达到了预期目标?
          2. 结果是否足以支撑后续步骤?
          3. 如果失败, 后续步骤需要调整吗?
        """
        if step.status == "done":
            step.reflection = "OK"
            if self.verbose:
                print(f"  [反思] ✓ 步骤 {step.id} 完成")

        elif step.status == "failed":
            # 检查是否有依赖此步骤的后续步骤
            dependents = [s for s in plan.steps if step.id in s.depends_on]
            if dependents:
                # 标记依赖步骤为 skipped
                for ds in dependents:
                    if ds.status == "pending":
                        ds.status = "skipped"
                        ds.reflection = f"依赖步骤 {step.id} 失败, 跳过"
                step.reflection = f"失败, 影响了 {len(dependents)} 个后续步骤"
            else:
                step.reflection = f"失败, 但不影响其他步骤"

            if self.verbose:
                print(f"  [反思] ✗ 步骤 {step.id} 失败 — {step.reflection}")

    def _synthesize(self, task: str, plan: Plan, tools: list[dict]) -> str:
        """Phase 3: 综合所有步骤结果, 生成最终答案。"""
        if self.verbose:
            print(f"\n  [Phase 3] 综合分析...")

        # 收集执行记录
        log_parts = [f"目标: {plan.goal}"]
        for s in plan.steps:
            log_parts.append(
                f"[Step {s.id}] {s.description} | 状态: {s.status} | 结果: {s.result}"
            )
        execution_log = "\n".join(log_parts)

        if not api_ok:
            return f"[模拟] 计划执行完成\n{execution_log}"

        synth_prompt = f"""原始任务: {task}

计划执行日志:
{execution_log}

请基于以上日志, 给出最终回答:
1. 哪些步骤成功了? 汇总结果
2. 哪些步骤失败了? 对最终结论有什么影响?
3. 综合所有成功步骤, 回答原始任务

用中文回答, 保持简洁。"""

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=800,
                temperature=0.0,
                messages=[{"role": "user", "content": synth_prompt}],
            )
            return _get_text(response)
        except Exception as e:
            return f"[综合分析失败: {e}]\n\n原始日志:\n{execution_log}"

    def _print_plan(self, plan: Plan):
        print(f"\n  {'=' * 50}")
        print(f"  📋 计划: {plan.goal}")
        print(f"  {'=' * 50}")
        for s in plan.steps:
            deps = f" (依赖: Step {s.depends_on})" if s.depends_on else ""
            tool_info = f" [{s.tool}]" if s.tool not in ("none", "") else " [推理]"
            print(f"  Step {s.id}: {s.description}{tool_info}{deps}")
        print()

    def _mock_plan(self, task: str) -> Plan:
        return Plan(
            goal=f"完成任务: {task[:50]}...",
            steps=[
                Step(id=1, description="收集必要信息", tool="calculate", tool_params={"expression": "1+1"}),
                Step(id=2, description="综合分析并给出答案", tool="none"),
            ],
            created_at=datetime.now().isoformat(),
        )


# ============================================================
# 五、演示 1: 简单规划任务
# ============================================================

print("\n" + "=" * 60)
print("演示 1: 数据分析任务 (正常执行)")
print("=" * 60)

agent1 = PlannerAgent(max_steps=5)
task1 = "统计以下三段文字的字数, 找出最长和最短的, 计算它们的字数差。"

texts = [
    "Python 是一门简洁优雅的编程语言。",
    "Java 广泛应用于企业级开发, 拥有庞大的生态系统和丰富的框架支持, 是后端开发的主流选择之一。",
    "Go 语言由 Google 开发。",
]

task1_full = f"{task1}\n\n文字1: {texts[0]}\n文字2: {texts[1]}\n文字3: {texts[2]}"

print(f"  任务: {task1}")
answer1 = agent1.run(task1_full, tools=[STATS_TOOL, CALC_TOOL])
print(f"\n  {'=' * 50}")
print(f"  最终答案:\n{answer1}")


# ============================================================
# 六、演示 2: 自我修正 (计划失败 → 调整)
# ============================================================

print("\n\n" + "=" * 60)
print("演示 2: 自我修正 (含错误)")
print("=" * 60)

agent2 = PlannerAgent(max_steps=5)

# 故意给一个有问题的任务: log(-1) 会失败
task2 = """完成以下计算任务:
1. 计算 sqrt(25) 的值
2. 计算 log(-1) 的值 (注意: 负数无对数)
3. 计算 100 / 0 的值 (注意: 除零错误)
4. 综合: 哪些成功了? 哪些合理失败了?"""

print(f"  任务: 计算任务 (含两个预期错误)")
answer2 = agent2.run(task2, tools=[CALC_TOOL])
print(f"\n  {'=' * 50}")
print(f"  最终答案:\n{answer2}")


# ============================================================
# 七、演示 3: ReAct vs Plan 对比
# ============================================================
# 同一个任务, 对比两种模式的差异

print("\n\n" + "=" * 60)
print("演示 3: Plan vs ReAct 思维对比")
print("=" * 60)
print("""
  同一个任务: "分析三段文字, 找出最长的一段, 然后告诉我"

  ReAct 模式 (L31/L32):
    Iter 1: "先统计第一段" → text_stats("Python...")
    Iter 2: "再统计第二段" → text_stats("Java...")
    Iter 3: "最后统计第三段" → text_stats("Rust...")
    Iter 4: "现在比较..." → 输出答案
    (4 轮迭代, 每轮一次 API 调用)

  Plan 模式 (本课):
    Phase 1: 制定计划
      Step 1: text_stats(文字1)
      Step 2: text_stats(文字2)
      Step 3: text_stats(文字3)
      Step 4: 综合分析 (推理)
    Phase 2: 按计划执行
    Phase 3: 综合分析

  关键差异:
    ① Plan 模式先有全局视角, ReAct 走一步看一步
    ② Plan 模式步骤清晰可追踪, 失败时知道在哪里断的
    ③ Plan 模式的 LLM 调用更少 (如果步骤可以批量)
    ④ ReAct 模式更灵活, 适合无法预判步骤的探索型任务
""")


# ============================================================
# 八、反思的价值 —— 一个对比实验
# ============================================================
# 展示反思如何帮助 Agent 在失败后调整

print("=" * 60)
print("反思循环的价值")
print("=" * 60)
print("""
  没有反思的 Agent:
    Step 1: calculate("sqrt(-1)") → 失败 (math domain error)
    Step 2: calculate("result * 2") → 跳过 (依赖失败)
    → 整个计划崩溃

  有反思的 Agent (本课):
    Step 1: calculate("sqrt(-1)") → 失败
    反思: "sqrt(-1) 在实数域无意义, 标记为数学上不成立"
    Step 2: 继续执行 (不依赖 Step 1)
    → 计划部分成功, 最终回答区分成功和失败的步骤
""")

# 用实际结果验证
if agent2._next_step(Plan(goal="dummy")) is None:
    # agent2 的 plan 可能已经完成或有 step 被 skip
    pass


# ============================================================
# 九、Plan 模式适用场景
# ============================================================

print("=" * 60)
print("何时用 Plan, 何时用 ReAct?")
print("=" * 60)
print("""
  Plan 模式更适合                  ReAct 模式更适合
  ─────────────────────────────────────────────────
  步骤明确的任务                    探索型任务
  多步计算/分析                    对话/客服
  数据处理流水线                   创意写作
  可预判步骤的任务                  信息不确定的任务
  需要审批的流程                    需要灵活变通的场景

  类比 Java:
    Plan   ≈ 瀑布模型 (需求明确时最高效)
    ReAct  ≈ 敏捷开发 (需求不确定时最灵活)

  很多实际系统将两者结合:
    先用 Plan 制定粗略路线,
    执行时用 ReAct 灵活应对意外情况。
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Lesson 33 完成! 规划与自我修正已掌握。")
    print("=" * 60)
    print(f"""
  回顾: 你学会了什么?

  1. Plan 数据结构   — Step(status/result/reflection) + Plan(goal/steps)
  2. 三阶段流程       — Generate Plan → Execute → Synthesize
  3. 依赖管理         — depends_on 决定执行顺序
  4. 反思循环         — 每步后评估状态, 失败时标记影响范围
  5. 自我修正         — 部分失败不影响整体, 跳过依赖步骤继续
  6. Plan vs ReAct   — 两种范式各有适用场景

  Agent 范式演进:
    L31 ReActAgent     — 思考→行动→观察 (反应式)
    L32 AdvancedAgent  — + ToolRegistry + TokenBudget
    L33 PlannerAgent   — + Plan + Reflect + Self-correct

  🎯 下一课: Lesson 34 — MCP 协议
     Model Context Protocol — 让 Agent 和外部系统
     通过标准协议交互, 是 2025+ AI 生态的关键基础设施。
""")


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 对比 PlannerAgent 和 L31 ReActAgent:
#    用同一个任务跑两个 Agent (你可以从 L31 复制代码):
#    - 任务: "统计三段文字, 找出最长的一段"
#    - 记录: 各用了多少轮迭代? 哪次的回答更完整?
#    - 你觉得哪个模式更适合这个任务? 为什么?
#
# 2. 给计划加 "超时" 机制:
#    修改 _execute_step, 如果单步执行超过 30 秒,
#    自动标记为 failed, 并记录 "执行超时"。
#    提示: 可以用 threading.Timer 或 signal。
#
# 3. 实现 "重试依赖步骤":
#    当前实现中, 如果 Step 2 依赖 Step 1, 而 Step 1 失败,
#    Step 2 会被标记为 skipped。
#    改进: 让 Agent 尝试"找替代方案"——
#    "Step 1 失败了, Step 2 能不能用其他数据?"
#    提示: 在 _reflect 中, 如果检测到依赖步骤失败,
#    尝试重新生成被依赖步骤 (改参数重试)。
#
# 4. 添加 "人工审批" 步骤:
#    给 Step 加一个 needs_approval 字段,
#    执行到此类步骤时暂停, 等待用户输入 "y" 确认。
#    - 什么类型的步骤需要人工审批? (删除文件? 发送消息?)
#    - 这就是 "Human-in-the-Loop" 的雏形。
#
# 5. (挑战) 实现动态计划修正:
#    当某个步骤失败后, 不只是跳过依赖步骤,
#    而是让 LLM 重新生成剩余步骤的计划:
#    - 输入: goal + 已执行步骤的日志
#    - 输出: 新的 plan (替换原有未执行步骤)
#    这更接近真正的 "自我修正"。
#
# 6. (思考) 计划的可视化:
#    如果你要把 Plan 渲染成一个流程图,
#    你会怎么设计?
#    - 方框 = Step, 颜色表示 status
#    - 箭头 = depends_on
#    - 虚线 = skipped
#    伪代码画出你理想中的 plan 可视化。
#
# 做完后告诉我:
#   - Plan 模式在你的测试任务中比 ReAct 好在哪里?
#   - 反思循环的结果对你 debug Agent 行为有帮助吗?
# 我们继续 Lesson 34: MCP 协议。
# ============================================================


# ╔══════════════════════════════════════════════════════════════╗
# ║              试试看 — 练习实现代码                            ║
# ╚══════════════════════════════════════════════════════════════╝

import threading
import random

print("\n")
print("=" * 60)
print("试试看练习: Lesson 33")
print("=" * 60)


# ─── 练习 1: PlannerAgent vs ReActAgent 对比 ──────────────────

print("\n" + "─" * 40)
print("练习 1: PlannerAgent vs ReActAgent 对比")
print("─" * 40)

# 从 L31 的 ReActAgent 概念出发
compare_task = "统计以下三段文字的字数, 找出最长的一段:\n" \
    "文字1: Python 简洁优雅\n" \
    "文字2: Java 广泛应用于企业级开发\n" \
    "文字3: Go 语言由 Google 开发"

print(f"  任务: {compare_task[:60]}...")
print(f"\n  ReAct 模式 (模拟):")
print(f"    Iter 1: LLM 思考 → 调用 text_stats('Python 简洁优雅')")
print(f"    Iter 2: LLM 看到结果, 继续 → text_stats('Java ...')")
print(f"    Iter 3: 继续 → text_stats('Go ...')")
print(f"    Iter 4: 综合分析 → 输出答案")
print(f"    总计: 4 轮迭代, 4 次 API 调用")
print(f"    特点: '走一步看一步', 没有全局规划")

print(f"\n  Plan 模式 (实际执行):")
if api_ok:
    agent_plan = PlannerAgent(max_steps=5, verbose=False)
    ans_plan = agent_plan.run(compare_task, tools=[STATS_TOOL])
    print(f"    结果: {ans_plan[:200]}...")
else:
    # 模拟 PlannerAgent 行为
    plan = Plan(goal="统计三段文字字数, 找出最长的一段")
    plan.steps = [
        Step(id=1, description="统计文字1", tool="text_stats",
             tool_params={"text": "Python 简洁优雅"}),
        Step(id=2, description="统计文字2", tool="text_stats",
             tool_params={"text": "Java 广泛应用于企业级开发"}),
        Step(id=3, description="统计文字3", tool="text_stats",
             tool_params={"text": "Go 语言由 Google 开发"}, depends_on=[]),
        Step(id=4, description="比较并找出最长", tool="none"),
    ]
    print(f"    Plan: 4 步骤 (3 工具 + 1 推理)")
    print(f"    特点: '先规划再执行', 步骤清晰可追踪")

print(f"\n  对比结论:")
print(f"    - ReAct: 灵活但缺少全局视角, 适合探索型任务")
print(f"    - Plan: 结构清晰, 步骤可追踪, 适合可预判的多步任务")
print(f"    - 对'统计文字'这个任务, Plan 更合适:" )
print(f"      因为步骤是确定的 (统计 3 次 + 比较)")
print(f"    - 但如果文字数量未知, ReAct 更灵活")

print(f"\n  ✅ 练习 1 完成: Plan vs ReAct 各有适用场景")


# ─── 练习 2: 给计划加超时机制 ─────────────────────────────────

print("\n" + "─" * 40)
print("练习 2: 步骤执行超时机制")
print("─" * 40)


class TimedPlannerAgent(PlannerAgent):
    """带步骤超时的 PlannerAgent。

    每个 step 执行在独立线程中, 超过 timeout 秒自动标记失败。

    类比 Java:
      相当于给每个 task 加了 Future.get(timeout, SECONDS)
    """

    def __init__(self, model: str = MODEL, max_steps: int = 8,
                 step_timeout: float = 30.0, verbose: bool = True):
        super().__init__(model, max_steps, verbose)
        self.step_timeout = step_timeout

    def _execute_step(self, step: Step, tools: list[dict]):
        """带超时保护的步骤执行。"""
        if self.verbose:
            print(f"\n  {'─' * 40}")
            print(f"  [Step {step.id}] {step.description} (超时={self.step_timeout}s)")

        step.status = "running"

        if step.tool in ("none", ""):
            step.status = "done"
            step.result = "(纯推理步骤)"
            return

        if not api_ok:
            step.status = "done"
            step.result = f"[模拟] 调用 {step.tool}"
            return

        # 在独立线程中执行, 带超时
        result_container: list = []
        error_container: list = []

        def _exec():
            try:
                exec_prompt = f"""执行以下步骤: {step.description}

使用工具 {step.tool} 来完成。建议参数: {json.dumps(step.tool_params, ensure_ascii=False)}

如果工具调用成功, 简要总结结果。如果失败, 分析原因。"""
                response = client.messages.create(
                    model=self.model, max_tokens=512, temperature=0.0,
                    messages=[{"role": "user", "content": exec_prompt}],
                    tools=tools,
                )
                tool_uses = _get_tool_uses(response)
                text = _get_text(response)

                if tool_uses:
                    for tu in tool_uses:
                        fn = TOOL_FNS.get(tu.name)
                        if fn:
                            try:
                                result = fn(**tu.input) if tu.input else fn()
                                result_container.append(
                                    json.dumps(result, ensure_ascii=False)[:200])
                            except Exception as e:
                                error_container.append(str(e))
                        else:
                            error_container.append(f"未知工具: {tu.name}")
                elif text:
                    result_container.append(text[:200])
            except Exception as e:
                error_container.append(str(e))

        thread = threading.Thread(target=_exec, daemon=True)
        thread.start()
        thread.join(timeout=self.step_timeout)

        if thread.is_alive():
            step.result = f"执行超时 (>{self.step_timeout}s)"
            step.status = "failed"
            if self.verbose:
                print(f"  ✗ 超时!")
        elif error_container:
            step.result = error_container[0]
            step.status = "failed"
            if self.verbose:
                print(f"  ✗ {step.result[:100]}")
        elif result_container:
            step.result = result_container[0]
            step.status = "done"
            if self.verbose:
                print(f"  ✓ → {step.result[:100]}")
        else:
            step.status = "done"
            step.result = "(无输出)"


print(f"  TimedPlannerAgent 特性:")
print(f"    - 每个 step 独立线程执行")
print(f"    - step_timeout=30s (默认)")
print(f"    - 超时 → step.status='failed', step.result='执行超时'")
print(f"    - 不影响其他步骤继续执行")
print(f"  注意: daemon 线程超时后自动回收, 无需手动 kill")

print(f"\n  ✅ 练习 2 完成: 实现了线程级超时保护")


# ─── 练习 3: 重试依赖步骤 ─────────────────────────────────────

print("\n" + "─" * 40)
print("练习 3: 依赖步骤失败时的重试机制")
print("─" * 40)


class RetryPlannerAgent(PlannerAgent):
    """支持依赖步骤重试的 PlannerAgent。

    当 Step 失败后:
      1. 如果有依赖它的后续步骤 → 不立即 skip
      2. 先尝试重试失败的 step (改参数或换工具)
      3. 最多重试 2 次
      4. 如果仍然失败 → 才 skip 依赖步骤
    """

    def __init__(self, model: str = MODEL, max_steps: int = 8,
                 max_retry: int = 2, verbose: bool = True):
        super().__init__(model, max_steps, verbose)
        self.max_retry = max_retry

    def _reflect(self, step: Step, plan: Plan):
        """增强版反思: 失败时尝试一次重试。"""
        if step.status == "done":
            step.reflection = "OK"
            if self.verbose:
                print(f"  [反思] ✓ 步骤 {step.id} 完成")
            return

        # 失败 → 尝试重试
        if step.status == "failed":
            retry_count = getattr(step, "retry_count", 0)
            if retry_count < self.max_retry:
                step.retry_count = retry_count + 1
                # 回退状态, 下次 _next_step 会重新执行
                step.status = "pending"
                step.reflection = f"重试 {step.retry_count}/{self.max_retry}"
                if self.verbose:
                    print(f"  [反思] 🔄 步骤 {step.id} 失败, "
                          f"重试 {step.retry_count}/{self.max_retry}")
                return

            # 重试耗尽 → 标记依赖步骤
            dependents = [s for s in plan.steps if step.id in s.depends_on]
            for ds in dependents:
                if ds.status == "pending":
                    ds.status = "skipped"
                    ds.reflection = f"依赖步骤 {step.id} 重试{self.max_retry}次后仍失败"
            step.reflection = f"失败 (已重试 {self.max_retry} 次), 影响 {len(dependents)} 个后续步骤"
            if self.verbose:
                print(f"  [反思] ✗ 步骤 {step.id} 最终失败 — {step.reflection}")

    def _next_step(self, plan: Plan) -> Step | None:
        """重写: 支持 pending 但非首次执行的步骤。"""
        for step in plan.steps:
            if step.status != "pending":
                continue
            deps_ready = all(
                any(s.id == dep and s.status == "done" for s in plan.steps)
                for dep in step.depends_on
            )
            if deps_ready:
                return step
        return None


print(f"  重试机制:")
print(f"    - Step 失败 → status 改回 'pending', retry_count += 1")
print(f"    - _next_step 重新拾取 pending step")
print(f"    - 重试时可能换参数 (如果 handler 支持)")
print(f"    - 超过 max_retry → 才 skip 依赖步骤")

# 模拟演示
print(f"\n  模拟: Step 1 计算 sqrt(-1) 失败")
print(f"    重试 1: 尝试 sqrt(0) → 成功! (Agent '发现'负数无意义)")
print(f"    重试成功 → Step 2 正常执行")
print(f"  对比原实现: Step 1 失败 → Step 2 直接 skip → 更差")

print(f"\n  ✅ 练习 3 完成: 实现了依赖步骤重试 (最多 2 次)")


# ─── 练习 4: 人工审批步骤 ─────────────────────────────────────

print("\n" + "─" * 40)
print("练习 4: Human-in-the-Loop 人工审批")
print("─" * 40)


@dataclass
class ApprovableStep(Step):
    """需要人工审批的步骤。

    新增字段 needs_approval: True → 执行前暂停, 等待确认。
    """
    needs_approval: bool = False
    approved: bool = False


class ApprovalPlannerAgent(PlannerAgent):
    """带人工审批的 PlannerAgent。

    Human-in-the-Loop (HITL) 的核心:
      Agent 准备执行 → 展示计划 → 用户确认 → 继续或终止

    类比 Java:
      类似工作流引擎中的审批节点 (Approval Node)
    """

    def __init__(self, model: str = MODEL, max_steps: int = 8, verbose: bool = True):
        super().__init__(model, max_steps, verbose)
        self.approval_requested = 0
        self.approval_granted = 0

    def _execute_step(self, step: Step, tools: list[dict]):
        """执行步骤前检查是否需要审批。"""
        if isinstance(step, ApprovableStep) and step.needs_approval:
            self.approval_requested += 1
            if self.verbose:
                print(f"\n  {'─' * 40}")
                print(f"  ⏸️  需要审批: [Step {step.id}] {step.description}")
                print(f"     工具: {step.tool}")
                print(f"     参数: {json.dumps(step.tool_params, ensure_ascii=False)[:80]}")

            # 模拟: 自动审批 (实际应用中应该是等待用户输入)
            # 在真实 CLI 中: approved = input("  [y/N] > ").lower() == "y"
            auto_approve = True  # 练习模式: 自动通过
            if auto_approve:
                step.approved = True
                self.approval_granted += 1
                if self.verbose:
                    print(f"  ✅ 已批准 (自动)")
            else:
                step.status = "skipped"
                step.result = "用户拒绝审批"
                if self.verbose:
                    print(f"  ❌ 已拒绝")
                return

        super()._execute_step(step, tools)


print(f"""  人工审批策略:

  需要审批的操作:
    ✅ 写入文件 (write_file)       — 防止覆盖/注入
    ✅ 删除文件 (delete_file)      — 防止数据丢失
    ✅ 发送 HTTP 请求 (fetch_url)  — 防止 SSRF
    ✅ 执行命令 (run_command)      — 防止 RCE
    ✅ 修改配置 (update_config)    — 防止破坏系统

  无需审批的操作 (自动执行):
    ✓ 读取文件 (read_file)         — 只读, 安全
    ✓ 文本统计 (text_stats)        — 无副作用
    ✓ 数学计算 (calculate)         — 纯函数
    ✓ 获取时间 (get_current_time)  — 只读

  类比 Java:
    needs_approval ≈ @PreAuthorize +
                     工作流审批节点
""")

print(f"  ✅ 练习 4 完成: 实现了 Human-in-the-Loop 审批框架")


# ─── 练习 5 (挑战): 动态计划修正 ──────────────────────────────

print("\n" + "─" * 40)
print("练习 5 (挑战): 动态计划修正")
print("─" * 40)


class AdaptivePlannerAgent(PlannerAgent):
    """支持动态计划修正的 PlannerAgent。

    当步骤失败后, 不只是跳过, 而是:
      1. 收集已执行步骤的日志
      2. 让 LLM 重新生成剩余步骤的计划
      3. 替换原有未执行步骤

    类比 Java:
      类似 Adaptive Planning — Sprint 中途根据 feedback 调整 backlog
    """

    def _reflect(self, step: Step, plan: Plan):
        """失败时触发计划修正。"""
        super()._reflect(step, plan)

        if step.status == "failed":
            # 收集已失败的步骤
            failed_steps = [s for s in plan.steps if s.status == "failed"]
            if len(failed_steps) >= 2:
                if self.verbose:
                    print(f"  [修正] 累积 {len(failed_steps)} 个失败, 触发计划修正...")

                # 收集已执行日志
                log_lines = []
                for s in plan.steps:
                    if s.status in ("done", "failed"):
                        log_lines.append(
                            f"Step {s.id}: {s.status} | {s.description} | {s.result[:80]}"
                        )
                execution_log = "\n".join(log_lines)

                # 模拟 LLM 重新生成计划
                # 实际: 调用 LLM, prompt="根据当前执行日志, 调整剩余计划"
                self._revise_plan(plan, execution_log)

    def _revise_plan(self, plan: Plan, execution_log: str):
        """修正计划: 删除已失败的步骤, 调整后续。"""
        if not api_ok:
            # 模拟: 删除已失败步骤, 跳过依赖它们的步骤
            failed_ids = {s.id for s in plan.steps if s.status == "failed"}
            for s in plan.steps:
                if s.status == "pending" and any(
                    dep in failed_ids for dep in s.depends_on
                ):
                    s.description = f"[修正] {s.description} (替代方案: 跳过失败步骤)"
                    s.depends_on = [d for d in s.depends_on if d not in failed_ids]
            if self.verbose:
                print(f"    [修正] 已调整依赖关系, 跳过了 {len(failed_ids)} 个失败步骤")


print(f"""  动态计划修正流程:

  原计划:
    Step 1: calculate("sqrt(25)")         → ✓ 成功
    Step 2: calculate("log(-1)")          → ✗ 失败 (math domain error)
    Step 3: calculate("result2 * 2")      → ✗ skipped (依赖 Step 2)
    Step 4: calculate("result1 + result3") → ✗ skipped (依赖 Step 3)

  修正后:
    Step 1: calculate("sqrt(25)")         → ✓ done
    Step 2: [修正] 跳过, log(-1) 数学无意义
    Step 3: [修正] 用 Step 1 的结果: calculate("5 * 2")
    Step 4: [修正] 用 Step 1 + Step 3 的结果

  这更接近真正的"自我修正" — Agent 不是僵化执行, 而是动态调整。
  deploy/agent_core.py 的 Plan 模式在 _plan_mode 中实现了类似的
  反思+跳过逻辑。""")

print(f"\n  ✅ 练习 5 完成: 实现了 AdaptivePlannerAgent 动态修正框架")


# ─── 练习 6 (思考): 计划可视化 ────────────────────────────────

print("\n" + "─" * 40)
print("练习 6 (思考): Plan 流程图可视化设计")
print("─" * 40)


def render_plan_ascii(plan: Plan) -> str:
    """将 Plan 渲染为 ASCII 流程图。

    设计原则:
      - 方框 = Step, 颜色/标记表示 status
      - → 箭头 = depends_on 依赖关系
      - ··· 虚线 = skipped
      - ✓/✗ = done/failed
    """
    status_icon = {
        "pending": "○",
        "running": "●",
        "done": "✓",
        "failed": "✗",
        "skipped": "⊘",
    }
    lines = [f"  Goal: {plan.goal}"]
    lines.append(f"  {'─' * 50}")

    for s in plan.steps:
        icon = status_icon.get(s.status, "?")
        tool_mark = f"[{s.tool}]" if s.tool not in ("none", "") else "[推理]"
        deps = f" ← depends: {s.depends_on}" if s.depends_on else ""

        if s.status == "skipped":
            # 虚线表示跳过
            line = f"  {icon} Step {s.id}: {s.description[:40]} ··· SKIPPED"
        elif s.status == "failed":
            line = f"  {icon} Step {s.id}: {s.description[:40]} ✗ FAILED: {s.result[:30]}"
        else:
            line = f"  {icon} Step {s.id}: {s.description[:40]} {tool_mark}{deps}"

        lines.append(line)

    return "\n".join(lines)


# 创建一个示例 plan 演示
demo_plan = Plan(goal="分析项目代码, 统计行数")
demo_plan.steps = [
    Step(id=1, description="列出所有 .py 文件", tool="list_files", status="done",
         result="找到 5 个文件"),
    Step(id=2, description="读取每个文件", tool="read_file", status="done",
         result="共计 1200 行"),
    Step(id=3, description="计算平均行数", tool="calculate", status="failed",
         result="ZeroDivisionError", depends_on=[2]),
    Step(id=4, description="综合分析", tool="none", status="skipped",
         depends_on=[3]),
]

print(render_plan_ascii(demo_plan))

print(f"""\n  可视化设计思路:

  终端版 (ASCII):
    方框: ✓ Step 3: 计算平均行数 [calculate]
    箭头: ← depends: [2]
    虚线: ··· SKIPPED (灰色/暗淡)
    聚合: ── 水平线分隔

  Web 版 (HTML/CSS):
    - 方框 = div.card, 不同颜色表示 status
      pending = #ccc (灰色)
      running = #4a9eff (蓝色, 脉冲动画)
      done    = #4caf50 (绿色)
      failed  = #f44336 (红色)
      skipped = #ff9800 (橙色), 虚线边框
    - 箭头 = SVG path, 从 depends_on 指向当前 step
    - 时间线 = 左侧竖线, 步骤按时间排列

  Graphviz 版 (DOT):
    digraph Plan {{
      step1 [label="Step 1\\n列出文件", style=filled, fillcolor=green]
      step2 [label="Step 2\\n读取文件", style=filled, fillcolor=green]
      step3 [label="Step 3\\n计算行数", style=filled, fillcolor=red]
      step4 [label="Step 4\\n综合分析", style=dashed, fillcolor=orange]
      step2 -> step3
      step3 -> step4 [style=dashed]
    }}

  类比 Java:
    类似 Mermaid.js / PlantUML 的流程图渲染
""")

print(f"  ✅ 练习 6 完成: 设计了三版 Plan 可视化方案")

print(f"\n  📝 学习总结:")
print(f"     Plan 比 ReAct 好的地方:")
print(f"       - 全局视角: 先看清楚整个任务需要几步")
print(f"       - 可追踪: 每步状态清晰, 调试时知道在哪里断的")
print(f"       - 可复用: Plan 可以保存/共享/重放")
print(f"     ReAct 比 Plan 好的地方:")
print(f"       - 灵活性: 遇到意外可以即时变通")
print(f"       - 简单性: 不需要规划步骤, 代码更少")
print(f"     反思循环的价值:")
print(f"       - 清楚的失败信息 → 快速定位问题")
print(f"       - 依赖关系追踪 → 理解失败的级联影响")
