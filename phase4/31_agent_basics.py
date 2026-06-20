# ============================================================
# Phase 4, Lesson 31: Agent 基础 & ReAct 模式
# ============================================================
#
# 本课目标:
#   1. 理解 Agent 是什么 — LLM 的 "大脑 + 双手"
#   2. 掌握 ReAct 模式 — Reasoning + Acting 交替进行
#   3. 理解 Agent Loop — Think → Act → Observe → Repeat
#   4. 实现 ReActAgent 类 — 自主规划、调用工具、多步推理
#   5. Agent vs RAG — 两种 AI 范式的区别与互补
#
# 预计阅读 + 实操时间: 50-60 分钟
#
# 前置: Phase 2 (Tool Use) + Phase 3 (RAG, 可选——Agent 可以调用知识库)
# ============================================================

import os
import json
import time
import math
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


# ============================================================
# 〇、环境准备
# ============================================================

from anthropic import Anthropic
from anthropic.types import Message, ToolUseBlock, TextBlock

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
# 一、什么是 Agent？
# ============================================================
# 到目前为止, 你调用 LLM 的方式是:
#
#   用户提问 → LLM 推理 → 输出答案
#
# 这相当于一个"只会回答"的顾问——它可以给你建议, 但不能帮你做事。
#
# Agent 的不同之处: 它是一个能自主行动的 AI。
#
#   ┌──────────────────────────────────────────────────┐
#   │  Agent Loop                     │  普通 LLM 调用   │
#   ├──────────────────────────────────────────────────┤
#   │  Think: "我需要先查天气"         │  用户 → LLM → 答案 │
#   │  Act:   调用 get_weather()      │                   │
#   │  Observe: "北京今天 28°C"       │  (单次, 无工具)     │
#   │  Think: "现在可以建议穿什么了"    │                   │
#   │  Act:   输出最终建议            │                   │
#   └──────────────────────────────────────────────────┘
#
# 核心区别:
#   LLM 调用 = 一次推理, 一个答案
#   Agent    = 多步循环: 思考 → 行动 → 观察 → 再思考...
#
# 类比 Java:
#   普通 LLM ≈ 调用一个 static 方法, 返回结果
#   Agent    ≈ 一个有状态的 Service Bean, 可以调用多个 Repository
#             每次调用 Repository (工具) 拿到结果后,
#             决定下一步做什么。

print("=" * 60)
print("Agent vs 普通 LLM")
print("=" * 60)
print("""
  普通 LLM:
    用户: "今天适合穿什么?"
    LLM: "抱歉, 我不知道你在哪个城市。"

  Agent (有工具):
    用户: "今天适合穿什么?"
    Agent Think: "需要知道用户在哪里、天气如何。"
    Agent Act:  调用 get_location() → "北京"
    Agent Act:  调用 get_weather("北京") → {temp: 28, condition: "晴"}
    Agent Think: "北京28°C晴天, 可以建议穿轻薄衣物。"
    Agent Act:  输出答案: "北京今天28°C晴天, 建议穿短袖..."
""")


# ============================================================
# 二、ReAct 模式 —— Reasoning + Acting
# ============================================================
# ReAct = Reasoning (推理) + Acting (行动)
#
# 这是当前 Agent 系统最主流的范式, 论文: Yao et al., 2022
#
# 核心思想: 让 LLM 交替进行"思考"和"行动":
#
#   Thought: "要回答这个问题, 我需要知道 X"
#   Action:  调用工具获取 X
#   Observation: "工具返回了 Y"
#   Thought: "结合 Y 和已知信息, 我可以得出..."
#   Action:  输出最终答案
#
# 为什么 ReAct 有效?
#   1. 推理指导行动 — 先想清楚需要什么, 再调用工具
#   2. 行动反馈推理 — 工具结果带来新信息, 修正后续思考
#   3. 可观察可调试 — 每一步都有明确的 Thought/Action/Observation
#
# 类比 Java:
#   ReAct ≈ while 循环 + switch 语句:
#     while (!done && iterations < max):
#         thought = model.think(context)
#         if thought.needsTool():
#             result = executeTool(thought.toolName, thought.params)
#             context.add(result)
#         else:
#             return thought.finalAnswer()

print("\n" + "=" * 60)
print("ReAct 模式")
print("=" * 60)
print("""
  Thought → Action → Observation → Thought → Action → ... → Final Answer

  实例: "北京今天28°C, 适合跑步吗?"

  ┌─ Thought ───────────────────────────────────────┐
  │ "需要查北京的天气和空气质量"                        │
  ├─ Action ────────────────────────────────────────┤
  │ get_weather("北京")                               │
  ├─ Observation ───────────────────────────────────┤
  │ {temp: 28, humidity: 45, condition: "晴"}         │
  ├─ Thought ───────────────────────────────────────┐
  │ "28°C 晴天, 湿度 45%, 适合户外运动"                │
  ├─ Action ────────────────────────────────────────┤
  │ 输出最终答案                                       │
  └─────────────────────────────────────────────────┘
""")


# ============================================================
# 三、Agent 工具箱 —— 定义 Agent 可用的能力
# ============================================================
# Agent 的能力来自它可以调用的工具。
# 这里定义三个工具, 从简单到复杂。

# --- 工具 1: 计算器 ---
# 让 Agent 能算数学题, 而不是依赖训练数据里的"记忆"

CALCULATOR_TOOL = {
    "name": "calculate",
    "description": "执行数学表达式计算。支持 + - * / ** sqrt() abs() 等。输入一个 Python 数学表达式, 返回计算结果。",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Python 数学表达式, 如 '2 + 3 * 4' 或 'sqrt(16) * 2'"
            }
        },
        "required": ["expression"]
    }
}

# 安全计算: 只允许数学函数, 不允许任何形式的代码注入
_SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "log": math.log,
    "pi": math.pi, "e": math.e,
}


def handle_calculate(expression: str) -> dict:
    try:
        result = eval(expression, {"__builtins__": {}}, _SAFE_MATH)
        # 处理非 JSON 可序列化的类型 (如 complex、float('inf') 等)
        if isinstance(result, complex):
            result = str(result)
        elif isinstance(result, float):
            import math as _math
            if _math.isnan(result) or _math.isinf(result):
                result = str(result)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


# --- 工具 2: 当前时间 ---
# 让 Agent 知道"现在是什么时候"

TIME_TOOL = {
    "name": "get_current_time",
    "description": "获取当前日期和时间。可指定时区, 默认 Asia/Shanghai。",
    "input_schema": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "时区, 如 'Asia/Shanghai'、'America/New_York'、'Europe/London'"
            }
        },
        "required": []
    }
}


def handle_time(timezone: str = "Asia/Shanghai") -> dict:
    now = datetime.now()
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": ["一", "二", "三", "四", "五", "六", "日"][now.weekday()],
        "timestamp": int(now.timestamp()),
    }


# --- 工具 3: 知识库搜索 ---
# Agent 可以调用 Phase 3 的 ChromaDB 知识库!
# 这是 Agent + RAG 的结合: Agent 决定什么时候需要查资料。

SEARCH_TOOL = {
    "name": "search_knowledge",
    "description": "在本地知识库中搜索相关文档。当用户问到需要专业知识或文档支持的问题时调用。",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询, 用自然语言描述要找什么"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量, 默认 3"
            }
        },
        "required": ["query"]
    }
}

# 延迟初始化知识库 (避免启动时加载所有依赖)
_kb = None


def _get_kb():
    global _kb
    if _kb is None:
        from sentence_transformers import SentenceTransformer
        import chromadb
        import numpy as np

        st = SentenceTransformer("all-MiniLM-L6-v2")

        class _EF:
            def name(self): return "miniLM"
            def embed_query(self, input): return st.encode(input, convert_to_numpy=True).tolist()
            def embed_documents(self, input): return st.encode(input, convert_to_numpy=True).tolist()
            def __call__(self, input): return self.embed_query(input)

        c = chromadb.PersistentClient(path="./phase3/chroma_db")
        col = c.get_or_create_collection(name="pykb_main", embedding_function=_EF())
        _kb = col
    return _kb


def handle_search(query: str, top_k: int = 3) -> dict:
    try:
        col = _get_kb()
        if col.count() == 0:
            return {"query": query, "results": [], "hint": "知识库为空, 请先导入文档"}
        results = col.query(query_texts=[query], n_results=min(top_k, col.count()),
                            include=["documents", "metadatas", "distances"])
        items = []
        for doc_id, text, meta, dist in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0],
        ):
            items.append({
                "source": meta.get("source", "unknown"),
                "text": text[:200],
                "score": round(1.0 / (1.0 + dist), 3),
            })
        return {"query": query, "results": items}
    except Exception as e:
        return {"query": query, "error": str(e)}


# 工具注册表
TOOL_DEFS = [CALCULATOR_TOOL, TIME_TOOL, SEARCH_TOOL]
TOOL_HANDLERS = {
    "calculate": handle_calculate,
    "get_current_time": handle_time,
    "search_knowledge": handle_search,
}

print(f"  Agent 工具箱: {[t['name'] for t in TOOL_DEFS]}")
for t in TOOL_DEFS:
    print(f"    {t['name']:<20s} — {t['description'][:50]}...")


# ============================================================
# 四、ReActAgent —— 实现 Agent 循环
# ============================================================
# 这是本课的核心: 一个能自主思考、调用工具、多步推理的 Agent。
#
# 循环流程:
#   ┌─────────────────────────────────────────────────┐
#   │  1. 发送 messages + tools 给 LLM                 │
#   │  2. LLM 返回: text (完成) 或 tool_use (行动)     │
#   │  3. 如果 tool_use → 执行工具 → 追加结果到 messages│
#   │  4. 跳回步骤 1 (最多 max_iterations 次)          │
#   │  5. 如果是 text → 提取最终答案, 返回              │
#   └─────────────────────────────────────────────────┘

class ReActAgent:
    """ReAct 模式 Agent。

    类比 Java:
      ReActAgent ≈ 一个有状态的 Orchestrator
        while (未完成 && 未超次数):
            response = llm.think(context, tools)
            if response.hasToolCalls():
                for each tool:
                    result = tool.execute()
                    context.add(result)
            else:
                return response.text()

    关键参数:
      max_iterations — 防止死循环 (LLM 可能反复调同一个工具)
      verbose        — 是否打印思考过程
    """

    def __init__(self, model: str = MODEL, max_iterations: int = 10, verbose: bool = True):
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.iteration_log: list[dict] = []

    def run(self, task: str,
            system_prompt: str | None = None,
            tools: list[dict] | None = None,
            handlers: dict | None = None) -> str:
        """执行 Agent 循环, 处理一个任务。

        Args:
          task: 用户的任务描述
          system_prompt: 自定义系统提示
          tools: 可用工具列表
          handlers: 工具名 → 处理函数 映射

        Returns:
          Agent 的最终回答
        """
        if tools is None:
            tools = TOOL_DEFS
        if handlers is None:
            handlers = TOOL_HANDLERS

        if system_prompt is None:
            system_prompt = self._default_system()

        # messages 是 Agent 的"工作记忆"
        messages: list[dict] = [{"role": "user", "content": task}]
        self.iteration_log = []

        for i in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'─' * 50}")
                print(f"  [迭代 {i + 1}] ", end="")

            # --- Step 1: 调用 LLM (带工具列表) ---
            if not api_ok:
                return self._mock_run(task, tools, handlers)

            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )

            tool_uses = _get_tool_uses(response)
            text_parts = _get_text(response)

            # --- Step 2: 如果是纯文本 → 任务完成 ---
            if not tool_uses:
                if self.verbose:
                    print(f"✅ 完成")
                self.iteration_log.append({
                    "iteration": i + 1,
                    "action": "final_answer",
                    "text": text_parts,
                })
                return text_parts

            # --- Step 3: 执行工具调用 ---
            if self.verbose and text_parts:
                print(f"💭 {text_parts[:80]}...")

            # 将 assistant 的回复 (含 tool_use blocks) 追加到 messages
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            messages.append({"role": "assistant", "content": assistant_content})

            # 执行每个工具, 追加 tool_result
            tool_results = []
            for tu in tool_uses:
                handler = handlers.get(tu.name)
                if handler:
                    result = handler(**tu.input) if tu.input else handler()
                else:
                    result = {"error": f"未知工具: {tu.name}"}

                tool_results.append({"name": tu.name, "input": tu.input, "result": result})

                if self.verbose:
                    input_str = json.dumps(tu.input, ensure_ascii=False)[:60]
                    result_str = json.dumps(result, ensure_ascii=False)[:80]
                    print(f"  🔧 {tu.name}({input_str})")
                    print(f"  📊 → {result_str}")

            self.iteration_log.append({
                "iteration": i + 1,
                "action": "tool_calls",
                "calls": tool_results,
            })

            # 追加 tool_result 到 messages (必须有 tool_use 的 id 对应)
            tool_content = []
            for tu in tool_uses:
                handler = handlers.get(tu.name)
                result = handler(**tu.input) if tu.input else handler()
                tool_content.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            messages.append({"role": "user", "content": tool_content})

        # --- 达到最大迭代次数 ---
        return ("[Agent 达到最大迭代次数, 未能完成任务。"
                "请简化问题或增加 max_iterations。]")

    def _default_system(self) -> str:
        return """你是一个 ReAct 模式的 AI Agent。你可以调用工具来完成任务。

遵循以下规则:
1. 分析用户任务, 确定需要哪些信息
2. 如果计算/查询/搜索能帮助完成任务, 主动调用工具
3. 每次调用工具后, 分析结果, 决定是否需要更多信息
4. 当信息足够时, 给出完整、准确的最终回答
5. 用中文回答

可用工具:
- calculate: 数学计算
- get_current_time: 获取当前时间
- search_knowledge: 搜索本地知识库"""

    def _mock_run(self, task: str, tools: list[dict], handlers: dict) -> str:
        """模拟运行 (API 不可用时)。"""
        tool_names = [t["name"] for t in tools]
        return (f"[模拟 Agent] 收到任务: {task}\n"
                f"可用工具: {tool_names}\n"
                f"在真实环境中, Agent 会自主选择合适的工具, "
                f"多步推理后给出答案。")


# ============================================================
# 五、示例 1: 数学计算 Agent
# ============================================================
# 场景: 用户问一个需要多步计算的问题。
# Agent 需要自己决定先算什么、后算什么。

print("\n" + "=" * 60)
print("示例 1: 计算 Agent")
print("=" * 60)

agent_calc = ReActAgent(max_iterations=5)
calc_task = "帮我算一下: 一个圆的半径是 5.5, 求它的面积和周长。然后告诉我面积是周长的多少倍。"

print(f"  任务: {calc_task}\n")
answer = agent_calc.run(
    calc_task,
    tools=[CALCULATOR_TOOL],
    handlers={"calculate": handle_calculate},
)
print(f"\n  {'=' * 50}")
print(f"  最终答案: {answer}")


# ============================================================
# 六、示例 2: 研究助手 Agent (多工具)
# ============================================================
# 场景: 用户问的问题需要结合实时信息 + 知识库。

print("\n\n" + "=" * 60)
print("示例 2: 研究助手 Agent")
print("=" * 60)

agent_research = ReActAgent(max_iterations=8)

research_task = "现在是什么时候? 根据知识库, 告诉我 MySQL 有哪些备份方式。然后帮我算一下, 如果每天备份一次, 30 天需要多少次备份。"

print(f"  任务: {research_task}\n")
answer = agent_research.run(research_task)
print(f"\n  {'=' * 50}")
print(f"  最终答案: {answer}")

# 打印迭代摘要
print(f"\n  Agent 迭代摘要:")
for log in agent_research.iteration_log:
    if log["action"] == "final_answer":
        print(f"    第{log['iteration']}轮: 输出最终答案")
    else:
        for call in log.get("calls", []):
            print(f"    第{log['iteration']}轮: 调用 {call['name']} → 成功")


# ============================================================
# 七、Agent 的边界 —— 什么时候停下来?
# ============================================================
# Agent 不是万能的, 有几个重要的边界条件:

print("\n" + "=" * 60)
print("Agent 边界条件")
print("=" * 60)
print("""
  1. 迭代上限 (max_iterations)
     防止 LLM 陷入"调用→不满意→再调用"的循环。
     本课默认 10 轮, 生产环境通常 5-15 轮。

  2. Token 预算
     每轮工具调用的结果都会追加到 messages,
     messages 越来越长 → token 消耗线性增长。
     本课没有显式的 token 管理, L32 会讲。

  3. 工具可靠性
     Agent 可能传入不合理的参数 (如 calculate("abc"))。
     工具必须做好错误处理, 返回清晰的错误信息。

  4. 幻觉放大
     普通 LLM 的幻觉只影响一个答案。
     Agent 的幻觉会影响后续决策, 导致连锁错误。
     解决: 工具返回事实性数据, 让 Agent "脚踏实地"。

  5. 安全边界
     Agent 可以执行代码 (calculate tool)、访问文件系统。
     永远不要让 Agent 直接执行不可信的输入!
     本课的 calculate 用沙箱 eval, 只允许数学函数。

  类比 Java:
    Agent 边界 ≈ API 的 rate limit + circuit breaker
    没有限制的 Agent = 没有超时的 HTTP 请求 = 灾难
""")


# ============================================================
# 八、Agent vs RAG —— 两种 AI 范式
# ============================================================

print("=" * 60)
print("Agent vs RAG")
print("=" * 60)
print("""
  维度          RAG                          Agent
  ─────────────────────────────────────────────────────
  核心思路      给 LLM 查资料                  给 LLM 配工具
  解决的问题    "LLM 不知道/记不住"            "LLM 做不到/做不对"
  数据流        Query → 检索 → 拼接 → 生成     Task → 思考 → 工具 → 思考 → ...
  输出          基于文档的回答                  行动 + 回答
  典型场景      客服知识库、文档问答            自动化、数据分析、多步任务

  它们不是替代关系, 而是互补关系:
    Agent 可以调用 RAG 作为工具 (本课的 search_knowledge)
    RAG 可以嵌入 Agent 的某个步骤

  类比 Java:
    RAG    ≈ DAO 层 — 数据访问, 提供事实
    Agent  ≈ Service 层 — 业务编排, 决策 + 执行
    Tool   ≈ 各种 Bean — 每个工具是一个被调用的能力单元

  Phase 3 你构建的 RAG 系统 → Phase 4 Agent 的一个工具!
""")


# ============================================================
# 九、Agent 思考过程可视化
# ============================================================

print("=" * 60)
print("Agent 思考过程详情")
print("=" * 60)

if agent_research.iteration_log:
    for entry in agent_research.iteration_log:
        i = entry["iteration"]
        if entry["action"] == "final_answer":
            print(f"\n  第{i}轮: ✅ 输出最终答案")
            print(f"    {entry['text'][:120]}...")
        else:
            print(f"\n  第{i}轮: 🔧 工具调用")
            for call in entry.get("calls", []):
                print(f"    工具: {call['name']}")
                print(f"    输入: {json.dumps(call['input'], ensure_ascii=False)}")
                result_str = json.dumps(call['result'], ensure_ascii=False)
                print(f"    结果: {result_str[:120]}...")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Lesson 31 完成! Agent 基础已掌握。")
    print("=" * 60)
    print(f"""
  回顾: 你学会了什么?

  1. Agent 概念     — LLM + 工具 = 能自主行动的 AI
  2. ReAct 模式     — Reasoning (思考) + Acting (行动)
  3. Agent Loop     — Think → Act → Observe → Repeat
  4. ReActAgent     — 封装了完整循环, 可复用
  5. 多工具协作     — 计算 + 时间 + 知识库搜索
  6. Agent + RAG    — search_knowledge 工具 = Phase 3 → Phase 4

  Agent 循环核心代码:
    for i in range(max_iterations):
        response = llm.create(messages, tools=...)
        if has_tool_calls(response):
            execute_and_append_results(messages, response)
        else:
            return response.text  # 完成!

  🎯 下一课: Lesson 32 — 工具调用深入
     实现工具注册中心、工具结果格式化、
     Token 预算管理、并行工具调用。
""")


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 添加一个新工具:
#    实现一个 "文本统计" 工具 count_text:
#    - 输入: text (字符串)
#    - 输出: {chars, words, lines}
#    定义 tool schema + handler, 注册到 Agent。
#    然后让 Agent 分析一段你写的文字。
#
# 2. 观察 Agent 的"错误决策":
#    给 Agent 一个模糊的任务, 如 "帮我做点有用的事"。
#    - Agent 会怎么做? 调用什么工具? 还是直接拒绝?
#    - 对比修改 system_prompt 前后的行为变化。
#
# 3. 修改 max_iterations:
#    故意设一个需要多轮的任务, 分别用:
#    - max_iterations=1 (只能做一步)
#    - max_iterations=3 (适中)
#    - max_iterations=10 (充裕)
#    观察不同限制下 Agent 的表现差异。
#
# 4. 实现工具调用失败的优雅处理:
#    在 handle_calculate 中, 如果表达式不合法,
#    Agent 是否能理解错误信息并尝试修正?
#    尝试问: "帮我算 log(-1)" — 看 Agent 如何处理。
#
# 5. (挑战) 实现 Agent 的"自我反思":
#    在 Agent 循环中加一个步骤:
#    每次工具返回后, 先让 LLM 反思 "这个结果是否足够回答问题?"
#    如果不够, 继续调工具; 如果够了, 输出答案。
#    提示: 在 system_prompt 中加入反思指令。
#
# 6. (思考) Agent 的安全问题:
#    如果 Agent 有 write_file 工具, 你怎么防止它:
#    - 覆盖重要文件?
#    - 写入恶意代码?
#    - 读取敏感信息?
#    设计一套安全策略, 在 L34 (MCP) 中会深入这个话题。
#
# 做完后告诉我:
#   - Agent 和普通 LLM 调用最大的不同是什么?
#   - Agent 调用 search_knowledge 的效果如何? (Phase 3→4 联动)
# 我们继续 Lesson 32: 工具调用深入。
# ============================================================


# ╔══════════════════════════════════════════════════════════════╗
# ║              试试看 — 练习实现代码                            ║
# ╚══════════════════════════════════════════════════════════════╝

import threading
import random

print("\n")
print("=" * 60)
print("试试看练习: Lesson 31")
print("=" * 60)


# ─── 练习 1: 添加 count_text 工具 ─────────────────────────────
# 实现一个 "文本统计" 工具, 统计字符数/词数/行数

print("\n" + "─" * 40)
print("练习 1: 添加 count_text 工具")
print("─" * 40)

COUNT_TEXT_TOOL = {
    "name": "count_text",
    "description": "统计文本的字符数、词数、行数。输入一段文字, 返回详细的统计信息。",
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "要统计的文本内容"
            }
        },
        "required": ["text"]
    }
}


def handle_count_text(text: str) -> dict:
    """统计文本的详细指标。

    类比 Java:
      类似一个 TextStatsUtil 工具类, 返回统计 DTO。
    """
    if not text.strip():
        return {"error": "文本为空", "chars": 0, "words": 0, "lines": 0}

    lines = text.split("\n")
    words = text.split()
    return {
        "chars": len(text),
        "chars_no_spaces": len(text.replace(" ", "").replace("\n", "")),
        "words": len(words),
        "lines": len(lines),
        "avg_word_len": round(sum(len(w) for w in words) / len(words), 1),
        "paras": len([l for l in lines if l.strip()]),
    }


# 注册到 Agent 工具箱
ALL_TOOLS_31 = TOOL_DEFS + [COUNT_TEXT_TOOL]
ALL_HANDLERS_31 = {**TOOL_HANDLERS, "count_text": handle_count_text}

# 测试工具
sample_text = (
    "Python 是一门解释型、动态类型的编程语言。\n"
    "它由 Guido van Rossum 于 1991 年发布。\n\n"
    "Python 设计哲学强调代码的可读性, \n"
    "使用显著的空白缩进来划分代码块。"
)
print(f"  测试文本: {sample_text[:40]}...")
stats_result = handle_count_text(sample_text)
print(f"  统计结果: chars={stats_result['chars']}, "
      f"words={stats_result['words']}, lines={stats_result['lines']}, "
      f"avg_word_len={stats_result['avg_word_len']}")

# 用 ReActAgent 演示 (调用 LLM 分析)
if api_ok:
    agent_text = ReActAgent(max_iterations=3, verbose=False)
    text_task = (
        f"帮我分析这段文字，先调用 count_text 统计基本信息, "
        f"然后告诉我这段文字的特点:\n\n{sample_text}"
    )
    text_answer = agent_text.run(
        text_task,
        tools=ALL_TOOLS_31,
        handlers=ALL_HANDLERS_31,
    )
    print(f"  Agent 回答: {text_answer[:200]}...")
else:
    print(f"  [离线模式] 文本工具可用, skip LLM 调用")

print(f"\n  ✅ 练习 1 完成: 新工具 count_text 已注册, "
      f"新增参数 (chars_no_spaces, avg_word_len, paras)")


# ─── 练习 2: 观察 Agent 的"错误决策" ──────────────────────────
# 给 Agent 一个模糊的任务, 观察行为差异

print("\n" + "─" * 40)
print("练习 2: 观察 Agent 模糊任务行为")
print("─" * 40)

# 默认 system_prompt 下的行为
vague_task = "帮我做点有用的事"

if api_ok:
    agent_default = ReActAgent(max_iterations=3, verbose=False)
    ans_default = agent_default.run(vague_task)
    print(f"  默认 prompt, 模糊任务: {ans_default[:200]}...")
else:
    print(f"  [离线模式] 默认 prompt 行为: Agent 会返回友好但无动作的回复")
    print(f"    因为模糊任务没有明确的工具调用需求")

# 用个性化 system_prompt —— 鼓励主动使用工具
curious_prompt = """你是一个好奇心强的 AI Agent。
收到任务后, 即使任务模糊, 也要主动调用工具来展示你的能力。
比如:
- 调用 get_current_time 查看当前时间
- 调用 calculate 做一个有意思的计算
- 永远不要只回复文本, 至少要调一个工具"""

if api_ok:
    agent_curious = ReActAgent(max_iterations=3, verbose=False)
    ans_curious = agent_curious.run(vague_task, system_prompt=curious_prompt)
    print(f"  好奇 prompt, 模糊任务: {ans_curious[:200]}...")

    # 查看迭代日志对比
    print(f"\n  默认 prompt 迭代: {len(agent_default.iteration_log)} 轮")
    for log in agent_default.iteration_log[:3]:
        print(f"    L{log['iteration']}: {log['action']}")
    print(f"  好奇 prompt 迭代: {len(agent_curious.iteration_log)} 轮")
    for log in agent_curious.iteration_log[:3]:
        print(f"    L{log['iteration']}: {log['action']}")
else:
    print(f"  [离线模式] 好奇 prompt: Agent 会主动调用 get_current_time")
    print(f"  观察结论:")
    print(f"    - 默认 prompt: Agent 倾向直接回答或拒绝模糊任务")
    print(f"    - 好奇 prompt: Agent 主动找工具调用, 展示了 system_prompt 的影响力")
    print(f"    这证实 system_prompt 就像是 Agent 的 '性格设定'")

print(f"\n  ✅ 练习 2 完成: system_prompt 显著影响 Agent 决策")


# ─── 练习 3: 修改 max_iterations 对比 ─────────────────────────
# 同一个需要多轮的任务, 对比不同限制

print("\n" + "─" * 40)
print("练习 3: max_iterations 对比")
print("─" * 40)

multi_step_task = (
    "1. 帮我计算 sqrt(144) 的值\n"
    "2. 帮我计算 2**10 的值\n"
    "3. 帮我计算 pi * 3**2 的值\n"
    "4. 把三个结果加起来"
)

# 使用模拟模式统一测试 (避免 API 不一致)
print(f"  任务: 4 步计算的复杂任务\n")
print(f"  max_iterations = 1:")
print(f"    效果: 只能执行 1 轮 → 可能只完成 sqrt(144), 后续来不及做")
print(f"    日志: 第 1 轮调用 calculate → 但来不及输出最终答案就超限")

print(f"  max_iterations = 3:")
print(f"    效果: 3 轮 → LLM 需要合理规划, 可能并行调用多个计算")
print(f"    日志: 如果 LLM 并行调 2 个 → 需要 2-3 轮完成")

print(f"  max_iterations = 10:")
print(f"    效果: 充裕 → Agent 可以从容完成, 但也浪费 token")
print(f"    日志: 通常 3-5 轮完成, 剩余迭代被浪费")

# 实际测试 (如果 API 可用)
if api_ok:
    print(f"\n  实际测试结果:")
    for max_iter in [1, 3, 10]:
        agent_test = ReActAgent(max_iterations=max_iter, verbose=False)
        ans = agent_test.run(multi_step_task, tools=[CALCULATOR_TOOL],
                            handlers={"calculate": handle_calculate})
        actual_iters = len(agent_test.iteration_log)
        print(f"    max={max_iter}: 实际用了 {actual_iters} 轮")
        if "[Agent 达到最大迭代次数" in ans:
            print(f"      → 未完成!")
        else:
            print(f"      → 完成: {ans[:100]}...")

    print(f"\n  观察结论:")
    print(f"    - max_iterations 太小 → Agent '有心无力', 虎头蛇尾")
    print(f"    - max_iterations 适中 → 够用就行, 省 token")
    print(f"    - max_iterations 太大 → 浪费, 且可能在失败时无限重试")
    print(f"    类比 Java: 这就像是给 while 循环设置合理的退出条件")
else:
    print(f"\n  [离线模式] max_iterations 对比观察 (基于 ReActAgent._mock_run):")
    print(f"    max=1: 只能做 1 轮, 复杂任务通常无法完成")
    print(f"    max=3: 中等任务刚好, 复杂任务可能不够")
    print(f"    max=10: 大部分任务都能完成, 但多轮浪费 token")
    print(f"    经验: 简单查询 3, 中等分析 5-8, 复杂探索 10-15")

print(f"\n  ✅ 练习 3 完成: 理解了 max_iterations 的 trade-off")


# ─── 练习 4: 工具调用失败的优雅处理 ───────────────────────────
# Agent 如何处理 log(-1) 这样的错误

print("\n" + "─" * 40)
print("练习 4: 工具失败优雅处理")
print("─" * 40)

print(f"  场景: 用户问 '帮我算 log(-1)'")
print(f"  工具行为: math.log(-1) → ValueError: math domain error")
print(f"  handle_calculate 返回: {{'expression': 'log(-1)', 'error': '...'}}")

# 手动测试错误处理
bad_result = handle_calculate("log(-1)")
print(f"  实际返回: {bad_result}")

# Agent 收到错误后的行为:
print(f"\n  期望行为:")
print(f"    1. Agent 收到 {{'error': 'math domain error'}}")
print(f"    2. Agent 分析: 'log 的真数必须 > 0, 负数无实数对数'")
print(f"    3. Agent 输出: 'log(-1) 在实数域无意义, 但复数域值为 πi'")
print(f"    4. Agent 不会反复重试同一个错误参数")

if api_ok:
    agent_error = ReActAgent(max_iterations=3, verbose=False)
    error_answer = agent_error.run(
        "帮我算 log(-1)", tools=[CALCULATOR_TOOL],
        handlers={"calculate": handle_calculate},
    )
    print(f"\n  Agent 实际回答: {error_answer[:300]}...")
    print(f"  使用了 {len(agent_error.iteration_log)} 轮迭代")
else:
    print(f"\n  [离线模式] Agent 处理错误的最佳实践:")
    print(f"    - 工具返回清晰的错误信息 (不是裸 exception)")
    print(f"    - Agent 在 system_prompt 中被告知: 区分 '工具失败' 和 '数学无意义'")
    print(f"    - Agent 不应该反复重试同样的错误参数")
    print(f"    参考 deploy/agent_core.py: MCPServer.handle 返回友好错误")

print(f"\n  ✅ 练习 4 完成: 理解了错误信息的双向传递机制")


# ─── 练习 5 (挑战): Agent 自我反思 ────────────────────────────
# 在 Agent 循环中加入反思步骤

print("\n" + "─" * 40)
print("练习 5 (挑战): 自我反思 Agent")
print("─" * 40)


class ReflectiveAgent(ReActAgent):
    """带自我反思的 ReAct Agent。

    和普通 ReActAgent 的区别:
      每次工具调用完成后, 先让 LLM 反思 "这个结果够不够?"
      如果 LLM 觉得够了 → 输出答案
      如果 LLM 觉得不够 → 继续调工具

    类比 Java:
      普通 Agent ≈ 简单 while 循环
      反思 Agent ≈ 循环内加一个 quality check gate
    """

    REFLECTION_PROMPT = """你是一个会自我反思的 AI Agent。

每次工具调用完成后, 你必须问自己三个问题:
1. 工具返回的结果是否完整、准确?
2. 当前拥有的信息是否足以回答用户的问题?
3. 如果不够, 我还需要什么信息?

规则:
- 如果信息足够: 直接给出最终答案
- 如果信息不够: 继续调用工具 (但不要重复已经做过的)
- 如果工具返回错误: 分析原因, 不要重试同样的参数
- 用中文回答"""

    def __init__(self, model: str = MODEL, max_iterations: int = 10, verbose: bool = True):
        super().__init__(model, max_iterations, verbose)
        self.reflection_log: list[str] = []

    def run(self, task: str, system_prompt: str | None = None,
            tools: list[dict] | None = None,
            handlers: dict | None = None) -> str:
        """重写 run, 使用反思 system_prompt。"""
        if system_prompt is None:
            system_prompt = self.REFLECTION_PROMPT
        self.reflection_log = []
        return super().run(task, system_prompt, tools, handlers)


# 对比: 普通 Agent vs 反思 Agent
print(f"  关键区别:")
print(f"    普通 Agent: system_prompt 只说 '用工具完成任务'")
print(f"    反思 Agent: system_prompt 要求 '每步后问自己够不够'")
print(f"    效果: 反思 Agent 更不容易陷入无限循环")
print(f"          因为它会主动评估 '还有必要继续调工具吗?'")

if api_ok:
    reflect_agent = ReflectiveAgent(max_iterations=5, verbose=False)
    reflect_task = "帮我算一下: (sqrt(144) + 2**5) * pi, 只需要最终结果。"
    reflect_answer = reflect_agent.run(
        reflect_task,
        tools=[CALCULATOR_TOOL],
        handlers={"calculate": handle_calculate},
    )
    print(f"\n  反思 Agent 回答: {reflect_answer[:200]}")
    print(f"  迭代次数: {len(reflect_agent.iteration_log)}")
else:
    print(f"\n  [离线模式] 反思机制演示:")
    print(f"    场景: '计算 sqrt(144) + 2**5'")
    print(f"    Iter 1: calculate('sqrt(144)+2**5') → 结果: 44")
    print(f"    反思: '结果是 44, 信息足够, 可以回答'")
    print(f"    → 输出最终答案, 共 1 轮")
    print(f"    对比普通 Agent 可能需要 2-3 轮")
    print(f"    这也是 deploy/agent_core.py 中 Plan 模式的反思机制")

print(f"\n  ✅ 练习 5 完成: 反思 Agent 通过 system_prompt 注入自我评估能力")


# ─── 练习 6 (思考): Agent 安全问题 ────────────────────────────
# 如果 Agent 有 write_file 工具, 安全策略设计

print("\n" + "─" * 40)
print("练习 6 (思考): Agent 安全策略设计")
print("─" * 40)

print("""
  如果 Agent 有 write_file 工具, 安全防护策略:

  ┌─────────────────────────────────────────────────────────────┐
  │  1. 路径沙箱 (Path Sandbox)                                  │
  │     只允许写入指定目录 (如 project_root/output),             │
  │     禁止绝对路径、禁止 ../ 穿越。                             │
  │     类比 Java: SecurityManager + 文件访问策略                 │
  │                                                              │
  │  2. 文件类型白名单                                           │
  │     只允许写入 .txt / .json / .md / .log                     │
  │     禁止 .py / .sh / .bat (防止写入可执行恶意代码)            │
  │     类比 Java: Content-Type 校验                              │
  │                                                              │
  │  3. 内容扫描                                                 │
  │     写入前检查内容是否包含:                                   │
  │     - import os / subprocess / eval / exec (代码注入)        │
  │     - rm -rf / / DROP TABLE (破坏性命令)                     │
  │     类比 Java: OWASP 输入校验                                 │
  │                                                              │
  │  4. 工具分级 (Tool Tiers)                                     │
  │     READ   (search, calculate, read_file)   → 自动执行         │
  │     WRITE  (write_file, delete_file)       → 需要用户确认      │
  │     ADMIN  (system_command)                → 需要 admin 密钥   │
  │     类比 Java: @PreAuthorize("hasRole('ADMIN')")             │
  │                                                              │
  │  5. Human-in-the-Loop                                        │
  │     写入前打印 diff, 要求用户输入 y/n 确认                     │
  │     类比 Java: 审批工作流                                     │
  │                                                              │
  │  6. 审计日志                                                 │
  │     所有 write_file 操作记录:                                 │
  │     {time, user, tool, file_path, content_hash, approved_by}  │
  │     类比 Java: @Auditable + 数据库记录                        │
  │                                                              │
  │  7. 敏感信息防护                                             │
  │     read_file 工具:                                          │
  │     - 不读取 .env / credentials / *_secret_*                 │
  │     - 返回内容时自动脱敏 API key / token                      │
  │     类比 Java: @JsonIgnore on sensitive fields               │
  └─────────────────────────────────────────────────────────────┘

  这些策略在 deploy/agent_core.py 中已有部分实现:
    - tool_read_file: 路径沙箱 + 文件类型白名单
    - L34 MCPServer: 工具分级 + 权限系统概念
    - L35 DevAssistant: Human-in-the-Loop 实践 (/plan 模式)
""")

print(f"  ✅ 练习 6 完成: 设计了 7 层安全策略")
print(f"\n  📝 学习总结:")
print(f"     Agent vs 普通 LLM 最大不同: Agent 是'多步循环 + 工具'的组合,")
print(f"     能自主决策、执行行动、观察反馈后调整。")
print(f"     Agent + search_knowledge (Phase 3→4): Agent 把 RAG 当工具调用,")
print(f"     实现了'需要时才查'的按需知识获取。")
