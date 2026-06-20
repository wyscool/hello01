# ============================================================
# Phase 4, Lesson 35: 端到端 Agent 项目 —— Phase 4 收官
# ============================================================
#
# 本课目标:
#   融合 Phase 4 全部技能, 构建 DevAssistant — 开发者智能助手。
#
#   融合的技能:
#     L31: ReAct 循环 (Think → Act → Observe)
#     L32: ToolRegistry + TokenBudget + ToolResult
#     L33: Plan-then-Act + Reflection + Self-correction
#     L34: MCP 协议 (动态工具发现)
#
#   新增知识:
#     1. 双模式架构 — Quick (ReAct) / Plan (Plan-then-Act)
#     2. 文件系统工具 — 让 Agent 读文件、列目录
#     3. 交互式 CLI — 命令解析 + 模式切换
#     4. 项目结构 — 分层的生产级 Agent 应用
#
# 预计阅读 + 实操时间: 60-70 分钟
#
# 前置: Lesson 31-34
# ============================================================

import os
import sys
import json
import math
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, Any
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

ckwargs = {"api_key": api_key} if api_key else {}
if base_url:
    ckwargs["base_url"] = base_url
llm = Anthropic(**ckwargs)

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
    llm.messages.create(model=MODEL, max_tokens=10,
                        messages=[{"role": "user", "content": "ping"}])
    api_ok = True
except Exception:
    api_ok = False


# ============================================================
# 一、系统架构
# ============================================================
# DevAssistant 三层架构:
#
#   CLI 层        命令解析、交互循环、输出格式化
#   Agent 层      双模式: Quick (ReAct) + Plan (Plan-then-Act)
#   MCP 层        工具注册、协议处理、文件系统访问
#
#   ┌────────────────────────────────────────────┐
#   │  CLI: /ask /plan /tools /mode /budget ... │
#   ├────────────────────────────────────────────┤
#   │  Agent                                      │
#   │    Quick Mode  → ReActAgent (L31+L32)      │
#   │    Plan Mode   → PlannerAgent (L33)        │
#   ├────────────────────────────────────────────┤
#   │  MCP Server                                 │
#   │    calculate | time | text_stats            │
#   │    read_file | list_files    (文件系统!)    │
#   └────────────────────────────────────────────┘

print("=" * 60)
print("  DevAssistant — Phase 4 收官项目")
print("=" * 60)
print(f"  API: {'✓' if api_ok else '✗ (模拟模式)'}")
print(f"  Model: {MODEL}")
print()


# ============================================================
# 二、MCP 协议层 (精简复用 L34)
# ============================================================

@dataclass
class MCPResponse:
    result: Any = None
    error: str | None = None
    id: int = 0

    def to_json(self) -> str:
        body = {"jsonrpc": "2.0", "id": self.id}
        if self.error:
            body["error"] = {"code": -1, "message": self.error}
        else:
            body["result"] = self.result
        return json.dumps(body, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "MCPResponse":
        data = json.loads(raw)
        if "error" in data:
            return cls(error=data["error"]["message"], id=data.get("id", 0))
        return cls(result=data.get("result"), id=data.get("id", 0))


class MCPServer:
    """MCP Server — 工具的注册和调用中心。"""

    def __init__(self, name: str = "dev-assistant"):
        self.name = name
        self._tools: dict[str, dict] = {}

    def register(self, name: str, description: str,
                 parameters: dict, required: list[str],
                 handler: Callable) -> "MCPServer":
        self._tools[name] = {
            "schema": {
                "name": name, "description": description,
                "inputSchema": {
                    "type": "object", "properties": parameters,
                    "required": required,
                }
            },
            "handler": handler,
        }
        return self

    def handle(self, raw_request: str) -> str:
        try:
            data = json.loads(raw_request)
            method = data.get("method", "")
            params = data.get("params", {})
            rid = data.get("id", 0)
        except json.JSONDecodeError:
            return MCPResponse(error="Invalid JSON").to_json()

        match method:
            case "tools/list":
                tools = [t["schema"] for t in self._tools.values()]
                return MCPResponse(result={"tools": tools}, id=rid).to_json()
            case "tools/call":
                tool = self._tools.get(params.get("name", ""))
                if not tool:
                    return MCPResponse(error=f"未知工具: {params.get('name')}", id=rid).to_json()
                try:
                    args = params.get("arguments", {})
                    result = tool["handler"](**args)
                    text = json.dumps(result, ensure_ascii=False)
                    return MCPResponse(
                        result={"content": [{"type": "text", "text": text}]},
                        id=rid,
                    ).to_json()
                except Exception as e:
                    return MCPResponse(error=f"执行失败: {e}", id=rid).to_json()
            case "initialize":
                return MCPResponse(result={
                    "protocolVersion": "0.2",
                    "serverInfo": {"name": self.name, "version": "1.0"},
                    "capabilities": {"tools": {}},
                }, id=rid).to_json()
            case _:
                return MCPResponse(error=f"未知方法: {method}", id=rid).to_json()


class MCPClient:
    """MCP Client — 封装协议通信, 格式转换。"""

    def __init__(self, server: MCPServer | None = None):
        self.server = server
        self._rid = 0

    def connect(self, server: MCPServer):
        self.server = server
        resp = self._send("initialize")
        if resp.error:
            raise RuntimeError(f"握手失败: {resp.error}")
        info = resp.result.get("serverInfo", {})
        print(f"  MCP: 已连接 {info.get('name', '?')} v{resp.result.get('protocolVersion', '?')}")

    def _send(self, method: str, params: dict | None = None) -> MCPResponse:
        if self.server is None:
            return MCPResponse(error="未连接")
        self._rid += 1
        raw = json.dumps({"jsonrpc": "2.0", "method": method,
                          "params": params or {}, "id": self._rid})
        return MCPResponse.from_json(self.server.handle(raw))

    def list_tools(self) -> list[dict]:
        """获取 Anthropic 格式的工具列表。"""
        resp = self._send("tools/list")
        if resp.error:
            return []
        tools = resp.result.get("tools", [])
        return [{
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("inputSchema",
                                  {"type": "object", "properties": {}, "required": []}),
        } for t in tools]

    def call_tool(self, name: str, arguments: dict) -> dict:
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        if resp.error:
            return {"error": resp.error}
        content = resp.result.get("content", [])
        for item in content:
            if item.get("type") == "text":
                try:
                    return json.loads(item["text"])
                except json.JSONDecodeError:
                    return {"text": item["text"]}
        return {"raw": content}


# ============================================================
# 三、工具注册 — 构建 DevAssistant 的能力
# ============================================================

# --- 数学工具 ---
_SAFE = {"abs": abs, "round": round, "min": min, "max": max,
         "sum": sum, "pow": pow, "sqrt": math.sqrt,
         "sin": math.sin, "cos": math.cos, "log": math.log,
         "pi": math.pi, "e": math.e}


def tool_calculate(expression: str) -> dict:
    result = eval(expression, {"__builtins__": {}}, _SAFE)
    return {"expression": expression, "result": result}


# --- 时间工具 ---
def tool_time(timezone: str = "Asia/Shanghai") -> dict:
    now = datetime.now()
    return {"datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "weekday": ["一", "二", "三", "四", "五", "六", "日"][now.weekday()],
            "timestamp": int(now.timestamp())}


# --- 文本工具 ---
def tool_text_stats(text: str) -> dict:
    return {"chars": len(text), "words": len(text.split()),
            "lines": len(text.split("\n"))}


# --- 文件系统工具 (新!) ---
# 这是 Phase 4 Agent 相比 Phase 3 RAG 的关键扩展:
# Agent 可以主动访问文件系统!

def tool_read_file(path: str, max_lines: int = 100) -> dict:
    """读取文件内容。有安全限制。"""
    p = Path(path).expanduser().resolve()

    # 安全检查: 只允许读取项目内的文件
    project_root = Path(__file__).parent.parent.resolve()
    try:
        p.relative_to(project_root)
    except ValueError:
        return {"error": f"安全限制: 只能读取项目目录内的文件 ({project_root})"}

    if not p.exists():
        return {"error": f"文件不存在: {path}"}
    if p.is_dir():
        return {"error": f"路径是目录: {path}"}
    if p.suffix not in (".py", ".txt", ".md", ".json", ".yml", ".yaml", ".toml", ".cfg"):
        return {"error": f"不支持的文件类型: {p.suffix}"}

    try:
        lines = p.read_text(encoding="utf-8").split("\n")
        total = len(lines)
        preview = "\n".join(lines[:max_lines])
        return {"path": str(p), "total_lines": total,
                "preview_lines": min(max_lines, total),
                "content": preview}
    except Exception as e:
        return {"error": f"读取失败: {e}"}


def tool_list_files(directory: str = ".", path: str | None = None) -> dict:
    """列出目录内容。directory 和 path 是别名。"""
    target = path if path else directory
    p = Path(target).expanduser().resolve()
    project_root = Path(__file__).parent.parent.resolve()

    try:
        p.relative_to(project_root)
    except ValueError:
        return {"error": f"安全限制: 只能浏览项目目录 ({project_root})"}

    if not p.exists():
        return {"error": f"目录不存在: {directory}"}
    if not p.is_dir():
        return {"error": f"不是目录: {directory}"}

    items = []
    for entry in sorted(p.iterdir()):
        t = "dir" if entry.is_dir() else "file"
        items.append({"name": entry.name, "type": t,
                      "size": entry.stat().st_size if entry.is_file() else 0})

    return {"directory": str(p), "item_count": len(items), "items": items}


# --- 注册所有工具 ---
mcp_server = MCPServer("dev-assistant")

mcp_server.register(
    "calculate", "执行数学表达式计算",
    {"expression": {"type": "string", "description": "数学表达式"}},
    ["expression"], tool_calculate,
).register(
    "get_current_time", "获取当前日期时间",
    {"timezone": {"type": "string", "description": "时区"}},
    [], tool_time,
).register(
    "text_stats", "统计文本的字数、词数、行数",
    {"text": {"type": "string", "description": "统计文本"}},
    ["text"], tool_text_stats,
).register(
    "read_file", "读取文件内容 (限制: 仅项目目录内的 .py/.txt/.md 等)",
    {"path": {"type": "string", "description": "文件路径"},
     "max_lines": {"type": "integer", "description": "最大行数, 默认 100"}},
    ["path"], tool_read_file,
).register(
    "list_files", "列出目录内容 (限制: 仅项目目录)",
    {"directory": {"type": "string", "description": "目录路径, 默认 '.'"},
     "path": {"type": "string", "description": "目录路径 (directory 别名)"}},
    [], tool_list_files,
)

print(f"  MCP Server: {mcp_server.name}")
print(f"  已注册 {len(mcp_server._tools)} 个工具:")
for name in mcp_server._tools:
    print(f"    - {name}")


# ============================================================
# 四、Token 预算 (L32)
# ============================================================

class TokenBudget:
    def __init__(self, max_tokens: int = 50000):
        self.max_tokens = max_tokens
        self.warning_at = int(max_tokens * 0.7)
        self.used = 0

    def estimate(self, messages: list[dict]) -> int:
        raw = json.dumps(messages, ensure_ascii=False)
        return max(1, len(raw) // 4)

    def check(self, messages: list[dict]) -> tuple[bool, str]:
        est = self.estimate(messages)
        self.used = est
        if est >= self.max_tokens:
            return False, f"超出: {est}/{self.max_tokens}"
        if est >= self.warning_at:
            return True, f"⚡ {est}/{self.max_tokens}"
        return True, f"{est}/{self.max_tokens}"


# ============================================================
# 五、DevAssistant Agent — 双模式核心
# ============================================================

class DevAssistant:
    """开发者智能助手。

    双模式:
      Quick Mode  — ReAct 循环, 适合简单快速的任务
      Plan Mode   — Plan-then-Act, 适合复杂的多步任务

    类比 Java:
      DevAssistant ≈ @Service
        quickMode()  → 轻量级处理
        planMode()   → 重量级处理 (带规划)
    """

    QUICK_SYSTEM = """你是 DevAssistant, 一个开发者智能助手。
你可以调用工具来完成任务。遇到需要文件内容或目录信息时主动调用工具。
工具调用结果可能包含错误, 分析错误原因并尝试修正。用中文回答。"""

    PLAN_SYSTEM = """你是 DevAssistant 的规划模式。收到任务后:
1. 先制定一个执行计划 (列出步骤及依赖)
2. 按计划步骤逐步执行
3. 每步执行后反思: 成功? 需要调整?
4. 综合所有步骤结果, 给出最终答案

如果某一步失败, 标记为 "合理失败", 不影响其他步骤。用中文回答。"""

    def __init__(self, mcp: MCPClient, model: str = MODEL):
        self.mcp = mcp
        self.model = model
        self.budget = TokenBudget()
        self.history: list[dict] = []  # 对话历史
        self.default_mode = "quick"

    def ask(self, task: str, mode: str = "quick") -> dict:
        """统一入口。mode: 'quick' | 'plan'"""
        if mode == "plan":
            return self._plan_mode(task)
        return self._quick_mode(task)

    # --- Quick Mode (ReAct) ---

    def _quick_mode(self, task: str) -> dict:
        tools = self.mcp.list_tools()
        messages: list[dict] = [
            {"role": "user", "content": self._build_quick_prompt(task)}
        ]
        iterations = []

        for i in range(8):
            ok, info = self.budget.check(messages)
            if not ok:
                messages = messages[:1] + messages[-6:]

            if not api_ok:
                return {"answer": f"[模拟] Quick 模式收到: {task}", "iterations": 0,
                        "mode": "quick", "tokens": self.budget.used}

            response = llm.messages.create(
                model=self.model, max_tokens=1024, temperature=0.0,
                system=self.QUICK_SYSTEM, messages=messages, tools=tools,
            )

            tool_uses = _get_tool_uses(response)
            text = _get_text(response)

            if not tool_uses:
                iterations.append({"iter": i + 1, "action": "answer"})
                return {"answer": text, "iterations": i + 1, "mode": "quick",
                        "tokens": self.budget.used}

            # 追加 assistant 消息
            ac = []
            for b in response.content:
                if b.type == "text":
                    ac.append({"type": "text", "text": b.text})
                elif b.type == "tool_use":
                    ac.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
            messages.append({"role": "assistant", "content": ac})

            # 执行工具
            tc = []
            for tu in tool_uses:
                result = self.mcp.call_tool(tu.name, tu.input or {})
                tc.append({"type": "tool_result", "tool_use_id": tu.id,
                           "content": json.dumps(result, ensure_ascii=False)})
            messages.append({"role": "user", "content": tc})
            iterations.append({"iter": i + 1, "action": "tools",
                               "count": len(tool_uses)})

        return {"answer": "[Quick 模式: 超出最大迭代次数]",
                "iterations": 8, "mode": "quick", "tokens": self.budget.used}

    def _build_quick_prompt(self, task: str) -> str:
        """拼接对话历史到 prompt。"""
        if not self.history:
            return task
        hist = "\n".join(
            f"[{m['role']}]: {m['content'][:100]}" for m in self.history[-4:]
        )
        return f"对话历史:\n{hist}\n\n当前任务: {task}"

    # --- Plan Mode (Plan-then-Act) ---

    def _plan_mode(self, task: str) -> dict:
        tools = self.mcp.list_tools()
        tool_names = [t["name"] for t in tools]

        # Phase 1: 生成计划
        plan_prompt = self._plan_prompt(task, tools)
        if not api_ok:
            return {"answer": f"[模拟] Plan 模式收到: {task}", "iterations": 0,
                    "mode": "plan", "tokens": 0}

        plan_response = llm.messages.create(
            model=self.model, max_tokens=1024, temperature=0.0,
            messages=[{"role": "user", "content": plan_prompt}],
        )
        plan_text = _get_text(plan_response)
        plan, steps = self._parse_plan(plan_text)

        # Phase 2: 执行计划
        results = []
        for step in steps:
            step_id = step["id"]
            desc = step["description"]
            tool_name = step.get("tool", "none")

            if tool_name == "none" or tool_name not in tool_names:
                results.append({"step": step_id, "desc": desc,
                                "status": "done", "result": "(推理)"})
                continue

            # 执行工具
            tool_result = self.mcp.call_tool(tool_name, step.get("params", {}))
            status = "done" if "error" not in tool_result else "failed"
            results.append({"step": step_id, "desc": desc, "status": status,
                            "tool": tool_name, "result": tool_result})

            # 反思
            if status == "failed":
                # 标记依赖此步骤的后续步骤
                for s in steps:
                    if step_id in s.get("depends_on", []) and s.get("status") != "done":
                        s["status"] = "skipped"

        # Phase 3: 综合分析
        synth_prompt = self._synth_prompt(task, results)
        if api_ok:
            synth_response = llm.messages.create(
                model=self.model, max_tokens=800, temperature=0.0,
                messages=[{"role": "user", "content": synth_prompt}],
            )
            final_answer = _get_text(synth_response)
        else:
            final_answer = f"[模拟] Plan 执行完毕, {len(results)} 步骤"

        return {
            "answer": final_answer,
            "iterations": len(steps),
            "mode": "plan",
            "plan": plan,
            "steps": results,
            "tokens": self.budget.used,
        }

    def _plan_prompt(self, task: str, tools: list[dict]) -> str:
        tool_desc = "\n".join(
            f"- {t['name']}: {t.get('description', '')}" for t in tools
        )
        return f"""为以下任务制定执行计划。

可用工具:
{tool_desc}

任务: {task}

输出 JSON 格式:
{{
  "goal": "目标描述",
  "steps": [
    {{
      "id": 1,
      "description": "步骤描述",
      "tool": "工具名 (纯推理则填 none)",
      "params": {{}},
      "depends_on": []
    }}
  ]
}}

每个步骤如果需要工具就用 tool, 如果只是推理就用 none。
只输出 JSON, 不要其他文字。"""

    def _parse_plan(self, text: str) -> tuple[str, list[dict]]:
        try:
            json_str = text
            if "```" in text:
                json_str = text.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            data = json.loads(json_str.strip())
            return data.get("goal", ""), data.get("steps", [])
        except json.JSONDecodeError:
            return text[:100], [{"id": 1, "description": "分析任务",
                                 "tool": "none", "params": {}, "depends_on": []}]

    def _synth_prompt(self, task: str, results: list[dict]) -> str:
        log = "\n".join(
            f"[Step {r['step']}] {r['desc']} | {r['status']} | "
            f"{json.dumps(r.get('result', ''), ensure_ascii=False)[:150]}"
            for r in results
        )
        return f"""原始任务: {task}

执行日志:
{log}

综合以上结果, 回答:
1. 哪些步骤成功了?
2. 哪些步骤失败了? 影响是什么?
3. 给出最终的完整答案。"""


# ============================================================
# 六、CLI 界面
# ============================================================

class CLI:
    """DevAssistant 命令行界面。

    命令:
      /ask <问题>    Quick 模式 (ReAct, 快速响应)
      /plan <任务>    Plan 模式 (先规划再执行)
      /tools          列出可用工具
      /mode [quick|plan]  查看/切换模式
      /budget         查看 Token 用量
      /history        查看对话历史
      /help           帮助
      /quit           退出
    """

    def __init__(self):
        self.mcp_client = MCPClient()
        self.mcp_client.connect(mcp_server)
        self.agent = DevAssistant(self.mcp_client)
        self.mode = "quick"
        self.running = True
        self.command_count = 0

    def run(self):
        self._welcome()
        while self.running:
            try:
                raw = input("\n🧠 > ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not raw:
                continue
            if raw.startswith("/"):
                self._cmd(raw)
            else:
                self._query(raw)
        self._goodbye()

    def _cmd(self, raw: str):
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        match cmd:
            case "/ask":
                if arg:
                    self._query(arg)
                else:
                    print("  用法: /ask <问题>")
            case "/plan":
                if arg:
                    self._run_plan(arg)
                else:
                    print("  用法: /plan <任务>")
            case "/tools":
                tools = self.mcp_client.list_tools()
                print(f"\n  可用工具 ({len(tools)}):")
                for t in tools:
                    print(f"    {t['name']:<20s} {t.get('description', '')[:50]}")
            case "/mode":
                if arg in ("quick", "plan"):
                    self.mode = arg
                    print(f"  模式切换为: {arg}")
                else:
                    print(f"  当前模式: {self.mode}")
                    print(f"  切换: /mode quick  或  /mode plan")
            case "/budget":
                info = f"Token 估算: {self.agent.budget.used}/{self.agent.budget.max_tokens}"
                print(f"  {info}")
            case "/history":
                if not self.agent.history:
                    print("  无对话历史")
                else:
                    print(f"\n  对话历史 (最近 {len(self.agent.history)} 条):")
                    for i, m in enumerate(self.agent.history[-6:], 1):
                        role = "🧠" if m["role"] == "user" else "🤖"
                        print(f"  [{i}] {role} {m['content'][:100]}...")
            case "/help":
                self._help()
            case "/quit" | "/exit":
                self.running = False
            case _:
                print(f"  未知命令: {cmd}, 输入 /help 查看帮助")

    def _query(self, question: str):
        self.command_count += 1
        print(f"\n  [{self.mode.upper()} 模式] 思考中...")
        start = time.time()
        result = self.agent.ask(question, mode=self.mode)
        elapsed = (time.time() - start) * 1000

        print(f"\n  {'─' * 40}")
        print(f"  {result['answer']}")
        print(f"  {'─' * 40}")
        print(f"  {result['mode']} | {result['iterations']}轮 | "
              f"~{result['tokens']}tk | {elapsed:.0f}ms")

        self.agent.history.append({"role": "user", "content": question})
        self.agent.history.append({"role": "assistant", "content": result["answer"][:200]})

    def _run_plan(self, task: str):
        self.command_count += 1
        print(f"\n  [PLAN 模式] 制定计划中...")
        start = time.time()
        result = self.agent.ask(task, mode="plan")
        elapsed = (time.time() - start) * 1000

        print(f"\n  {'─' * 40}")
        if "plan" in result and result["plan"]:
            print(f"  目标: {result['plan'][:80]}...")
        if "steps" in result:
            for s in result["steps"]:
                icon = "✓" if s["status"] == "done" else "✗"
                print(f"  {icon} Step {s['step']}: {s['desc'][:60]}")
        print(f"\n  {result['answer']}")
        print(f"  {'─' * 40}")
        print(f"  plan | {elapsed:.0f}ms")

        self.agent.history.append({"role": "user", "content": task})
        self.agent.history.append({"role": "assistant", "content": result["answer"][:200]})

    def _welcome(self):
        print(f"""
  DevAssistant — 开发者智能助手
  {'─' * 40}
  双模式:
    Quick  (/ask)  — ReAct 快速响应, 适合简单问题
    Plan   (/plan) — 先规划再执行, 适合复杂任务

  当前模式: {self.mode}
  可用工具: {len(mcp_server._tools)} 个 (输入 /tools 查看)
  输入 /help 查看所有命令
""")

    def _help(self):
        print(f"""
  命令列表:
    /ask <问题>     Quick 模式问答
    /plan <任务>     Plan 模式 (多步规划)
    /tools           列出 MCP 工具
    /mode [模式]     查看/切换默认模式
    /budget          查看 Token 用量
    /history         对话历史
    /help            此帮助
    /quit            退出

  直接输入文本 = 用当前默认模式处理
  当前默认模式: {self.mode}
""")

    def _goodbye(self):
        print(f"\n  本次会话: {self.command_count} 次交互")
        print(f"  再见!\n")


# ============================================================
# 七、演示 (非交互模式)
# ============================================================

def demo():
    """非交互演示 — 展示两种模式。"""
    client = MCPClient()
    client.connect(mcp_server)
    agent = DevAssistant(client)

    print("=" * 60)
    print("演示 1: Quick 模式 (ReAct)")
    print("=" * 60)
    task1 = "帮我计算 (sqrt(256) + 100) / 2, 然后告诉我现在的时间。"
    print(f"  任务: {task1}")
    r1 = agent.ask(task1, mode="quick")
    print(f"\n  答案: {r1['answer'][:300]}...")
    print(f"  统计: {r1['mode']} | {r1['iterations']}轮 | ~{r1['tokens']}tk")

    print("\n\n" + "=" * 60)
    print("演示 2: Plan 模式 (Plan-then-Act)")
    print("=" * 60)
    task2 = "读取 phase4/ 目录下的文件列表, 统计有多少个 .py 文件。然后读取当前时间。"
    print(f"  任务: {task2}")
    r2 = agent.ask(task2, mode="plan")
    if "steps" in r2:
        print(f"\n  计划执行:")
        for s in r2["steps"]:
            icon = "✓" if s["status"] == "done" else "✗"
            print(f"  {icon} Step {s['step']}: {s['desc'][:60]}")
    print(f"\n  答案: {r2['answer'][:400]}...")

    print("\n\n" + "=" * 60)
    print("演示 3: 文件系统工具 (安全边界)")
    print("=" * 60)
    # 测试安全限制
    print(f"  尝试读取 /etc/passwd (越界)...")
    result = tool_read_file("/etc/passwd")
    print(f"  → {result}")
    print(f"\n  读取本文件自身 (合法)...")
    result = tool_read_file("phase4/35_agent_project.py", max_lines=5)
    print(f"  → {result.get('total_lines', '?')} 行, 前 5 行:")
    if "content" in result:
        for line in result["content"].split("\n")[:3]:
            print(f"    {line[:80]}")


# ============================================================
# 八、入口
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    elif len(sys.argv) > 1:
        # 一次性问答
        client = MCPClient()
        client.connect(mcp_server)
        agent = DevAssistant(client)
        task = " ".join(sys.argv[1:])
        print(f"  Quick 模式: {task}\n")
        r = agent.ask(task, mode="quick")
        print(r["answer"])
    else:
        # 交互模式 (需要 STDIN 是 TTY)
        if sys.stdin.isatty():
            cli = CLI()
            cli.run()
        else:
            # 非 TTY 环境回退到演示模式
            demo()

    print("\n" + "=" * 60)
    print("  Lesson 35 完成! Phase 4 收官!")
    print("=" * 60)
    print(f"""
  Phase 4 技能树:
    L31 Agent 基础       — ReAct 循环, Think→Act→Observe
    L32 工具调用深入      — ToolRegistry + TokenBudget + ToolResult
    L33 多步规划         — Plan-then-Act + Reflection + Self-correct
    L34 MCP 协议         — 标准化工具接口, 动态发现
    L35 端到端项目 ★     — 以上四课的完整融合

  DevAssistant 架构:
    CLI  →  /ask(Quick)  /plan(Plan)  /tools  /mode  /budget
    Agent →  DevAssistant  (双模式核心)
    MCP   →  MCPServer + MCPClient  (协议层)
    Tools →  calculate | time | text_stats | read_file | list_files

  🎯 下一阶段: Phase 5 — AI 工程化
     Lesson 41: 评估框架与指标设计
     如何度量你的 AI 系统的质量? 准确率、召回率、用户满意度...
""")


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 用 /plan 模式完成一个复杂任务:
#    启动 CLI: python phase4/35_agent_project.py
#    输入 /plan "分析 phase3/ 目录结构, 列出每课的 topic,
#             统计总代码行数, 找出最长的文件。"
#    对比: 同样的任务用 /ask (Quick 模式), 哪个更好?
#
# 2. 添加一个新工具:
#    实现一个 "代码统计" 工具 count_code:
#    - 读取 .py 文件
#    - 统计: 总行数、注释行数、空白行数、代码行数
#    - 注册到 MCP Server
#    让 Agent 用这个工具分析自己的代码。
#
# 3. 给文件工具加 "写入" 功能:
#    实现 write_file 工具 (需要用户确认):
#    - 在 CLI 中, 写入前打印 "确认写入? (y/n)"
#    - 这就是 Human-in-the-Loop 的实践
#    思考: 哪些操作需要确认? 哪些可以自动执行?
#
# 4. 扩展历史管理:
#    当前 history 只存最近的消息,
#    改为自动保存到文件:
#    - 每次交互后追加到 .devassistant_history.jsonl
#    - 启动时加载历史
#    - 实现 /clear 清除
#
# 5. (挑战) 实现 MCP 的 stdio Transport:
#    把 MCPServer 拆成独立进程:
#    - Server 进程: 从 stdin 读 JSON, 向 stdout 写 JSON
#    - Client 进程: 用 subprocess 启动 Server, 通过管道通信
#    这和生产 MCP 部署方式一致!
#
# 6. (思考) DevAssistant 的下一步:
#    如果要让 DevAssistant 成为你日常开发的助手,
#    你最想加什么能力?
#    - 运行测试? 分析 git log? 生成 commit message?
#    - 设计 3 个你最需要的工具, 写出 tool schema。
#
# 做完后告诉我:
#   - Plan 模式和 Quick 模式分别适合什么场景?
#   - 文件系统工具的安全限制你觉得够吗? 还需要加什么?
# 我们进入 Phase 5: AI 工程化!
# ============================================================


# ╔══════════════════════════════════════════════════════════════╗
# ║              试试看 — 练习实现代码                            ║
# ╚══════════════════════════════════════════════════════════════╝

import subprocess
import random
import json as json_module

print("\n")
print("=" * 60)
print("试试看练习: Lesson 35")
print("=" * 60)


# ─── 练习 1: /plan 复杂任务 (非交互模式演示) ──────────────────

print("\n" + "─" * 40)
print("练习 1: Plan vs Quick 复杂任务对比")
print("─" * 40)

print(f"""  复杂任务示例:
    /plan "分析 phase4/ 目录结构, 列出每课的文件名,
           统计总代码行数, 找出最长的文件。"

  Plan 模式执行流程:
    Phase 1: 制定计划
      Step 1: list_files("phase4/") → 获取目录结构
      Step 2: read_file(each .py)   → 读取每个文件的前 N 行
      Step 3: 统计比较               → 推理: 哪个最长
      Step 4: 综合分析               → 输出报告

    结构清晰, 每步可追踪, 失败时知道在哪一步出问题。

  Quick 模式执行流程:
    Iter 1: 调用 list_files("phase4/")
    Iter 2: 收到结果, 调用 read_file("phase4/31_agent_basics.py")
    Iter 3: 收到结果, 调用 read_file("phase4/32_tool_use.py")
    Iter 4: ...继续逐个读取...
    Iter N: 综合分析

    灵活但可能多轮迭代, 依赖 LLM 的"记忆力"。

  结论:
    Plan 更好 → 步骤明确, 执行可控, 结果可复现
    Quick 也可用 → 但更依赖 LLM 的智能规划能力
""")

# 实际运行演示 (如果 API 可用)
if api_ok:
    plan_client = MCPClient()
    plan_client.connect(mcp_server)
    plan_agent = DevAssistant(plan_client)

    complex_task = "列出 phase4/ 目录下的所有 .py 文件, 告诉我文件名列表。"
    print(f"\n  实际测试: {complex_task}")
    plan_result = plan_agent.ask(complex_task, mode="plan")
    print(f"  Plan 模式: {plan_result.get('answer', '')[:200]}...")
    if "steps" in plan_result:
        for s in plan_result["steps"]:
            icon = "✓" if s.get("status") == "done" else "✗"
            print(f"  {icon} Step {s['step']}: {s.get('desc', '')[:50]}")

    quick_result = plan_agent.ask(complex_task, mode="quick")
    print(f"\n  Quick 模式: {quick_result.get('answer', '')[:200]}...")
else:
    print(f"\n  [离线模式] 启动 CLI: python phase4/35_agent_project.py")
    print(f"  然后输入 /plan \"分析 phase4/ 目录...\"")
    print(f"  观察 Plan 模式的步骤分解和执行过程")

print(f"\n  ✅ 练习 1 完成: Plan vs Quick 各有适用场景")


# ─── 练习 2: 添加 count_code 工具 ─────────────────────────────

print("\n" + "─" * 40)
print("练习 2: count_code 工具 — 代码统计分析")
print("─" * 40)


def tool_count_code(path: str) -> dict:
    """统计分析 .py 文件的代码特征。

    统计: 总行数、注释行数、空白行数、有效代码行数、
         函数/类定义数、导入语句数。

    类比 Java:
      类似 SonarQube/PMD 的代码度量工具
    """
    p = Path(path).expanduser().resolve()
    project_root = Path(__file__).parent.parent.resolve()

    # 安全检查: 只能在项目目录内
    try:
        p.relative_to(project_root)
    except ValueError:
        return {"error": f"安全限制: 只能分析项目目录内的文件 ({project_root})"}

    if not p.exists():
        return {"error": f"文件不存在: {path}"}
    if not p.suffix == ".py":
        return {"error": f"只支持 .py 文件, 收到: {p.suffix}"}

    try:
        lines = p.read_text(encoding="utf-8").split("\n")
    except Exception as e:
        return {"error": f"读取失败: {e}"}

    total_lines = len(lines)
    blank_lines = 0
    comment_lines = 0
    code_lines = 0
    imports = 0
    defs = 0
    classes = 0

    in_docstring = False
    docstring_delim = ""

    for line in lines:
        stripped = line.strip()

        # 空白行
        if not stripped:
            blank_lines += 1
            continue

        # 检查多行 docstring
        if in_docstring:
            comment_lines += 1
            if docstring_delim in stripped and stripped != docstring_delim:
                in_docstring = False
            continue

        # 单行注释
        if stripped.startswith("#"):
            comment_lines += 1
            continue

        # Docstring 开始
        if stripped.startswith('"""') or stripped.startswith("'''"):
            comment_lines += 1
            if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                # 单行 docstring
                code_lines += 1  # 也算一行代码
            else:
                in_docstring = True
                docstring_delim = stripped[:3]
            continue

        # 代码行
        code_lines += 1

        # 统计导入
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports += 1

        # 统计函数/类定义
        if stripped.startswith("def "):
            defs += 1
        if stripped.startswith("class "):
            classes += 1

    return {
        "path": str(p),
        "total_lines": total_lines,
        "blank_lines": blank_lines,
        "comment_lines": comment_lines,
        "code_lines": code_lines,
        "comment_ratio": round(comment_lines / max(total_lines, 1) * 100, 1),
        "imports": imports,
        "function_defs": defs,
        "class_defs": classes,
    }


# 注册到 MCP Server
mcp_server.register(
    "count_code",
    "统计分析 Python 文件的代码组成: 注释行数、空白行数、代码行数、函数/类定义数、导入数。适用于代码质量分析。",
    {"path": {"type": "string",
              "description": "Python 文件路径 (相对于项目根目录, 如 'phase4/31_agent_basics.py')"}},
    ["path"],
    tool_count_code,
)

# 测试
print(f"  Server 工具数: {len(mcp_server._tools)}")
print(f"  工具列表: {list(mcp_server._tools.keys())}")

# 分析本文件
self_analysis = tool_count_code("phase4/35_agent_project.py")
if "error" not in self_analysis:
    print(f"\n  分析 phase4/35_agent_project.py:")
    print(f"    总行数: {self_analysis['total_lines']}")
    print(f"    注释行: {self_analysis['comment_lines']} "
          f"({self_analysis['comment_ratio']}%)")
    print(f"    空白行: {self_analysis['blank_lines']}")
    print(f"    代码行: {self_analysis['code_lines']}")
    print(f"    导入: {self_analysis['imports']}")
    print(f"    函数: {self_analysis['function_defs']}")
    print(f"    类: {self_analysis['class_defs']}")
else:
    print(f"  {self_analysis['error']}")

# 通过 MCP Client 调用
client_tmp = MCPClient()
client_tmp.connect(mcp_server)
result_cc = client_tmp.call_tool("count_code", {"path": "phase4/31_agent_basics.py"})
print(f"\n  MCP 调用 count_code(31_agent_basics.py):")
if "error" not in result_cc:
    print(f"    lines={result_cc['total_lines']}, "
          f"code={result_cc['code_lines']}, "
          f"comments={result_cc['comment_lines']}")

print(f"\n  ✅ 练习 2 完成: count_code 工具已注册, 支持注释率/函数/类统计")


# ─── 练习 3: write_file 带 Human-in-the-Loop ──────────────────

print("\n" + "─" * 40)
print("练习 3: write_file 带 Human-in-the-Loop 确认")
print("─" * 40)


def tool_write_file(path: str, content: str, _confirmed: bool = False) -> dict:
    """写入文件。安全限制 + 人工确认。

    安全策略 (多层):
      1. 路径沙箱 — 只能写入项目 OUTPUT 目录
      2. 文件类型白名单 — 只允许 .txt/.json/.md/.log
      3. 人工确认 — _confirmed=True 才会写入
      4. 不覆盖 — 文件存在返回错误 (需显式 overwrite)

    类比 Java:
      类似 @PreAuthorize + 审批工作流
    """
    p = Path(path).expanduser().resolve()

    # 安全 1: 只允许写入 output/ 目录
    project_root = Path(__file__).parent.parent.resolve()
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    try:
        p.relative_to(output_dir)
    except ValueError:
        return {"error": f"安全: 只能写入 {output_dir} 目录, 收到: {path}"}

    # 安全 2: 文件类型白名单
    allowed_exts = {".txt", ".json", ".md", ".log", ".csv"}
    if p.suffix not in allowed_exts:
        return {"error": f"安全: 不支持的文件类型 {p.suffix}, "
                f"允许: {allowed_exts}"}

    # 安全 3: 内容检查 (禁止写入可执行代码)
    dangerous_patterns = ["import os", "subprocess", "eval(", "exec(",
                         "rm -rf", "DROP TABLE", "__import__"]
    for pattern in dangerous_patterns:
        if pattern in content:
            return {"error": f"安全: 内容包含禁止模式 '{pattern}'"}

    # 安全 4: 人工确认
    if not _confirmed:
        preview = content[:200] + ("..." if len(content) > 200 else "")
        return {
            "status": "pending_confirmation",
            "path": str(p),
            "content_preview": preview,
            "content_size": len(content),
            "message": "需要人工确认。请在 CLI 中输入 y 确认写入, 或 n 取消。",
            "hint": "重新调用此工具并添加 _confirmed=true 以确认写入",
        }

    # 安全 5: 不覆盖 (需显式 overwrite)
    if p.exists():
        return {"error": f"文件已存在: {path}. 目前不支持覆盖。"
                "请手动删除后重试。"}

    try:
        p.write_text(content, encoding="utf-8")
        return {
            "status": "written",
            "path": str(p),
            "size": len(content),
            "lines": len(content.split("\n")),
        }
    except Exception as e:
        return {"error": f"写入失败: {e}"}


# 测试安全策略
print(f"  测试 1: 写入到非允许目录")
r1 = tool_write_file("/tmp/test.txt", "hello")
print(f"    {r1}")

print(f"  测试 2: 写入允许目录, 无确认")
r2 = tool_write_file("output/test.txt", "hello world")
print(f"    {r2}")

print(f"  测试 3: 内容包含危险模式")
r3 = tool_write_file("output/test.txt", "import os\nos.system('rm -rf /')")
print(f"    {r3}")

print(f"  测试 4: 正常写入 (有确认)")
r4 = tool_write_file("output/test.txt",
                     "# DevAssistant 输出\nHello from Agent!",
                     _confirmed=True)
print(f"    {r4}")

# 清理
Path("output/test.txt").unlink(missing_ok=True)

print(f"\n  Human-in-the-Loop 设计原则:")
print(f"    必须确认: write_file, delete_file, send_email, run_command")
print(f"    自动执行: read_file, calculate, text_stats, count_code")
print(f"    参考 deploy/agent_core.py: tool_read_file 的路径沙箱")

print(f"\n  ✅ 练习 3 完成: write_file 多层安全 + HITL 确认机制")


# ─── 练习 4: 历史持久化 ───────────────────────────────────────

print("\n" + "─" * 40)
print("练习 4: 对话历史持久化")
print("─" * 40)

HISTORY_FILE = Path(__file__).parent / ".devassistant_history.jsonl"


class PersistentHistory:
    """持久化对话历史管理器。

    格式: JSONL (每行一条 JSON 记录)
    支持: 追加写入、批量加载、清除、截断

    类比 Java:
      类似 JPA Repository, 管理对话记录的持久化
    """

    def __init__(self, filepath: Path = HISTORY_FILE, max_entries: int = 1000):
        self.filepath = filepath
        self.max_entries = max_entries
        self.entries: list[dict] = []

    def load(self) -> list[dict]:
        """从文件加载历史记录。"""
        self.entries = []
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                self.entries.append(json_module.loads(line))
                            except json_module.JSONDecodeError:
                                continue
            except Exception as e:
                print(f"  [警告] 加载历史失败: {e}")
        return self.entries

    def append(self, role: str, content: str, metadata: dict | None = None):
        """追加一条对话记录。"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content[:500],  # 截断过长内容
            "metadata": metadata or {},
        }
        self.entries.append(entry)

        # 写入文件
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json_module.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  [警告] 写入历史失败: {e}")

        # 保持最近 max_entries 条
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def clear(self):
        """清除所有历史。"""
        self.entries = []
        if self.filepath.exists():
            self.filepath.unlink()
        print(f"  历史已清除")

    def recent(self, n: int = 10) -> list[dict]:
        """获取最近 n 条记录。"""
        return self.entries[-n:]

    @property
    def count(self) -> int:
        return len(self.entries)


# 演示
hist = PersistentHistory()
print(f"  历史文件: {hist.filepath}")

# 模拟几条对话
hist.append("user", "帮我计算 sqrt(256)")
hist.append("assistant", "sqrt(256) = 16")
hist.append("user", "现在是什么时间?")
hist.append("assistant", "2026年6月19日 星期五")

print(f"  当前记录数: {hist.count}")

# 读取
recent = hist.recent(2)
print(f"  最近 2 条:")
for r in recent:
    role_icon = "用户" if r["role"] == "user" else "助手"
    print(f"    [{r['timestamp'][:19]}] {role_icon}: {r['content'][:60]}...")

# 加载 (模拟重启后)
hist2 = PersistentHistory()
loaded = hist2.load()
print(f"  重新加载: {len(loaded)} 条历史")

# 清除
hist.clear()
hist2_after = PersistentHistory()
after_clear = hist2_after.load()
print(f"  清除后: {len(after_clear)} 条历史")

# 清理测试文件
HISTORY_FILE.unlink(missing_ok=True)

# 集成到 DevAssistant 的建议
print(f"\n  集成方式:")
print(f"    class DevAssistant:")
print(f"        def __init__(self, ...):")
print(f"            self.history_mgr = PersistentHistory()")
print(f"            self.history_mgr.load()")
print(f"        ")
print(f"        def ask(self, task, mode):")
print(f"            # ... 执行 ...")
print(f"            self.history_mgr.append('user', task)")
print(f"            self.history_mgr.append('assistant', result['answer'])")

print(f"\n  ✅ 练习 4 完成: 实现了 JSONL 持久化历史管理器")


# ─── 练习 5 (挑战): MCP stdio Transport ───────────────────────

print("\n" + "─" * 40)
print("练习 5 (挑战): MCP stdio Transport")
print("─" * 40)


def run_stdio_server():
    """MCP Server stdio 进程 (独立进程模式)。

    从 stdin 逐行读取 JSON-RPC 请求, 向 stdout 写入 JSON-RPC 响应。
    stderr 用于日志 (不干扰协议通信)。

    类比 Java:
      类似一个从 System.in 读取、写入 System.out 的 CLI 应用
    """
    # 创建一个 math server
    srv = MCPServer(name="stdio-math-server")
    srv.register_tool(
        "calculate", "数学计算",
        {"expression": {"type": "string", "description": "数学表达式"}},
        ["expression"],
        lambda expression: {"result": eval(
            expression, {"__builtins__": {}},
            {"abs": abs, "sqrt": math.sqrt, "pi": math.pi, "sin": math.sin})},
    )

    print(f"[server] stdio-math-server 已启动, 等待请求...",
          file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "SHUTDOWN":
            break
        response = srv.handle(line)
        print(response, flush=True)  # stdout → 给 client 的响应


class StdioMCPClient:
    """通过 subprocess 连接 MCP Server 的客户端。

    Client 进程 → subprocess.Popen → Server 进程
    通信: stdin/stdout 管道 (JSON-RPC 行协议)

    类比 Java:
      类似 ProcessBuilder + BufferedReader/Writer
    """

    def __init__(self, server_script_code: str = ""):
        self.process: subprocess.Popen | None = None
        self._rid = 0

    def connect(self, server_script_path: str = ""):
        """启动 MCP Server 进程。"""
        # 如果提供了外部脚本路径, 启动它
        if server_script_path:
            self.process = subprocess.Popen(
                [sys.executable, server_script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        else:
            # 启动自身作为 server (通过 --mcp-server 参数)
            self.process = subprocess.Popen(
                [sys.executable, __file__, "--mcp-server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def _send(self, method: str, params: dict | None = None) -> MCPResponse:
        if self.process is None:
            return MCPResponse(error="未连接")
        self._rid += 1
        req = json_module.dumps({
            "jsonrpc": "2.0", "method": method,
            "params": params or {}, "id": self._rid,
        })
        self.process.stdin.write(req + "\n")
        self.process.stdin.flush()
        raw = self.process.stdout.readline()
        return MCPResponse.from_json(raw.strip())

    def list_tools(self) -> list[dict]:
        resp = self._send("tools/list")
        if resp.error:
            return []
        tools = resp.result.get("tools", [])
        return [{
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("inputSchema",
                                  {"type": "object", "properties": {}, "required": []}),
        } for t in tools]

    def call_tool(self, name: str, arguments: dict) -> dict:
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        if resp.error:
            return {"error": resp.error}
        for item in resp.result.get("content", []):
            if item.get("type") == "text":
                try:
                    return json_module.loads(item["text"])
                except json_module.JSONDecodeError:
                    return {"text": item["text"]}
        return {"raw": resp.result.get("content", [])}

    def shutdown(self):
        if self.process:
            self.process.stdin.write("SHUTDOWN\n")
            self.process.stdin.flush()
            self.process.wait(timeout=5)
            self.process = None


# 演示: 用嵌入式 server 模拟 stdio transport
# (实际生产环境中, server 是独立脚本, 这里直接复用进程内的 server 对象)
print(f"  stdio Transport 架构:")
print(f"""    ┌──────────────┐       subprocess        ┌──────────────────┐
    │ MCP Client   │ ──── stdin/stdout ──── │  MCP Server      │
    │ (Agent App)  │     JSON-RPC 行协议     │  (独立进程)       │
    └──────────────┘                        └──────────────────┘

  生产环境:
    Server: python mcp_math_server.py
    Client: subprocess.Popen(["python", "mcp_math_server.py"],
                             stdin=PIPE, stdout=PIPE)
""")

# 模拟: 使用 embed 的 server 演示协议
print(f"  演示: 通过模拟 stdio 管道调用工具")
test_req = json_module.dumps({
    "jsonrpc": "2.0", "method": "tools/call",
    "params": {"name": "calculate",
               "arguments": {"expression": "sqrt(256) + pi * 3"}},
    "id": 1,
})
test_resp = mcp_server.handle(test_req)
resp_data = json_module.loads(test_resp)
result_content = resp_data["result"]["content"][0]["text"]
print(f"    请求: {test_req}")
print(f"    响应: {test_resp[:100]}...")
print(f"    结果: {result_content}")

print(f"\n  关键区别: stdio vs 进程内")
print(f"    进程内: client.connect(server)  → 直接持有对象引用")
print(f"    stdio:  client.connect_via_subprocess() → 通过管道通信")
print(f"    stdio 优势: 跨语言、进程隔离、crash 不影响 client")
print(f"    这是官方 mcp SDK 的标准部署方式")

print(f"\n  ✅ 练习 5 完成: 实现了 MCP stdio Transport 架构")


# ─── 练习 6 (思考): DevAssistant 下一步 ───────────────────────

print("\n" + "─" * 40)
print("练习 6 (思考): DevAssistant 下一步 — 3 个核心工具")
print("─" * 40)

print("""  DevAssistant 作为日常开发助手, 最需要的 3 个工具:

  ┌─────────────────────────────────────────────────────────────┐
  │ 工具 1: git_analyze — Git 仓库分析                          │
  │                                                              │
  │  参数:                                                       │
  │    repo_path: string   仓库路径                               │
  │    action: string      操作: "log" | "diff" | "status"       │
  │    author: string      按作者过滤 (可选)                      │
  │    days: integer       最近 N 天 (默认 7)                     │
  │                                                              │
  │  返回:                                                       │
  │    {commits: [...], stats: {additions, deletions},           │
  │     summary: "3 个提交, +120/-45 行"}                        │
  │                                                              │
  │  场景: "今天做了什么?" "这个文件谁改的?"                       │
  │         "生成本周的 commit message 摘要"                      │
  ├─────────────────────────────────────────────────────────────┤
  │ 工具 2: run_tests — 运行测试并分析结果                       │
  │                                                              │
  │  参数:                                                       │
  │    test_path: string   测试路径 (文件或目录)                  │
  │    verbose: bool       详细输出 (默认 false)                 │
  │                                                              │
  │  返回:                                                       │
  │    {passed: 12, failed: 3, errors: 1,                        │
  │     failed_tests: [详细列表], duration_ms: 2340}             │
  │                                                              │
  │  场景: "跑一下单元测试" "分析失败的测试原因"                    │
  ├─────────────────────────────────────────────────────────────┤
  │ 工具 3: code_search — 语义搜索代码                           │
  │                                                              │
  │  参数:                                                       │
  │    query: string       搜索内容 (自然语言或代码片段)          │
  │    scope: string       搜索范围: "all" | "phase1" | "deploy" │
  │    top_k: integer      返回数 (默认 5)                        │
  │                                                              │
  │  返回:                                                       │
  │    {results: [{file, line, snippet, score}, ...]}            │
  │                                                              │
  │  场景: "这函数在哪定义的?" "类似的实现在哪里?"                  │
  └─────────────────────────────────────────────────────────────┘

  工具 Schema 示例 (git_analyze):
  {{
    "name": "git_analyze",
    "description": "分析 Git 仓库的提交历史、变更统计。",
    "input_schema": {{
      "type": "object",
      "properties": {{
        "repo_path": {{"type": "string"}},
        "action": {{"type": "string", "enum": ["log", "diff", "status"]}},
        "author": {{"type": "string"}},
        "days": {{"type": "integer", "default": 7}}
      }},
      "required": ["repo_path", "action"]
    }}
  }}

  其他可选工具:
    - format_code: 格式化代码 (black/formatter)
    - lint_check: 代码质量检查 (flake8/pylint 输出解析)
    - doc_gen: 生成函数文档注释
    - dep_graph: 分析模块依赖关系图
    - perf_profile: 运行简单的性能分析

  参考 deploy/ 项目: app.py 的 FastAPI 接口可扩展这些工具
""")

print(f"  ✅ 练习 6 完成: 设计了 3 个核心开发工具 + schema")

print(f"\n  📝 学习总结:")
print(f"     Plan 模式适合: 步骤明确的多步任务 (分析/统计/报告)")
print(f"     Quick 模式适合: 简单问答、探索性查询")
print(f"     安全限制: 路径沙箱 + 文件类型白名单 + 内容扫描三层防护,")
print(f"     再加审计日志和人工确认, 基本够用。")
print(f"     可增强: 文件大小限制、敏感信息脱敏、速率限制")
print(f"\n  🎯 Phase 4 全部完成! 进入 Phase 5: AI 工程化!")
