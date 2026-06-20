# ============================================================
# deploy/agent_core.py — 核心 Agent 层
# ============================================================
# 提取自 phase4/35_agent_project.py，做以下重构:
#   1. LlmClient 封装 Anthropic SDK (消除模块级全局变量)
#   2. project_root 参数化 (不再硬编码 Path(__file__).parent.parent)
#   3. 所有组件通过依赖注入连接
#
# 类比 Java:
#   LlmClient     ≈ @Repository (数据访问层)
#   MCPServer     ≈ @Service (工具注册)
#   DevAssistant  ≈ @Service (业务逻辑)
# ============================================================

import json
import math
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, Any

from anthropic import Anthropic
from anthropic.types import Message, ToolUseBlock


# ============================================================
# 一、LlmClient — LLM 调用封装
# ============================================================

class LlmClient:
    """Anthropic SDK 的轻量封装。

    消除 L35 中的模块级全局变量 (llm, api_ok, _get_text, _get_tool_uses)。
    支持依赖注入，方便测试和切换模型。

    类比 Java:
      类似一个 Repository，封装外部 API 调用。
    """

    def __init__(self, api_key: str = "", base_url: str = "",
                 model: str = "claude-sonnet-4-6",
                 max_retries: int = 3, timeout: float = 60.0):
        self.model = model
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        kwargs["max_retries"] = max_retries
        kwargs["timeout"] = timeout
        self._client = Anthropic(**kwargs)
        self._healthy: bool | None = None

    @property
    def is_healthy(self) -> bool:
        """检查 API 连通性 (带缓存，只检查一次)。"""
        if self._healthy is None:
            try:
                self._client.messages.create(
                    model=self.model, max_tokens=10,
                    messages=[{"role": "user", "content": "ping"}],
                )
                self._healthy = True
            except Exception:
                self._healthy = False
        return self._healthy

    def create(self, messages: list[dict], tools: list[dict] | None = None,
               system: str = "", max_tokens: int = 1024,
               temperature: float = 0.0) -> Message:
        """调用 LLM API。

        Args:
            messages: 对话消息列表
            tools: Anthropic 格式工具定义
            system: 系统提示词
            max_tokens: 最大输出 token
            temperature: 生成温度

        Returns:
            anthropic.types.Message
        """
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        return self._client.messages.create(**kwargs)

    @staticmethod
    def get_text(response: Message) -> str:
        """提取响应中的文本内容。"""
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts)

    @staticmethod
    def get_tool_uses(response: Message) -> list[ToolUseBlock]:
        """提取响应中的工具调用。"""
        return [b for b in response.content if b.type == "tool_use"]


# ============================================================
# 二、MCP 协议层
# ============================================================

@dataclass
class MCPResponse:
    """JSON-RPC 2.0 响应。"""
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

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

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
                    return MCPResponse(
                        error=f"未知工具: {params.get('name')}", id=rid
                    ).to_json()
                try:
                    args = params.get("arguments", {})
                    result = tool["handler"](**args)
                    text = json.dumps(result, ensure_ascii=False)
                    return MCPResponse(
                        result={"content": [{"type": "text", "text": text}]},
                        id=rid,
                    ).to_json()
                except Exception as e:
                    return MCPResponse(
                        error=f"执行失败: {e}", id=rid
                    ).to_json()
            case "initialize":
                return MCPResponse(result={
                    "protocolVersion": "0.2",
                    "serverInfo": {"name": self.name, "version": "1.0"},
                    "capabilities": {"tools": {}},
                }, id=rid).to_json()
            case _:
                return MCPResponse(
                    error=f"未知方法: {method}", id=rid
                ).to_json()


class MCPClient:
    """MCP Client — 封装协议通信，格式转换。"""

    def __init__(self, server: MCPServer | None = None):
        self.server = server
        self._rid = 0
        self.connected = False

    def connect(self, server: MCPServer):
        self.server = server
        resp = self._send("initialize")
        if resp.error:
            raise RuntimeError(f"MCP 握手失败: {resp.error}")
        self.connected = True
        return resp.result

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
# 三、Token 预算
# ============================================================

class TokenBudget:
    """Token 预算估算器。"""

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
            return True, f"高: {est}/{self.max_tokens}"
        return True, f"正常: {est}/{self.max_tokens}"


# ============================================================
# 四、DevAssistant — 双模式 Agent
# ============================================================

class DevAssistant:
    """开发者智能助手。

    双模式:
      Quick  — ReAct 循环, 适合简单任务
      Plan   — Plan-then-Act, 适合复杂多步任务
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

    def __init__(self, mcp: MCPClient, llm: LlmClient):
        self.mcp = mcp
        self.llm = llm
        self.budget = TokenBudget()
        self.history: list[dict] = []
        self.default_mode = "quick"

    def ask(self, task: str, mode: str = "quick") -> dict:
        """统一入口。"""
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

            if not self.llm.is_healthy:
                return {
                    "answer": f"[离线模式] Quick 模式收到: {task}",
                    "iterations": 0, "mode": "quick",
                    "tokens": self.budget.used,
                }

            response = self.llm.create(
                messages=messages, tools=tools,
                system=self.QUICK_SYSTEM, max_tokens=1024,
            )

            tool_uses = self.llm.get_tool_uses(response)
            text = self.llm.get_text(response)

            if not tool_uses:
                iterations.append({"iter": i + 1, "action": "answer"})
                return {
                    "answer": text, "iterations": i + 1,
                    "mode": "quick", "tokens": self.budget.used,
                }

            # 追加 assistant 消息
            ac = []
            for b in response.content:
                if b.type == "text":
                    ac.append({"type": "text", "text": b.text})
                elif b.type == "tool_use":
                    ac.append({
                        "type": "tool_use", "id": b.id,
                        "name": b.name, "input": b.input,
                    })
            messages.append({"role": "assistant", "content": ac})

            # 执行工具
            tc = []
            for tu in tool_uses:
                result = self.mcp.call_tool(tu.name, tu.input or {})
                tc.append({
                    "type": "tool_result", "tool_use_id": tu.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            messages.append({"role": "user", "content": tc})
            iterations.append({
                "iter": i + 1, "action": "tools",
                "count": len(tool_uses),
            })

        return {
            "answer": "[Quick 模式: 超出最大迭代次数]",
            "iterations": 8, "mode": "quick",
            "tokens": self.budget.used,
        }

    def _build_quick_prompt(self, task: str) -> str:
        if not self.history:
            return task
        hist = "\n".join(
            f"[{m['role']}]: {m['content'][:100]}"
            for m in self.history[-4:]
        )
        return f"对话历史:\n{hist}\n\n当前任务: {task}"

    # --- Plan Mode ---

    def _plan_mode(self, task: str) -> dict:
        tools = self.mcp.list_tools()
        tool_names = [t["name"] for t in tools]

        # Phase 1: 生成计划
        plan_prompt = self._plan_prompt(task, tools)
        if not self.llm.is_healthy:
            return {
                "answer": f"[离线模式] Plan 模式收到: {task}",
                "iterations": 0, "mode": "plan", "tokens": 0,
            }

        plan_response = self.llm.create(
            messages=[{"role": "user", "content": plan_prompt}],
            max_tokens=1024,
        )
        plan_text = self.llm.get_text(plan_response)
        plan, steps = self._parse_plan(plan_text)

        # Phase 2: 执行计划
        results = []
        for step in steps:
            step_id = step["id"]
            desc = step["description"]
            tool_name = step.get("tool", "none")

            if tool_name == "none" or tool_name not in tool_names:
                results.append({
                    "step": step_id, "desc": desc,
                    "status": "done", "result": "(推理)",
                })
                continue

            tool_result = self.mcp.call_tool(
                tool_name, step.get("params", {})
            )
            status = "done" if "error" not in tool_result else "failed"
            results.append({
                "step": step_id, "desc": desc, "status": status,
                "tool": tool_name, "result": tool_result,
            })

            # 失败时级联标记
            if status == "failed":
                for s in steps:
                    if (step_id in s.get("depends_on", [])
                            and s.get("status") != "done"):
                        s["status"] = "skipped"

        # Phase 3: 综合分析
        synth_prompt = self._synth_prompt(task, results)
        if self.llm.is_healthy:
            synth_response = self.llm.create(
                messages=[{"role": "user", "content": synth_prompt}],
                max_tokens=800,
            )
            final_answer = self.llm.get_text(synth_response)
        else:
            final_answer = f"[离线模式] Plan 执行完毕, {len(results)} 步骤"

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
            return text[:100], [{
                "id": 1, "description": "分析任务",
                "tool": "none", "params": {}, "depends_on": [],
            }]

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
# 五、工具函数
# ============================================================

_SAFE = {"abs": abs, "round": round, "min": min, "max": max,
         "sum": sum, "pow": pow, "sqrt": math.sqrt,
         "sin": math.sin, "cos": math.cos, "log": math.log,
         "pi": math.pi, "e": math.e}


def tool_calculate(expression: str) -> dict:
    result = eval(expression, {"__builtins__": {}}, _SAFE)
    return {"expression": expression, "result": result}


def tool_get_current_time(timezone: str = "Asia/Shanghai") -> dict:
    now = datetime.now()
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": ["一", "二", "三", "四", "五", "六", "日"][now.weekday()],
        "timestamp": int(now.timestamp()),
    }


def tool_text_stats(text: str) -> dict:
    return {
        "chars": len(text),
        "words": len(text.split()),
        "lines": len(text.split("\n")),
    }


def tool_read_file(path: str, max_lines: int = 100,
                   project_root: str = ".") -> dict:
    """读取文件内容。安全限制: 仅项目目录内。"""
    p = Path(path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()

    try:
        p.relative_to(root)
    except ValueError:
        return {"error": f"安全限制: 只能读取项目目录内的文件 ({root})"}

    if not p.exists():
        return {"error": f"文件不存在: {path}"}
    if p.is_dir():
        return {"error": f"路径是目录: {path}"}
    if p.suffix not in (".py", ".txt", ".md", ".json",
                        ".yml", ".yaml", ".toml", ".cfg"):
        return {"error": f"不支持的文件类型: {p.suffix}"}

    try:
        lines = p.read_text(encoding="utf-8").split("\n")
        total = len(lines)
        preview = "\n".join(lines[:max_lines])
        return {
            "path": str(p), "total_lines": total,
            "preview_lines": min(max_lines, total),
            "content": preview,
        }
    except Exception as e:
        return {"error": f"读取失败: {e}"}


def tool_list_files(directory: str = ".", path: str | None = None,
                    project_root: str = ".") -> dict:
    """列出目录内容。directory 和 path 是别名。"""
    target = path if path else directory
    p = Path(target).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()

    try:
        p.relative_to(root)
    except ValueError:
        return {"error": f"安全限制: 只能浏览项目目录 ({root})"}

    if not p.exists():
        return {"error": f"目录不存在: {target}"}
    if not p.is_dir():
        return {"error": f"不是目录: {target}"}

    items = []
    for entry in sorted(p.iterdir()):
        t = "dir" if entry.is_dir() else "file"
        items.append({
            "name": entry.name, "type": t,
            "size": entry.stat().st_size if entry.is_file() else 0,
        })

    return {"directory": str(p), "item_count": len(items), "items": items}


# ============================================================
# 六、工厂函数
# ============================================================

def create_mcp_server(name: str = "dev-assistant",
                      project_root: str = ".") -> MCPServer:
    """创建并注册好所有工具的 MCPServer。

    这是部署时的组装入口:
      server = create_mcp_server(project_root=config.project_root)
    """
    server = MCPServer(name)

    # 用 functools.partial 绑定 project_root 参数
    from functools import partial

    read_file_bound = partial(tool_read_file, project_root=project_root)
    list_files_bound = partial(tool_list_files, project_root=project_root)

    server.register(
        "calculate", "执行数学表达式计算",
        {"expression": {"type": "string", "description": "数学表达式"}},
        ["expression"], tool_calculate,
    ).register(
        "get_current_time", "获取当前日期时间",
        {"timezone": {"type": "string", "description": "时区"}},
        [], tool_get_current_time,
    ).register(
        "text_stats", "统计文本的字数、词数、行数",
        {"text": {"type": "string", "description": "统计文本"}},
        ["text"], tool_text_stats,
    ).register(
        "read_file", "读取文件内容 (限制: 仅项目目录内的 .py/.txt/.md 等)",
        {"path": {"type": "string", "description": "文件路径"},
         "max_lines": {"type": "integer", "description": "最大行数, 默认 100"}},
        ["path"], read_file_bound,
    ).register(
        "list_files", "列出目录内容 (限制: 仅项目目录)",
        {"directory": {"type": "string", "description": "目录路径, 默认 '.'"},
         "path": {"type": "string", "description": "目录路径 (directory 别名)"}},
        [], list_files_bound,
    )

    return server


def create_agent(config=None) -> DevAssistant:
    """创建完整配置的 DevAssistant。

    这是便捷组装入口:
      agent = create_agent(AppConfig.from_env())
    """
    if config is None:
        from .config import AppConfig
        config = AppConfig.from_env()

    llm = LlmClient(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
        max_retries=config.llm_max_retries,
        timeout=config.llm_timeout_seconds,
    )

    server = create_mcp_server(project_root=config.project_root)
    client = MCPClient()
    client.connect(server)

    agent = DevAssistant(client, llm)
    return agent
