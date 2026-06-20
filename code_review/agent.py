# ============================================================
# code_review/agent.py — ReviewAgent
# ============================================================
# 复用 deploy/agent_core.py 的 MCP 基础设施，
# ReviewAgent 专为代码审查优化: Plan 模式 + 代码专用 System Prompt。
# ============================================================

import json
import math
from pathlib import Path
from functools import partial
from typing import Callable, Any

from anthropic import Anthropic
from anthropic.types import Message, ToolUseBlock

from code_review.tools import (
    tool_check_style,
    tool_detect_patterns,
    tool_read_file,
)


# ============================================================
# 一、LlmClient — 轻量 LLM 封装
# ============================================================

class LlmClient:
    """Anthropic SDK 封装。"""

    def __init__(self, api_key: str = "", base_url: str = "",
                 model: str = "claude-sonnet-4-6",
                 max_retries: int = 2, timeout: float = 90.0):
        self.model = model
        kwargs = {"max_retries": max_retries, "timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = Anthropic(**kwargs)
        self._healthy: bool | None = None

    @property
    def is_healthy(self) -> bool:
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
        kwargs: dict = {
            "model": self.model, "max_tokens": max_tokens,
            "temperature": temperature, "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        return self._client.messages.create(**kwargs)

    @staticmethod
    def get_text(response: Message) -> str:
        return "\n".join(
            b.text for b in response.content if b.type == "text"
        )

    @staticmethod
    def get_tool_uses(response: Message) -> list[ToolUseBlock]:
        return [b for b in response.content if b.type == "tool_use"]


# ============================================================
# 二、MCP 协议层 (精简版)
# ============================================================

class MCPResponse:
    def __init__(self, result=None, error=None, id=0):
        self.result = result
        self.error = error
        self.id = id

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
    def __init__(self, name: str = "code_review"):
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
                return MCPResponse(
                    result={"tools": tools}, id=rid
                ).to_json()
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
    def __init__(self, server: MCPServer | None = None):
        self.server = server
        self._rid = 0

    def connect(self, server: MCPServer):
        self.server = server
        resp = self._send("initialize")
        if resp.error:
            raise RuntimeError(f"MCP 握手失败: {resp.error}")
        return resp.result

    def _send(self, method: str, params: dict | None = None) -> MCPResponse:
        if self.server is None:
            return MCPResponse(error="未连接")
        self._rid += 1
        raw = json.dumps({
            "jsonrpc": "2.0", "method": method,
            "params": params or {}, "id": self._rid,
        })
        return MCPResponse.from_json(self.server.handle(raw))

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
        resp = self._send("tools/call",
                         {"name": name, "arguments": arguments})
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
# 三、ReviewAgent
# ============================================================

class ReviewAgent:
    """AI 代码审查 Agent。

    默认使用 Plan 模式 (系统性分析流程):
      Step 1: 风格检查 (check_style)
      Step 2: Bug 模式检测 (detect_patterns)
      Step 3: 深度分析 (LLM 逻辑/安全审查)
      Step 4: 综合报告 (汇总 + 评分)

    类比 Java:
      就像同时跑了 Checkstyle + SpotBugs + 一个资深同事 review。
    """

    REVIEW_SYSTEM = """你是一个资深代码审查专家，精通 Java 和 Python。

审查标准:
  1. 逻辑正确性 — 边界条件、null 安全、异常处理
  2. 安全性 — SQL 注入、XSS、敏感信息泄露、输入校验
  3. 性能 — 不必要的对象创建、算法复杂度、资源释放
  4. 可读性 — 命名、注释、函数长度、代码结构
  5. 最佳实践 — 设计模式、SOLID 原则、惯用法

输出要求:
  - 每个发现标注严重级别: critical / warning / info
  - 标注具体行号
  - 给出具体的修复建议和示例代码
  - 用中文回答

审查态度: 建设性而非批判性。指出问题时给出 "为什么不好" + "怎么改更好" + "改进后的示例"。"""

    def __init__(self, mcp: MCPClient, llm: LlmClient):
        self.mcp = mcp
        self.llm = llm

    def review(self, code: str, language: str = "java",
               focus: list[str] | None = None) -> dict:
        """审查入口。

        Args:
            code: 源代码文本
            language: java | python
            focus: 关注领域列表, 如 ["security", "performance"]

        Returns:
            结构化审查报告 dict
        """
        if focus is None:
            focus = ["security", "logic", "performance",
                     "readability", "best_practice"]

        return self._plan_review(code, language, focus)

    def _plan_review(self, code: str, language: str,
                     focus: list[str]) -> dict:
        """Plan 模式 — 按步骤系统性审查。"""
        tools = self.mcp.list_tools()
        tool_names = [t["name"] for t in tools]

        if not self.llm.is_healthy:
            return {
                "summary": "[离线模式] 无法进行 LLM 审查",
                "findings": [],
                "score": 0,
            }

        # === Phase 1: 生成审查计划 ===
        plan_prompt = self._build_plan_prompt(code, language, focus, tools)
        plan_response = self.llm.create(
            messages=[{"role": "user", "content": plan_prompt}],
            max_tokens=800,
        )
        plan_text = self.llm.get_text(plan_response)
        parsed = self._parse_json(plan_text)
        if isinstance(parsed, dict):
            goal = parsed.get("goal", "代码审查")
            steps = parsed.get("steps", [])
        else:
            goal, steps = parsed

        # === Phase 2: 执行步骤 ===
        results = []
        for step in steps:
            step_id = step.get("id", len(results) + 1)
            desc = step.get("description", "")
            tool_name = step.get("tool", "none")

            if tool_name == "none" or tool_name not in tool_names:
                results.append({
                    "step": step_id, "desc": desc,
                    "status": "done", "result": "(推理)",
                })
                continue

            # 对需要代码的调用，注入 code 和 language
            params = step.get("params", {})
            if tool_name in ("check_style", "detect_patterns"):
                params.setdefault("code", code)
                params.setdefault("language", language)

            tool_result = self.mcp.call_tool(tool_name, params)
            status = "done" if "error" not in tool_result else "failed"
            results.append({
                "step": step_id, "desc": desc, "status": status,
                "tool": tool_name, "result": tool_result,
            })

        # === Phase 3: 综合报告 ===
        synth_prompt = self._build_synth_prompt(
            code, language, focus, results
        )
        synth_response = self.llm.create(
            messages=[{"role": "user", "content": synth_prompt}],
            max_tokens=2000,
        )
        report_text = self.llm.get_text(synth_response)

        # 尝试解析结构化报告
        report = self._parse_json(report_text)
        if isinstance(report, dict) and "findings" in report:
            return report

        # 解析失败，返回原始文本
        return {
            "summary": report_text[:500],
            "findings": [],
            "steps": results,
            "language": language,
        }

    def _build_plan_prompt(self, code: str, language: str,
                           focus: list[str], tools: list[dict]) -> str:
        tool_desc = "\n".join(
            f"- {t['name']}: {t.get('description', '')}" for t in tools
        )
        focus_str = ", ".join(focus)
        # 截断过长代码
        code_preview = code[:3000]
        code_note = (
            f"\n(代码共 {len(code)} 字符, 显示前 3000)"
            if len(code) > 3000 else ""
        )

        return f"""为以下 {language} 代码制定审查计划。

关注领域: {focus_str}

可用工具:
{tool_desc}

代码:
```{language}
{code_preview}
```{code_note}

制定审查计划, 输出 JSON:
{{
  "goal": "审查目标",
  "steps": [
    {{
      "id": 1,
      "description": "步骤描述",
      "tool": "工具名 (check_style / detect_patterns / none)",
      "params": {{}}
    }}
  ]
}}

建议步骤:
  1. 用 check_style 检查代码风格
  2. 用 detect_patterns 检测常见 Bug 模式
  3. 深度分析: 逻辑、安全、性能 (tool: none, 纯推理)
  4. 综合以上发现, 生成审查报告

只输出 JSON。"""

    def _build_synth_prompt(self, code: str, language: str,
                            focus: list[str],
                            results: list[dict]) -> str:
        log = "\n".join(
            f"[Step {r['step']}] {r['desc']} | {r['status']} | "
            f"{json.dumps(r.get('result', ''), ensure_ascii=False)[:200]}"
            for r in results
        )
        focus_str = ", ".join(focus)
        code_preview = code[:2000]

        return f"""综合以下审查结果, 生成最终审查报告。

语言: {language}
关注领域: {focus_str}
代码:
```{language}
{code_preview}
```

审查执行日志:
{log}

输出 JSON 格式的审查报告:
{{
  "summary": "整体评价 (2-3 句话)",
  "findings": [
    {{
      "severity": "critical|warning|info",
      "category": "security|logic|performance|readability|best_practice",
      "line": 行号 (整数),
      "title": "简短标题",
      "description": "详细说明",
      "suggestion": "修复建议",
      "code_example": "改进后的代码示例 (可选)"
    }}
  ],
  "score": 0-10 的评分
}}

只输出 JSON, 不要其他文字。"""

    def _parse_json(self, text: str) -> tuple[str, list[dict]] | dict:
        """从 LLM 输出中提取 JSON。"""
        try:
            json_str = text
            if "```" in text:
                parts = text.split("```")
                json_str = parts[1] if len(parts) > 1 else parts[0]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            return json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError):
            # 返回原始文本和目标
            if "steps" not in text.lower():
                return {
                    "goal": "代码审查",
                    "steps": [
                        {"id": 1, "description": "风格检查",
                         "tool": "check_style", "params": {}},
                        {"id": 2, "description": "Bug 模式检测",
                         "tool": "detect_patterns", "params": {}},
                        {"id": 3, "description": "深度分析",
                         "tool": "none", "params": {}},
                        {"id": 4, "description": "综合报告",
                         "tool": "none", "params": {}},
                    ],
                }
            return {"summary": text[:500], "findings": []}


# ============================================================
# 四、工厂函数
# ============================================================

def create_mcp_server(project_root: str = ".") -> MCPServer:
    """创建注册好代码审查工具的 MCPServer。"""
    server = MCPServer("code_review")

    read_bound = partial(tool_read_file, project_root=project_root)

    server.register(
        "check_style", "规则引擎 — 检查命名规范、行长度、方法长度等",
        {
            "code": {"type": "string", "description": "源代码"},
            "language": {"type": "string", "description": "java 或 python"},
        },
        ["code"], tool_check_style,
    ).register(
        "detect_patterns",
        "模式匹配 — 检测常见 Bug 模式 (空catch/SQL注入/可变默认参数等)",
        {
            "code": {"type": "string", "description": "源代码"},
            "language": {"type": "string", "description": "java 或 python"},
        },
        ["code"], tool_detect_patterns,
    ).register(
        "read_file", "读取文件内容 (限制: 项目目录内)",
        {
            "path": {"type": "string", "description": "文件路径"},
            "max_lines": {"type": "integer", "description": "最大行数, 默认 200"},
        },
        ["path"], read_bound,
    )

    return server


def create_review_agent(config=None) -> ReviewAgent:
    """创建完整配置的 ReviewAgent。"""
    if config is None:
        from code_review.config import AppConfig
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

    return ReviewAgent(client, llm)
