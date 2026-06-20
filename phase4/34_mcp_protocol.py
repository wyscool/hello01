# ============================================================
# Phase 4, Lesson 34: MCP 协议 —— AI 与工具的标准化接口
# ============================================================
#
# 本课目标:
#   1. 理解 MCP 是什么 — AI 生态的 "USB 协议"
#   2. 掌握 MCP 架构 — Client / Server / Transport
#   3. 理解三种原语 — Tools / Resources / Prompts
#   4. 实现最小 MCP Server + Client (无需第三方 SDK)
#   5. MCPAgent — 通过 MCP 协议动态发现和调用工具
#   6. MCP vs 直接 Tool Use — 协议层的价值
#
# 预计阅读 + 实操时间: 55-65 分钟
#
# 前置: Lesson 31-33 (Agent + Tool Use)
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
llm = Anthropic(**client_kwargs)

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
    print("✅ API 连接正常\n")
except Exception:
    api_ok = False
    print("⚠️  API 不可用, 将以模拟模式运行\n")


# ============================================================
# 一、为什么需要 MCP?
# ============================================================
# L31-L33 中, 工具是直接硬编码在 Agent 代码里的:
#
#   TOOL_DEFS = [CALC_TOOL, TIME_TOOL, ...]    ← 写死在 Agent 里
#   TOOL_FNS  = {"calculate": fn, ...}          ← 也在 Agent 里
#
# 问题:
#   - 工具和 Agent 紧耦合 → 换工具要改 Agent 代码
#   - 每个 AI 框架定义工具的方式不同 → 工具不可复用
#   - 没有标准传输层 → Agent 只能调用本进程的工具
#
# MCP (Model Context Protocol) 解决的就是这个问题:
#   定义一个标准协议, 让任何 AI 客户端都能发现和调用
#   任何 MCP 服务端的工具。
#
# 类比:
#   L31-L33 直接 Tool Use ≈ 直接 JDBC 连数据库
#   MCP                   ≈ 统一的 REST API 层
#
#   USB 统一了外设接口, MCP 统一了 AI 工具接口。
#
# MCP 由 Anthropic 于 2024 年底发布,
# 目前已被 OpenAI、Google、Microsoft 等支持。

print("=" * 60)
print("MCP: AI 工具接口的标准化")
print("=" * 60)
print("""
  没有 MCP:
    ┌──────────┐     各自定义接口     ┌──────────────┐
    │ Agent A  │ ──→ (自定义格式) ──→ │ Tool: 计算器  │
    │ Agent B  │ ──→ (另一种格式) ──→ │ Tool: 搜索    │
    └──────────┘                      └──────────────┘

  有 MCP:
    ┌──────────┐                      ┌──────────────┐
    │ Agent A  │ ──┐              ┌── │ Tool: 计算器  │
    └──────────┘   │  MCP 协议    │   └──────────────┘
                   ├──────────────┤
    ┌──────────┐   │              │   ┌──────────────┐
    │ Agent B  │ ──┘              └── │ Tool: 搜索    │
    └──────────┘                      └──────────────┘
""")


# ============================================================
# 二、MCP 架构核心
# ============================================================
# MCP 采用 Client-Server 架构:
#
#   ┌───────────┐     MCP Protocol      ┌───────────────┐
#   │  Client   │ ←──────────────────→  │    Server     │
#   │ (AI App)  │   JSON-RPC over       │  (Tool Host)  │
#   │           │   stdio / HTTP / SSE  │               │
#   └───────────┘                       └───────────────┘
#
# 三种核心原语 (Primitives):
#
#   1. Tools    — 可执行的函数 (类比: REST API 的 POST)
#     - tools/list    → 获取可用工具列表
#     - tools/call    → 调用工具
#
#   2. Resources — 可读取的数据 (类比: REST API 的 GET)
#     - resources/list   → 获取资源列表
#     - resources/read   → 读取资源内容
#
#   3. Prompts  — 预定义的 prompt 模板
#     - prompts/list  → 获取 prompt 模板列表
#     - prompts/get   → 获取具体 prompt
#
# 本课聚焦 Tools (最常用), Resources 和 Prompts 作为概念了解。

print("\n" + "=" * 60)
print("MCP 架构")
print("=" * 60)
print("""
  Host (你的应用)
   ├── MCP Client ────── stdio/HTTP ────── MCP Server (工具提供方)
   │                                          ├── Tool: "calculator"
   │                                          ├── Tool: "file_reader"
   │                                          └── Resource: "file://docs/"
   │
   └── LLM (Anthropic API)
        └── 拿到 MCP tools → 调用 → 返回结果

  类比 Java:
    MCP Server  ≈ Microservice (通过标准 API 暴露能力)
    MCP Client  ≈ Service Consumer (FeignClient / RestTemplate)
    MCP Protocol ≈ OpenAPI Spec (定义接口契约)
""")


# ============================================================
# 三、MCP 消息格式 —— 简化的 JSON-RPC 2.0
# ============================================================
# MCP 基于 JSON-RPC 2.0, 核心消息类型:

@dataclass
class MCPRequest:
    """MCP 请求。

    类比: HTTP Request
      method ≈ HTTP method + path
      params ≈ request body
      id     ≈ request ID (用于匹配响应)
    """
    method: str
    params: dict = field(default_factory=dict)
    id: int = 1

    def to_json(self) -> str:
        return json.dumps({
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.id,
        }, ensure_ascii=False)


@dataclass
class MCPResponse:
    """MCP 响应。

    类比: HTTP Response
      result ≈ response body (200)
      error  ≈ error body (4xx/5xx)
    """
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


# 演示: 消息格式
print("\n" + "=" * 60)
print("MCP 消息格式")
print("=" * 60)

req = MCPRequest(method="tools/list", id=1)
print(f"  请求: {req.to_json()}")

resp = MCPResponse(result={"tools": [{"name": "calculate"}]}, id=1)
print(f"  响应: {resp.to_json()}")

err_resp = MCPResponse(error="Tool not found: bad_tool", id=2)
print(f"  错误: {err_resp.to_json()}")


# ============================================================
# 四、MCP Server —— 工具提供方
# ============================================================
# MCP Server 负责:
#   1. 注册工具 (name + description + schema + handler)
#   2. 处理 tools/list 请求 → 返回工具列表
#   3. 处理 tools/call 请求 → 执行工具并返回结果

class MCPServer:
    """最小 MCP Server 实现。

    支持:
      - tools/list  — 列出所有已注册工具
      - tools/call  — 调用指定工具

    类比 Java:
      MCPServer ≈ @RestController
        @GetMapping("/tools") → tools/list
        @PostMapping("/tools/{name}") → tools/call
    """

    def __init__(self, name: str = "mcp-server"):
        self.name = name
        self._tools: dict[str, dict] = {}      # tool_name → {schema, handler}
        self._resources: dict[str, dict] = {}   # resource_uri → {content, mime_type}

    # --- 工具注册 ---

    def register_tool(
        self, name: str, description: str,
        parameters: dict, required: list[str],
        handler: Callable,
    ) -> "MCPServer":
        """注册工具。支持链式调用。"""
        self._tools[name] = {
            "schema": {
                "name": name,
                "description": description,
                "inputSchema": {
                    "type": "object",
                    "properties": parameters,
                    "required": required,
                }
            },
            "handler": handler,
        }
        return self

    # --- 资源注册 ---

    def register_resource(self, uri: str, content: str, mime_type: str = "text/plain"):
        """注册资源 (只读数据)。"""
        self._resources[uri] = {"content": content, "mime_type": mime_type}
        return self

    # --- 消息处理 (核心) ---

    def handle(self, raw_request: str) -> str:
        """处理一个 MCP 请求, 返回 JSON 响应。

        这是 MCP 的 "路由" 层: 根据 method 分发到具体处理逻辑。
        """
        try:
            data = json.loads(raw_request)
            method = data.get("method", "")
            params = data.get("params", {})
            req_id = data.get("id", 0)
        except json.JSONDecodeError:
            return MCPResponse(error="Invalid JSON", id=0).to_json()

        match method:
            case "tools/list":
                return self._handle_list(req_id)
            case "tools/call":
                return self._handle_call(params, req_id)
            case "resources/list":
                return self._handle_resources_list(req_id)
            case "resources/read":
                return self._handle_resource_read(params, req_id)
            case "initialize":
                return MCPResponse(result={
                    "protocolVersion": "0.2",
                    "serverInfo": {"name": self.name, "version": "1.0"},
                    "capabilities": {"tools": {}, "resources": {}},
                }, id=req_id).to_json()
            case _:
                return MCPResponse(error=f"未知方法: {method}", id=req_id).to_json()

    def _handle_list(self, req_id: int) -> str:
        tools = [t["schema"] for t in self._tools.values()]
        return MCPResponse(result={"tools": tools}, id=req_id).to_json()

    def _handle_call(self, params: dict, req_id: int) -> str:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        tool = self._tools.get(tool_name)
        if not tool:
            return MCPResponse(error=f"未知工具: {tool_name}", id=req_id).to_json()

        try:
            result = tool["handler"](**arguments)
            # MCP 要求返回 content 列表
            return MCPResponse(result={
                "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
            }, id=req_id).to_json()
        except Exception as e:
            return MCPResponse(error=f"工具执行失败: {e}", id=req_id).to_json()

    def _handle_resources_list(self, req_id: int) -> str:
        resources = [
            {"uri": uri, "mimeType": r["mime_type"]}
            for uri, r in self._resources.items()
        ]
        return MCPResponse(result={"resources": resources}, id=req_id).to_json()

    def _handle_resource_read(self, params: dict, req_id: int) -> str:
        uri = params.get("uri", "")
        resource = self._resources.get(uri)
        if not resource:
            return MCPResponse(error=f"资源不存在: {uri}", id=req_id).to_json()
        return MCPResponse(result={
            "contents": [{"uri": uri, "mimeType": resource["mime_type"],
                          "text": resource["content"]}]
        }, id=req_id).to_json()


# ============================================================
# 五、MCP Client —— 协议消费方
# ============================================================
# MCP Client 封装了和 Server 的通信。
# 在本课中, Client 直接持有 Server 引用 (进程内通信),
# 在真实场景中则是通过 stdio / HTTP / SSE。

class MCPClient:
    """MCP 客户端。

    封装了:
      1. 连接 (本课简化: 直接引用 Server 对象)
      2. 工具发现 (list_tools)
      3. 工具调用 (call_tool)
      4. 资源读取 (read_resource)

    类比 Java:
      MCPClient ≈ FeignClient
        封装了对远程服务的调用, 暴露本地方法接口。
    """

    def __init__(self, server: MCPServer | None = None):
        self.server = server
        self._request_id = 0

    def connect(self, server: MCPServer):
        """连接 MCP Server。"""
        self.server = server
        # 发送 initialize 握手
        resp = self._send("initialize")
        if resp.error:
            raise RuntimeError(f"MCP 握手失败: {resp.error}")
        info = resp.result
        print(f"  已连接: {info.get('serverInfo', {}).get('name', 'unknown')} "
              f"(协议 v{info.get('protocolVersion', '?')})")

    def _send(self, method: str, params: dict | None = None) -> MCPResponse:
        """发送请求并获取响应。"""
        if self.server is None:
            return MCPResponse(error="未连接 MCP Server")
        self._request_id += 1
        req = MCPRequest(method=method, params=params or {}, id=self._request_id)
        raw_resp = self.server.handle(req.to_json())
        return MCPResponse.from_json(raw_resp)

    def list_tools(self) -> list[dict]:
        """获取可用工具列表。返回 Anthropic tool schema 格式。

        MCP 返回 inputSchema (camelCase), Anthropic API 需要 input_schema。
        这里做格式转换。
        """
        resp = self._send("tools/list")
        if resp.error:
            print(f"  list_tools 失败: {resp.error}")
            return []
        tools = resp.result.get("tools", [])
        # MCP 格式 → Anthropic 格式
        converted = []
        for t in tools:
            converted.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("inputSchema", {"type": "object", "properties": {}, "required": []}),
            })
        return converted

    def call_tool(self, name: str, arguments: dict) -> dict:
        """调用工具。返回解析后的结果 dict。"""
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        if resp.error:
            return {"error": resp.error}
        # 解析 MCP content 格式
        content = resp.result.get("content", [])
        for item in content:
            if item.get("type") == "text":
                try:
                    return json.loads(item["text"])
                except json.JSONDecodeError:
                    return {"text": item["text"]}
        return {"raw": content}

    def list_resources(self) -> list[dict]:
        resp = self._send("resources/list")
        if resp.error:
            return []
        return resp.result.get("resources", [])

    def read_resource(self, uri: str) -> str:
        resp = self._send("resources/read", {"uri": uri})
        if resp.error:
            return f"[错误: {resp.error}]"
        contents = resp.result.get("contents", [])
        if contents:
            return contents[0].get("text", "")
        return ""


# ============================================================
# 六、构建 MCP Server —— 注册工具和资源
# ============================================================

print("\n" + "=" * 60)
print("构建 MCP Server")
print("=" * 60)

# 工具: 计算器
def calc_tool(expression: str) -> dict:
    safe = {"abs": abs, "round": round, "sqrt": math.sqrt,
            "pi": math.pi, "sin": math.sin, "cos": math.cos}
    result = eval(expression, {"__builtins__": {}}, safe)
    return {"expression": expression, "result": result}


# 工具: 当前时间
def time_tool(timezone: str = "Asia/Shanghai") -> dict:
    now = datetime.now()
    return {"datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "weekday": ["一","二","三","四","五","六","日"][now.weekday()]}


# 工具: 文本统计
def stats_tool(text: str) -> dict:
    return {"chars": len(text), "words": len(text.split()),
            "lines": len(text.split("\n"))}


# 创建 Server + 注册
server = MCPServer(name="demo-server")

server.register_tool(
    "calculate", "执行数学表达式计算",
    {"expression": {"type": "string", "description": "数学表达式"}},
    ["expression"], calc_tool,
).register_tool(
    "get_current_time", "获取当前日期时间",
    {"timezone": {"type": "string", "description": "时区"}},
    [], time_tool,
).register_tool(
    "text_stats", "统计文本的字数、词数、行数",
    {"text": {"type": "string", "description": "要统计的文本"}},
    ["text"], stats_tool,
).register_resource(
    "info://about",
    "这是一个演示 MCP Server, 展示了 Tools 和 Resources 两种原语。",
).register_resource(
    "config://limits",
    json.dumps({"max_tokens": 4096, "timeout": 30}, ensure_ascii=False),
    "application/json",
)

print(f"  Server: {server.name}")
print(f"  工具: {list(server._tools.keys())}")
print(f"  资源: {list(server._resources.keys())}")


# ============================================================
# 七、MCP Client —— 连接与调用
# ============================================================

print("\n" + "=" * 60)
print("MCP Client 连接与调用")
print("=" * 60)

client = MCPClient()
client.connect(server)

# 列出工具
tools = client.list_tools()
print(f"\n  可用工具 ({len(tools)}):")
for t in tools:
    print(f"    - {t['name']}: {t['description'][:40]}...")

# 直接调用工具
print(f"\n  直接调用:")
result = client.call_tool("calculate", {"expression": "sqrt(144) + 5"})
print(f"    calculate(sqrt(144)+5) → {result}")

result = client.call_tool("get_current_time", {})
print(f"    get_current_time() → {result}")

# 读取资源
print(f"\n  读取资源:")
content = client.read_resource("info://about")
print(f"    info://about → {content}")

content = client.read_resource("config://limits")
print(f"    config://limits → {content}")


# ============================================================
# 八、MCPAgent —— 通过 MCP 协议使用工具
# ============================================================

class MCPAgent:
    """MCP Agent — 通过 MCP Client 动态获取工具。

    和 L31-L33 Agent 的关键区别:
      工具不是硬编码的, 而是通过 client.list_tools() 动态获取。
      Agent 启动时可能不知道有哪些工具可用 ——
      一切由 MCP Server 在握手时告知。

    类比 Java:
      MCPAgent ≈ Service 通过依赖注入获取 Repository
        而非 new Repository() 硬编码。
    """

    def __init__(self, mcp_client: MCPClient, model: str = MODEL,
                 max_iterations: int = 10, verbose: bool = True):
        self.client = mcp_client
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose

    def run(self, task: str) -> str:
        # 动态获取工具列表 (和 L31-L33 的最大区别!)
        tools = self.client.list_tools()

        if self.verbose:
            print(f"\n  从 MCP Server 获取了 {len(tools)} 个工具:")
            for t in tools:
                print(f"    - {t['name']}")

        if not tools:
            return "[MCP Server 没有提供任何工具]"

        system_prompt = self._build_system(tools)
        messages: list[dict] = [{"role": "user", "content": task}]

        for i in range(self.max_iterations):
            if self.verbose:
                print(f"\n  [迭代 {i + 1}] ", end="")

            if not api_ok:
                return f"[模拟] MCP Agent 收到: {task}, 可用工具: {[t['name'] for t in tools]}"

            response = llm.messages.create(
                model=self.model, max_tokens=1024, temperature=0.0,
                system=system_prompt, messages=messages, tools=tools,
            )

            tool_uses = _get_tool_uses(response)
            text_parts = _get_text(response)

            if not tool_uses:
                if self.verbose:
                    print("✅ 完成")
                return text_parts

            # 追加 assistant 消息
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use", "id": block.id,
                        "name": block.name, "input": block.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            # 通过 MCP Client 执行工具 (而非直接调用 handler!)
            tool_content = []
            for tu in tool_uses:
                params = tu.input or {}
                result = self.client.call_tool(tu.name, params)

                if self.verbose:
                    status = "✗" if "error" in result else "✓"
                    result_str = json.dumps(result, ensure_ascii=False)[:80]
                    print(f"\n  {status} {tu.name}({json.dumps(params, ensure_ascii=False)[:40]}) → {result_str}")

                tool_content.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            messages.append({"role": "user", "content": tool_content})

        return "[Agent 在最大迭代次数内未能完成任务]"

    def _build_system(self, tools: list[dict]) -> str:
        tool_desc = "\n".join(
            f"- {t['name']}: {t.get('description', '')}" for t in tools
        )
        return f"""你是一个 MCP Agent, 通过 MCP 协议调用工具。

可用工具 (由 MCP Server 提供):
{tool_desc}

规则:
1. 根据任务主动选择合适的工具
2. 区分工具调用失败和数学上无意义
3. 用中文回答"""


# ============================================================
# 九、演示: MCPAgent 实战
# ============================================================

print("\n\n" + "=" * 60)
print("演示 1: MCP Agent 基础调用")
print("=" * 60)

agent1 = MCPAgent(client, max_iterations=5)
task1 = "帮我计算 sqrt(225) + 100 的值, 然后告诉我现在是什么时间。"
print(f"  任务: {task1}")
answer1 = agent1.run(task1)
print(f"\n  最终答案:\n{answer1[:400]}...")


print("\n\n" + "=" * 60)
print("演示 2: 多工具综合任务")
print("=" * 60)

agent2 = MCPAgent(client, max_iterations=6)
task2 = """分析下面这段文字, 统计字数, 然后告诉我现在是几点:

'人工智能正在深刻改变软件开发的方式。从代码补全到自动化测试,
AI 工具让开发者的效率大幅提升。MCP 协议则让 AI 能够安全、
标准化地接入各种工具和数据源。'"""

print(f"  任务: {task2[:60]}...")
answer2 = agent2.run(task2)
print(f"\n  最终答案:\n{answer2[:400]}...")


# ============================================================
# 十、MCP vs 直接 Tool Use —— 对比总结
# ============================================================

print("\n\n" + "=" * 60)
print("MCP vs 直接 Tool Use")
print("=" * 60)
print("""
  维度              直接 Tool Use (L31-33)     MCP (本课)
  ──────────────────────────────────────────────────────────
  工具定义位置         Agent 代码中硬编码           MCP Server 中注册
  工具发现             Agent 启动前已知             client.list_tools() 动态获取
  工具调用             handler(**params)            client.call_tool(name, params)
  传输层               进程内直接调用               stdio / HTTP / SSE
  跨语言支持           不支持                      任何语言都能实现 MCP Server
  工具复用             Agent 间无法共享             一个 Server 服务多个 Agent
  热更新               需要重启 Agent               Server 更新后 Agent 自动感知

  什么时候用 MCP?
    ✅ 工具由不同团队维护
    ✅ 工具用不同语言实现
    ✅ 需要工具的热更新
    ✅ 一个工具服务多个 AI 应用

  什么时候直接 Tool Use?
    ✅ 原型阶段 / 单文件脚本
    ✅ 工具逻辑简单且固定
    ✅ 不想引入协议层的复杂度

  类比 Java:
    直接 Tool Use ≈ new 一个对象调方法 (简单场景)
    MCP           ≈ 通过 REST API 调微服务 (分布式场景)
""")


# ============================================================
# 十一、真实 MCP 生态
# ============================================================

print("=" * 60)
print("真实 MCP 生态")
print("=" * 60)
print("""
  本课实现的是 MCP 的"教学版" (最小可用实现)。
  真实 MCP SDK (pip install mcp) 提供:

  1. 多种 Transport:
     - stdio (标准输入输出, 用于本地工具)
     - SSE (Server-Sent Events, 用于远程工具)
     - Streamable HTTP (新, 替代 SSE)

  2. 完整生命周期:
     - initialize → initialized → notifications → shutdown

  3. 更多原语:
     - Sampling (Server 请求 Client 调用 LLM)
     - Notifications (事件推送)
     - Logging (日志流)

  4. 官方 Server 生态:
     - @anthropic/mcp-server-filesystem  (文件系统)
     - @anthropic/mcp-server-github     (GitHub API)
     - @anthropic/mcp-server-postgres   (PostgreSQL)
     - 社区贡献了数百个 MCP Server...

  类比:
    本课实现   ≈ 手写 HTTP Server (学习原理)
    mcp 官方 SDK ≈ Spring Boot (生产使用)
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Lesson 34 完成! MCP 协议已掌握。")
    print("=" * 60)
    print(f"""
  回顾: 你学会了什么?

  1. MCP 协议概念    — AI 工具接口的标准化, "AI 的 USB 协议"
  2. MCP 架构         — Client ↔ JSON-RPC ↔ Server
  3. 三种原语         — Tools (执行) / Resources (读取) / Prompts (模板)
  4. MCPRequest       — method + params + id
  5. MCPResponse      — result / error
  6. MCPServer        — register_tool / handle / list / call
  7. MCPClient        — connect / list_tools / call_tool
  8. MCPAgent         — 通过 MCP 动态发现和调用工具
  9. MCP vs 直接调用  — 协议层的价值: 解耦、标准化、跨语言

  Agent 范式演进:
    L31 ReActAgent     — 直接 dict 定义工具
    L32 AdvancedAgent  — + ToolRegistry (统一管理)
    L33 PlannerAgent   — + Plan + Reflect
    L34 MCPAgent       — + MCP 协议 (标准化接口)

  🎯 下一课: Lesson 35 — 端到端 Agent 项目
     Phase 4 收官之作! 融合所有 Agent 技能,
     构建一个能规划、调用 MCP 工具、自我修正的完整智能体。
""")


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 给 MCP Server 添加一个新工具:
#    实现一个 "随机数" 工具:
#    - 参数: min(int), max(int), count(int)
#    - 返回: {"numbers": [随机数列表], "sum": 求和}
#    注册到 server, 用 MCPAgent 测试: "生成 5 个 1-100 的随机数, 求和"
#
# 2. 模拟 MCP Server 的热更新:
#    在 Agent 运行过程中, 给 server 注册一个新工具,
#    Agent 能否在下一次 run() 中感知到?
#    - 提示: run() 每次都会调用 client.list_tools()
#    - 这就是 MCP 的 "动态发现" 优势
#
# 3. 对比 MCPAgent 和 L31 ReActAgent:
#    用同一个任务, 同一个工具集, 对比:
#    - MCPAgent.run(task)  vs  ReActAgent.run(task)
#    - 除了工具获取方式不同, 还有什么差异?
#    - 如果换一个 MCP Server (不同工具), MCPAgent 需要改代码吗?
#
# 4. 实现简单的 Resources 应用:
#    注册一个 "数据" 资源: resource://data
#    让 Agent 先读取资源, 再基于资源内容回答问题。
#    - 提示: 在 system_prompt 中告诉 Agent 可以 read_resource
#    - 需要添加 read_resource 作为工具暴露给 LLM
#
# 5. (挑战) 实现多 MCP Server:
#    创建两个 MCP Server:
#    - server_math: 数学工具
#    - server_text: 文本工具
#    让 MCPClient 连接两个 Server, Agent 同时使用两边的工具。
#    - 提示: client 维护 server 列表, list_tools 时合并
#    这就是 "MCP Gateway" 的雏形!
#
# 6. (思考) MCP 的安全模型:
#    如果 MCP Server 提供了 "删除文件" 工具,
#    你怎么防止 Agent 误删重要文件?
#    - 设计一个权限系统: 工具分级 (READ / WRITE / ADMIN)
#    - 敏感工具需要用户确认
#    - 限制工具可访问的路径范围
#    这和生产 MCP 部署直接相关。
#
# 做完后告诉我:
#   - MCP 协议相比直接 Tool Use, 你觉得最大的价值是什么?
#   - 如果你要给你的团队引入 MCP, 第一个 Server 会提供什么工具?
# 我们继续 Lesson 35: 端到端 Agent 项目 — Phase 4 大结局!
# ============================================================


# ╔══════════════════════════════════════════════════════════════╗
# ║              试试看 — 练习实现代码                            ║
# ╚══════════════════════════════════════════════════════════════╝

import random
import threading
from enum import Enum

print("\n")
print("=" * 60)
print("试试看练习: Lesson 34")
print("=" * 60)


# ─── 练习 1: 给 MCP Server 添加随机数工具 ─────────────────────

print("\n" + "─" * 40)
print("练习 1: 添加随机数工具到 MCP Server")
print("─" * 40)


def random_numbers_handler(min_val: int = 1, max_val: int = 100,
                           count: int = 3) -> dict:
    """生成指定数量的随机整数, 并求和。

    MCP 工具的限制:
      - 参数都是 JSON 类型 (int, str, bool, ...)
      - 返回值转为 JSON string (通过 json.dumps)
      - 错误通过 MCP error 返回, 不抛异常到协议层
    """
    if min_val > max_val:
        raise ValueError(f"min({min_val}) 不能大于 max({max_val})")
    if count <= 0:
        raise ValueError(f"count 必须 > 0, 收到: {count}")
    if count > 100:
        raise ValueError(f"count 不能超过 100, 收到: {count}")

    numbers = [random.randint(min_val, max_val) for _ in range(count)]
    return {
        "numbers": numbers,
        "sum": sum(numbers),
        "avg": round(sum(numbers) / len(numbers), 2),
        "min": min(numbers),
        "max": max(numbers),
        "count": count,
        "range": [min_val, max_val],
    }


# 注册到现有 server
server.register_tool(
    "random_numbers",
    "生成指定数量的随机整数, 返回数字列表和统计信息 (sum, avg, min, max)。",
    {
        "min_val": {"type": "integer", "description": "最小值 (含), 默认 1"},
        "max_val": {"type": "integer", "description": "最大值 (含), 默认 100"},
        "count": {"type": "integer", "description": "数量, 默认 3, 最大 100"},
    },
    [], random_numbers_handler,
)

print(f"  Server 工具: {list(server._tools.keys())}")

# 直接测试 MCP 调用
raw_req = json.dumps({
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {"name": "random_numbers",
               "arguments": {"min_val": 1, "max_val": 100, "count": 5}},
    "id": 999,
})
raw_resp = server.handle(raw_req)
resp_obj = json.loads(raw_resp)
result_text = resp_obj["result"]["content"][0]["text"]
result_data = json.loads(result_text)
print(f"  直接调用 random_numbers(count=5):")
print(f"    数字: {result_data['numbers']}")
print(f"    和: {result_data['sum']}, 平均: {result_data['avg']}")

# 通过 MCP Client 调用
result = client.call_tool("random_numbers", {"min_val": 1, "max_val": 100, "count": 3})
print(f"  Client 调用: {result}")

# Agent 测试
print(f"\n  Agent 测试:")
if api_ok:
    agent_rand = MCPAgent(client, max_iterations=5, verbose=False)
    rand_task = "帮我生成 5 个 1-100 的随机数, 然后求和。"
    rand_answer = agent_rand.run(rand_task)
    print(f"  Agent: {rand_answer[:250]}...")
else:
    # 模拟
    nums = [random.randint(1, 100) for _ in range(5)]
    print(f"  [模拟] 生成 {nums}, 和={sum(nums)}")

print(f"\n  ✅ 练习 1 完成: random_numbers 已注册到 MCP Server")


# ─── 练习 2: MCP Server 热更新模拟 ────────────────────────────

print("\n" + "─" * 40)
print("练习 2: MCP Server 热更新 (动态工具发现)")
print("─" * 40)

# 工具变更前后的对比
tools_before = client.list_tools()
print(f"  当前工具数: {len(tools_before)}")
print(f"  工具名: {[t['name'] for t in tools_before]}")

# 热更新: 注册新工具
server.register_tool(
    "uuid_generator",
    "生成一个随机的 UUID4 字符串。",
    {},
    [],
    lambda: {"uuid": str(__import__("uuid").uuid4())},
)
print(f"\n  [热更新] 注册了新工具: uuid_generator")
print(f"  Server 无需重启, Agent 无需重新初始化!")

# Agent 下次 run() 自动感知
tools_after = client.list_tools()
print(f"  热更新后工具数: {len(tools_after)}")
print(f"  工具名: {[t['name'] for t in tools_after]}")

print(f"\n  热更新关键点:")
print(f"    1. client.list_tools() 每次调用都实时查询 server")
print(f"    2. Agent.run() 开始时调用 list_tools(), 自动获取最新工具")
print(f"    3. 不需要重启 Agent, 不需要修改 Agent 代码")
print(f"    4. 这就是 MCP 相比硬编码的最大优势之一")

# 验证: 新工具可用
uuid_result = client.call_tool("uuid_generator", {})
print(f"\n  新工具调用: {uuid_result}")

print(f"\n  ✅ 练习 2 完成: 验证了 MCP 的动态工具发现能力")


# ─── 练习 3: MCPAgent vs ReActAgent 对比 ──────────────────────

print("\n" + "─" * 40)
print("练习 3: MCPAgent vs ReActAgent 深入对比")
print("─" * 40)

print(f"""  同一个任务: "计算 sqrt(256) + 100"

  ReActAgent (L31):
    工具来源: TOOL_DEFS + TOOL_HANDLERS (模块级变量)
    工具发现: 启动前已知 (硬编码)
    工具调用: handler(**params) (直接调用)
    切换工具: 修改 TOOL_DEFS → 重启 Agent

  MCPAgent (L34):
    工具来源: client.list_tools() → 从 MCP Server 动态获取
    工具发现: 每次 run() 实时查询
    工具调用: client.call_tool(name, params) (通过 MCP 协议)
    切换工具: 注册/注销到 server → Agent 自动感知 (无需重启!)

  关键差异不止是"获取方式不同":
    ① 解耦: Agent 不依赖具体工具实现
    ② 跨进程: MCP Server 可以跑在另一个进程/服务器
    ③ 跨语言: MCP Server 可以用任何语言实现
    ④ 共享: 一个 MCP Server 服务多个 Agent
    ⑤ 安全: 协议层可以做统一的权限/限流/审计

  类比 Java:
    ReActAgent ≈ 直接调用 local method
    MCPAgent    ≈ 通过 REST API 调用微服务
""")

print(f"  如果换一个 MCP Server:")
print(f"    MCPAgent 代码完全不变!")
print(f"    只需: new_client.connect(new_server)")
print(f"    然后: agent.run(task) → 自动使用新工具集")
print(f"    这就是 MCP 的 '接口标准' 价值")

print(f"\n  ✅ 练习 3 完成: 深入理解了 MCP vs 直接调用的差异")


# ─── 练习 4: Resources 应用 ───────────────────────────────────

print("\n" + "─" * 40)
print("练习 4: Resources 原语 — Agent 读取数据资源")
print("─" * 40)

# 注册资源
server.register_resource(
    "resource://company_faq",
    json.dumps({
        "公司名": "DeepThink AI",
        "成立": "2024",
        "核心产品": "智能编程助手 DevAssistant",
        "技术栈": ["Python", "FastAPI", "Anthropic Claude"],
        "团队规模": "15 人",
        "办公室": "北京、上海",
    }, ensure_ascii=False),
    "application/json",
)

# 将 read_resource 包装为可被 LLM 调用的 "工具"
# 这样 LLM 才能通过 Tool Use 来读取资源
READ_RESOURCE_TOOL_SCHEMA = {
    "name": "read_resource",
    "description": "读取 MCP Server 上的数据资源。可用于查询 FAQ、配置、文档等。",
    "input_schema": {
        "type": "object",
        "properties": {
            "uri": {
                "type": "string",
                "description": "资源 URI, 如 'resource://company_faq'"
            }
        },
        "required": ["uri"]
    }
}


class ResourceEnabledMCPAgent:
    """支持 Resources 的 MCPAgent。

    关键设计: 把 read_resource 包装成 LLM 可以调用的 "虚拟工具"。
    MCP 的 Resources 原语本身不是 Tool Use 的一部分,
    但我们可以把它暴露为工具给 LLM 使用。

    对比:
      Tools    — LLM 直接调用: calculate, text_stats
      Resources — LLM 通过 "read_resource 工具" 间接读取
    """

    def __init__(self, mcp_client: MCPClient, model: str = MODEL,
                 max_iterations: int = 10, verbose: bool = True):
        self.client = mcp_client
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose

    def run(self, task: str) -> str:
        tools = self.client.list_tools()
        # 把 read_resource 添加为虚拟工具
        tools = tools + [READ_RESOURCE_TOOL_SCHEMA]

        if self.verbose:
            print(f"  可用工具 + 资源读取: {len(tools)} 个")

        system_prompt = f"""你是 MCP Agent, 通过 MCP 协议调用工具和读取资源。

可用工具:
- calculate: 数学计算
- get_current_time: 获取时间
- text_stats: 文本统计
- random_numbers: 随机数
- read_resource: 读取 MCP 数据资源

可用资源 (通过 read_resource 读取):
- resource://company_faq  公司常见问题

规则:
1. 当需要查询公司信息时, 调用 read_resource(uri="resource://company_faq")
2. 综合分析工具结果和资源内容后回答
3. 用中文回答"""

        if not api_ok:
            # 模拟: 直接读取资源
            content = client.read_resource("resource://company_faq")
            return f"[模拟] 读取了资源: {content[:200]}..."

        messages: list[dict] = [{"role": "user", "content": task}]

        for i in range(self.max_iterations):
            response = llm.messages.create(
                model=self.model, max_tokens=1024, temperature=0.0,
                system=system_prompt, messages=messages, tools=tools,
            )
            tool_uses = _get_tool_uses(response)
            text_parts = _get_text(response)

            if not tool_uses:
                return text_parts

            # 构建 assistant 消息
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use", "id": block.id,
                        "name": block.name, "input": block.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            # 执行工具 + 资源读取
            tool_content = []
            for tu in tool_uses:
                if tu.name == "read_resource":
                    # 特殊处理: 走 resources/read 协议
                    uri = (tu.input or {}).get("uri", "")
                    content = self.client.read_resource(uri)
                    result_text = content
                else:
                    result = self.client.call_tool(tu.name, tu.input or {})
                    result_text = json.dumps(result, ensure_ascii=False)

                if self.verbose:
                    print(f"  {'✓' if 'error' not in str(result_text)[:50] else '✗'} "
                          f"{tu.name} → {str(result_text)[:80]}")

                tool_content.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_text,
                })
            messages.append({"role": "user", "content": tool_content})

        return "[Agent 在最大迭代次数内未能完成任务]"


# 测试 Resources
print(f"  资源列表: {[r['uri'] for r in client.list_resources()]}")

res_content = client.read_resource("resource://company_faq")
print(f"  直接读取 resource://company_faq:")
print(f"    {res_content[:200]}...")

if api_ok:
    res_agent = ResourceEnabledMCPAgent(client, max_iterations=4, verbose=False)
    res_task = "我们公司叫什么名字? 核心产品是什么? 成立几年了?"
    res_answer = res_agent.run(res_task)
    print(f"\n  Agent 基于资源回答: {res_answer[:300]}...")
else:
    print(f"\n  [模拟] Agent 读取 resource://company_faq → 回答公司信息")

print(f"\n  ✅ 练习 4 完成: Resources 原语和 Agent 集成")


# ─── 练习 5 (挑战): 多 MCP Server ─────────────────────────────

print("\n" + "─" * 40)
print("练习 5 (挑战): 多 MCP Server — MCP Gateway 雏形")
print("─" * 40)


class MultiMCPClient:
    """连接多个 MCP Server 的客户端。

    这是 MCP Gateway 的雏形:
      - 聚合多个 Server 的工具
      - 路由工具调用到正确的 Server
      - 不同的 Server 可以有不同的工具集

    类比 Java:
      MultiMCPClient ≈ API Gateway (Zuul / Spring Cloud Gateway)
        聚合多个微服务, 统一入口, 动态路由
    """

    def __init__(self):
        self._servers: list[MCPServer] = []
        self._rid = 0
        self._tool_routing: dict[str, int] = {}  # tool_name → server_index

    def add_server(self, server: MCPServer) -> "MultiMCPClient":
        """添加一个 MCP Server。"""
        self._servers.append(server)
        return self

    def _send(self, server: MCPServer, method: str, params: dict | None = None) -> MCPResponse:
        self._rid += 1
        raw = json.dumps({"jsonrpc": "2.0", "method": method,
                         "params": params or {}, "id": self._rid})
        return MCPResponse.from_json(server.handle(raw))

    def list_tools(self) -> list[dict]:
        """聚合所有 Server 的工具列表。"""
        all_tools = []
        self._tool_routing.clear()
        for idx, server in enumerate(self._servers):
            resp = self._send(server, "tools/list")
            if not resp.error:
                tools = resp.result.get("tools", [])
                for t in tools:
                    all_tools.append({
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "input_schema": t.get("inputSchema",
                                              {"type": "object", "properties": {}, "required": []}),
                    })
                    self._tool_routing[t["name"]] = idx
        return all_tools

    def call_tool(self, name: str, arguments: dict) -> dict:
        """路由工具调用到正确的 Server。"""
        server_idx = self._tool_routing.get(name)
        if server_idx is None or server_idx >= len(self._servers):
            return {"error": f"未知工具: {name}"}

        server = self._servers[server_idx]
        resp = self._send(server, "tools/call", {"name": name, "arguments": arguments})
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


# 创建两个独立的 MCP Server
server_math = MCPServer(name="math-server")
server_math.register_tool(
    "calculate", "数学表达式计算",
    {"expression": {"type": "string", "description": "数学表达式"}},
    ["expression"],
    lambda expression: {"result": eval(
        expression, {"__builtins__": {}},
        {"abs": abs, "sqrt": math.sqrt, "pi": math.pi})},
)

server_text = MCPServer(name="text-server")
server_text.register_tool(
    "text_stats", "统计文本信息",
    {"text": {"type": "string", "description": "要统计的文本"}},
    ["text"],
    lambda text: {"chars": len(text), "words": len(text.split())},
)

# 创建 Gateway Client
gw_client = MultiMCPClient()
gw_client.add_server(server_math)
gw_client.add_server(server_text)

tools_all = gw_client.list_tools()
print(f"  Gateway 聚合工具: {len(tools_all)} 个")
for t in tools_all:
    src = "math-server" if gw_client._tool_routing[t["name"]] == 0 else "text-server"
    print(f"    - {t['name']} (来自 {src})")

# 测试路由
r1 = gw_client.call_tool("calculate", {"expression": "sqrt(144)"})
print(f"\n  calculate → math-server: {r1}")

r2 = gw_client.call_tool("text_stats", {"text": "Hello MCP Gateway"})
print(f"  text_stats → text-server: {r2}")

# Agent 使用 Gateway
if api_ok:
    gw_agent = MCPAgent(
        type("GWClient", (), {  # 临时适配: 让 gw_client 模拟 MCPClient 接口
            "list_tools": gw_client.list_tools,
            "call_tool": gw_client.call_tool,
        })(),
        max_iterations=5,
        verbose=False,
    )
    gw_answer = gw_agent.run("统计 'Hello MCP Gateway' 的字数, 然后计算 sqrt(256)")
    print(f"\n  Gateway Agent: {gw_answer[:250]}...")
else:
    print(f"\n  [模拟] Gateway Agent 跨 Server 调用:")
    print(f"    Step 1: text_stats('Hello MCP Gateway') → {r2}")
    print(f"    Step 2: calculate('sqrt(256)') → {r1}")

print(f"\n  MCP Gateway 架构:")
print(f"""    ┌─────────────┐
    │   Agent     │
    └─────┬───────┘
          │
    ┌─────▼───────┐
    │ MCP Gateway │  ← 统一入口, 工具路由
    └──┬───────┬──┘
       │       │
  ┌────▼──┐ ┌──▼─────┐
  │ math  │ │ text   │  ← 独立 Server
  │ server│ │ server │
  └───────┘ └────────┘
""")

print(f"  ✅ 练习 5 完成: 实现了 MultiMCPClient / MCP Gateway 雏形")


# ─── 练习 6 (思考): MCP 安全模型 ──────────────────────────────

print("\n" + "─" * 40)
print("练习 6 (思考): MCP 安全模型设计")
print("─" * 40)


# 工具权限分级枚举 (与 L32 练习 6 呼应)
class MCPToolPermission(Enum):
    READ = "read"        # 只读工具: calculate, search, text_stats
    WRITE = "write"      # 写入工具: write_file, record_data
    DELETE = "delete"    # 删除工具: delete_file, drop_table
    EXECUTE = "execute"  # 执行工具: run_command, eval_code


@dataclass
class SecureToolDef:
    """带安全元数据的工具定义。"""
    name: str
    description: str
    parameters: dict
    required: list[str]
    handler: Callable
    permission: MCPToolPermission = MCPToolPermission.READ
    allow_auto: bool = True              # READ 工具自动执行
    require_confirmation: bool = False   # WRITE/DELETE 需要用户确认
    allowed_paths: list[str] = field(default_factory=list)   # 路径白名单
    allowed_patterns: list[str] = field(default_factory=list) # 文件名模式


class SecureMCPServer(MCPServer):
    """带安全策略的 MCP Server。

    每次 tools/call 时检查:
      1. 权限级别 → 是否允许自动执行
      2. 路径范围 → 是否在允许的目录内
      3. 操作审计 → 记录谁、何时、做了什么
    """

    def __init__(self, name: str = "secure-server",
                 allowed_root: str = "./safe_output/"):
        super().__init__(name)
        self.allowed_root = Path(allowed_root).resolve()
        self.audit_log: list[dict] = []

    def register_secure(self, tool: SecureToolDef) -> "SecureMCPServer":
        """注册安全工具。"""
        self._tools[tool.name] = {
            "schema": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "type": "object",
                    "properties": tool.parameters,
                    "required": tool.required,
                }
            },
            "handler": tool.handler,
            "permission": tool.permission,
            "allow_auto": tool.allow_auto,
            "require_confirmation": tool.require_confirmation,
            "allowed_paths": tool.allowed_paths,
        }
        return self

    def _handle_call(self, params: dict, req_id: int) -> str:
        """增强的 tools/call: 加安全检查。"""
        tool_name = params.get("name", "")
        tool_entry = self._tools.get(tool_name)

        if not tool_entry:
            self._audit(tool_name, "denied: unknown tool")
            return MCPResponse(error=f"未知工具: {tool_name}", id=req_id).to_json()

        permission = tool_entry["permission"]

        # 安全检查 1: EXECUTE 权限拒绝 (或需要 admin 令牌)
        if permission == MCPToolPermission.EXECUTE:
            self._audit(tool_name, "denied: EXECUTE permission required")
            return MCPResponse(
                error=f"安全: '{tool_name}' 需要 EXECUTE 权限, 已拒绝",
                id=req_id,
            ).to_json()

        # 安全检查 2: DELETE 权限需要用户确认
        if permission == MCPToolPermission.DELETE:
            if not params.get("_confirmed"):
                self._audit(tool_name, "pending: requires confirmation")
                return MCPResponse(
                    error=f"安全: '{tool_name}' 是删除操作, 需要 _confirmed=true",
                    id=req_id,
                ).to_json()

        # 安全检查 3: 路径白名单 (WRITE/DELETE 工具)
        if permission in (MCPToolPermission.WRITE, MCPToolPermission.DELETE):
            file_path = params.get("arguments", {}).get("path", "")
            if file_path:
                resolved = Path(file_path).resolve()
                try:
                    resolved.relative_to(self.allowed_root)
                except ValueError:
                    self._audit(tool_name, f"denied: path escape - {file_path}")
                    return MCPResponse(
                        error=f"安全: 路径 '{file_path}' 不在允许范围 {self.allowed_root}",
                        id=req_id,
                    ).to_json()

        self._audit(tool_name, f"allowed ({permission.value})")
        return super()._handle_call(params, req_id)

    def _audit(self, tool: str, action: str):
        self.audit_log.append({
            "time": datetime.now().isoformat(),
            "tool": tool,
            "action": action,
        })


print(f"""  MCP 安全模型设计:

  ┌──────────────┬──────────────────────────────────┐
  │  权限级别    │  安全策略                          │
  ├──────────────┼──────────────────────────────────┤
  │  READ        │  自动执行                          │
  │              │  • no side effects                │
  │              │  • 无需确认                        │
  ├──────────────┼──────────────────────────────────┤
  │  WRITE       │  路径白名单 + 用户确认              │
  │              │  • 只允许写入 ./safe_output/        │
  │              │  • 写入前展示 diff                  │
  │              │  • 禁止 .py/.sh/.bat 文件           │
  ├──────────────┼──────────────────────────────────┤
  │  DELETE      │  用户确认 + 二次确认                │
  │              │  • _confirmed=true 参数            │
  │              │  • 只允许删除 safe_output/ 内文件    │
  ├──────────────┼──────────────────────────────────┤
  │  EXECUTE     │  默认拒绝                          │
  │              │  • 需要 admin 令牌                  │
  │              │  • 完整沙箱环境                     │
  └──────────────┴──────────────────────────────────┘

  审计日志:
    每条 tools/call 记录: {{time, tool, action, params}}
    示例: 2026-06-19T10:30:00, delete_file, denied: requires confirmation

  类比 Java:
    SecureMCPServer ≈ @RestController + Spring Security
      READ    → @PreAuthorize("permitAll()")
      WRITE   → @PreAuthorize("hasRole('USER')")
      DELETE  → @PreAuthorize("hasRole('USER') and #params._confirmed")
      EXECUTE → @PreAuthorize("hasRole('ADMIN')")
    审计日志  → @Auditable + 审计表

  参考 deploy/agent_core.py: tool_read_file 已实现路径沙箱。""")

print(f"\n  ✅ 练习 6 完成: 设计了完整的 MCP 安全模型")

print(f"\n  📝 学习总结:")
print(f"     MCP 最大价值: 解耦 + 标准化")
print(f"       - Agent 不依赖工具实现 → 工具可独立变更")
print(f"       - 协议统一 → 不同团队/语言的工具可互操作")
print(f"       - 动态发现 → 热更新、热插拔")
print(f"       给团队引入 MCP, 第一个 Server:")
print(f"       提供代码仓库工具 (list_files, read_file, search_code)")
print(f"       → 所有 AI 工具共享同一个代码访问层")
