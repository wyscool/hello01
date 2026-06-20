# phase4/ — Agent + 工具使用 + MCP (5 课)

Phase 4 是最高级阶段: 从 ReAct Agent 循环开始，逐步构建多步规划 Agent，最终实现 MCP (Model Context Protocol) 标准协议。

## 学习目标

完成 Phase 4 后应能:
- 实现 ReAct Agent 循环 (Thought → Action → Observation → ...)
- 设计工具注册、调用和安全机制
- 实现 Plan-then-Act 多步推理模式
- 理解并实现 MCP (JSON-RPC 2.0) 协议
- 构建端到端 DevAssistant 项目 (capstone)

## 课程列表

| # | 文件 | 主题 | 核心内容 |
|---|------|------|---------|
| 31 | `31_agent_basics.py` | Agent 基础 | ReAct 循环 (Reason + Act)、工具定义 schema、工具调用处理、观察结果反馈 |
| 32 | `32_tool_use.py` | 工具使用进阶 | ToolRegistry 统一管理、工具编排、安全边界设计、MCP 前导概念 |
| 33 | `33_planning.py` | 多步规划 | Plan-then-Act 模式、子任务分解、执行结果反思、Plan → Execute → Reflect |
| 34 | `34_mcp_protocol.py` | MCP 协议 | JSON-RPC 2.0、`MCPServer` (tool/resource/prompt 注册)、`MCPClient` (连接/握手/调用)、`MCPAgent` (动态工具发现)、多服务器网关、安全模型 |
| 35 | `35_agent_project.py` | DevAssistant 项目 | 端到端 Agent 系统: CLI + Agent 层 (MCP 工具) + LLM 层。含 stdio 传输示例 |

## 课程递进关系

```
L31 ReActAgent
  └─ 直接硬编码工具定义 (dict, handler fn)
        ↓
L32 AdvancedAgent
  └─ ToolRegistry 统一管理 → 安全控制
        ↓
L33 PlannerAgent
  └─ Plan + Reflect 模式 → 复杂任务分解
        ↓
L34 MCPAgent
  └─ MCP 协议标准化 → 工具可跨进程/跨语言复用
        ↓
L35 DevAssistant (capstone)
  └─ CLI + MCP + LLM 三层架构 → 可部署的 AI 开发者助手
```

## 运行方式

```bash
# MCP 协议课 (最核心的一课)
python phase4/34_mcp_protocol.py

# Capstone 项目
python phase4/35_agent_project.py
```

## MCP vs 直接工具调用

| 维度 | 直接调用 (L31-33) | MCP (L34) |
|------|-------------------|-----------|
| 工具定义 | 硬编码在 Agent 代码中 | 注册在 MCP Server 中 |
| 工具发现 | Agent 启动前已知 | `client.list_tools()` 动态获取 |
| 工具调用 | `handler(**params)` | `client.call_tool(name, params)` |
| 传输层 | 进程内直接调用 | stdio / HTTP / SSE |
| 跨语言 | 不支持 | 任意语言可实现 Server |
| 工具复用 | 不能跨 Agent 共享 | 一个 Server 服务多个 Agent |
| 热更新 | 需重启 Agent | Agent 自动发现新工具 |

## 核心类 (L34 自实现 MCP)

| 类 | 职责 |
|----|------|
| `MCPRequest` / `MCPResponse` | JSON-RPC 2.0 消息格式。`to_json()` / `from_json()` |
| `MCPServer` | 工具注册中心。`register_tool()` 链式调用; `handle(raw_request)` 路由分发 |
| `MCPClient` | MCP 客户端。`connect(server)` → 握手; `list_tools()` → Anthropic 格式转换; `call_tool(name, args)` |
| `MCPAgent` | LLM Agent。`client.list_tools()` 动态发现工具 → ReAct 循环调用 |

## 前置要求

- 完成 Phase 1-3
- 理解 Agent 循环的基本概念 (观察 → 决策 → 行动)
- 了解 JSON-RPC 2.0 或 gRPC 等 RPC 协议思想
