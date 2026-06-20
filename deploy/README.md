# deploy/ — DevAssistant 共享基础设施包

本包是所有其他应用项目的**共享基础设施**，提供生产级的 LLM 客户端、MCP 协议、可观测性、成本控制和运维组件。被 `codebase_qa`、`rag_kb`、`log_analyzer`、`mcp_server` 等项目直接依赖。

类比 Java: 相当于公司内部的 common-lib / shared-infra 库，被所有微服务引用。

## 快速开始

```bash
pip install -r deploy/requirements.txt
# 各组件无独立启动入口，通过 FastAPI 服务组装使用:
uvicorn deploy.app:app --port 8000
```

## 架构

```
deploy/
├── __init__.py          # 包文档 (FastAPI + MCP 服务描述)
├── config.py            # AppConfig: 16 字段配置, from_env() 工厂
├── agent_core.py        # LlmClient + MCPServer/MCPClient + DevAssistant Agent
├── infrastructure.py    # RateLimiter + HealthChecker + GracefulShutdown + ServiceStats
├── observability.py     # JsonLogger + Span/Trace + TokenMonitor
├── cost_control.py      # ExactCache + SemanticCache + ModelRouter + SmartClient
├── app.py               # FastAPI 组装入口 (4 端点 + 4 中间件)
├── Dockerfile            # 生产容器镜像
├── requirements.txt      # anthropic, fastapi, uvicorn, python-dotenv, sentence-transformers, pydantic
└── tests/                # 6 个测试文件
```

## 各文件详解

### `config.py` — AppConfig

12-Factor 风格的全局配置，所有字段有默认值，通过 `from_env()` 从环境变量覆盖。

```python
@dataclass
class AppConfig:
    # Service
    host: str = "0.0.0.0"
    port: int = 8000
    # LLM
    llm_model: str = "claude-sonnet-4-6"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_max_retries: int = 3
    llm_timeout_seconds: float = 60.0
    # Limits
    max_concurrent_llm: int = 5
    rate_limit_per_minute: int = 30
    token_daily_budget: int = 1_000_000
    # Cache
    cache_enabled: bool = True
    cache_ttl_seconds: float = 300.0
    semantic_cache_threshold: float = 0.85
    project_root: str = ""
```

环境变量映射: `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `LLM_MODEL`, `LLM_MAX_RETRIES`, `LLM_TIMEOUT`, `MAX_CONCURRENT_LLM`, `RATE_LIMIT`, `HOST`, `PORT`, `TOKEN_DAILY_BUDGET`, `CACHE_ENABLED`, `CACHE_TTL`, `SEMANTIC_CACHE_THRESHOLD`, `PROJECT_ROOT`.

### `agent_core.py` — Agent 核心层

| 类 | 职责 |
|----|------|
| `LlmClient` | Anthropic SDK 封装。`create(messages, tools, system)` → `Message`; 静态方法 `get_text()`, `get_tool_uses()`; `is_healthy` 属性 |
| `MCPServer` | 工具注册中心。`register(name, desc, params, required, handler)` → 链式调用; `handle(raw)` → JSON-RPC 路由 (`tools/list`, `tools/call`, `initialize`) |
| `MCPClient` | MCPServer 的客户端。`connect(server)` → MCP 握手; `list_tools()` → Anthropic 格式; `call_tool(name, args)` → 调用结果 |
| `TokenBudget` | Token 预算估算器 (JSON 长度 / 4) |
| `DevAssistant` | 双模式 Agent: `ask(task, mode="quick|plan")` → `{answer, iterations, mode, tokens}` |
| `create_mcp_server()` | 工厂函数，注册 5 个内置工具 (calculate, get_current_time, text_stats, read_file, list_files) |
| `create_agent()` | 工厂函数，从 AppConfig 组装完整的 DevAssistant |

### `infrastructure.py` — 服务基础设施

| 类 | 职责 |
|----|------|
| `RateLimiter` | 滑动窗口速率限制。`allow() → bool`, `current_rate`, `stats()` |
| `HealthChecker` | 存活/就绪探针。`register_check(name, fn)` → 链式; `is_healthy`, `is_ready`; `status()` → `{alive, ready, uptime_seconds, checks}` |
| `GracefulShutdown` | 请求门控优雅关闭。`start_request()` / `end_request()`; `initiate(health)` → 设 not_ready → 等待活跃请求完成 → shutdown |
| `ServiceStats` | 请求统计。`record(status_code, latency_ms)`; `avg_latency_ms`, `p99_latency_ms`, `error_rate`, `rps` |

### `observability.py` — 可观测性

| 类 | 职责 |
|----|------|
| `JsonLogger` | 结构化 JSON 日志。`debug/info/warn/error(msg, **ctx)` → 单行 JSON 含 timestamp/level/logger/message |
| `Span` / `Trace` | 分布式追踪。`start_span(name)` → `end_span()`; `to_dict()` 序列化; `duration_ms` |
| `TokenMonitor` | Token 消耗追踪。`record(model, input_tokens, output_tokens)`; `total_used`, `budget_remaining`, `by_model()` |

### `cost_control.py` — 成本控制

| 类 | 职责 |
|----|------|
| `ExactCache` | LRU+TTL 精确缓存 (SHA-256 key)。`get(prompt, **params)`, `set(prompt, value, **params)`, `hit_rate` |
| `SemanticCache` | 基于 embedding 的语义缓存 (余弦相似度 ≥ threshold)。`get(query) → (value, similarity)`, `set(query, result)` |
| `ModelRouter` | 按任务复杂度路由模型。`route(task) → cheap/normal/premium`。使用关键词信号判断复杂度 |
| `SmartClient` | 两层缓存 + LLM 编排。`call(task) → (result, metadata)`。流程: ExactCache → SemanticCache → ModelRouter → LLM |

### `app.py` — FastAPI 组装入口

**中间件** (按执行顺序):
1. `RequestIdMiddleware` — 注入 `X-Request-ID`
2. `RequestLoggingMiddleware` — 记录 method/path/status/duration
3. `ShutdownGateMiddleware` — 关闭中返回 503
4. `RateLimitMiddleware` — 超限返回 429

**端点**:

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务索引页 |
| GET | `/health` | 健康检查 (K8s 探针) |
| GET | `/status` | 聚合状态: health + stats + rate_limiter + token_monitor |
| GET | `/tools` | 列出所有已注册的 MCP 工具 |
| POST | `/ask` | 核心问答: `{task: "...", mode: "quick|plan"}` |

**Lifespan**: 启动 → 初始化 LlmClient → 创建 MCP Server (5 个工具) → 组装 DevAssistant Agent → 寄存器健康检查 → yield → 关闭

## API 调用示例

```bash
# 健康检查
curl http://localhost:8000/health

# 服务状态
curl http://localhost:8000/status

# 列出工具
curl http://localhost:8000/tools

# 执行任务
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"task": "读取并分析 CLAUDE.md 文件内容", "mode": "quick"}'
```

## 被依赖关系

```
deploy/
  ^
  +-- codebase_qa/    (LlmClient, RateLimiter, HealthChecker, JsonLogger, Trace, ExactCache)
  +-- rag_kb/         (同上)
  +-- log_analyzer/   (LlmClient, MCPServer, MCPClient, TokenBudget, 基础设施)
  +-- mcp_server/     (LlmClient, ExactCache)
  └── code_review/    独立 (自有 LlmClient/MCPServer, 不依赖 deploy/)
```

## Docker

```bash
docker build -t dev-assistant -f deploy/Dockerfile .
docker run -d --name assistant -p 8000:8000 --env-file .env dev-assistant
```

## 测试

```bash
pytest deploy/tests/ -v    # 6 个文件, 覆盖 agent / config / cost_control / infrastructure / observability
```

## 设计模式

- **依赖注入**: 所有组件通过构造函数接收依赖，无全局变量
- **工厂函数**: `create_mcp_server()` 和 `create_agent()` 封装复杂的组装逻辑
- **lifespan 管理**: 资源在 async context manager 中初始化/清理
- **中间件链**: RequestId → Logging → ShutdownGate → RateLimit
