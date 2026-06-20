# Hello01 — Python AI 应用开发学习项目

一个从零开始的 **Python AI 应用开发**学习项目，涵盖 Python 基础、LLM API、RAG、Agent、MCP 协议等 5 个阶段 29 节课，以及 6 个可直接运行的生产级实战项目。

学习者背景：Java 后端工程师，Python 零基础，系统化学习 AI 应用开发。

## 环境

| 项目 | 值 |
|------|-----|
| Python | 3.12.13 |
| 环境管理 | miniforge / conda 环境 `myhello` |
| IDE | PyCharm |
| LLM API | DeepSeek (兼容 Anthropic SDK) |
| 操作系统 | macOS (Apple Silicon) |

### 环境变量

项目根目录 `.env` 文件包含 LLM API 配置：

```
ANTHROPIC_API_KEY=sk-xxx
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.deepseek.com
```

各应用包通过 `python-dotenv` 的 `load_dotenv()` 自动加载。

## 项目结构

```
hello01/
├── .env                          # API 密钥配置
├── CLAUDE.md                     # Claude Code 项目指南
├── curriculum.py                 # 课程进度追踪器
├── main.py                       # PyCharm 默认入口
│
├── phase1/                       # Python 基础 + 工程化 (10 课)
├── phase2/                       # LLM API + Prompt 工程 (5 课)
├── phase3/                       # RAG + 向量数据库 (5 课)
├── phase4/                       # Agent + 工具使用 + MCP (5 课)
├── phase5/                       # AI 工程化 (4 课)
│
├── deploy/                       # [项目1] DevAssistant 基础设施包
├── codebase_qa/                  # [项目2] AST 代码库问答助手
├── rag_kb/                       # [项目3] RAG 知识库服务
├── log_analyzer/                 # [项目4] 智能日志分析 Agent
├── code_review/                  # [项目5] AI 代码审查助手
├── mcp_server/                   # [项目6] Codebase QA MCP Server
└── output/                       # 临时输出目录
```

## 课程体系

5 个阶段 29 节课，每节课是独立可运行的 `.py` 文件，带详细注释和 Java 类比。

运行 `python curriculum.py` 查看进度。

### Phase 1 — Python 基础 + 工程化 (10 课)

| 课 | 文件 | 内容 |
|----|------|------|
| 01 | `01_basics.py` | 变量、类型、字符串、f-string |
| 02 | `02_control_flow.py` | 条件判断、循环、推导式 |
| 03 | `03_collections.py` | list/dict/set/tuple、切片、生成器 |
| 04 | `04_functions.py` | 函数定义、参数、装饰器、闭包 |
| 05 | `05_classes.py` | 类、继承、魔术方法、property、dataclass |
| 06 | `06_modules_packages.py` | import、包结构、`__init__.py` |
| 07 | `07_errors.py` | try/except、自定义异常、上下文管理器 |
| 08 | `08_files_json.py` | 文件读写、JSON、序列化 |
| 09 | `09_async_basics.py` | asyncio、协程、异步上下文管理器 |
| 10 | `10_pytest_basics.py` | pytest、fixture、mock、参数化 |

### Phase 2 — LLM API + Prompt 工程 (5 课)

| 课 | 文件 | 内容 |
|----|------|------|
| 11 | `11_llm_api_intro.py` | Anthropic SDK、Message API、基本调用 |
| 12 | `12_prompt_engineering.py` | System prompt、few-shot、结构化提示 |
| 13 | `13_structured_output.py` | JSON mode、tool use、结构化输出 |
| 14 | `14_streaming.py` | SSE 流式响应、实时输出 |
| 15 | `15_chat_app.py` | 多轮对话、上下文管理、简单 chatbot |

### Phase 3 — RAG + 向量数据库 + Embedding (5 课)

| 课 | 文件 | 内容 |
|----|------|------|
| 21 | `21_embeddings.py` | 语义嵌入、sentence-transformers、余弦相似度 |
| 22 | `22_vector_db.py` | ChromaDB、向量检索、metadata 过滤 |
| 23 | `23_document_processing.py` | 文档加载、清洗、递归分块 |
| 24 | `24_retrieval_pipeline.py` | 检索 + 重排 (MMR) + LLM 生成 |
| 25 | `25_knowledge_base.py` | 知识库封装、增删查改 API |

### Phase 4 — Agent + 工具使用 + MCP (5 课)

| 课 | 文件 | 内容 |
|----|------|------|
| 31 | `31_agent_basics.py` | ReAct Agent 循环、工具定义与调用 |
| 32 | `32_tool_use.py` | ToolRegistry、工具编排与安全 |
| 33 | `33_planning.py` | Plan-then-Act 模式、多步推理 |
| 34 | `34_mcp_protocol.py` | MCP 协议 (JSON-RPC 2.0)、server/client 实现 |
| 35 | `35_agent_project.py` | DevAssistant 端到端项目 (capstone) |

### Phase 5 — AI 工程化 (4 课)

| 课 | 文件 | 内容 |
|----|------|------|
| 41 | `41_evaluation.py` | 评估框架、指标计算、评估 pipeline |
| 42 | `42_observability.py` | 结构化日志、分布式追踪、Token 监控 |
| 43 | `43_cost_control.py` | 缓存策略、模型路由、成本计算 |
| 44 | `44_production.py` | FastAPI、中间件、Docker 部署、健康检查 |

---

## 实战项目

6 个可独立运行的项目，从教学阶段的知识点构建而来，遵循统一的设计模式。

### 通用设计模式

所有项目共享以下模式（见 `deploy/` 基础设施包）：

| 组件 | 来源 | 用途 |
|------|------|------|
| `LlmClient` | `deploy/agent_core.py` | Anthropic SDK 封装，含健康检查和重试 |
| `JsonLogger` | `deploy/observability.py` | 结构化 JSON 日志 |
| `Trace` | `deploy/observability.py` | 请求级分布式追踪 |
| `RateLimiter` | `deploy/infrastructure.py` | 滑动窗口速率限制 |
| `HealthChecker` | `deploy/infrastructure.py` | 存活/就绪探针 |
| `GracefulShutdown` | `deploy/infrastructure.py` | 优雅关闭 (请求门控 + 超时) |
| `ServiceStats` | `deploy/infrastructure.py` | 延迟统计 (avg, P99) |
| `ExactCache` | `deploy/cost_control.py` | LRU+TTL 查询缓存 |
| `AppConfig` | 每个包各有一个 | 12-Factor 配置，`from_env()` 工厂方法 |

---

### 项目 1: deploy/ — DevAssistant 生产部署 (基础设施)

**一句话**: 将教学阶段的 ReAct Agent 包装为生产级 FastAPI 服务，附带完整的可观测性和成本控制。

**端口**: 8000

**启动**:
```bash
uvicorn deploy.app:app --port 8000
```

**API**:
```
GET  /health    — 健康检查 (K8s 探针兼容)
GET  /status    — 服务状态 + 统计 + Token 消耗
GET  /tools     — 列出已注册的 MCP 工具
POST /ask       — Agent 任务执行 {task: "...", mode: "quick|plan"}
```

**核心文件**:

| 文件 | 职责 |
|------|------|
| `agent_core.py` | `LlmClient` (LLM 封装)、`MCPServer`/`MCPClient` (JSON-RPC 2.0)、`DevAssistant` (双模式 Agent)、5 个内置工具 (`calculate`/`get_current_time`/`text_stats`/`read_file`/`list_files`) |
| `infrastructure.py` | `RateLimiter`、`HealthChecker`、`GracefulShutdown`、`ServiceStats` |
| `observability.py` | `JsonLogger`、`Span`/`Trace`、`TokenMonitor` |
| `cost_control.py` | `ExactCache` (LRU+TTL)、`SemanticCache` (embedding 相似度匹配)、`ModelRouter` (任务复杂度 → cheap/normal/premium 模型路由)、`SmartClient` (两层缓存 + LLM 编排) |
| `config.py` | `AppConfig`: 16 个字段，环境变量管理 |
| `app.py` | FastAPI 组装入口，4 个中间件 (RequestId → Logging → ShutdownGate → RateLimit) |

**Docker**:
```bash
docker build -t dev-assistant -f deploy/Dockerfile .
docker run -d --name assistant -p 8000:8000 --env-file .env dev-assistant
```

---

### 项目 2: codebase_qa/ — AST 代码库问答助手

**一句话**: 用 Python `ast` 标准库将源码按函数/类/方法边界结构化分块，存入 ChromaDB，支持自然语言搜索并返回精确到文件:行号的答案。

**核心创新**: AST 结构化分块 (vs 通用文本的滑动窗口分块)。每个块 = 一个函数/类/方法，包含完整签名、docstring 和源码。

**端口**: 8003

**架构**:
```
.py 源码目录
  │
  ▼
CodeIndexer
  ├─ walk_directory() — 递归发现 .py 文件
  ├─ hash_file()      — SHA-256 去重，支持增量索引
  └─ parse_file()     — ast.parse → FunctionDef/ClassDef/methods → CodeChunk[]
  │
  ▼
ChromaDB (PersistentClient)
  └─ EmbeddingFunction (BAAI/bge-m3, 1024d)
  │
  ▼
Retriever.search() → Reranker (MMR) → AnswerGenerator (LlmClient)
  │
  ▼
QAResponse { answer, sources: [{file_path, start_line, end_line, score}], latency_ms }
```

**启动**:
```bash
# FastAPI 服务
uvicorn codebase_qa.app:app --port 8003

# Streamlit Web UI
streamlit run codebase_qa/ui/app.py
```

**API**:
```
GET  /health    — 健康检查 (LLM + ChromaDB + Embedding)
GET  /status    — 索引统计 + 缓存统计 + 配置快照
POST /index     — 索引目录 {dirs: ["./phase1", "./deploy"]}
POST /query     — 提问 {question: "retry 装饰器在哪里？", top_k: 5, filter_type: "function"}
```

**核心文件**:

| 文件 | 职责 |
|------|------|
| `indexer.py` | `CodeChunk` 数据类 (name/type/file_path/start_line/end_line/full_source/docstring/signature/SHA-256)。`CodeIndexer`: AST 解析 → 提取函数/异步函数/类/方法/模块级代码 |
| `retriever.py` | `SearchResult` 数据类。`Retriever` (ChromaDB 封装，支持 type 过滤)。`Reranker` (阈值过滤 + 贪心 MMR 多样性重排) |
| `generator.py` | `AnswerGenerator`: 构建带源码引用的 prompt → 调用 LLM → 返回文件:行号答案 |
| `pipeline.py` | `EmbeddingFunction` (sentence-transformers → ChromaDB 适配器)。`QAPipeline` 编排器 (Retrieve → Filter → MMR → Generate) |
| `config.py` | `AppConfig`: 22 个字段，覆盖 Service/LLM/Embedding/ChromaDB/Index/Retrieval/Cache |
| `app.py` | FastAPI 组装。4 个端点 + 4 个中间件 + lifespan 初始化 |
| `ui/app.py` | Streamlit Web UI: 搜索框 + 可折叠源码卡片 + 配置侧边栏 |

**Docker**:
```bash
docker build -t codebase-qa -f codebase_qa/Dockerfile .
docker run -d --name cqa -p 8003:8003 --env-file .env codebase-qa
```

**测试**:
```bash
pytest codebase_qa/tests/ -v        # 52 个测试 (AST 解析 + 检索 + 端到端)
```

---

### 项目 3: rag_kb/ — RAG 知识库问答服务

**一句话**: 通用文档 RAG 服务，支持导入 txt/md/代码文件，递归分块 → ChromaDB 向量检索 → LLM 生成带引用的答案。

**端口**: 8002

**架构**:
```
文件上传 (.txt/.md/.py)
  │
  ▼
DocumentProcessor
  └─ load → clean → recursive_split (256 chars, 64 overlap)
  │
  ▼
ChromaDB (PersistentClient)
  └─ EmbeddingFunction (all-MiniLM-L6-v2, 384d)
  │
  ▼
Retriever → Reranker (MMR) → ContextBuilder → Generator (LLM)
  │
  ▼
答案 + 来源引用 (source + chunk index)
```

**启动**:
```bash
uvicorn rag_kb.app:app --port 8002
```

**API**:
```
GET  /health         — 健康检查
GET  /status         — 知识库统计
GET  /kb             — 列出所有知识库来源
GET  /kb/{source}    — 查看来源详情
DELETE /kb/{source}  — 删除来源
POST /ingest         — 导入文档 {source: "...", text: "..."}
POST /query          — 提问 {query: "...", top_k: 5}
```

**核心文件**:

| 文件 | 职责 |
|------|------|
| `pipeline.py` | 完整 RAG 流水线: `EmbeddingFunction` → `DocumentProcessor` → `Retriever` → `Reranker` → `ContextBuilder` → `Generator` → `RAGPipeline` |
| `knowledge_base.py` | `KnowledgeBase`: ChromaDB CRUD 封装 (add/get/list/remove/search/stats) |
| `config.py` | `AppConfig`: 22 个字段，含文件大小限制 (max_file_size_mb=10) |
| `app.py` | FastAPI 服务组装 |

**Docker**:
```bash
docker build -t rag-kb -f rag_kb/Dockerfile .
docker run -d --name rag -p 8002:8002 --env-file .env rag-kb
```

---

### 项目 4: log_analyzer/ — 智能日志分析 Agent

**一句话**: LLM 驱动的日志分析 Agent。解析多种日志格式 (JSON/text/Java 堆栈)，通过 ReAct 循环自主分析故障原因。

**端口**: 8003 (注意与 codebase_qa 冲突，需设 `LOG_ANALYZER_PORT`)

**架构**:
```
日志文件 (.log/.json/.txt, 最大 50MB)
  │
  ▼
LogParser
  ├─ 自动编码检测 (utf-8 → gbk → latin-1)
  ├─ 多格式解析 (JSON lines / 标准日志 / Java 堆栈 / 纯文本)
  └─ → LogEntry[]
  │
  ▼
MCPServer (7 个日志分析工具)
  ├─ tool_stats       — 统计概览 (条目数、时间范围、级别分布)
  ├─ tool_top_errors  — TOP N 错误消息
  ├─ tool_search      — 关键词搜索
  ├─ tool_count       — 正则匹配计数
  ├─ tool_sample      — 随机采样
  ├─ tool_timeline    — 时间轴分布
  └─ tool_get_section — 按行号范围提取
  │
  ▼
LogAnalysisAgent
  └─ ReAct 循环: stats → top_errors → search → get_section → 根因报告
```

**启动**:
```bash
# CLI
python log_analyzer/cli.py analyze /path/to/app.log
python log_analyzer/cli.py stats /path/to/app.log       # 纯统计，不调用 LLM

# FastAPI 服务
uvicorn log_analyzer.app:app --port 8004
```

**API**:
```
POST /analyze         — 分析服务器端日志文件 {path: "/var/log/app.log"}
POST /analyze/upload  — 上传并分析日志文件 (multipart/form-data)
GET  /cache           — 缓存统计
```

**核心文件**:

| 文件 | 职责 |
|------|------|
| `parser.py` | `LogEntry` 数据类。`LogParser`: 4 种格式解析 + 自动编码检测 |
| `tools.py` | 7 个日志分析工具，通过 `functools.partial` 绑定到 `MCPServer` |
| `agent.py` | `LogAnalysisAgent`: ReAct 循环 (最多 10 次迭代) |
| `config.py` | `AppConfig`: 17 个字段，含 max_file_mb=50, agent_max_iterations=10 |
| `cli.py` | CLI 入口 (analyze + stats 子命令) |
| `app.py` | FastAPI 服务 |

**测试**:
```bash
pytest log_analyzer/tests/ -v      # 日志解析 + Agent + 工具注册测试
```

---

### 项目 5: code_review/ — AI 代码审查助手

**一句话**: 对 Java/Python 代码执行多层 AI 审查：静态规则检查 → LLM 模式检测 → 深度分析 → 综合报告。

**端口**: 8001

**架构**:
```
源代码 (Java/Python)
  │
  ▼
Step 1: 静态规则检查 (零 LLM 成本)
  ├─ 命名规范 (class PascalCase, function snake_case)
  ├─ 行长度检查 (>120 chars)
  ├─ 方法长度检查 (>50 lines)
  ├─ 空 catch 块检测
  ├─ System.out.print 检测 (Java)
  └─ 可变默认参数检测 (Python)
  │
  ▼
Step 2: LLM 辅助模式检测
  ├─ SQL 注入检测
  ├─ 资源泄漏检测
  ├─ 空指针风险检测
  └─ 不安全 eval() 检测
  │
  ▼
Step 3: 深度 LLM 分析
  └─ 逻辑错误 / 安全漏洞 / 性能问题 / 可读性
  │
  ▼
Step 4: 综合报告 (规则 + 模式 + 深度分析 + 建议)
```

**启动**:
```bash
# CLI — 文件审查
python code_review/cli.py path/to/file.py

# CLI — 交互模式
python code_review/cli.py
# 粘贴代码，输入 END 结束

# CLI — 管道
cat file.py | python code_review/cli.py

# FastAPI 服务
uvicorn code_review.app:app --port 8001
```

**API**:
```
POST /review  — 审查代码 {code: "...", language: "python|java", focus: ["security", "performance"]}
```

**核心文件**:

| 文件 | 职责 |
|------|------|
| `agent.py` | `ReviewAgent`: Plan-then-Act 4 步审查流程 |
| `tools.py` | `tool_check_style` (静态规则) + `tool_detect_patterns` (LLM 辅助模式匹配) |
| `cli.py` | CLI 入口，三种模式 (交互/文件/管道) |
| `config.py` | `AppConfig`: 7 个字段 |
| `app.py` | FastAPI 服务 |

**注**: `code_review/` 是唯一不依赖 `deploy/` 的包，拥有独立的 `LlmClient`/`MCPServer` 实现。

---

### 项目 6: mcp_server/ — Codebase QA MCP Server

**一句话**: 使用官方 `mcp` Python SDK (FastMCP) 将 codebase_qa 的核心能力包装为 MCP 工具，通过 stdio 与 Claude Desktop 等 MCP Host 通信。

**传输方式**: stdio (标准输入/输出)

**架构**:
```
Claude Desktop / MCP Host
  │
  ▼ (JSON-RPC over stdin/stdout)
FastMCP Server (stdio)
  ├─ @mcp.tool codebase_search → QAPipeline.ask()
  ├─ @mcp.tool codebase_index  → CodeIndexer.index_directory()
  └─ @mcp.tool codebase_status → ChromaDB 状态 + 配置快照
  │
  ▼
codebase_qa (复用全部组件)
  └─ QAPipeline / Retriever / Reranker / CodeIndexer / EmbeddingFunction
```

**启动**:
```bash
python -m mcp_server
```

**MCP 工具**:

| 工具 | 参数 | 返回 |
|------|------|------|
| `codebase_search` | `query` (必填), `top_k=5`, `filter_type=""` | `{question, answer, sources, latency_ms, cached}` |
| `codebase_index` | `dirs: list[str]` | `{status, indexed_files, total_chunks, errors}` |
| `codebase_status` | 无 | `{service, index: {files_indexed, total_chunks, ...}, config: {...}}` |

**Claude Desktop 配置** (合并到 `~/.config/claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "codebase-qa": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "/path/to/hello01",
      "env": {
        "ANTHROPIC_API_KEY": "sk-xxx",
        "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
        "CHROMA_PERSIST_DIR": "./codebase_qa/chroma_db",
        "EMBEDDING_MODEL": "BAAI/bge-m3"
      }
    }
  }
}
```

**核心文件**:

| 文件 | 职责 |
|------|------|
| `server.py` | FastMCP 组装，`@mcp.tool()` 注册 3 个工具 |
| `lifecycle.py` | `AppContext` 容器 + `create_lifespan()` async context manager |
| `tools.py` | 3 个工具处理函数，通过 `ctx.request_context.lifespan_context` 获取依赖 |
| `config.py` | `McpServerConfig`: 包装 `AppConfig`，新增 MCP 专用字段 |
| `demo_client.py` | MCP 客户端演示脚本，模拟 Claude Desktop 的完整交互流程 |

**测试**:
```bash
pytest mcp_server/tests/ -v        # 21 个测试 (单元 + 服务器 + 集成)
python mcp_server/demo_client.py   # 端到端演示
```

---

## 依赖关系图

```
deploy/  (共享基础设施 — LlmClient, JsonLogger, Trace, RateLimiter,
           HealthChecker, GracefulShutdown, ExactCache, ModelRouter, ...)
  ^
  |
  +── codebase_qa/    import: deploy.agent_core, deploy.infrastructure,
  |                           deploy.observability, deploy.cost_control
  |
  +── rag_kb/         import: deploy.agent_core, deploy.infrastructure,
  |                           deploy.observability, deploy.cost_control
  |
  +── log_analyzer/   import: deploy.agent_core, deploy.infrastructure,
  |                           deploy.observability, deploy.cost_control
  |
  +── mcp_server/     import: deploy.agent_core.LlmClient,
  |                           deploy.cost_control.ExactCache
  |                     also: codebase_qa.pipeline, codebase_qa.indexer,
  |                           codebase_qa.retriever, codebase_qa.generator
  |
  └── code_review/    独立实现 (有自己的 LlmClient/MCPServer，不依赖 deploy/)
```

## 技术栈

| 技术 | 用途 |
|------|------|
| **FastAPI** | 所有 HTTP 服务的框架，含 lifespan + 中间件 |
| **Uvicorn** | ASGI 服务器 |
| **ChromaDB** | 向量数据库，PersistentClient 模式 |
| **sentence-transformers** | 文本嵌入 (MiniLM 384d / bge-m3 1024d) |
| **Anthropic SDK** | LLM 调用 (通过 DeepSeek API 兼容代理) |
| **ast** (标准库) | Python 源码结构化分块，零额外依赖 |
| **mcp (>=1.0.0)** | 官方 MCP Python SDK，stdio 传输 |
| **Streamlit** | Web UI (纯 Python，零前端代码) |
| **Docker** | 容器化部署，CPU 版 PyTorch |
| **pytest** | 测试框架，含 pytest-asyncio |

## 快速开始

```bash
# 1. 激活环境
conda activate myhello

# 2. 安装所有依赖 (各包按需安装)
pip install -r deploy/requirements.txt
pip install -r codebase_qa/requirements.txt
pip install -r rag_kb/requirements.txt
pip install -r log_analyzer/requirements.txt
pip install -r code_review/requirements.txt
pip install -r mcp_server/requirements.txt
pip install streamlit pytest-asyncio

# 3. 运行测试
pytest deploy/tests/ -v
pytest codebase_qa/tests/ -v
pytest rag_kb/tests/ -v
pytest code_review/tests/ -v
pytest log_analyzer/tests/ -v
pytest mcp_server/tests/ -v

# 4. 启动服务 (示例)
uvicorn codebase_qa.app:app --port 8003  # Codebase Q&A
streamlit run codebase_qa/ui/app.py       # Web UI
python -m mcp_server                      # MCP Server
```

## 常见问题

**Q: 第一次启动为什么很慢？**
A: sentence-transformers 会自动从 Hugging Face 下载 embedding 模型 (bge-m3 约 2.2GB)。模型会缓存到本地，后续启动秒级完成。

**Q: 为什么都用 DeepSeek API 而不是 Anthropic 直连？**
A: DeepSeek 提供 Anthropic SDK 兼容的 API 端点 (`/anthropic/v1/messages`)，且价格更低。直接改 `ANTHROPIC_BASE_URL` 即可切回 Anthropic 官方 API。

**Q: log_analyzer 和 codebase_qa 默认都是端口 8003？**
A: 是的，需要调整其中一个。建议设环境变量 `LOG_ANALYZER_PORT=8004` 或改代码中的默认值。

**Q: code_review 为什么不依赖 deploy？**
A: code_review 是最早的实战项目，为了独立可运行，包含了自给自足的 LlmClient/MCPServer 实现。后续项目抽取公共代码到 deploy/，体现了"先写具体再提取抽象"的学习路径。
