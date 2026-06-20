# mcp_server/ — Codebase QA MCP Server

使用官方 `mcp` Python SDK (FastMCP) 将 `codebase_qa` 的核心能力包装为标准 MCP 工具，通过 stdio 与 Claude Desktop 等 MCP Host 通信。

## 快速开始

```bash
pip install -r mcp_server/requirements.txt
python -m mcp_server
# stderr 输出: Embedding dim=1024 ✓  ChromaDB chunks=1163 ✓  Ready ✓
```

配置到 Claude Desktop (`~/.config/claude/claude_desktop_config.json`):

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

重启 Claude Desktop 后即可使用 `codebase_search`、`codebase_index`、`codebase_status` 三个工具。

## 架构

```
Claude Desktop / MCP Host
  │
  ▼ (JSON-RPC over stdin/stdout)
FastMCP Server
  ├─ @mcp.tool codebase_search  → QAPipeline.ask()                 → 自然语言搜索代码库
  ├─ @mcp.tool codebase_index   → CodeIndexer.index_directory()    → 索引代码目录
  └─ @mcp.tool codebase_status  → collection.count() + 配置快照    → 查看索引状态
  │
  ▼
codebase_qa (复用全部组件)
  └─ QAPipeline / Retriever / Reranker / CodeIndexer / EmbeddingFunction
```

## 项目结构

```
mcp_server/
├── __init__.py              # 包文档: 官方 mcp SDK + stdio 传输
├── __main__.py              # 入口点: python -m mcp_server
├── config.py                # McpServerConfig: 包装 AppConfig + MCP 专用字段
├── lifecycle.py             # AppContext + create_lifespan() (async context manager)
├── server.py                # FastMCP 组装 + @mcp.tool() 注册
├── tools.py                 # 3 个工具处理函数
├── demo_client.py           # MCP 客户端演示脚本
├── claude_desktop_config.json  # Claude Desktop 配置示例
├── requirements.txt         # mcp>=1.0.0 + anthropic + sentence-transformers + chromadb + numpy + python-dotenv
└── tests/
    ├── test_tools.py        # 13 个单元测试 (mock AppContext)
    ├── test_server.py       # 5 个初始化测试 (tool schema 验证)
    └── test_integration.py   # 3 个端到端测试 (ClientSession + stdio_client)
```

## 核心组件

### lifecycle.py — 生命周期管理

`create_lifespan()` async context manager 按顺序初始化:

```
1. 加载 Embedding 模型 (BAAI/bge-m3, 1024d)        ← 最耗时 (~10s)
2. 连接 ChromaDB (PersistentClient)                  ← 复用 codebase_qa 的数据
3. 创建 CodeIndexer (排除 tests/venv/.git 等)        ← AST 解析器
4. 创建 LlmClient (DeepSeek API, 健康检查)           ← LLM 可用性确认
5. 组装 QAPipeline (Retriever → Reranker → Generator) ← 业务流水线
6. 创建 ExactCache (LRU+TTL, 1000 条)                ← 查询缓存
→ yield AppContext                                    ← 就绪
→ finally: 清理 + 打印统计                             ← 优雅关闭
```

**AppContext**: 数据类容器，类似 Spring 的 `ApplicationContext`。所有初始化的组件放入其中，tool 函数通过 `ctx.request_context.lifespan_context` 访问。

### tools.py — 工具处理函数

| 函数 | 输入 | 返回 (JSON) | 关键实现 |
|------|------|-------------|---------|
| `handle_codebase_search` | query, top_k=5, filter_type="" | `{question, answer, sources, latency_ms, cached}` | 缓存检查 → `asyncio.to_thread(pipeline.ask)` → 格式化 sources → 写缓存 |
| `handle_codebase_index` | dirs: list[str] | `{status, indexed_files, total_chunks, collection_count, errors}` | 遍历目录 → `asyncio.to_thread(indexer.index_directory)` → ChromaDB upsert |
| `handle_codebase_status` | — | `{service, version, index: {...}, config: {...}}` | 读取 `collection.count()` + 配置快照 |

**关键模式**: `QAPipeline.ask()` 和 `CodeIndexer.index_directory()` 是同步阻塞操作，通过 `asyncio.to_thread()` 在线程池中运行，避免阻塞 MCP event loop。

### server.py — FastMCP 组装

```python
mcp = FastMCP(name="codebase-qa", lifespan=create_lifespan)

@mcp.tool(name="codebase_search", description="...")
async def codebase_search(ctx, query: str, top_k=5, filter_type="") -> str:
    return await handle_codebase_search(ctx, query, top_k, filter_type)
# ... 同理注册 codebase_index 和 codebase_status
```

## MCP 工具 Schema

### codebase_search

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | string | ✓ | — | 自然语言问题 |
| `top_k` | integer | | 5 | 返回结果数 |
| `filter_type` | string | | `""` | 代码类型过滤: function/class/method/module_level |

### codebase_index

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `dirs` | array[string] | ✓ | — | 要索引的目录列表 |

### codebase_status

无输入参数。

## 演示客户端

```bash
python mcp_server/demo_client.py
```

模拟 Claude Desktop 的完整交互流程:
1. `initialize()` 握手 → Server info + Protocol version
2. `list_tools()` → 3 个工具的完整 schema
3. `call_tool("codebase_status")` → 索引统计
4. `call_tool("codebase_search", {"query": "retry 在哪里"})` → 精确答案 + 来源
5. 重复查询验证缓存 → `0ms 缓存命中!`

## 与 codebase_qa FastAPI 的对比

| 维度 | codebase_qa FastAPI | mcp_server |
|------|---------------------|------------|
| 协议 | HTTP REST | JSON-RPC 2.0 (MCP) |
| 传输 | HTTP/TCP :8003 | stdio (标准输入输出) |
| 客户端 | curl / SDK / Web UI | Claude Desktop / MCP Host |
| 工具发现 | API 文档 (手动) | `list_tools()` 自动 |
| 部署 | 独立进程监听端口 | 作为 MCP Host 的子进程 |
| 生命周期 | FastAPI lifespan | MCP lifespan (async context manager) |

## 依赖

| 来源 | 用途 |
|------|------|
| `mcp>=1.0.0` | 官方 MCP Python SDK (FastMCP + stdio transport) |
| `codebase_qa.*` | QAPipeline, CodeIndexer, Retriever, Reranker, EmbeddingFunction, AppConfig |
| `deploy.agent_core.LlmClient` | LLM API 封装 |
| `deploy.cost_control.ExactCache` | 查询缓存 |
