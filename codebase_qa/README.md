# codebase_qa/ — AST 代码库问答助手

用 Python `ast` 标准库将源码按函数/类/方法边界结构化分块，存入 ChromaDB，支持自然语言搜索并返回带精确文件:行号的答案。

## 核心创新

**AST 结构化分块** vs 通用文本滑动窗口分块: 每个 chunk = 一个完整的函数/类/方法，含签名 + docstring + 源码，而非按字符数固定切分。

## 快速开始

```bash
# 安装依赖
pip install -r codebase_qa/requirements.txt

# FastAPI 服务
uvicorn codebase_qa.app:app --port 8003

# Streamlit Web UI
streamlit run codebase_qa/ui/app.py

# Docker
docker build -t codebase-qa -f codebase_qa/Dockerfile .
docker run -d --name cqa -p 8003:8003 --env-file .env codebase-qa
```

## 架构

```
.py 源码目录
  │
  ▼
CodeIndexer
  ├─ walk_directory()   — 递归发现 .py 文件 (跳过 tests/venv/.git 等)
  ├─ hash_file()        — SHA-256 去重, 支持增量索引
  └─ parse_file()       — ast.parse → FunctionDef/ClassDef/methods → CodeChunk[]
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

## 项目结构

```
codebase_qa/
├── __init__.py          # 包文档
├── config.py            # AppConfig: 22 个字段, Service/LLM/Embedding/ChromaDB/Index/Retrieval/Cache
├── indexer.py           # CodeChunk + CodeIndexer: AST 解析核心
├── retriever.py         # SearchResult + Retriever (ChromaDB 封装) + Reranker (MMR)
├── generator.py         # AnswerGenerator: 构建 prompt → 调用 LLM → 返回答案
├── pipeline.py          # EmbeddingFunction + QAPipeline + QAResponse
├── app.py               # FastAPI 服务 (4 端点 + 4 中间件 + lifespan)
├── requirements.txt     # anthropic, fastapi, uvicorn, python-dotenv, sentence-transformers, chromadb, numpy
├── Dockerfile            # python:3.12-slim + torch cpu + 预下载 bge-m3 (~2.2GB)
├── chroma_db/           # ChromaDB 持久化目录
├── ui/
│   └── app.py            # Streamlit Web UI
└── tests/
    ├── test_indexer.py    # AST 解析测试 (25 个)
    ├── test_retriever.py  # 检索/重排测试 (13 个)
    └── test_pipeline.py   # 端到端集成测试 (14 个)
```

## 各文件详解

### `indexer.py` — AST 结构化分块 (核心模块)

**CodeChunk 数据类**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | `"retry"` / `"MyClass"` / `"MyClass.method"` |
| `type` | str | `"function"` / `"class"` / `"method"` / `"module_level"` |
| `file_path` | str | 相对路径 |
| `start_line` / `end_line` | int | 1-based 行号 |
| `code_text` | str | 完整源码 |
| `docstring` | str | 提取的文档字符串 |
| `signature` | str | def/class 第一行 (含装饰器) |
| `file_hash` | str | SHA-256，用于增量索引去重 |

**CodeIndexer 核心方法**:

| 方法 | 说明 |
|------|------|
| `walk_directory(root)` | 递归发现 .py 文件，跳过排除目录|
| `hash_file(file_path)` | SHA-256 文件哈希 |
| `is_unchanged(file_path)` | 检查文件是否自上次索引后未修改 |
| `parse_file(file_path)` | ast.parse → 提取函数/类/方法/模块级代码 |
| `index_directory(root)` | walk + parse → 增量索引 |
| `chunk_to_metadata(chunk)` | CodeChunk → ChromaDB metadata dict |

**AST 解析逻辑**:
```
ast.parse(source) → tree
  ├─ FunctionDef/AsyncFunctionDef → CodeChunk(type="function")
  ├─ ClassDef → CodeChunk(type="class") + 遍历 body 提取 methods
  └─ 剩余未覆盖行 → CodeChunk(type="module_level")
```

### `retriever.py` — 检索 + 重排

| 类 | 方法 | 说明 |
|----|------|------|
| `SearchResult` | — | 数据类: doc_id, text, score, metadata |
| `Retriever` | `search(query, top_k, filter_type)` | ChromaDB 查询封装，score = 1/(1+distance) |
| `Reranker` | `filter_by_threshold(results, min_score)` (static) | 阈值过滤 |
| `Reranker` | `mmr_rerank(query, results, top_k, lambda)` | 贪心 MMR 多样性重排 |

### `generator.py` — 答案生成

`AnswerGenerator` 封装 `LlmClient`:
- `build_user_prompt(query, results)` — 格式化为 `[N] file:line (score) \n```code``` `
- `generate(query, results, max_tokens=1024)` — 调用 LLM → 返回带引用的答案

### `pipeline.py` — 流水线编排

| 组件 | 说明 |
|------|------|
| `EmbeddingFunction` | sentence-transformers → ChromaDB 适配器。BGE 模型自动加查询前缀 |
| `QAResponse` | 数据类: query, answer, sources, latency_ms |
| `QAPipeline.ask()` | 检索(2x top_k) → 阈值过滤 → MMR 重排 → LLM 生成 |

### `config.py` — 配置

```python
@dataclass
class AppConfig:
    # Service: host="0.0.0.0", port=8003
    # LLM: model="claude-sonnet-4-6", api_key, base_url, max_retries=2, timeout=90s
    # Rate limit: 30/min
    # Embedding: model="BAAI/bge-m3"
    # ChromaDB: persist_dir="./codebase_qa/chroma_db", collection="codebase_main"
    # Index: project_dirs=".", exclude="tests,venv,.git,..."
    # Retrieval: top_k=5, min_score=0.3, use_mmr=True, mmr_lambda=0.7
    # Cache: enabled=True, ttl=300s, max_size=1000
```

## API 参考

### `POST /index` — 索引代码目录

```bash
curl -X POST http://localhost:8003/index \
  -H "Content-Type: application/json" \
  -d '{"dirs": ["./phase1", "./deploy"]}'
# → {"status": "ok", "indexed_files": 45, "total_chunks": 523}
```

### `POST /query` — 自然语言搜索

```bash
curl -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{"question": "异步重试装饰器 retry 在哪里？", "top_k": 5}'
# → {
#     "question": "...",
#     "answer": "在 [phase1/04_functions.py:429-450] 找到了 retry 装饰器...",
#     "sources": [{ "file_path": "...", "start_line": 429, "score": 0.92 }],
#     "latency_ms": 6500,
#     "cached": false
#   }
```

支持 `filter_type` 参数: `"function"` / `"class"` / `"method"` / `"module_level"`。

### `GET /health` — 健康检查

```bash
curl http://localhost:8003/health
# → {"status": "ok", "alive": true, "ready": true, "checks": {...}}
```

### `GET /status` — 详细状态

```bash
curl http://localhost:8003/status
# → {"index_stats": {...}, "cache_stats": {...}, "config": {...}}
```

## ChromaDB Metadata Schema

| 字段 | 说明 |
|------|------|
| `name` | 函数/类/方法名 |
| `type` | function / class / method / module_level |
| `file_path` | 源文件相对路径 |
| `start_line` / `end_line` | 1-based 行范围 |
| `signature` | def/class 第一行 (含装饰器和类型注解) |

嵌入文本 (document) = `"function: def retry(max_attempts: int = 3):\nAsync retry decorator.\n实际代码..."`

## 环境变量

所有 `AppConfig` 字段均可通过环境变量覆盖:

| 变量 | 对应字段 | 默认值 |
|------|---------|--------|
| `EMBEDDING_MODEL` | embedding_model | `BAAI/bge-m3` |
| `CHROMA_PERSIST_DIR` | chroma_persist_dir | `./codebase_qa/chroma_db` |
| `TOP_K` | top_k | `5` |
| `MIN_SCORE` | min_score | `0.3` |
| `USE_MMR` | use_mmr | `true` |
| `CACHE_ENABLED` | cache_enabled | `true` |

## 依赖

| 包 | 用途 |
|----|------|
| `deploy.agent_core.LlmClient` | LLM API 封装 |
| `deploy.infrastructure` | RateLimiter、HealthChecker、GracefulShutdown、ServiceStats |
| `deploy.observability` | JsonLogger、Trace |
| `deploy.cost_control.ExactCache` | 查询缓存 |
| `chromadb` | 向量数据库 |
| `sentence-transformers` | 文本嵌入 |
| `numpy` | 向量计算 (余弦相似度) |
| `ast` (标准库) | Python 源码解析 |
