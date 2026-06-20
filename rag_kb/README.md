# rag_kb/ — RAG 知识库问答服务

通用文档 RAG (检索增强生成) 服务，支持导入 txt/md/代码文件 → 递归分块 → ChromaDB 向量检索 → LLM 生成带引用答案。

与 `codebase_qa/` 的区别: rag_kb 面向**通用文档** (递归文本分块)，codebase_qa 面向**Python 代码** (AST 结构化分块)。

## 快速开始

```bash
pip install -r rag_kb/requirements.txt
uvicorn rag_kb.app:app --port 8002
```

## 架构

```
文件上传 (.txt/.md/.py, 最大 10MB)
  │
  ▼
DocumentProcessor
  ├─ load()   — 读取文件 + 编码检测
  ├─ clean()  — 清洗空白/特殊字符
  └─ chunk()  — 递归分块 (chunk_size=256, overlap=64)
  │
  ▼
EmbeddingFunction (all-MiniLM-L6-v2, 384d)
  │
  ▼
ChromaDB (PersistentClient)
  │
  ▼
Retriever → Reranker (MMR) → ContextBuilder → Generator (LLM)
  │
  ▼
答案 + 来源引用 (source + chunk_index)
```

## 项目结构

```
rag_kb/
├── __init__.py          # 包文档
├── config.py            # AppConfig: 22 个字段
├── pipeline.py          # 完整 RAG 流水线 (EmbeddingFunction + RAGPipeline)
├── knowledge_base.py    # KnowledgeBase: ChromaDB CRUD 封装
├── app.py               # FastAPI 服务 (7 端点)
├── Dockerfile            # 生产镜像
├── requirements.txt      # anthropic, fastapi, uvicorn, python-dotenv, sentence-transformers, chromadb, numpy
└── tests/
    ├── test_config.py
    ├── test_knowledge_base.py
    └── test_pipeline.py
```

## 核心组件

### pipeline.py — RAG 流水线

| 组件 | 职责 |
|------|------|
| `EmbeddingFunction` | sentence-transformers → ChromaDB 适配器 |
| `Chunk` | 文档块数据类: index, text, metadata |
| `DocumentProcessor` | 加载 → 清洗 → 递归分块 (双换行 → 单换行 → 句号 → 固定大小) |
| `SearchResult` | 检索结果: doc_id, text, score, metadata |
| `Retriever` | ChromaDB 查询封装, score = 1/(1+distance) |
| `Reranker` | 阈值过滤 + MMR 多样性重排 |
| `ContextBuilder` | 构建 LLM prompt: results 列表 → 格式化文本 |
| `Generator` | 调用 LLM 生成答案 (含 source 引用) |
| `RAGPipeline` | 编排器: Retrieve → Rerank → BuildContext → Generate |

### knowledge_base.py — 知识库 CRUD

```python
kb = KnowledgeBase(collection, embed_fn)

kb.add(source="doc1", text="...")      # 导入文档
kb.get(source="doc1")                  # 查看来源
kb.list()                               # 列出所有来源
kb.remove(source="doc1")               # 删除来源
kb.search(query, top_k=5)              # 语义搜索
kb.stats()                              # 统计信息
```

## API 参考

### `POST /ingest` — 导入文档

```bash
curl -X POST http://localhost:8002/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "claude_docs", "text": "Claude Code is an AI-powered CLI..."}'
# → {"status": "ok", "chunks": 3, "source": "claude_docs"}
```

### `POST /query` — 提问

```bash
curl -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Claude Code?", "top_k": 5}'
# → {"question": "...", "answer": "...", "sources": [...], "latency_ms": 1200}
```

### `GET /kb` — 列出所有知识库来源

```bash
curl http://localhost:8002/kb
# → {"sources": ["claude_docs", "python_tutorial"]}
```

### `GET /kb/{source}` — 查看来源详情

```bash
curl http://localhost:8002/kb/claude_docs
# → {"source": "claude_docs", "chunk_count": 3, "total_chars": 450}
```

### `DELETE /kb/{source}` — 删除来源

```bash
curl -X DELETE http://localhost:8002/kb/claude_docs
# → {"status": "deleted", "source": "claude_docs"}
```

## 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 256 | 文本分块大小 (字符数) |
| `chunk_overlap` | 64 | 块间重叠大小 |
| `max_file_size_mb` | 10 | 单文件导入上限 |
| `embedding_model` | `all-MiniLM-L6-v2` | 嵌入模型 (384d，轻量快速) |
| `top_k` | 5 | 检索返回数 |
| `min_score` | 0.3 | 最低相关度阈值 |
| `use_mmr` | True | 是否启用 MMR 多样性重排 |
| `mmr_lambda` | 0.7 | MMR 参数 (越大越偏向相关性) |

## 与 codebase_qa 的对比

| 维度 | rag_kb | codebase_qa |
|------|--------|-------------|
| 分块策略 | 递归文本分块 (按字符数) | AST 结构化分块 (按语法) |
| 适用内容 | 通用文档 (txt/md/代码) | Python 源码 |
| 粒度 | chunk (无语义边界) | function/class/method (语义完整) |
| 答案引用 | source + chunk_index | file_path:start_line-end_line |
| 嵌入模型 | MiniLM 384d | bge-m3 1024d |
| 端口 | 8002 | 8003 |
