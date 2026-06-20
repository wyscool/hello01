# phase3/ — RAG + 向量数据库 + Embedding (5 课)

Phase 3 学习 RAG (检索增强生成) 的完整技术栈: 语义嵌入、向量数据库、文档处理、检索排序和知识库构建。

## 学习目标

完成 Phase 3 后应能:
- 使用 sentence-transformers 生成语义嵌入向量
- 用 ChromaDB 构建向量检索系统
- 设计文档分块策略 (递归分块、语义分块)
- 实现检索 → 重排 (MMR) → LLM 生成的完整 RAG pipeline
- 封装知识库: 文档导入、存储、检索、管理

## 课程列表

| # | 文件 | 主题 | 核心内容 |
|---|------|------|---------|
| 21 | `21_embeddings.py` | 语义嵌入 | sentence-transformers、向量维度、余弦相似度、嵌入可视化 |
| 22 | `22_vector_db.py` | 向量数据库 | ChromaDB PersistentClient、collection CRUD、metadata 过滤、相似度搜索 |
| 23 | `23_document_processing.py` | 文档处理 | 文件加载、文本清洗、递归分块 (chunk_size + overlap)、多格式支持 |
| 24 | `24_retrieval_pipeline.py` | 检索流水线 | 向量检索 → 阈值过滤 → MMR 重排 → LLM 生成答案 + 引用 |
| 25 | `25_knowledge_base.py` | 知识库封装 | KnowledgeBase 类: add/get/list/remove/search/stats 完整 API |

## 运行方式

```bash
# 首次运行时 sentence-transformers 会下载模型 (约 120MB)
python phase3/21_embeddings.py

# 22 课会创建本地 ChromaDB 数据库
python phase3/22_vector_db.py
```

## 关键技术点

### RAG Pipeline 流程

```
文档
  ↓ DocumentProcessor.load()     — 读取文件
  ↓ DocumentProcessor.clean()    — 清洗文本
  ↓ DocumentProcessor.chunk()    — 递归分块 (256 chars, 64 overlap)
  ↓ EmbeddingFunction.__call__() — 文本 → 向量
  ↓ ChromaDB.add()               — 存入向量库
  ...
用户查询
  ↓ EmbeddingFunction.__call__() — 查询 → 向量
  ↓ ChromaDB.query()             — 相似度检索 Top-K
  ↓ Reranker.filter()            — 阈值过滤 (min_score)
  ↓ Reranker.mmr_rerank()        — MMR 多样性重排
  ↓ Generator.generate()         — LLM 生成答案 + 引用
  → 答案
```

### 嵌入模型

课程默认使用 `all-MiniLM-L6-v2` (384 维)，是 sentence-transformers 生态中最轻量的模型，适合本地开发和快速迭代。

## 辅助文件

| 文件 | 说明 |
|------|------|
| `docs/` | 教学用文档目录 (5 个 Markdown/TXT 文件) |
| `chroma_db/` | ChromaDB 持久化存储目录 |
| `phase3/chroma_db/` | 另一个 ChromaDB 存储目录 (课程内代码可能创建) |

## 前置要求

- 完成 Phase 1-2
- 理解向量、余弦相似度的基本概念
- `pip install sentence-transformers chromadb numpy`
