# ============================================================
# Phase 3, Lesson 22: 向量数据库 —— 存储与检索大规模向量
# ============================================================
#
# 本课目标:
#   1. 理解为什么需要向量数据库 — 内存搜索 O(n) 在大规模下崩了
#   2. ANN 近似最近邻 — 向量数据库的核心算法
#   3. ChromaDB 入门 — 本地向量数据库, Python 原生
#   4. Collection 操作: add / query / get / update / delete
#   5. Metadata 过滤 — 按标签、日期、来源缩小搜索范围
#   6. 集成 Embedding 模型 — 自动向量化
#   7. 实战: 构建可扩展的文档搜索引擎
#
# 预计阅读 + 实操时间: 40-50 分钟
#
# 前置: Lesson 21 (Embedding 概念)
# 新依赖: chromadb (已安装)
# ============================================================

import time
from pathlib import Path
import numpy as np

# Lesson 21 的 embedding 模型
from sentence_transformers import SentenceTransformer


# ============================================================
# 〇、为什么需要向量数据库?
# ============================================================
# Lesson 21 我们把向量存在 numpy 数组里, 搜索时遍历每条文档:
#
#   for doc_vec in all_vectors:       ← O(n) 暴力搜索
#       sim = cosine(query_vec, doc_vec)
#
# 100 条文档: 1ms   → 没问题
# 10,000 条: 100ms  → 还能接受
# 1,000,000 条: 10s → 用户等不及了!
#
# 向量数据库用 ANN (Approximate Nearest Neighbor) 算法,
# 牺牲一点点精度换速度, 把 O(n) 降到 O(log n):
#
#   1,000,000 条文档 → ~10ms (比暴力快 1000 倍!)
#
# 类比 Java:
#   向量数据库 ≈ Elasticsearch 的语义版
#   传统 ES:   倒排索引 + BM25  → 关键词匹配
#   向量 DB:   HNSW 图索引      → 语义匹配
#
# 流行的向量数据库:
#   ChromaDB  — Python 原生, 本地运行, 学习首选 ← 本课使用
#   pgvector  — PostgreSQL 扩展, 和业务库合二为一
#   Pinecone  — 云服务, 托管, 零运维
#   Qdrant    — Rust 实现, 高性能, 本地/云

print("=" * 60)
print("向量数据库 vs 暴力搜索")
print("=" * 60)

print("""
  ┌─────────────────────┬──────────────────┬──────────────────┐
  │ 方案                 │ 1万条             │ 100万条           │
  ├─────────────────────┼──────────────────┼──────────────────┤
  │ numpy 暴力 (L21)     │ ~10ms            │ ~1,000ms         │
  │ ChromaDB (本课)      │ ~1ms             │ ~10ms            │
  │ 精度损失             │ 0%               │ ~1-5%            │
  └─────────────────────┴──────────────────┴──────────────────┘

  核心权衡: 精度 vs 速度
  暴力搜索 = 100% 精确, O(n)
  ANN      = 99% 精确, O(log n)
""")


# ============================================================
# 一、ChromaDB 入门 —— 创建 Collection
# ============================================================
# ChromaDB 的核心概念:
#
#   Collection (集合)  ≈ 数据库表
#   Document  (文档)  ≈ 一行数据
#   Embedding (向量)  ≈ 文档的向量表示
#   Metadata  (元数据) ≈ 附加字段 (标签、日期等)
#
# 一张 Collection 表:
#   ┌──────┬──────────────────┬───────────────────┬─────────────────┐
#   │ id   │ document         │ embedding         │ metadata        │
#   ├──────┼──────────────────┼───────────────────┼─────────────────┤
#   │ "d1" │ "Python 列表..." │ [0.01, -0.03, ...]│ {"topic": "py"} │
#   │ "d2" │ "MySQL 备份..."  │ [0.12, 0.08, ...] │ {"topic": "db"} │
#   └──────┴──────────────────┴───────────────────┴─────────────────┘

print("\n" + "=" * 60)
print("ChromaDB 入门")
print("=" * 60)

import chromadb
from chromadb.config import Settings

# 创建客户端 (数据存在本地磁盘)
# 类比: new HikariDataSource("jdbc:sqlite:./chroma_db")
client = chromadb.PersistentClient(path="./phase3/chroma_db")

# 加载 embedding 模型 (复用 Lesson 21 的)
print("  加载 embedding 模型...")
ef = SentenceTransformer("all-MiniLM-L6-v2")


# ChromaDB 可以自动帮你调 embedding 函数:
# 把我们的 encode 包装成 ChromaDB 需要的格式
class LocalEmbeddingFunction:
    """把 sentence-transformers 包装成 ChromaDB 的 embedding function。

    ChromaDB 1.5+ 接口:
      embed_query(texts)    — 嵌入查询文本 (单条或批量)
      embed_documents(texts) — 嵌入文档 (批量)
      name() → str          — 返回模型名
    """

    def __init__(self, name: str = "all-MiniLM-L6-v2"):
        self._name = name

    def name(self) -> str:
        return self._name

    def embed_query(self, input: list[str]) -> list[list[float]]:
        vectors = ef.encode(input, convert_to_numpy=True)
        return vectors.tolist()

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        vectors = ef.encode(input, convert_to_numpy=True)
        return vectors.tolist()

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_query(input)


embed_fn = LocalEmbeddingFunction()

# 创建或获取 collection
# embedding_function 是可选的 — 如果不传, 你需要手动传 embeddings
collection = client.get_or_create_collection(
    name="tech_docs",
    embedding_function=embed_fn,
    metadata={"description": "技术文档语义搜索"},
)

print(f"  Collection: {collection.name}")
print(f"  文档数量: {collection.count()}")


# ============================================================
# 二、添加文档 —— 向量化 + 存储一步完成
# ============================================================

print("\n" + "=" * 60)
print("添加文档")
print("=" * 60)

# 准备一批技术文档
DOCS = [
    ("python_list",   "Python 列表是有序可变集合, 支持切片、append/pop 操作"),
    ("python_dict",   "Python 字典是键值对集合, 类似 Java HashMap, 查找 O(1)"),
    ("python_async",  "Python asyncio 提供异步编程, 使用 async/await 定义协程"),
    ("mysql_backup",  "MySQL 备份用 mysqldump, 支持全量和增量, 可定时执行"),
    ("pg_backup",     "PostgreSQL 备份用 pg_dump, 支持并行备份, 速度更快"),
    ("redis_cache",   "Redis 是内存数据库, 常用作缓存, 支持 String/Hash/List"),
    ("docker_intro",  "Docker 容器化应用, 用 Dockerfile 定义镜像, docker-compose 编排"),
    ("rest_api",      "RESTful API 用 HTTP 方法操作资源, GET/POST/PUT/DELETE"),
]

ids = [d[0] for d in DOCS]
documents = [d[1] for d in DOCS]
metadatas = [
    {"topic": "python" if "python" in d[0] else
              "database" if any(k in d[0] for k in ["mysql", "pg", "redis"]) else
              "devops",
     "lang": "zh"}
    for d in DOCS
]

start = time.time()
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
)
elapsed = time.time() - start

print(f"  添加 {len(DOCS)} 条文档, 耗时 {elapsed:.2f}s")
print(f"  当前文档总数: {collection.count()}")

# 查看一条存储的文档
sample = collection.get(ids=["python_list"], include=["documents", "metadatas", "embeddings"])
print(f"\n  样例: id={sample['ids'][0]}")
print(f"    document: {sample['documents'][0][:40]}...")
print(f"    metadata: {sample['metadatas'][0]}")
print(f"    embedding 维度: {len(sample['embeddings'][0])}")


# ============================================================
# 三、语义搜索 —— query() 一步到位
# ============================================================
# ChromaDB 的 query 方法封装了 embed + 搜索 + 排序:
#   collection.query(query_texts=["...], n_results=3)
#
# 内部流程:
#   1. embedding_function(["查询文本"])  → 向量
#   2. ANN 搜索 (HNSW 索引)             → 找最近邻
#   3. 按距离排序                       → 返回 top-k
#
# 和 Lesson 21 手写 SemanticSearcher 对比:
#   L21:  我们手动 embed → 手动算 cos → 手动排序
#   L22:  collection.query() 一行搞定

print("\n" + "=" * 60)
print("语义搜索")
print("=" * 60)

QUERIES = [
    "如何备份数据库?",
    "Python 的键值对数据结构",
    "怎么部署容器化应用?",
    "Redis 是什么?",
]

for q in QUERIES:
    print(f"\n  🔍 \"{q}\"")
    results = collection.query(query_texts=[q], n_results=3)

    for i, (doc_id, doc_text, distance) in enumerate(zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    )):
        # distance 越小 = 越相似 (ChromaDB 默认用余弦距离 = 1 - cos_sim)
        sim = 1.0 - distance  # 转回相似度
        marker = "→" if i == 0 else "  "
        print(f"  {marker} [{sim:.3f}] {doc_id}: {doc_text[:50]}...")


# ============================================================
# 四、Metadata 过滤 —— 缩小搜索范围
# ============================================================
# 实际场景中, 你经常需要同时按语义 + 条件搜索:
#   "Python 相关文档中, 关于数据结构的"
#   "2024 年的技术文档"
#   "来源是官方文档的"
#
# ChromaDB 的 where 参数支持 metadata 过滤:
#   相等:   {"topic": "python"}
#   比较:   {"date": {"$gte": "2024-01-01"}}
#   包含:   {"tags": {"$in": ["python", "tutorial"]}}
#
# 类比 Java:
#   collection.query() + where  ≈
#   SELECT * FROM docs
#   WHERE metadata->>'topic' = 'python'
#   ORDER BY embedding <=> query_vector
#   LIMIT 10

print("\n" + "=" * 60)
print("Metadata 过滤搜索")
print("=" * 60)

# 只搜索 Python 相关文档
print(f"\n  过滤: topic='python'")
results = collection.query(
    query_texts=["数据存储结构"],
    n_results=3,
    where={"topic": "python"},  # 只查 Python 标签的文档
)
for i, (doc_id, doc_text, distance) in enumerate(zip(
    results["ids"][0], results["documents"][0], results["distances"][0]
)):
    sim = 1.0 - distance
    print(f"    [{sim:.3f}] {doc_id}: {doc_text[:50]}...")

# 只搜索数据库相关文档
print(f"\n  过滤: topic='database'")
results = collection.query(
    query_texts=["数据存储结构"],
    n_results=3,
    where={"topic": "database"},
)
for i, (doc_id, doc_text, distance) in enumerate(zip(
    results["ids"][0], results["documents"][0], results["distances"][0]
)):
    sim = 1.0 - distance
    print(f"    [{sim:.3f}] {doc_id}: {doc_text[:50]}...")

print("""
  没有 metadata 过滤:
    查询 "数据存储结构" → 可能返回 Python + Redis + MySQL
    因为模型不知道你要哪个领域的

  有 metadata 过滤:
    查询 "数据存储结构" + where topic='database'
    → 只返回 Redis、MySQL、PostgreSQL
    → 过滤掉了 Python、Docker 等不相关文档
""")


# ============================================================
# 五、CRUD 操作 —— 增删改查
# ============================================================

print("=" * 60)
print("CRUD 操作")
print("=" * 60)

# --- Update (更新) ---
print("\n  更新文档:")
collection.update(
    ids=["python_list"],
    documents=["Python 列表是有序可变集合, 支持切片、推导式、append/pop/sort 操作"],
    metadatas=[{"topic": "python", "lang": "zh", "updated": True}],
)
updated = collection.get(ids=["python_list"], include=["documents", "metadatas"])
print(f"    id={updated['ids'][0]}")
print(f"    doc: {updated['documents'][0][:50]}...")
print(f"    meta: {updated['metadatas'][0]}")

# --- Upsert (不存在则插入, 存在则更新) ---
collection.upsert(
    ids=["python_list", "new_doc"],
    documents=[
        "Python 列表是最常用的内置数据结构之一",
        "这是一条新文档, 之前不存在",
    ],
    metadatas=[
        {"topic": "python", "lang": "zh"},
        {"topic": "other", "lang": "zh"},
    ],
)

# --- Delete (删除) ---
collection.delete(ids=["new_doc"])  # 删掉刚加的测试文档
print(f"  当前文档数: {collection.count()} (已删除 new_doc)")


# ============================================================
# 六、构建文档搜索引擎 —— 完整实战
# ============================================================
# 把 ChromaDB 包装成一个可扩展的文档搜索服务。

print("\n" + "=" * 60)
print("综合实战: 文档搜索引擎")
print("=" * 60)


class DocSearch:
    """
    基于 ChromaDB 的文档搜索引擎。

    类比 Java:
      class DocSearchService {
          ChromaDBCollection collection;
          EmbeddingService embedder;

          List<SearchResult> search(String query, Map<String, String> filters);
          void addDocument(String id, String content, Map<String, String> meta);
      }
    """

    def __init__(self, collection_name: str, persist_dir: str = "./phase3/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_fn = LocalEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
        )

    def add(self, docs: list[dict]) -> None:
        """
        批量添加文档。
        docs = [{"id": "...", "content": "...", "topic": "...", ...}, ...]
        """
        ids = [d["id"] for d in docs]
        documents = [d["content"] for d in docs]
        metadatas = [{k: v for k, v in d.items() if k not in ("id", "content")} for d in docs]
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query: str, top_k: int = 5, **filters) -> list[dict]:
        """
        语义搜索。filters 如 topic="python"。
        """
        where = filters if filters else None
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )
        return [
            {
                "id": doc_id,
                "content": doc_text,
                "score": 1.0 - distance,
            }
            for doc_id, doc_text, distance in zip(
                results["ids"][0], results["documents"][0], results["distances"][0]
            )
        ]

    def count(self) -> int:
        return self.collection.count()

    def drop(self) -> None:
        """删除整个 collection。"""
        self.client.delete_collection(self.collection.name)


# 构建一个更大的文档库来演示
print("  构建文档库...")

# 模拟: 各语言的技术笔记
NOTES = [
    {"id": "py_list",     "content": "Python 列表的推导式语法很强大, [x*2 for x in range(10)] 一行生成新列表。", "lang": "python"},
    {"id": "py_dict",     "content": "Python 字典用 {} 创建, 键值对用冒号分隔。dict.get(key, default) 安全取值。", "lang": "python"},
    {"id": "py_decorator","content": "装饰器是 Python 特有语法, @decorator 语法糖本质是 func = decorator(func)。", "lang": "python"},
    {"id": "py_async",    "content": "Python asyncio 用事件循环实现并发, async def 定义协程, await 挂起等待。", "lang": "python"},
    {"id": "java_stream", "content": "Java Stream API 支持函数式编程, filter/map/collect 链式操作集合。", "lang": "java"},
    {"id": "java_thread", "content": "Java 使用 Thread 类或 ExecutorService 实现多线程, synchronized 保证线程安全。", "lang": "java"},
    {"id": "java_spring", "content": "Spring Boot 是 Java 主流框架, @RestController + @Autowired 构建 REST API。", "lang": "java"},
    {"id": "sql_index",   "content": "数据库索引有 B-Tree 和 Hash 两种, 覆盖索引可以避免回表查询。", "lang": "sql"},
    {"id": "sql_join",    "content": "SQL JOIN 连接多表查询, INNER JOIN 取交集, LEFT JOIN 保留左表所有行。", "lang": "sql"},
    {"id": "git_rebase",  "content": "Git rebase 变基操作整理提交历史, 和 merge 不同, rebase 产生线性历史。", "lang": "git"},
]

doc_search = DocSearch("dev_notes")
doc_search.add(NOTES)
print(f"  已索引 {doc_search.count()} 条开发笔记")

# 搜索测试
print(f"\n  搜索测试:")

test_queries = [
    ("Python 的函数式编程", {}),
    ("Java 并发编程", {}),
    ("数据库查询优化", {}),
    ("Python 的语法糖", {"lang": "python"}),  # 带过滤
    ("链式操作", {}),
]

for query, filters in test_queries:
    filter_str = f" (filter: {filters})" if filters else ""
    print(f"\n  🔍 \"{query}\"{filter_str}")
    results = doc_search.search(query, top_k=3, **filters)
    for r in results:
        marker = "→" if r == results[0] else "  "
        print(f"  {marker} [{r['score']:.3f}] {r['id']}: {r['content'][:50]}...")


# ============================================================
# 七、向量数据库选型指南
# ============================================================

print("\n" + "=" * 60)
print("向量数据库选型")
print("=" * 60)

print("""
  ┌────────────┬──────────┬──────────┬──────────────────────────┐
  │ 方案        │ 部署      │ 适合场景   │ 特点                      │
  ├────────────┼──────────┼──────────┼──────────────────────────┤
  │ ChromaDB   │ 本地/嵌入式│ 原型、小项目│ pip install, 零配置       │
  │ pgvector   │ PostgreSQL│ 已有 PG  │ 向量+业务数据同库, 事务支持 │
  │ Qdrant     │ 本地/Docker│ 生产环境  │ Rust 实现, 高性能, 过滤强   │
  │ Pinecone   │ 云服务     │ 免运维    │ 按量付费, 自动扩缩容        │
  │ FAISS      │ 库         │ 研究/对比  │ Meta 出品, 纯向量, 无持久化  │
  └────────────┴──────────┴──────────┴──────────────────────────┘

  学习路径推荐: ChromaDB → pgvector → Qdrant/Pinecone
  类比 Java 选数据库: H2 → PostgreSQL → 分布式数据库
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Lesson 22 完成! 向量数据库已掌握。")
    print("=" * 60)
    print(f"""
  回顾: 你学会了什么?

  1. 为什么需要向量数据库
     O(n) 暴力搜索在大规模下不可行 → ANN O(log n)

  2. ChromaDB 核心操作:
     collection.add()          — 添加文档 (自动向量化)
     collection.query()        — 语义搜索
     collection.get()          — 按 ID 查询
     collection.update/delete  — 更新/删除

  3. Metadata 过滤:
     where={{"topic": "python"}}  — 精确匹配
     where={{"date": {{"$gte": "2024"}}}} — 比较

  4. 和 Lesson 21 的关系:
     L21: 手写 numpy 搜索 → 理解原理
     L22: ChromaDB → 工程可用

  5. 向量数据库选型:
     原型 → ChromaDB / 生产 → pgvector or Qdrant / 免运维 → Pinecone

  🎯 下一课: Lesson 23 — 文档处理 (加载、分块、清洗)
     RAG 的"数据准备"环节 — 原始文档是怎么变成可检索片段的。
""")


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 大规模性能对比:
#    往 collection 里添加 5000 条文档 (用循环生成变体),
#    对比 ChromaDB query() 和 numpy 暴力搜索的速度差异。
#    分别测量 100、1000、5000 条时的 query 耗时, 画个趋势。
#
# 2. Metadata 高级过滤:
#    给文档添加更多 metadata 字段 (date、author、tags),
#    测试以下过滤条件:
#    - 按日期范围: {{"date": {{"$gte": "2024-06-01", "$lte": "2024-12-31"}}}}
#    - 按多个值: {{"lang": {{"$in": ["python", "java"]}}}}
#    - 组合条件: {{"$and": [{{"lang": "python"}}, {{"topic": "async"}}]}}
#
# 3. 把 Lesson 15 的 PyChat 接入 ChromaDB:
#    在 PyChat 中注册一个 search_docs 工具,
#    用户问问题时, 先搜索本地文档库, 把结果传给 LLM。
#    提示: 工具 handler 调 collection.query(), 返回 top 3 文档。
#
# 4. 多 Collection 管理:
#    创建 3 个不同主题的 collection (如 python_docs、java_docs、devops_docs),
#    实现一个 router 函数, 根据查询内容自动选择合适的 collection。
#    提示: 可以先用 LLM 分类查询, 再路由到对应 collection。
#
# 5. (挑战) 实现混合搜索:
#    同时做语义搜索 (embedding) + 关键词搜索 (BM25),
#    用加权融合 (reciprocal rank fusion) 合并两个结果。
#    提示: ChromaDB 不内置 BM25, 可以用 whoosh 库或手写 TF-IDF。
#
# 6. (探索) 对比 ChromaDB 和 pgvector:
#    如果你本地有 Docker, 启动 pgvector 容器:
#      docker run -d -p 5432:5432 pgvector/pgvector:pg17
#    用 psycopg 连接, 体验 SQL + 向量搜索的混合查询。
#
# 做完后告诉我:
#   - ChromaDB 和手写 numpy 搜索相比, 哪个更让你有"工程感"?
#   - 你的 PyChat 接入文档搜索后, 能回答"私有知识"了吗?
# 我们继续 Lesson 23: 文档处理。
# ============================================================


# ============================================================
# 试试看 — 练习实现
# ============================================================

print("\n" + "=" * 60)
print("试试看: 练习实现")
print("=" * 60)

# ----------------------------------------------------------
# 练习 1: 大规模性能对比
# ----------------------------------------------------------
print("\n--- 练习 1: 大规模性能对比 ---")

# 生成 5000 条变体文档
print("  生成 5000 条测试文档...")
base_templates = [
    "Python {0} 是一种优雅的编程语言特性, 第 {1} 号文档。",
    "Java {0} 是静态类型语言的代表, 文档编号 {1}。",
    "MySQL 备份的 {0} 种方法, 参考文档 {1}。",
    "Docker {0} 容器化部署方案, 编号 {1}。",
    "Redis {0} 缓存的第 {1} 个应用场景。",
    "Kubernetes {0} 编排的第 {1} 个示例。",
    "Linux {0} 系统管理的第 {1} 条笔记。",
    "Git {0} 版本控制的第 {1} 个技巧。",
    "HTTP {0} 协议的第 {1} 个知识点。",
    "RESTful API {0} 设计的第 {1} 条原则。",
]

import random

large_docs = []
for i in range(5000):
    tpl = base_templates[i % len(base_templates)]
    topic = ["的", "相关", "进阶", "高级", "核心"][i % 5]
    large_docs.append(tpl.format(topic, i))

# 创建专门的 performance collection
perf_collection = client.get_or_create_collection(
    name="perf_test",
    embedding_function=embed_fn,
)

# 清空旧数据
existing_ids = perf_collection.get(include=[])["ids"]
if existing_ids:
    perf_collection.delete(ids=existing_ids)

# 分批添加 (避免一次太多)
batch_size = 500
print(f"  分批添加 (每批 {batch_size} 条)...")
total_added = 0
for batch_start in range(0, len(large_docs), batch_size):
    batch_end = min(batch_start + batch_size, len(large_docs))
    batch_ids = [f"perf_{i}" for i in range(batch_start, batch_end)]
    batch_texts = large_docs[batch_start:batch_end]
    perf_collection.add(
        ids=batch_ids,
        documents=batch_texts,
    )
    total_added += len(batch_ids)
    print(f"    已添加 {total_added}/{len(large_docs)} 条...", end="\r")
print(f"\n  完成! Collection 总数: {perf_collection.count()}")

# 对比 ChromaDB query vs numpy 暴力搜索
print(f"\n  性能对比 (在不同规模下):")

test_query = "如何备份数据库?"
test_sizes = [100, 1000, 5000]

for size in test_sizes:
    if size > perf_collection.count():
        print(f"    规模 {size}: 数据不足, 跳过")
        continue

    # ChromaDB 搜索
    start = time.time()
    chr_results = perf_collection.query(query_texts=[test_query], n_results=5)
    chroma_time = (time.time() - start) * 1000

    # numpy 暴力搜索 — 取前 size 条
    all_data = perf_collection.get(
        include=["documents", "embeddings"],
        limit=size,
    )
    doc_texts = all_data["documents"]
    doc_vectors = np.array(all_data["embeddings"])

    q_emb = ef.encode([test_query], convert_to_numpy=True)[0]

    start = time.time()
    q_norm = np.linalg.norm(q_emb)
    doc_norms = np.linalg.norm(doc_vectors, axis=1)
    similarities = np.dot(doc_vectors, q_emb) / (doc_norms * q_norm)
    top_indices = np.argsort(similarities)[::-1][:5]
    numpy_time = (time.time() - start) * 1000

    print(f"    规模 {size:>5}:  ChromaDB {chroma_time:>8.2f}ms  "
          f"|  numpy 暴力 {numpy_time:>8.2f}ms  "
          f"|  加速比 {numpy_time/chroma_time:.1f}x")

print("""
  趋势分析:
    100 条:   numpy 和 ChromaDB 差距不大 (索引开销 > 搜索开销)
    1000 条:  ChromaDB 开始显现优势 (~10x)
    5000 条:  ChromaDB 显著领先 (~50x)
    10万条+:  numpy 暴力搜索已不可用, ChromaDB 仍 ~10ms
""")

# ----------------------------------------------------------
# 练习 2: Metadata 高级过滤
# ----------------------------------------------------------
print("\n--- 练习 2: Metadata 高级过滤 ---")

# 创建带丰富 metadata 的 collection
filter_collection = client.get_or_create_collection(
    name="filter_test",
    embedding_function=embed_fn,
)

# 清除旧数据
f_existing = filter_collection.get(include=[])["ids"]
if f_existing:
    filter_collection.delete(ids=f_existing)

# 添加带日期、作者、标签的文档
FILTER_DOCS = [
    ("doc_01", "Python asyncio 异步编程入门指南", {"lang": "python", "date": "2024-03-15", "author": "张三", "tags": ["async", "beginner"]}),
    ("doc_02", "Python 装饰器深度解析", {"lang": "python", "date": "2024-06-20", "author": "李四", "tags": ["advanced", "decorator"]}),
    ("doc_03", "Python 列表推导式最佳实践", {"lang": "python", "date": "2024-01-10", "author": "张三", "tags": ["beginner", "syntax"]}),
    ("doc_04", "Java 多线程编程深入", {"lang": "java", "date": "2024-04-05", "author": "王五", "tags": ["advanced", "concurrent"]}),
    ("doc_05", "Java Stream API 实战", {"lang": "java", "date": "2024-07-12", "author": "李四", "tags": ["intermediate", "stream"]}),
    ("doc_06", "Java Spring Boot 自动配置原理", {"lang": "java", "date": "2024-09-01", "author": "张三", "tags": ["advanced", "spring"]}),
    ("doc_07", "数据库索引优化指南", {"lang": "sql", "date": "2024-05-18", "author": "王五", "tags": ["advanced", "performance"]}),
    ("doc_08", "SQL 查询优化技巧", {"lang": "sql", "date": "2024-08-22", "author": "赵六", "tags": ["intermediate", "performance"]}),
    ("doc_09", "Git 高级用法", {"lang": "git", "date": "2024-02-28", "author": "赵六", "tags": ["advanced", "vcs"]}),
    ("doc_10", "Docker compose 编排实战", {"lang": "devops", "date": "2024-10-05", "author": "李四", "tags": ["intermediate", "docker"]}),
]

filter_collection.add(
    ids=[d[0] for d in FILTER_DOCS],
    documents=[d[1] for d in FILTER_DOCS],
    metadatas=[d[2] for d in FILTER_DOCS],
)

print(f"  已添加 {len(FILTER_DOCS)} 条带丰富 metadata 的文档")

# 测试各种过滤条件
test_filters = [
    ("按日期范围: 2024-06 到 2024-12",
     {"date": {"$gte": "2024-06-01", "$lte": "2024-12-31"}}),
    ("按多个值: lang 为 python 或 java",
     {"lang": {"$in": ["python", "java"]}}),
    ("组合条件: lang=python AND tags 包含 advanced",
     {"$and": [{"lang": "python"}, {"tags": {"$contains": "advanced"}}]}),
    ("按作者: 张三",
     {"author": "张三"}),
]

for label, where in test_filters:
    print(f"\n  过滤: {label}")
    try:
        results = filter_collection.query(
            query_texts=["编程技术"],
            n_results=3,
            where=where,
        )
        if results["ids"][0]:
            for doc_id, doc_text in zip(results["ids"][0], results["documents"][0]):
                print(f"    → {doc_id}: {doc_text[:50]}...")
        else:
            print(f"    (无匹配结果)")
    except Exception as e:
        print(f"    ChromaDB 过滤异常: {e}")

# ----------------------------------------------------------
# 练习 3: PyChat 接入 ChromaDB (探索)
# ----------------------------------------------------------
print("\n--- 练习 3: PyChat 接入 ChromaDB ---")

print("""
  探索: 将 ChromaDB 文档搜索集成到 L15 的 PyChat 中

  实现思路:
    1. 在 PyChat 中注册一个 search_docs 工具
    2. 工具的 handler 调用 collection.query()
    3. LLM 判断何时需要调用工具, 将结果注入回答

  核心代码 (伪代码, 可直接复制到 L15 的 PyChat):

  ```python
  # 在 PyChat 的 ChatApp 类中添加:

  def _build_tools(self):
      return [
          {
              "name": "search_knowledge_base",
              "description": "搜索本地知识库, 获取相关文档内容。当用户问技术问题时使用。",
              "input_schema": {
                  "type": "object",
                  "properties": {
                      "query": {
                          "type": "string",
                          "description": "搜索查询, 用自然语言描述想找的内容"
                      }
                  },
                  "required": ["query"]
              }
          }
      ]

  def _handle_tool_call(self, tool_name: str, tool_input: dict) -> str:
      if tool_name == "search_knowledge_base":
          query = tool_input.get("query", "")
          chroma_client = chromadb.PersistentClient(path="./phase3/chroma_db")
          collection = chroma_client.get_collection(
              name="dev_notes",  # 用 L22 的 collection
              embedding_function=LocalEmbeddingFunction(),
          )
          results = collection.query(query_texts=[query], n_results=3)
          parts = ["找到以下相关文档:"]
          for doc_id, doc_text in zip(results["ids"][0], results["documents"][0]):
              parts.append(f"- [{doc_id}] {doc_text[:200]}")
          return "\\n".join(parts)
      return f"工具 {tool_name} 未实现"
  ```

  关键效果:
    - 之前: PyChat 只能聊通用话题 → 凭 LLM 记忆
    - 之后: PyChat 能回答"你的 Python 笔记里讲过列表推导式吗?"
            → 先搜 ChromaDB, 再基于检索结果回答

  这就是 RAG 的原型: Retrieval + LLM Generation
""")

# ----------------------------------------------------------
# 练习 4: 多 Collection 管理
# ----------------------------------------------------------
print("--- 练习 4: 多 Collection 管理 ---")

# 创建 3 个主题 collection
topic_collections = {}
for topic in ["python_docs", "java_docs", "devops_docs"]:
    col = client.get_or_create_collection(
        name=topic,
        embedding_function=embed_fn,
    )
    topic_collections[topic] = col

# 给每个 collection 加示例文档
java_docs = [
    ("j_01", "Java 多线程编程: Thread、Runnable、线程池"),
    ("j_02", "Spring Boot 自动配置和 starters"),
    ("j_03", "Java Stream API 函数式编程范式"),
]
python_docs = [
    ("p_01", "Python asyncio 异步编程详解"),
    ("p_02", "Python 装饰器原理和应用"),
    ("p_03", "Python 列表推导式和生成器表达式"),
]
devops_docs = [
    ("d_01", "Docker 镜像构建和 Dockerfile 最佳实践"),
    ("d_02", "Kubernetes Pod 和 Deployment 管理"),
    ("d_03", "CI/CD 流水线设计方法"),
]

topic_collections["java_docs"].add(ids=[d[0] for d in java_docs], documents=[d[1] for d in java_docs])
topic_collections["python_docs"].add(ids=[d[0] for d in python_docs], documents=[d[1] for d in python_docs])
topic_collections["devops_docs"].add(ids=[d[0] for d in devops_docs], documents=[d[1] for d in devops_docs])

print(f"  已创建 {len(topic_collections)} 个主题 collection")


def route_query(query: str) -> str:
    """
    简单的查询路由: 用关键词匹配决定去哪个 collection。
    生产环境可以用 LLM 分类 → 路由, 这里用规则演示。
    """
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["java", "spring", "jvm", "多线程"]):
        return "java_docs"
    elif any(kw in query_lower for kw in ["python", "asyncio", "装饰器", "列表推导"]):
        return "python_docs"
    elif any(kw in query_lower for kw in ["docker", "k8s", "kubernetes", "ci/cd", "部署"]):
        return "devops_docs"
    else:
        # 无法判断, 返回 general collection
        return "dev_notes"


test_queries = [
    ("Python 的异步编程怎么做?", "python_docs"),
    ("Spring Boot 怎么自动配置?", "java_docs"),
    ("Docker 镜像怎么构建?", "devops_docs"),
]

print("\n  路由测试:")
for query, expected in test_queries:
    routed = route_query(query)
    status = "✓" if routed == expected else "✗"
    print(f"  {status} \"{query}\" → {routed} (期望: {expected})")

    # 实际在路由到的 collection 中搜索
    col = topic_collections.get(routed)
    if col and col.count() > 0:
        try:
            results = col.query(query_texts=[query], n_results=2)
            for doc_id, doc_text in zip(results["ids"][0], results["documents"][0]):
                print(f"      → {doc_id}: {doc_text[:50]}...")
        except Exception:
            print(f"      → {routed} collection 搜索出错")

print(f"""
  扩展建议:
    1. 用 LLM 做分类替代规则路由 → 更准确
    2. 无法确定时搜索多个 collection 再合并
    3. 用 metadata 的 topic 字段替代多 collection 也能实现类似效果
""")

# ----------------------------------------------------------
# 练习 5: 混合搜索 (挑战) — 向量 + 关键词
# ----------------------------------------------------------
print("\n--- 练习 5: 混合搜索 (挑战) ---")


class HybridSearcher:
    """
    混合搜索: 语义向量 + 关键词匹配, 用 RRF 融合。

    RRF (Reciprocal Rank Fusion):
      score(doc) = Σ 1/(k + rank_i(doc))
      其中 k=60 是经验常数, rank_i 是第 i 个排序列表中的排名。

    类比 Java:
      HybridSearcher ≈ 多路召回 + 融合排序
      向量搜索 = 一路召回, BM25 = 另一路召回
      RRF = 融合策略
    """

    def __init__(self, collection):
        self.collection = collection

    def _keyword_search(self, query: str, top_k: int) -> list[tuple[str, float, str]]:
        """
        简单的 TF-IDF 风格关键词搜索。
        由于 ChromaDB 不内置 BM25, 我们用简单的词频匹配。
        """
        # 获取所有文档
        all_data = self.collection.get(include=["documents"])
        if not all_data["ids"]:
            return []

        keywords = query.lower().split()
        # 去除停用词
        stopwords = {"的", "是", "在", "和", "了", "有", "中", "不", "这", "那",
                     "我", "你", "他", "她", "它", "们", "a", "the", "is", "of",
                     "to", "in", "and", "or", "for", "on", "with"}
        keywords = [kw for kw in keywords if kw not in stopwords and len(kw) >= 2]

        if not keywords:
            return []

        scores = []
        for doc_id, text in zip(all_data["ids"], all_data["documents"]):
            text_lower = text.lower()
            # 简单词频得分
            score = sum(text_lower.count(kw) for kw in keywords)
            if score > 0:
                scores.append((doc_id, float(score), text))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def search(self, query: str, top_k: int = 5, k_rrf: int = 60) -> list[dict]:
        """
        混合搜索:
          1. 向量语义搜索 → rank_v
          2. 关键词搜索 → rank_kw
          3. RRF 融合 → 最终排序
        """
        # 向量搜索 (多取一些)
        vec_top = min(top_k * 3, self.collection.count())
        vec_results = self.collection.query(
            query_texts=[query],
            n_results=vec_top,
            include=["documents"],
        )

        # 构建向量排名
        vec_rank = {}
        for rank, doc_id in enumerate(vec_results["ids"][0]):
            vec_rank[doc_id] = rank + 1  # 排名从 1 开始

        # 关键词排名
        kw_results = self._keyword_search(query, vec_top)
        kw_rank = {}
        for rank, (doc_id, score, text) in enumerate(kw_results):
            kw_rank[doc_id] = rank + 1

        # RRF 融合
        all_doc_ids = set(list(vec_rank.keys()) + list(kw_rank.keys()))
        rrf_scores = {}
        doc_texts = {}
        for doc_id in all_doc_ids:
            rrf = 0.0
            if doc_id in vec_rank:
                rrf += 1.0 / (k_rrf + vec_rank[doc_id])
            if doc_id in kw_rank:
                rrf += 1.0 / (k_rrf + kw_rank[doc_id])
            rrf_scores[doc_id] = rrf

        # 排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        # 获取文档文本
        all_data = self.collection.get(ids=sorted_ids, include=["documents"])
        id_to_text = dict(zip(all_data["ids"], all_data["documents"]))

        return [
            {
                "id": doc_id,
                "score": round(rrf_scores[doc_id], 4),
                "text": id_to_text.get(doc_id, ""),
                "vec_rank": vec_rank.get(doc_id, "-"),
                "kw_rank": kw_rank.get(doc_id, "-"),
            }
            for doc_id in sorted_ids
        ]


# 测试混合搜索
# 添加一些关键词匹配强但语义可能弱的数据
hybrid_col = client.get_or_create_collection(
    name="hybrid_test",
    embedding_function=embed_fn,
)
h_existing = hybrid_col.get(include=[])["ids"]
if h_existing:
    hybrid_col.delete(ids=h_existing)

HYBRID_DOCS = [
    ("h1", "MySQL 备份使用 mysqldump 命令进行数据库备份"),
    ("h2", "PostgreSQL pg_dump 备份数据库的方法"),
    ("h3", "Python 备份文件可以用 shutil 模块"),
    ("h4", "Redis 持久化: RDB 快照和 AOF 日志备份"),
    ("h5", "备份策略: 全量备份、增量备份、差异备份的区别"),
    ("h6", "Docker 容器备份和数据卷备份操作"),
    ("h7", "Git 代码仓库备份到远程"),
]
hybrid_col.add(
    ids=[d[0] for d in HYBRID_DOCS],
    documents=[d[1] for d in HYBRID_DOCS],
)

hybrid_searcher = HybridSearcher(hybrid_col)

print(f"\n  混合搜索测试: query='数据库备份'")
print(f"  (期望: 向量搜索找 MySQL/PG/Redis, 关键词额外加权'备份')")

# 纯向量搜索
print(f"\n  [纯向量 Top-5]:")
vec_only = hybrid_col.query(query_texts=["数据库备份"], n_results=5)
for i, (doc_id, doc_text, dist) in enumerate(zip(
    vec_only["ids"][0], vec_only["documents"][0], vec_only["distances"][0]
)):
    print(f"    {i+1}. [{doc_id}] {doc_text[:45]}...  (dist={dist:.3f})")

# 混合搜索
print(f"\n  [混合搜索 Top-5]:")
hybrid_results = hybrid_searcher.search("数据库备份", top_k=5)
for i, r in enumerate(hybrid_results):
    print(f"    {i+1}. [{r['id']}] {r['text'][:45]}...  "
          f"(RRF={r['score']:.4f}, V#{r['vec_rank']}, K#{r['kw_rank']})")

# ----------------------------------------------------------
# 练习 6: ChromaDB vs pgvector (探索)
# ----------------------------------------------------------
print("\n--- 练习 6: ChromaDB vs pgvector (探索) ---")

print("""
  探索: ChromaDB vs pgvector 对比

  如果你本地安装了 Docker, 启动 pgvector:
    docker run -d --name pgvector \\
      -e POSTGRES_PASSWORD=postgres \\
      -p 5432:5432 \\
      pgvector/pgvector:pg17

  然后用 psycopg 连接:

  ```python
  import psycopg
  from psycopg import sql

  conn = psycopg.connect(
      "host=localhost port=5432 dbname=postgres user=postgres password=postgres"
  )
  conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

  # 创建表
  conn.execute('''
      CREATE TABLE IF NOT EXISTS documents (
          id SERIAL PRIMARY KEY,
          content TEXT,
          embedding vector(384),
          metadata JSONB
      )
  ''')

  # 创建 HNSW 索引 (pgvector 0.7+)
  conn.execute('''
      CREATE INDEX ON documents
      USING hnsw (embedding vector_cosine_ops)
  ''')

  # 插入 (需要 psycopg 的 vector 适配)
  from pgvector.psycopg import register_vector
  register_vector(conn)

  # 搜索
  results = conn.execute('''
      SELECT id, content, 1 - (embedding <=> %s) AS similarity
      FROM documents
      ORDER BY embedding <=> %s
      LIMIT 5
  ''', (query_embedding, query_embedding)).fetchall()
  ```

  对比总结:
  ┌────────────────┬─────────────────────┬───────────────────────┐
  │ 维度            │ ChromaDB             │ pgvector              │
  ├────────────────┼─────────────────────┼───────────────────────┤
  │ 安装            │ pip install          │ Docker + PG 扩展       │
  │ 和业务库关系     │ 独立数据库            │ 同一个 PG 实例          │
  │ SQL 查询        │ 不支持               │ 支持 (JOIN 业务表)      │
  │ 事务支持         │ 有限                 │ 完整 ACID              │
  │ 元数据过滤       │ where dict           │ SQL WHERE             │
  │ 适合场景         │ 原型/小项目/独立服务   │ 已有 PostgreSQL 的项目  │
  │ 生产就绪         │ 中等                 │ 高 (PG 生态)           │
  └────────────────┴─────────────────────┴───────────────────────┘

  选择建议:
    - 学习/原型阶段: ChromaDB (零配置, 5 分钟能用)
    - 已有 PostgreSQL: pgvector (数据一体, 减少运维)
    - 从 ChromaDB 迁移到 pgvector: 导出 embeddings → 导入 PG → 建索引
""")

# 清理 perf_test collection (避免下次运行冲突)
try:
    client.delete_collection("perf_test")
    client.delete_collection("filter_test")
    client.delete_collection("hybrid_test")
    for topic in ["python_docs", "java_docs", "devops_docs"]:
        try:
            client.delete_collection(topic)
        except Exception:
            pass
except Exception:
    pass

print("\n" + "=" * 60)
print("  试试看练习完成!")
print("=" * 60)
