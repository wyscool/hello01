# ============================================================
# Phase 3, Lesson 24: 检索流水线 —— 串联 RAG 全流程
# ============================================================
#
# 本课目标:
#   1. 理解 RAG 完整流程: Query → Retrieve → Rerank → Generate
#   2. 检索: 向量搜索 + Metadata 过滤
#   3. 重排序: 相似度阈值过滤 + MMR 多样性重排
#   4. 上下文构建: 把检索结果拼成 LLM 能理解的 prompt
#   5. 生成: 带上下文的 LLM 调用 + 源引用
#   6. RAGPipeline: 一条龙编排
#   7. 端到端演示: 用 Phase 3 三课的文档构建知识库问答
#
# 预计阅读 + 实操时间: 50-60 分钟
#
# 前置: Lesson 21 (Embedding) + 22 (ChromaDB) + 23 (文档处理)
# 整合: Phase 2 的 Anthropic API 调用
# ============================================================

import os
import sys
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# ============================================================
# 〇、环境准备
# ============================================================

# --- Anthropic API ---
from anthropic import Anthropic
from anthropic.types import Message

api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")

client_kwargs = {"api_key": api_key} if api_key else {}
if base_url:
    client_kwargs["base_url"] = base_url
llm_client = Anthropic(**client_kwargs)


def _get_text(response: Message) -> str:
    parts = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts)


def llm_ask(prompt: str, system: str | None = None,
            model: str = "claude-sonnet-4-6",
            max_tokens: int = 1024, temperature: float = 0.0) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        response = llm_client.messages.create(
            model=model, max_tokens=max_tokens,
            temperature=temperature, messages=messages,
        )
        return _get_text(response)
    except Exception as e:
        return f"[LLM 调用失败: {e}]"


try:
    llm_ask("ping", max_tokens=10)
    api_ok = True
    print("✅ API 连接正常\n")
except Exception:
    api_ok = False
    print("⚠️  API 不可用, 生成环节将以模拟模式运行\n")

# --- Embedding 模型 ---
from sentence_transformers import SentenceTransformer

print("  加载 embedding 模型...")
st_model = SentenceTransformer("all-MiniLM-L6-v2")


class EmbedFn:
    """ChromaDB 1.5+ embedding function 包装。"""

    def name(self) -> str:
        return "all-MiniLM-L6-v2"

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return st_model.encode(input, convert_to_numpy=True).tolist()

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return st_model.encode(input, convert_to_numpy=True).tolist()

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_query(input)


# --- ChromaDB ---
import chromadb

db_client = chromadb.PersistentClient(path="./phase3/chroma_db")

# --- numpy (用于 MMR 的相似度计算) ---
import numpy as np


# ============================================================
# 一、RAG 全流程概览
# ============================================================
# RAG (Retrieval-Augmented Generation) 是解决 LLM "幻觉" 的核心方案:
#
#   用户问: "MySQL 怎么备份?"
#   ┌─────────────────────────────────────────────────────┐
#   │                                                     │
#   │  ① Query:  "MySQL 怎么备份?"                        │
#   │       │                                             │
#   │       ▼                                             │
#   │  ② Embed:  [0.12, -0.34, 0.56, ...]  (向量化)      │
#   │       │                                             │
#   │       ▼                                             │
#   │  ③ Search: ChromaDB ANN 搜索  → Top-K 相关文档      │
#   │       │                                             │
#   │       ▼                                             │
#   │  ④ Rerank: 过滤低分 + MMR 多样性重排                 │
#   │       │                                             │
#   │       ▼                                             │
#   │  ⑤ Build Context: 拼接检索结果 → Prompt              │
#   │       │                                             │
#   │       ▼                                             │
#   │  ⑥ Generate: LLM 基于上下文生成答案                   │
#   │       │                                             │
#   │       ▼                                             │
#   │  ⑦ 返回: "MySQL 用 mysqldump 备份，命令是..."       │
#   │                                                     │
#   └─────────────────────────────────────────────────────┘
#
# 类比 Java:
#   ① Query       ≈ 用户 HTTP 请求到达 Controller
#   ② Embed       ≈ 把请求参数转成内部表示 (DTO → Domain)
#   ③ Search      ≈ Repository.findBySimilarity()  (DAO 层)
#   ④ Rerank      ≈ Service 层的后处理/排序逻辑
#   ⑤ Build       ≈ 模板引擎渲染 (Thymeleaf / JSP)
#   ⑥ Generate    ≈ 调用外部 API, 获取结果
#
# 和直接问 LLM 的关键区别:
#   直接问: LLM 凭记忆回答 → 可能过时、可能编造 (幻觉)
#   RAG:    LLM 基于我们提供的文档回答 → 可溯源、可更新

print("=" * 60)
print("RAG 检索流水线")
print("=" * 60)
print("""
  Query → Embed → Search → Rerank → Context → Generate → Answer
  ①       ②       ③        ④         ⑤          ⑥         ⑦
""")


# ============================================================
# 二、数据结构定义
# ============================================================

@dataclass
class SearchResult:
    """单条检索结果。

    类比 Java: 一个 POJO, 对应数据库查询的一行结果。
    """
    doc_id: str
    text: str
    score: float            # 相似度 (距离的倒数, 越高越相关)
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"Result(id={self.doc_id!r}, score={self.score:.4f}, "
                f"text={self.text[:50]!r}...)")


@dataclass
class RAGResponse:
    """RAG 完整响应。

    不仅包含答案, 还携带引用来源 — 这是 RAG 区别于普通 LLM 回答的关键。
    """
    query: str
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    latency_ms: float = 0.0

    def print(self):
        print(f"\n{'=' * 60}")
        print(f"  Q: {self.query}")
        print(f"{'=' * 60}")
        print(f"\n{self.answer}\n")
        if self.sources:
            print(f"{'=' * 60}")
            print(f"  参考来源 ({len(self.sources)}):")
            for i, src in enumerate(self.sources, 1):
                title = src.metadata.get("source", src.doc_id)
                print(f"  [{i}] {title}  (相关度: {src.score:.3f})")
        print(f"  耗时: {self.latency_ms:.0f}ms")


# ============================================================
# 三、检索模块 —— Retriever
# ============================================================
# 封装 ChromaDB 查询, 提供多种检索方式。
#
# 三种检索策略:
#   1. search()          — 基础向量搜索 (Top-K)
#   2. search_with_filter() — 带 metadata 过滤
#   3. search_mmr()      — MMR 多样性搜索 (后面讲)

class Retriever:
    """向量检索器。

    封装 ChromaDB collection 的 query 操作,
    把原始结果转为 SearchResult 列表。

    类比 Java:
      Retriever ≈ Spring Data Repository
      collection.query() ≈ JPA EntityManager.createQuery()
    """

    def __init__(self, collection):
        self.collection = collection

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """基础语义搜索。

        内部流程:
          1. ChromaDB 自动 embed query
          2. ANN 搜索
          3. 返回 top_k 个最相似文档
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        for doc_id, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB 默认返回 L2 距离, 越小越相似
            # 转为相似度分数: score = 1 / (1 + distance)
            score = 1.0 / (1.0 + dist)
            search_results.append(SearchResult(
                doc_id=doc_id,
                text=text,
                score=round(score, 4),
                metadata=meta or {},
            ))

        return search_results

    def search_with_filter(
        self, query: str, top_k: int = 5, where: dict | None = None
    ) -> list[SearchResult]:
        """带 metadata 过滤的搜索。

        where 参数示例:
          {"source": "redis_guide.md"}          → 只在 Redis 文档里搜
          {"source": {"$in": ["a.md", "b.md"]}} → 在多个文档里搜
          {"chunk": {"$gte": 2}}                → 只要 chunk_index >= 2

        类比: SQL WHERE 子句 — 缩小搜索范围
          SELECT * FROM docs WHERE source = 'redis_guide.md' ORDER BY similarity DESC
        """
        kwargs = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        search_results = []
        for doc_id, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 / (1.0 + dist)
            search_results.append(SearchResult(
                doc_id=doc_id,
                text=text,
                score=round(score, 4),
                metadata=meta or {},
            ))

        return search_results


# ============================================================
# 四、重排序模块 —— Reranker
# ============================================================
# 为什么需要重排序?
#   1. 向量相似度 ≠ 真实相关性 — 长的文本容易被"撞"到
#   2. Top-K 可能都是同一主题 — 缺乏多样性
#   3. 低分噪声 — 相关度太低的文档反而是干扰
#
# 两种后处理策略:
#   1. 相似度阈值过滤 — 去掉得分太低的
#   2. MMR (Maximal Marginal Relevance) — 平衡相关性和多样性

class Reranker:
    """检索结果后处理。

    类比 Java:
      Reranker ≈ Stream API 的 .filter().sorted().distinct()
      原始搜索结果 → filter(去噪) → sort(重排) → 最终结果
    """

    @staticmethod
    def filter_by_threshold(
        results: list[SearchResult], min_score: float = 0.5
    ) -> list[SearchResult]:
        """过滤低分结果。

        相似度太低 = 和查询基本不相关, 喂给 LLM 反而是噪声。
        """
        filtered = [r for r in results if r.score >= min_score]
        return filtered

    @staticmethod
    def mmr_rerank(
        query: str,
        results: list[SearchResult],
        top_k: int = 3,
        lambda_param: float = 0.7,
    ) -> list[SearchResult]:
        """MMR (Maximal Marginal Relevance) 多样性重排。

        MMR 的核心思想:
          每轮选一个 chunk, 既要和 query 相关, 又要和已选的 chunk 不重复。

          得分 = λ × sim(query, chunk) - (1-λ) × max(sim(chunk, 已选))

          λ=1.0 → 只看相关性 (退化为原始排序)
          λ=0.0 → 只看多样性 (全是不同主题)
          λ=0.7 → 偏相关性, 但避免重复 (推荐默认)

        为什么需要 MMR?
          假设你问 "Python 有什么特性?",
          Top-5 可能全是 python_intro.txt 的不同 chunk,
          而忽略了 database_backup.md 里也有 Python 相关内容。
          MMR 会让结果更多样, 覆盖不同来源。

        类比 Java:
          这就是搜索结果的 "去重 + 多样化",
          类似电商搜索 "手机" → 不要全是同一型号的不同颜色。
        """
        if len(results) <= top_k:
            return results

        # 为每个结果计算 embedding (用于 MMR 比较)
        texts = [r.text for r in results]
        embeddings = st_model.encode(texts, convert_to_numpy=True)
        query_emb = st_model.encode([query], convert_to_numpy=True)[0]

        selected: list[int] = []       # 已选索引
        remaining = list(range(len(results)))  # 候选索引

        for _ in range(min(top_k, len(results))):
            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                # 和 query 的相似度 (相关性)
                rel = float(np.dot(query_emb, embeddings[idx])
                            / (np.linalg.norm(query_emb) * np.linalg.norm(embeddings[idx])))

                # 和已选结果的最大相似度 (冗余度)
                red = 0.0
                if selected:
                    red = max(
                        float(np.dot(embeddings[idx], embeddings[s])
                              / (np.linalg.norm(embeddings[idx]) * np.linalg.norm(embeddings[s])))
                        for s in selected
                    )

                mmr = lambda_param * rel - (1 - lambda_param) * red
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return [results[i] for i in selected]


# ============================================================
# 五、上下文构建器 —— ContextBuilder
# ============================================================
# 把检索结果拼接成 LLM 能理解的 prompt。
#
# 这是 RAG 最关键的一步: 上下文怎么组织, 直接影响生成质量。
#
# 核心要素:
#   1. 系统指令 — 告诉 LLM 它的角色和规则
#   2. 检索上下文 — 文档片段 + 来源标记
#   3. 用户问题 — 原始 query

class ContextBuilder:
    """构建 RAG prompt。

    把 SearchResult[] 转成标准格式的 prompt。

    类比 Java:
      ContextBuilder ≈ Thymeleaf 模板引擎
      SearchResult[]  ≈ Model 数据
      system_prompt   ≈ 模板文件
    """

    DEFAULT_SYSTEM = """你是一个基于文档的知识库助手。回答规则:
1. 只根据提供的文档内容回答, 不要使用文档外的知识
2. 如果文档中没有相关信息, 明确说 "文档中未找到相关信息"
3. 回答时引用具体的文档来源
4. 保持简洁、准确"""

    @staticmethod
    def build(
        query: str,
        results: list[SearchResult],
        system_prompt: str | None = None,
    ) -> tuple[str, str]:
        """构建 system prompt 和 user prompt。

        Returns:
          (system_prompt, user_prompt) — 直接喂给 LLM
        """
        if system_prompt is None:
            system_prompt = ContextBuilder.DEFAULT_SYSTEM

        # 拼接检索到的文档
        context_parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown")
            context_parts.append(f"[文档{i}] 来源: {source}\n{r.text}")

        context = "\n\n---\n\n".join(context_parts)

        # 构建 user prompt
        user_prompt = f"""以下是相关的参考文档:

{context}

---
基于以上文档, 请回答用户的问题。

用户问题: {query}

请在你的回答末尾列出引用的文档编号。"""

        return system_prompt, user_prompt


# ============================================================
# 六、生成模块 —— Generator
# ============================================================
# 调用 LLM, 传入构建好的上下文, 生成最终答案。

class Generator:
    """LLM 生成器。

    类比 Java:
      Generator ≈ RestTemplate / WebClient
      封装了 HTTP 调用细节, 只暴露 ask() 接口。
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def generate(
        self, system_prompt: str, user_prompt: str,
        max_tokens: int = 1024, temperature: float = 0.0,
    ) -> str:
        """调用 LLM 生成回答。"""
        if not api_ok:
            return self._mock_generate(user_prompt)

        return llm_ask(
            prompt=user_prompt,
            system=system_prompt,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _mock_generate(self, user_prompt: str) -> str:
        """模拟生成 (API 不可用时)。"""
        # 从 prompt 中提取文档内容作为 "回答"
        lines = user_prompt.split("\n")
        doc_lines = []
        in_doc = False
        for line in lines:
            if line.startswith("[文档"):
                in_doc = True
            if in_doc:
                doc_lines.append(line)
        return ("[模拟模式] 基于检索到的文档, 以下是相关摘要:\n\n"
                + "\n".join(doc_lines[:8])
                + "\n\n(API 可用时将生成完整回答)")


# ============================================================
# 七、RAGPipeline —— 一条龙编排
# ============================================================
# 把 Retriever → Reranker → ContextBuilder → Generator 串起来。
#
# 这才是业务代码真正用的接口:
#
#   pipeline = RAGPipeline(collection)
#   response = pipeline.ask("MySQL 怎么备份?")
#   response.print()

class RAGPipeline:
    """RAG 完整流水线。

    封装了检索→重排→构建→生成的全流程。

    类比 Java:
      RAGPipeline ≈ @Service 类
      内部调用 Repository(Retriever) + Util(Reranker) + Template(ContextBuilder) + Client(Generator)
      对外只暴露一个 ask() 方法。
    """

    def __init__(
        self,
        collection,
        model: str = "claude-sonnet-4-6",
        top_k: int = 5,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
        min_score: float = 0.3,
    ):
        self.retriever = Retriever(collection)
        self.generator = Generator(model=model)

        self.top_k = top_k
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.min_score = min_score

        # 缓存 embedding 模型 (用于 MMR)
        self._ef = st_model

    def ask(
        self,
        query: str,
        where: dict | None = None,
        system_prompt: str | None = None,
    ) -> RAGResponse:
        """执行一次完整的 RAG 查询。

        Args:
          query: 用户问题
          where: 可选的 metadata 过滤 (缩小搜索范围)
          system_prompt: 自定义系统指令 (None 用默认)

        Returns:
          RAGResponse: 包含答案、来源、耗时
        """
        start = time.time()

        # ① 检索
        if where:
            results = self.retriever.search_with_filter(
                query, top_k=self.top_k * 2, where=where  # 多取一些, 给重排留空间
            )
        else:
            results = self.retriever.search(query, top_k=self.top_k * 2)

        if not results:
            latency = (time.time() - start) * 1000
            return RAGResponse(
                query=query,
                answer="未找到相关文档。",
                latency_ms=latency,
            )

        # ② 过滤低分
        results = Reranker.filter_by_threshold(results, self.min_score)

        if not results:
            latency = (time.time() - start) * 1000
            return RAGResponse(
                query=query,
                answer="找到的文档相关度太低, 无法给出可靠回答。",
                latency_ms=latency,
            )

        # ③ MMR 重排 (可选)
        if self.use_mmr:
            results = Reranker.mmr_rerank(
                query, results, top_k=self.top_k, lambda_param=self.mmr_lambda
            )
        else:
            results = results[:self.top_k]

        # ④ 构建上下文
        system_p, user_p = ContextBuilder.build(query, results, system_prompt)

        # ⑤ 生成
        answer = self.generator.generate(system_p, user_p)

        latency = (time.time() - start) * 1000
        return RAGResponse(
            query=query,
            answer=answer,
            sources=results,
            latency_ms=latency,
        )


# ============================================================
# 八、模块级演示 —— 检视每个环节
# ============================================================

print("\n" + "=" * 60)
print("模块演示: 检视每个环节")
print("=" * 60)

# 准备测试数据: 复用 Lesson 22 的 collection, 确保有数据
test_collection = db_client.get_or_create_collection(
    name="rag_demo",
    embedding_function=EmbedFn(),
)

# 如果为空则初始化
if test_collection.count() == 0:
    print("  初始化测试文档...")
    DEMO_DOCS = [
        ("py_list", "Python 列表是有序可变集合, 支持切片、append/pop 操作, 索引从 0 开始"),
        ("py_dict", "Python 字典是键值对集合, 类似 Java HashMap, 查找时间复杂度 O(1)"),
        ("py_async", "Python asyncio 提供异步编程能力, 使用 async/await 定义协程, event loop 调度执行"),
        ("mysql_bak", "MySQL 备份主要用 mysqldump 工具, 支持全量备份和增量备份, 可配合 crontab 定时执行"),
        ("pg_bak", "PostgreSQL 备份用 pg_dump, 支持并行备份 (-j 参数), 自定义格式 (-Fc) 支持压缩"),
        ("redis_intro", "Redis 是开源内存数据库, 核心数据结构包括 String、Hash、List、Set、Sorted Set"),
        ("redis_cache", "Redis 常用作缓存, 支持 TTL 过期, LRU 淘汰策略, 可大幅降低数据库负载"),
        ("docker_intro", "Docker 是容器化平台, 用 Dockerfile 定义镜像构建步骤, docker-compose 管理多容器应用"),
    ]
    test_collection.add(
        ids=[d[0] for d in DEMO_DOCS],
        documents=[d[1] for d in DEMO_DOCS],
        metadatas=[{"topic": (
            "python" if d[0].startswith("py") else
            "database" if any(k in d[0] for k in ["mysql", "pg", "redis"]) else
            "devops"
        )} for d in DEMO_DOCS],
    )

print(f"  Collection: {test_collection.name}, 文档数: {test_collection.count()}")

# --- 演示 1: 基础检索 ---
retriever = Retriever(test_collection)

print("\n  ① 基础检索: query='Python 如何异步编程?'")
results = retriever.search("Python 如何异步编程?", top_k=3)
for r in results:
    print(f"    {r}")

# --- 演示 2: Metadata 过滤检索 ---
print("\n  ② 过滤检索: query='备份方法', where={{'topic': 'database'}}")
filtered = retriever.search_with_filter(
    "备份方法", top_k=3, where={"topic": "database"}
)
for r in filtered:
    print(f"    {r}")

# --- 演示 3: MMR 重排 ---
print("\n  ③ MMR 重排: query='数据库相关技术'")
many_results = retriever.search("数据库相关技术", top_k=6)
print(f"    原始 Top-6:")
for r in many_results:
    print(f"      {r.doc_id:15s} score={r.score:.4f}")
mmr_results = Reranker.mmr_rerank("数据库相关技术", many_results, top_k=3, lambda_param=0.7)
print(f"    MMR 重排后 Top-3:")
for r in mmr_results:
    print(f"      {r.doc_id:15s} score={r.score:.4f}")

# --- 演示 4: 上下文构建 ---
print("\n  ④ 上下文构建预览:")
sys_p, usr_p = ContextBuilder.build("Python 异步怎么用?", results[:2])
print(f"    System prompt ({len(sys_p)} 字符): {sys_p[:80]}...")
print(f"    User prompt ({len(usr_p)} 字符):")
print(f"    {usr_p[:300]}...")

# --- 演示 5: 生成 ---
print("\n  ⑤ 生成 (LLM):")
if api_ok:
    gen = Generator()
    answer = gen.generate(sys_p, usr_p, max_tokens=300)
    print(f"    {answer[:300]}")
else:
    print("    (跳过, API 不可用)")


# ============================================================
# 九、端到端演示 —— 完整的 RAG 问答
# ============================================================
# 直接用前面已初始化的 test_collection (rag_demo),
# 对同一个 collection 跑完整流水线: 检索→重排→构建→生成

print("\n\n" + "=" * 60)
print("端到端演示: RAG 问答")
print("=" * 60)

pipeline = RAGPipeline(
    test_collection,
    model="claude-sonnet-4-6",
    top_k=3,
    use_mmr=True,
    mmr_lambda=0.7,
    min_score=0.3,
)

# 选 2 个有代表性的问题做端到端演示
questions = [
    "Python 的异步编程怎么实现?",
    "怎样备份 MySQL 数据库?",
]

for q in questions:
    response = pipeline.ask(q)
    response.print()
    print()  # 问题之间的间隔

print("  (端到端演示完成 — 检索→重排→构建→生成 全链路)\n")

# 补充: 如果有 L23 的 processed_docs 集合, 展示跨集合检索
try:
    kb_col = db_client.get_collection(name="processed_docs")
    if kb_col.count() > 0:
        print("  💡 检测到 L23 的 processed_docs 集合")
        print(f"     包含 {kb_col.count()} 个文档片段, 可以直接用 RAGPipeline 检索:")
        print(f"     pipeline2 = RAGPipeline(db_client.get_collection('processed_docs'))")
        print(f"     pipeline2.ask('Redis 支持哪些数据结构?')")
except Exception:
    pass  # processed_docs 可能不存在


# ============================================================
# 十、模拟模式 —— API 不可用时的完整演示
# ============================================================

print("\n" + "=" * 60)
print("模拟模式演示 (不依赖 LLM API)")
print("=" * 60)

# 用更简单的 collection 演示全流程
simple_pipeline = RAGPipeline(
    test_collection,  # 复用前面创建的 collection
    top_k=3,
    use_mmr=True,
)

test_questions = [
    "Python 列表怎么用?",
    "怎样备份数据库?",
    "Redis 和 Docker 有什么关系?",
]

for q in test_questions:
    print(f"\n  ❓ {q}")

    # ① 检索
    raw = retriever.search(q, top_k=5)
    print(f"  ① 检索: {len(raw)} 条结果")

    # ② 过滤
    filtered = Reranker.filter_by_threshold(raw, min_score=0.3)
    print(f"  ② 过滤: {len(filtered)} 条 (>0.3)")

    # ③ 重排
    final = Reranker.mmr_rerank(q, filtered, top_k=3)
    print(f"  ③ MMR: {len(final)} 条")

    # ④ 上下文
    sys_p, usr_p = ContextBuilder.build(q, final)
    print(f"  ④ 上下文: system={len(sys_p)}字, user={len(usr_p)}字")

    # ⑤ 显示检索到的文档片段
    for i, r in enumerate(final, 1):
        print(f"  [{i}] {r.doc_id}: {r.text[:60]}...")

print(f"""
  ✅ 以上展示了 RAG 流水线的完整检索链路 (不含 LLM 生成)。
  启用生成只需: pipeline.ask("问题") → 自动包含 LLM 回答 + 源引用
""")


# ============================================================
# 十一、和直接问 LLM 的对比
# ============================================================

print("=" * 60)
print("RAG vs 直接 LLM —— 关键区别")
print("=" * 60)
print("""
  维度        直接 LLM              RAG (本课)
  ──────────────────────────────────────────────
  知识来源     训练数据 (静态)        你的文档 (可更新)
  准确性       可能编造 (幻觉)        基于原文 (可溯源)
  时效性       截止训练日期           随时更新文档即可
  可控性       只能改 prompt          换文档 = 换知识
  成本         每次全量推理           检索缩小上下文 → 更省 token
  适用场景     通用问答               企业知识库、产品文档

  并非 RAG 替代 LLM, 而是增强 LLM:
    LLM 提供语言理解和生成能力
    RAG 提供准确、可更新的知识

  类比 Java:
    直接 LLM ≈ 纯内存计算 (快但不持久)
    RAG      ≈ 带数据库的 Service (数据持久、可查询)
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Lesson 24 完成! 检索流水线已掌握。")
    print("=" * 60)
    print(f"""
  回顾: 你学会了什么?

  1. Retriever      — 封装 ChromaDB, 支持基础搜索 + 过滤搜索
  2. Reranker       — 相似度阈值过滤 + MMR 多样性重排
  3. ContextBuilder — 检索结果 → 结构化 prompt
  4. Generator      — 带上下文的 LLM 调用
  5. RAGPipeline    — 一条龙编排, 一个 ask() 搞定全流程
  6. 端到端演示     — 从 docs/ 到智能问答

  RAG 完整架构:
    docs/ → DocumentProcessor(L23) → ChromaDB(L22)
                                          ↓
    用户提问 → Retriever → Reranker → ContextBuilder → LLM → 答案 + 来源

  🎯 下一课: Lesson 25 — 知识库 Q&A 系统
     Phase 3 收官之作! 把 21-24 课整合为完整的知识库问答应用,
     支持命令行交互、多种文档格式、流式输出。
""")


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 调整 MMR 参数观察效果:
#    用同一个 query, 对比 lambda_param=0.3, 0.7, 1.0 的结果:
#    - lambda=1.0: 最相关, 但可能重复
#    - lambda=0.3: 最多样, 但可能不够相关
#    - lambda=0.7: 推荐平衡值
#    什么时候你会用 lambda=0.3? (提示: 探索型搜索)
#
# 2. 实现简单的关键词 + 向量混合搜索:
#    在 Retriever 中加一个 hybrid_search 方法:
#    - 先向量搜索取 top_k*3 条
#    - 再用关键词匹配 (if keyword in text) 加分
#    - 最后按综合分数重排
#    对比纯向量搜索的结果。
#
# 3. 扩展 ContextBuilder 支持对话历史:
#    修改 build() 方法, 加上 messages_history 参数:
#    - 把历史问答对也拼进 prompt
#    - 让 LLM 知道上下文, 不会重复回答
#    提示: 在 user_prompt 开头加 "对话历史: ..."
#
# 4. 给 RAGResponse 加一个 "置信度":
#    根据检索结果的 score 计算整体置信度:
#    - confidence = avg(top_k scores)
#    - 如果 confidence < 0.4, 在回答中提示 "答案可信度较低"
#
# 5. (挑战) 实现查询改写 (Query Rewriting):
#    在检索之前, 先用 LLM 改写用户查询:
#    - "咋备份?" → "如何进行数据库备份?"
#    - "那个内存的数据库" → "Redis"
#    对比改写前后的检索结果, 哪个更准确?
#    提示: 调用 LLM 做改写, 不增加用户感知延迟。
#
# 6. (思考) 检索质量评估:
#    你怎么判断"检索结果好不好"?
#    - 写 5 个查询和它们的 "理想文档"
#    - 用 Retriever 搜索, 看理想文档是否在 Top-3
#    - 这其实就是 Lesson 41 要讲的评估 (evaluation)
#    在你的学习笔记中设计一个简单的评估方案。
#
# 做完后告诉我:
#   - MMR 在你的知识库里效果如何? 有没有比纯 Top-K 更好?
#   - 你觉得 RAG 最大的挑战是什么? (提示: 不是技术, 是数据)
# 我们继续 Lesson 25: 知识库 Q&A 系统 — Phase 3 大结局!
# ============================================================


# ============================================================
# 试试看 — 练习实现
# ============================================================

print("\n" + "=" * 60)
print("试试看: 练习实现")
print("=" * 60)

# 确保有可用的 collection 做测试
try:
    test_col = db_client.get_collection("rag_demo")
except Exception:
    test_col = db_client.get_or_create_collection(
        name="rag_demo",
        embedding_function=EmbedFn(),
    )

if test_col.count() == 0:
    print("  初始化测试数据...")
    DEMO_DOCS = [
        ("py_list", "Python 列表是有序可变集合, 支持切片、append/pop 操作, 索引从 0 开始"),
        ("py_dict", "Python 字典是键值对集合, 类似 Java HashMap, 查找时间复杂度 O(1)"),
        ("py_async", "Python asyncio 提供异步编程能力, 使用 async/await 定义协程, event loop 调度执行"),
        ("py_decorator", "Python 装饰器是修改函数行为的语法糖, @decorator 本质是 func = decorator(func)"),
        ("mysql_bak", "MySQL 备份主要用 mysqldump 工具, 支持全量备份和增量备份, 可配合 crontab 定时执行"),
        ("pg_bak", "PostgreSQL 备份用 pg_dump, 支持并行备份 (-j 参数), 自定义格式 (-Fc) 支持压缩"),
        ("redis_intro", "Redis 是开源内存数据库, 核心数据结构包括 String、Hash、List、Set、Sorted Set"),
        ("redis_cache", "Redis 常用作缓存, 支持 TTL 过期, LRU 淘汰策略, 可大幅降低数据库负载"),
        ("docker_intro", "Docker 是容器化平台, 用 Dockerfile 定义镜像构建步骤, docker-compose 管理多容器应用"),
        ("k8s_intro", "Kubernetes 是容器编排平台, 管理 Pod、Service、Deployment 等资源对象"),
    ]
    test_col.add(
        ids=[d[0] for d in DEMO_DOCS],
        documents=[d[1] for d in DEMO_DOCS],
        metadatas=[{"topic": d[0].split("_")[0]} for d in DEMO_DOCS],
    )

retriever = Retriever(test_col)

# ----------------------------------------------------------
# 练习 1: MMR 参数调优
# ----------------------------------------------------------
print("\n--- 练习 1: MMR 参数调优 ---")

# 同一个 query, 对比不同的 lambda
mmr_test_query = "Python 有哪些核心特性?"

# 先取更多结果作为候选
raw_results = retriever.search(mmr_test_query, top_k=10)
print(f"  Query: \"{mmr_test_query}\"")
print(f"\n  原始 Top-10:")
for i, r in enumerate(raw_results):
    print(f"    {i+1}. [{r.doc_id}] {r.text[:50]}... (score={r.score:.4f})")

print(f"\n  MMR 重排对比 (不同 λ):")
for lam in [1.0, 0.7, 0.3]:
    mmr_results = Reranker.mmr_rerank(
        mmr_test_query, raw_results, top_k=5, lambda_param=lam
    )
    sources = set()
    for r in mmr_results:
        sources.add(r.doc_id.split("_")[0] if "_" in r.doc_id else r.doc_id)
    print(f"\n  λ={lam}: {len(mmr_results)} 条, 来源: {sources}")
    for i, r in enumerate(mmr_results):
        print(f"    {i+1}. [{r.doc_id}] score={r.score:.4f} {r.text[:50]}...")

print(f"""
  MMR λ 参数指南:
    λ=1.0: 纯相关性 — 所有结果可能来自同一文档 (缺乏多样性)
            适合: "精确查找" (我就知道要看哪个文档)
    λ=0.7: 平衡 — 优先相关, 但避免重复 (推荐默认)
            适合: 大多数问答场景
    λ=0.3: 偏多样性 — 覆盖不同来源, 相关性可能降低
            适合: "探索型搜索" (我想了解有哪些相关内容)
""")

# ----------------------------------------------------------
# 练习 2: 关键词 + 向量混合搜索
# ----------------------------------------------------------
print("\n--- 练习 2: 关键词 + 向量混合搜索 ---")


class HybridRetriever(Retriever):
    """
    扩展 Retriever, 增加关键词 + 向量混合搜索。

    算法:
      1. 向量搜索取 top_k*3 条候选
      2. 对候选做关键词匹配, 计算加分
      3. 综合分数重新排序
    """

    def hybrid_search(self, query: str, top_k: int = 5,
                      kw_weight: float = 0.3) -> list[SearchResult]:
        """
        混合搜索: 向量相似度 + 关键词匹配。

        kw_weight: 关键词加分的权重 (0.0 = 纯向量, 1.0 = 纯关键词)
        """
        # ① 向量搜索 (多取一些)
        candidates = self.search(query, top_k=top_k * 3)

        if not candidates:
            return []

        # ② 提取关键词 (简单分词)
        keywords = [kw.lower() for kw in query.replace("?", " ").replace("?", " ").split()
                    if len(kw) >= 2]
        # 去除常见停用词
        stopwords = {"的", "是", "在", "和", "了", "有", "中", "不", "这", "那",
                     "怎么", "什么", "怎样", "如何", "为什么", "可以"}
        keywords = [kw for kw in keywords if kw not in stopwords]

        # ③ 计算关键词加分
        if keywords:
            for r in candidates:
                text_lower = r.text.lower()
                # 关键词命中率
                hits = sum(1 for kw in keywords if kw in text_lower)
                kw_score = hits / max(len(keywords), 1)
                # 综合分数: 向量分 + 关键词加分
                r.score = (1 - kw_weight) * r.score + kw_weight * kw_score

        # ④ 重新排序
        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[:top_k]


hybrid_retriever = HybridRetriever(test_col)

# 选一个关键词特征明显的查询
hybrid_query = "备份 MySQL 数据库"
print(f"  Query: \"{hybrid_query}\"")
print(f"  关键词: ['备份', 'MySQL', '数据库']")

print(f"\n  [纯向量搜索 Top-5]:")
vec_results = retriever.search(hybrid_query, top_k=5)
for i, r in enumerate(vec_results):
    print(f"    {i+1}. [{r.doc_id}] score={r.score:.4f} {r.text[:50]}...")

print(f"\n  [混合搜索 Top-5 (kw_weight=0.3)]:")
hyb_results = hybrid_retriever.hybrid_search(hybrid_query, top_k=5, kw_weight=0.3)
for i, r in enumerate(hyb_results):
    print(f"    {i+1}. [{r.doc_id}] score={r.score:.4f} {r.text[:50]}...")

print(f"""
  混合搜索原理:
    纯向量: 可能把 "备份" 相关的都找出来 (包括 PG、Redis)
    混合:   关键词 "MySQL" 加分 → MySQL 备份排得更靠前

  关键词权重调优:
    kw_weight=0.0: 纯向量 (语义理解强, 但可能忽略精确术语)
    kw_weight=0.3: 推荐 (语义 + 术语兼顾)
    kw_weight=0.7: 偏关键词 (精确匹配, 但失去语义理解)
""")

# ----------------------------------------------------------
# 练习 3: ContextBuilder 支持对话历史
# ----------------------------------------------------------
print("\n--- 练习 3: ContextBuilder 支持对话历史 ---")


class ContextBuilderWithHistory(ContextBuilder):
    """
    扩展 ContextBuilder, 支持多轮对话历史。

    把历史问答对也拼进 prompt, 让 LLM 知道上下文,
    避免重复回答或答非所问。
    """

    @staticmethod
    def build(
        query: str,
        results: list[SearchResult],
        system_prompt: str | None = None,
        history: list[dict] | None = None,
    ) -> tuple[str, str]:
        """
        构建 prompt, 加入对话历史。

        history 格式: [{"role": "user", "content": "..."},
                       {"role": "assistant", "content": "..."}]
        只保留最近的 N 轮, 避免 prompt 过长。
        """
        if system_prompt is None:
            system_prompt = ContextBuilder.DEFAULT_SYSTEM

        # 拼接检索到的文档 (和父类一样)
        context_parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown")
            context_parts.append(f"[文档{i}] 来源: {source}\n{r.text}")

        context = "\n\n---\n\n".join(context_parts) if context_parts else "(无相关文档)"

        # 构建历史摘要
        history_text = ""
        if history:
            recent = history[-6:]  # 最多 3 轮 (6 条消息)
            history_parts = []
            for msg in recent:
                role_label = "用户" if msg["role"] == "user" else "助手"
                content_preview = msg["content"][:150]
                history_parts.append(f"[{role_label}]: {content_preview}")
            if history_parts:
                history_text = "对话历史:\n" + "\n".join(history_parts) + "\n\n"

        # 构建 user prompt
        user_prompt = f"""{history_text}以下是相关的参考文档:

{context}

---
基于以上文档和对话历史, 请回答用户的问题。
如果问题是对上一轮回答的追问, 请结合历史理解意图。

用户问题: {query}

请在你的回答末尾列出引用的文档编号。"""

        return system_prompt, user_prompt


# 演示: 模拟多轮对话
print("  模拟多轮对话:")

# 第一轮
history = []
sys_p1, usr_p1 = ContextBuilderWithHistory.build(
    "Python 异步编程怎么实现?",
    retriever.search("Python 异步编程", top_k=2),
    history=history,
)
history.append({"role": "user", "content": "Python 异步编程怎么实现?"})
if api_ok:
    ans1 = llm_ask(usr_p1, system=sys_p1, max_tokens=200)
    print(f"  Q1: Python 异步编程怎么实现?")
    print(f"  A1: {ans1[:150]}...")
else:
    ans1 = "[模拟回答] 基于文档, Python asyncio 使用 async/await 定义协程。"
    print(f"  A1: {ans1}")
history.append({"role": "assistant", "content": ans1})

# 第二轮 (追问, 省略主语)
sys_p2, usr_p2 = ContextBuilderWithHistory.build(
    "那它的性能怎么样?",  # 省略了 "asyncio"
    retriever.search("asyncio 性能", top_k=2),
    history=history,
)
history.append({"role": "user", "content": "那它的性能怎么样?"})
print(f"\n  Q2: 那它的性能怎么样?  ← 省略主语, 依赖历史")
print(f"  [Prompt 中包含历史, 前 {len(history)} 条消息]")

# 展示 prompt 中历史部分的长度
history_in_prompt = "对话历史:" in usr_p2
print(f"  Prompt 包含历史: {history_in_prompt}")

print(f"""
  对话历史的价值:
    无历史: "那它的性能怎么样?" → LLM 困惑: "它"是什么?
    有历史: "那它的性能怎么样?" → LLM 知道: asyncio 的性能

  工程注意:
    1. 历史只保留最近 N 轮 (避免 prompt 过长)
    2. 历史内容做截断 (长回答取前 150 字)
    3. 如果用户明确切换话题, 清空历史
""")

# ----------------------------------------------------------
# 练习 4: RAGResponse 加置信度
# ----------------------------------------------------------
print("\n--- 练习 4: RAGResponse 置信度 ---")


@dataclass
class RAGResponseWithConfidence(RAGResponse):
    """
    扩展 RAGResponse, 增加置信度评估。
    """
    confidence: float = 0.0
    confidence_level: str = "unknown"

    @staticmethod
    def compute_confidence(sources: list[SearchResult]) -> tuple[float, str]:
        """
        根据检索结果的 score 计算置信度。

        算法: 取 Top-K scores 的加权平均
          confidence = avg(top_k scores)

        阈值:
          >= 0.7: high   (高度可信)
          >= 0.4: medium (中等可信)
          < 0.4:  low    (可信度较低)
        """
        if not sources:
            return 0.0, "none"

        scores = [s.score for s in sources]
        avg_score = sum(scores) / len(scores)

        # 也考虑最高分 (top-1 特别重要)
        top_score = scores[0] if scores else 0.0

        # 加权: 70% top-1 + 30% avg
        weighted = 0.7 * top_score + 0.3 * avg_score

        if weighted >= 0.7:
            level = "high"
        elif weighted >= 0.4:
            level = "medium"
        else:
            level = "low"

        return round(weighted, 4), level

    def print(self):
        super().print()
        level_emoji = {"high": "高", "medium": "中", "low": "低", "none": "无"}
        level_text = level_emoji.get(self.confidence_level, "?")
        print(f"  置信度: {self.confidence:.3f} ({level_text})")


# 测试置信度
print("  置信度测试:")

test_cases = [
    "Python 列表怎么用?",
    "量子计算机的原理是什么?",  # 知识库没有的内容
    "备份数据库",                # 可能中等
]

for query in test_cases:
    results = retriever.search(query, top_k=3)
    confidence, level = RAGResponseWithConfidence.compute_confidence(results)

    print(f"\n  Q: \"{query}\"")
    print(f"    检索到 {len(results)} 条, Top-1 score: {results[0].score:.4f}" if results else "    无结果")
    print(f"    置信度: {confidence:.3f} ({level})")

    # 演示 RAGResponse 使用
    response = RAGResponseWithConfidence(
        query=query,
        answer=f"[模拟] 关于 '{query}' 的回答",
        sources=results,
        confidence=confidence,
        confidence_level=level,
    )
    response.print()

# ----------------------------------------------------------
# 练习 5: 查询改写 (挑战)
# ----------------------------------------------------------
print("\n--- 练习 5: 查询改写 (挑战) ---")


class QueryRewriter:
    """
    查询改写器: 把用户的口语化/模糊查询改写成更精确的检索查询。

    改写策略:
      1. 口语 → 正式: "咋备份?" → "如何进行数据库备份"
      2. 代词 → 实体: "那个内存的数据库" → "Redis"
      3. 多义消歧: "Python 怎么跑?" → "Python 程序的执行方式" (不是蛇)

    可以用 LLM 或规则实现。这里演示两种方式。
    """

    @staticmethod
    def rule_based(query: str) -> str:
        """
        基于规则的查询改写 (简单, 零延迟)。

        适合常见的口语/缩写映射。
        """
        rewrites = {
            "咋": "怎么",
            "咋样": "怎么样",
            "啥": "什么",
            "弄": "操作",
            "跑": "执行/运行",
            "挂了": "出错/异常",
            "崩了": "崩溃",
            "那个内存的数据库": "Redis 内存数据库",
            "容器那个": "Docker 容器",
            "py": "Python",
            "pg": "PostgreSQL",
            "k8s": "Kubernetes",
        }

        result = query
        for slang, formal in rewrites.items():
            result = result.replace(slang, formal)
        return result

    @staticmethod
    def llm_based(query: str, model: str = "claude-sonnet-4-6") -> str:
        """
        基于 LLM 的查询改写 (更智能, 有延迟)。

        用 LLM 理解用户意图, 改写成精确的检索查询。
        """
        if not api_ok:
            return QueryRewriter.rule_based(query)

        prompt = f"""你是一个查询改写助手。把用户的口语化问题改写成适合向量检索的正式查询。

规则:
1. 保留原意, 不要添加新信息
2. 将口语/缩写转为正式术语
3. 将模糊指代转为明确实体
4. 输出改写后的查询, 不要额外解释

用户输入: "{query}"
改写输出:"""

        try:
            rewritten = llm_ask(prompt, max_tokens=100, temperature=0.0)
            # 清理输出 (去掉引号)
            rewritten = rewritten.strip().strip('"').strip("'")
            return rewritten if rewritten else query
        except Exception:
            return QueryRewriter.rule_based(query)


# 对比改写前后的检索效果
print("  查询改写对比:")
rewrite_test = [
    "咋备份数据库?",
    "py 怎么处理并发的?",
    "那个内存的缓存怎么用的?",
    "k8s 是干嘛的?",
]

for q in rewrite_test:
    # 规则改写
    rule_q = QueryRewriter.rule_based(q)

    # LLM 改写
    if api_ok:
        llm_q = QueryRewriter.llm_based(q)
    else:
        llm_q = rule_q

    print(f"\n  ┌─ 原始: \"{q}\"")
    print(f"  ├─ 规则: \"{rule_q}\"")
    print(f"  └─ LLM:  \"{llm_q}\"")

    # 对比检索结果
    if test_col.count() > 0:
        orig_results = retriever.search(q, top_k=3)
        rewritten_results = retriever.search(llm_q, top_k=3)

        orig_ids = [r.doc_id for r in orig_results]
        rew_ids = [r.doc_id for r in rewritten_results]
        overlap = set(orig_ids) & set(rew_ids)

        print(f"  原始查询 Top-3: {orig_ids}")
        print(f"  改写查询 Top-3: {rew_ids}")
        print(f"  重叠: {overlap if overlap else '无'}  ← 差异越大说明改写越有效")

print(f"""
  查询改写的价值:
    1. 口语 "k8s" → "Kubernetes"  → embedding 质量提升
    2. 模糊 "那个内存数据库" → "Redis" → 检索更精准
    3. 多义词 "Python 怎么跑" → "Python 代码执行方式" → 消除歧义

  工程实践:
    先用规则改写 (零延迟) → 如果 Top-1 score < 阈值, 再用 LLM 改写重搜
    这种"级联"策略兼顾速度和质量。
""")

# ----------------------------------------------------------
# 练习 6: 检索质量评估 (思考)
# ----------------------------------------------------------
print("\n--- 练习 6: 检索质量评估 (思考) ---")

print("""
  思考: 如何系统评估检索质量?

  1. 评估指标:
  ┌──────────────┬──────────────────────────────────────┐
  │ 指标          │ 含义                                  │
  ├──────────────┼──────────────────────────────────────┤
  │ Recall@K     │ 理想文档出现在 Top-K 中的比例            │
  │ Precision@K  │ Top-K 中相关文档的比例                  │
  │ MRR          │ 第一个相关文档排名的倒数均值             │
  │ NDCG@K       │ 考虑排名位置的加权相关度                 │
  └──────────────┴──────────────────────────────────────┘

  2. 评估数据集设计 (5 个查询 + 理想文档):
""")

# 实际的评估数据集
EVAL_DATASET: list[dict] = [
    {
        "query": "Python 异步编程怎么做?",
        "ideal_docs": ["py_async"],
        "description": "直接对应文档",
    },
    {
        "query": "如何做数据库备份?",
        "ideal_docs": ["mysql_bak", "pg_bak"],
        "description": "多个相关文档",
    },
    {
        "query": "Redis 支持什么数据结构?",
        "ideal_docs": ["redis_intro"],
        "description": "明确的主题",
    },
    {
        "query": "怎么把应用打包部署?",
        "ideal_docs": ["docker_intro"],
        "description": "语义搜索 (字面不匹配)",
    },
    {
        "query": "什么是键值对存储?",
        "ideal_docs": ["py_dict", "redis_intro"],
        "description": "跨领域概念",
    },
]

print("\n  评估结果:")
print(f"  {'Query':<30} {'Recall@3':<10} {'Top-1命中':<10} {'评价'}")
print(f"  {'─' * 60}")

total_recall = 0.0
top1_hits = 0

for case in EVAL_DATASET:
    query = case["query"]
    ideal = set(case["ideal_docs"])

    results = retriever.search(query, top_k=3)
    retrieved_ids = set(r.doc_id for r in results)

    # Recall@3: 理想文档中有多少出现在 Top-3
    recall = len(ideal & retrieved_ids) / max(len(ideal), 1)
    total_recall += recall

    # Top-1 是否命中
    top1_matched = results[0].doc_id in ideal if results else False
    if top1_matched:
        top1_hits += 1

    eval_icon = "优秀" if recall >= 1.0 else ("良好" if recall >= 0.5 else "待改进")
    print(f"  {query:<30} {recall:.2f}       {'是' if top1_matched else '否':<10} {eval_icon}")

avg_recall = total_recall / len(EVAL_DATASET)
print(f"\n  平均 Recall@3: {avg_recall:.2f}")
print(f"  Top-1 命中率: {top1_hits}/{len(EVAL_DATASET)} ({top1_hits/len(EVAL_DATASET):.0%})")

print(f"""
  3. 评估结果分析:

  如果 Recall@3 低 (如 < 0.5):
    可能原因:
      a. 文档分块策略不对 — 切碎了语义
      b. embedding 模型不合适 — 对中文理解不够
      c. query 表达到文档表达的 gap 太大

  如果 Recall@3 高但生成质量差:
    可能原因:
      a. Chunk 太大, 检索到了但噪声多
      b. LLM 没有严格遵守"只基于文档回答"

  4. 持续改进循环:
    评估 → 发现问题 → 调整 (分块/嵌入/检索) → 再评估

  这就是 Lesson 41 要讲的系统化评估方法论。

  类比 Java:
    检索评估 ≈ 单元测试
    EVAL_DATASET ≈ 预设的测试用例 + 预期输出
    Recall@K ≈ assertThat(results, containsInAnyOrder(expectedDocs))
""")

print("\n" + "=" * 60)
print("  试试看练习完成!")
print("=" * 60)
