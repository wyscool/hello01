# ============================================================
# rag_kb/pipeline.py — RAG 流水线
# ============================================================
# 完整 RAG 流水线:
#   DocumentProcessor → ChromaDB → Retriever → Reranker → ContextBuilder → Generator
#
# 所有类通过构造函数注入依赖 (DI), 不依赖模块级全局变量。
# 适配自 Phase 3 L22-L24 的教学代码。
# ============================================================

import re
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ============================================================
# 一、EmbeddingFunction — sentence-transformers → ChromaDB 适配器
# ============================================================

class EmbeddingFunction:
    """将 sentence-transformers 包装为 ChromaDB 兼容的嵌入函数。

    ChromaDB 1.5+ 要求 embedding function 实现:
      - name() → str
      - embed_query(input: list[str]) → list[list[float]]
      - embed_documents(input: list[str]) → list[list[float]]
      - __call__(input: list[str]) → list[list[float]]  (向后兼容)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension: int | None = None
        self._ready = False

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)
        self._dimension = self._model.get_embedding_dimension()
        self._ready = True

    def name(self) -> str:
        return self.model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load()
        return self._dimension or 384

    @property
    def is_ready(self) -> bool:
        return self._ready

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def __call__(self, input: list[str]) -> list[list[float]]:
        self._load()
        embeddings = self._model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

    def encode(self, texts: list[str]) -> np.ndarray:
        """供 MMR 重排使用, 直接返回 numpy 数组。"""
        self._load()
        return self._model.encode(texts, convert_to_numpy=True)


# ============================================================
# 二、Chunk — 文档片段
# ============================================================

@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    total_chunks: int
    metadata: dict = field(default_factory=dict)


# ============================================================
# 三、DocumentProcessor — 加载 → 清洗 → 分块
# ============================================================

class DocumentProcessor:
    """文档处理流水线: 原始文件 → Chunk 列表。"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    # --- 加载 ---

    @staticmethod
    def load_file(file_path: Path) -> str | None:
        """加载文本文件, 自动检测编码 (utf-8 → gbk → latin-1)。"""
        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return None

    # --- 清洗 ---

    @staticmethod
    def clean_text(text: str) -> str:
        """规范化文本: 统一换行符、压缩空白、移除控制字符。"""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 压缩连续空行
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 移除行内多余空白
        text = re.sub(r"[ \t]{2,}", " ", text)
        # 移除 ASCII 控制字符 (保留 \n \t)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()

    # --- 分块 ---

    @staticmethod
    def recursive_chunk(text: str, chunk_size: int = 500,
                        overlap: int = 50,
                        separators: list[str] | None = None) -> list[str]:
        """递归字符分割 — LangChain 风格。

        分隔符优先级: 段落 → 换行 → 句号 → 逗号 → 空格 → 字符
        """
        if separators is None:
            separators = ["\n\n", "\n", "。", "，", " ", ""]

        chunks: list[str] = []

        def _split(text: str, seps: list[str]):
            if not text:
                return
            sep = seps[0]
            remaining = seps[1:]

            if not sep:
                # 最后手段: 按字符分割
                for i in range(0, len(text), chunk_size - overlap):
                    ch = text[i:i + chunk_size]
                    if ch:
                        chunks.append(ch)
                return

            parts = text.split(sep)
            current = ""
            for part in parts:
                candidate = part if not current else current + sep + part
                if len(candidate) > chunk_size and remaining:
                    if current:
                        chunks.append(current)
                    _split(part, remaining)
                    current = ""
                else:
                    current = candidate
            if current:
                chunks.append(current)

        _split(text, separators)

        # 合并小片段
        merged: list[str] = []
        buf = ""
        for ch in chunks:
            if buf and len(buf) + len(ch) <= chunk_size:
                buf += " " + ch
            else:
                if buf:
                    merged.append(buf)
                buf = ch
        if buf:
            merged.append(buf)

        return merged

    # --- 处理 ---

    def process_file(self, file_path: Path) -> list[Chunk]:
        """处理单个文件 → Chunk 列表。"""
        raw = self.load_file(file_path)
        if raw is None:
            return []

        text = self.clean_text(raw)
        if not text:
            return []

        parts = self.recursive_chunk(text, self.chunk_size, self.overlap)
        total = len(parts)
        source = file_path.name

        return [
            Chunk(
                text=p,
                source=source,
                chunk_index=i,
                total_chunks=total,
                metadata={"source": source, "chunk": i,
                          "total": total},
            )
            for i, p in enumerate(parts)
        ]

    def process_directory(self, docs_dir: Path,
                          patterns: tuple[str, ...] = (
                              "*.txt", "*.md", "*.json", "*.py", "*.java",
                          )) -> list[Chunk]:
        """处理目录下所有匹配文件 → Chunk 列表。"""
        all_chunks: list[Chunk] = []
        for pattern in patterns:
            for fp in docs_dir.glob(pattern):
                if fp.is_file():
                    chunks = self.process_file(fp)
                    all_chunks.extend(chunks)
        return all_chunks

    def to_dicts(self, chunks: list[Chunk]) -> list[dict]:
        """Chunk 列表 → ChromaDB 格式的 dict 列表。"""
        return [{
            "text": c.text,
            "source": c.source,
            "chunk_index": c.chunk_index,
            "total_chunks": c.total_chunks,
        } for c in chunks]


# ============================================================
# 四、SearchResult — 检索结果
# ============================================================

@dataclass
class SearchResult:
    doc_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


# ============================================================
# 五、Retriever — 向量检索
# ============================================================

class Retriever:
    """封装 ChromaDB collection.query()。"""

    def __init__(self, collection):
        self._collection = collection

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        result = self._collection.query(
            query_texts=[query], n_results=top_k,
        )
        return self._to_results(result)

    def search_with_filter(self, query: str, top_k: int = 5,
                           where: dict | None = None) -> list[SearchResult]:
        kwargs = {"query_texts": [query], "n_results": top_k}
        if where:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)
        return self._to_results(result)

    def _to_results(self, chroma_result: dict) -> list[SearchResult]:
        """ChromaDB 返回格式 → SearchResult 列表。"""
        results: list[SearchResult] = []
        ids = chroma_result.get("ids", [[]])[0]
        docs = chroma_result.get("documents", [[]])[0]
        distances = chroma_result.get("distances", [[]])[0]
        metadatas = chroma_result.get("metadatas", [[]])[0]

        for i in range(len(ids)):
            dist = distances[i] if i < len(distances) else 1.0
            score = 1.0 / (1.0 + dist)
            meta = metadatas[i] if i < len(metadatas) else {}
            results.append(SearchResult(
                doc_id=ids[i],
                text=docs[i] if i < len(docs) else "",
                score=round(score, 4),
                metadata=meta or {},
            ))

        return results


# ============================================================
# 六、Reranker — 阈值过滤 + MMR 重排
# ============================================================

class Reranker:
    """检索结果后处理: 低分过滤 + MMR 多样性重排。"""

    def __init__(self, embedding_fn: EmbeddingFunction):
        self._embed = embedding_fn

    @staticmethod
    def filter_by_threshold(results: list[SearchResult],
                            min_score: float = 0.3) -> list[SearchResult]:
        return [r for r in results if r.score >= min_score]

    def mmr_rerank(self, query: str, results: list[SearchResult],
                   top_k: int = 3, lambda_param: float = 0.7,
                   ) -> list[SearchResult]:
        """MMR (Maximal Marginal Relevance) 重排。

        mmr = λ * relevance - (1-λ) * max_similarity_to_selected
        """
        if len(results) <= top_k:
            return results

        # 嵌入 query 和所有结果文本
        texts = [r.text for r in results]
        query_vec = self._embed.encode([query])[0]
        doc_vecs = self._embed.encode(texts)

        # 相关性: query 与每个文档的余弦相似度
        q_norm = np.linalg.norm(query_vec)
        relevance = np.dot(doc_vecs, query_vec) / (
            np.linalg.norm(doc_vecs, axis=1) * q_norm + 1e-9
        )

        # Greedy MMR 选择
        selected_idx: list[int] = []
        remaining = list(range(len(results)))

        # 第一轮: 选最相关的
        first = int(np.argmax(relevance))
        selected_idx.append(first)
        remaining.remove(first)

        while len(selected_idx) < min(top_k, len(results)):
            best_score = -float("inf")
            best_i = -1

            for i in remaining:
                rel = relevance[i]
                # 与已选结果的最大相似度
                sims = []
                for j in selected_idx:
                    cos_sim = np.dot(doc_vecs[i], doc_vecs[j]) / (
                        np.linalg.norm(doc_vecs[i]) *
                        np.linalg.norm(doc_vecs[j]) + 1e-9
                    )
                    sims.append(cos_sim)
                redundancy = max(sims) if sims else 0
                mmr = lambda_param * rel - (1 - lambda_param) * redundancy
                if mmr > best_score:
                    best_score = mmr
                    best_i = i

            if best_i >= 0:
                selected_idx.append(best_i)
                remaining.remove(best_i)
            else:
                break

        return [results[i] for i in selected_idx]


# ============================================================
# 七、ContextBuilder — Prompt 构建
# ============================================================

class ContextBuilder:
    """将检索结果拼接为 LLM prompt。"""

    DEFAULT_SYSTEM = (
        "你是一个基于文档的知识库助手。"
        "只根据提供的文档内容回答问题，不要编造信息。\n"
        "规则:\n"
        "  1. 每个回答必须引用来源 (用 [文档N] 标注)\n"
        "  2. 如果文档中没有相关信息，明确说 '文档中未找到相关信息'\n"
        "  3. 如果文档信息不完整，说明哪些是文档中的，哪些是推测的\n"
        "  4. 用中文回答，保持简洁专业"
    )

    @staticmethod
    def build(query: str, results: list[SearchResult],
              system_prompt: str | None = None) -> tuple[str, str]:
        """构建 (system_prompt, user_prompt)。"""
        if system_prompt is None:
            system_prompt = ContextBuilder.DEFAULT_SYSTEM

        if not results:
            user = (f"问题: {query}\n\n"
                    f"(没有找到相关文档, 请告知用户)")
            return system_prompt, user

        context_parts: list[str] = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "未知")
            context_parts.append(
                f"[文档{i}] 来源: {source}\n{r.text}"
            )

        context = "\n\n---\n\n".join(context_parts)
        user = (f"参考资料:\n\n{context}\n\n"
                f"---\n"
                f"问题: {query}\n\n"
                f"请根据以上参考资料回答。")

        return system_prompt, user


# ============================================================
# 八、Generator — LLM 调用
# ============================================================

class Generator:
    """封装 LlmClient 做 RAG 生成。"""

    def __init__(self, llm_client):
        self._llm = llm_client

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        response = self._llm.create(
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt, max_tokens=max_tokens,
            temperature=temperature,
        )
        return self._llm.get_text(response)


# ============================================================
# 九、RAGResponse
# ============================================================

@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    latency_ms: float = 0.0


# ============================================================
# 十、RAGPipeline — 流水线编排器
# ============================================================

class RAGPipeline:
    """RAG 流水线编排器: Retrieve → Rerank → Build → Generate。

    固定流水线, 单 ask() 入口。所有组件通过构造函数注入。
    """

    def __init__(self, retriever: Retriever, reranker: Reranker,
                 context_builder: ContextBuilder, generator: Generator,
                 top_k: int = 5, min_score: float = 0.3,
                 use_mmr: bool = True, mmr_lambda: float = 0.7):
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.generator = generator
        self.top_k = top_k
        self.min_score = min_score
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

    def ask(self, query: str, where: dict | None = None,
            system_prompt: str | None = None) -> RAGResponse:
        start = time.time()

        # Step 1: 检索
        search_top_k = self.top_k * 2  # 多检索一些给 MMR 筛选用
        if where:
            results = self.retriever.search_with_filter(
                query, top_k=search_top_k, where=where
            )
        else:
            results = self.retriever.search(query, top_k=search_top_k)

        # Step 2: 阈值过滤
        filtered = self.reranker.filter_by_threshold(results, self.min_score)

        # Step 3: MMR 重排 (可选)
        if self.use_mmr and len(filtered) > 1:
            final_results = self.reranker.mmr_rerank(
                query, filtered, top_k=self.top_k,
                lambda_param=self.mmr_lambda,
            )
        else:
            final_results = filtered[:self.top_k]

        # Step 4: 构建 prompt
        system, user = self.context_builder.build(
            query, final_results, system_prompt
        )

        # Step 5: LLM 生成
        answer = self.generator.generate(system, user)

        elapsed = (time.time() - start) * 1000
        return RAGResponse(
            query=query, answer=answer,
            sources=final_results, latency_ms=round(elapsed, 1),
        )
