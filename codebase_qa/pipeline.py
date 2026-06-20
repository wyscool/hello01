# ============================================================
# codebase_qa/pipeline.py — EmbeddingFunction + QAPipeline
# ============================================================
# 参考 rag_kb/pipeline.py 的 EmbeddingFunction 实现模式。
# QAPipeline: Retrieve → Rerank → Generate 编排器。
# ============================================================

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================
# 一、EmbeddingFunction — sentence-transformers → ChromaDB 适配器
# ============================================================

class EmbeddingFunction:
    """将 sentence-transformers 包装为 ChromaDB 兼容的嵌入函数。

    ChromaDB 要求:
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

    @property
    def query_prefix(self) -> str:
        """BGE 系列模型需要查询前缀才能达到最佳检索效果。"""
        if "bge" in self.model_name.lower():
            return "Represent this sentence for searching relevant passages: "
        return ""

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
        self._load()
        return self._dimension or 384

    @property
    def ready(self) -> bool:
        return self._ready

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """嵌入查询文本（带 BGE 前缀）。"""
        self._load()
        prefix = self.query_prefix
        if prefix:
            input = [prefix + t for t in input]
        result = self._model.encode(input, convert_to_numpy=True)
        return result.tolist()

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """嵌入文档（不带前缀）。"""
        self._load()
        result = self._model.encode(input, convert_to_numpy=True)
        return result.tolist()

    def __call__(self, input: list[str]) -> list[list[float]]:
        """ChromaDB 统一接口——add 和 query 都走这里，加前缀保持一致。"""
        return self.embed_query(input)

    def encode(self, texts: list[str]) -> np.ndarray:
        """numpy 数组输出，MMR 使用。不加前缀——MMR 在已检索结果间计算。"""
        self._load()
        return self._model.encode(texts, convert_to_numpy=True)


# ============================================================
# 二、QAResponse — 查询结果
# ============================================================

@dataclass
class QAResponse:
    query: str
    answer: str
    sources: list  # list[SearchResult]，延迟导入避免循环引用
    latency_ms: float = 0.0


# ============================================================
# 三、QAPipeline — 编排器
# ============================================================

class QAPipeline:
    """Codebase Q&A 流水线编排器。

    流程: Retriever.search() → Reranker.filter + MMR → Generator.generate()
    所有组件通过构造函数注入。
    """

    def __init__(
        self, retriever, reranker, generator,
        top_k: int = 5, min_score: float = 0.3,
        use_mmr: bool = True, mmr_lambda: float = 0.7,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k
        self.min_score = min_score
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

    def ask(self, query: str, filter_type: str | None = None) -> QAResponse:
        """执行一次完整的 Q&A。

        Args:
          query: 用户自然语言问题
          filter_type: 可选，限定代码类型 (function/class/method/module_level)

        Returns:
          QAResponse 含 answer + sources + latency_ms
        """
        start = time.time()

        # Step 1: 检索 (取 2x top_k，为 MMR 留选择空间)
        search_k = self.top_k * 2
        results = self.retriever.search(query, top_k=search_k, filter_type=filter_type)

        # Step 2: 阈值过滤
        filtered = self.reranker.filter_by_threshold(results, self.min_score)

        # Step 3: MMR 多样性重排
        if self.use_mmr and len(filtered) > 1:
            final_results = self.reranker.mmr_rerank(
                query, filtered, top_k=self.top_k, lambda_param=self.mmr_lambda,
            )
        else:
            final_results = filtered[:self.top_k]

        # Step 4: LLM 生成答案
        answer = self.generator.generate(query, final_results)

        elapsed = (time.time() - start) * 1000
        return QAResponse(
            query=query,
            answer=answer,
            sources=final_results,
            latency_ms=round(elapsed, 1),
        )
