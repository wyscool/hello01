# ============================================================
# codebase_qa/retriever.py — SearchResult + Retriever + Reranker
# ============================================================
# 参考 rag_kb/pipeline.py 的 Retriever/Reranker 模式。
# 适配代码搜索: metadata 含 file_path/start_line/end_line 等字段。
# ============================================================

from dataclasses import dataclass

import numpy as np


# ============================================================
# 一、SearchResult — 单条检索结果
# ============================================================

@dataclass
class SearchResult:
    doc_id: str            # chunk_id from CodeChunk
    text: str              # code snippet
    score: float           # 相关性分数 (0-1)
    metadata: dict         # {name, type, file_path, start_line, end_line, signature}


# ============================================================
# 二、Retriever — ChromaDB 查询封装
# ============================================================

class Retriever:
    """Wraps ChromaDB collection.query() 用于代码搜索。

    支持按代码类型过滤 (function/class/method/module_level)。
    """

    def __init__(self, collection):
        self._collection = collection

    def search(self, query: str, top_k: int = 5,
               filter_type: str | None = None) -> list[SearchResult]:
        """向量检索。

        Args:
          query: 自然语言查询
          top_k: 返回结果数
          filter_type: 可选，按 type 字段过滤
        """
        kwargs: dict = {"query_texts": [query], "n_results": top_k,
                        "include": ["documents", "metadatas", "distances"]}
        if filter_type:
            kwargs["where"] = {"type": filter_type}

        result = self._collection.query(**kwargs)
        return self._to_results(result)

    def _to_results(self, chroma_result: dict) -> list[SearchResult]:
        """ChromaDB 原始返回 → SearchResult 列表。

        score = 1.0 / (1.0 + distance)
        """
        ids = chroma_result.get("ids", [[]])[0]
        docs = chroma_result.get("documents", [[]])[0]
        distances = chroma_result.get("distances", [[]])[0]
        metadatas = chroma_result.get("metadatas", [[]])[0]

        results: list[SearchResult] = []
        for i in range(len(ids)):
            dist = distances[i] if i < len(distances) else 1.0
            score = round(1.0 / (1.0 + dist), 4)
            meta = metadatas[i] if i < len(metadatas) else {}
            results.append(SearchResult(
                doc_id=ids[i],
                text=docs[i] if i < len(docs) else "",
                score=score,
                metadata=meta or {},
            ))
        return results


# ============================================================
# 三、Reranker — 阈值过滤 + MMR 多样性重排
# ============================================================

class Reranker:
    """检索结果后处理: 低分过滤 + MMR 多样性重排。

    MMR = λ * relevance - (1-λ) * max_similarity_to_already_selected
    """

    def __init__(self, embedding_fn):
        self._embed = embedding_fn

    @staticmethod
    def filter_by_threshold(results: list[SearchResult],
                            min_score: float = 0.3) -> list[SearchResult]:
        return [r for r in results if r.score >= min_score]

    def mmr_rerank(self, query: str, results: list[SearchResult],
                   top_k: int = 3, lambda_param: float = 0.7,
                   ) -> list[SearchResult]:
        """Greedy MMR 重排。"""
        if len(results) <= top_k:
            return results

        texts = [r.text for r in results]
        query_vec = self._embed.encode([query])[0]
        doc_vecs = self._embed.encode(texts)

        # 相关性: query 与每个文档的余弦相似度
        q_norm = np.linalg.norm(query_vec)
        relevance = np.dot(doc_vecs, query_vec) / (
            np.linalg.norm(doc_vecs, axis=1) * q_norm + 1e-9
        )

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
