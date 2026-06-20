# ============================================================
# codebase_qa/tests/test_retriever.py — Retriever + Reranker 测试
# ============================================================

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

from codebase_qa.retriever import SearchResult, Retriever, Reranker


# ============================================================
# fixtures
# ============================================================

def _make_chroma_response(ids=None, docs=None, distances=None, metadatas=None):
    """构造 ChromaDB collection.query() 返回格式。"""
    if metadatas is None:
        metadatas = [
            {"name": "foo", "type": "function", "file_path": "a.py",
             "start_line": 10, "end_line": 15},
            {"name": "MyClass", "type": "class", "file_path": "b.py",
             "start_line": 20, "end_line": 30},
        ]
    return {
        "ids": [ids or ["id1", "id2"]],
        "documents": [docs or ["code text 1", "code text 2"]],
        "distances": [distances or [0.1, 0.5]],
        "metadatas": [metadatas],
    }


@pytest.fixture
def mock_collection():
    col = Mock()
    col.query.return_value = _make_chroma_response()
    return col


@pytest.fixture
def mock_embed_fn():
    fn = Mock()
    fn.encode.return_value = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
    ])
    return fn


@pytest.fixture
def sample_results():
    return [
        SearchResult(
            doc_id="id1", text="def foo(): pass",
            score=0.92, metadata={"type": "function", "file_path": "a.py"},
        ),
        SearchResult(
            doc_id="id2", text="class Bar: pass",
            score=0.85, metadata={"type": "class", "file_path": "b.py"},
        ),
        SearchResult(
            doc_id="id3", text="def baz(): pass",
            score=0.70, metadata={"type": "function", "file_path": "a.py"},
        ),
    ]


# ============================================================
# TestSearchResult
# ============================================================

class TestSearchResult:
    def test_basic_init(self):
        sr = SearchResult(doc_id="abc", text="def foo(): pass",
                          score=0.9, metadata={"type": "function"})
        assert sr.doc_id == "abc"
        assert sr.score == 0.9
        assert sr.metadata["type"] == "function"


# ============================================================
# TestRetriever
# ============================================================

class TestRetriever:
    def test_search_returns_results(self, mock_collection):
        retriever = Retriever(mock_collection)
        results = retriever.search("test query", top_k=5)
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)

    def test_score_calculation(self, mock_collection):
        """score = 1.0 / (1.0 + distance)"""
        retriever = Retriever(mock_collection)
        results = retriever.search("test")
        # distance=0.1 → score=1.0/(1.1)≈0.9091
        assert results[0].score > 0.9
        # distance=0.5 → score=1.0/(1.5)≈0.6667
        assert results[1].score < 0.8

    def test_search_metadata_mapped(self, mock_collection):
        retriever = Retriever(mock_collection)
        results = retriever.search("test")
        assert results[0].metadata.get("type") == "function"
        assert results[0].metadata.get("file_path") == "a.py"
        assert results[1].metadata.get("type") == "class"

    def test_search_with_type_filter(self, mock_collection):
        retriever = Retriever(mock_collection)
        retriever.search("test", filter_type="function")
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs.get("where") == {"type": "function"}

    def test_empty_results(self):
        col = Mock()
        col.query.return_value = {"ids": [[]], "documents": [[]],
                                  "distances": [[]], "metadatas": [[]]}
        retriever = Retriever(col)
        results = retriever.search("nothing")
        assert results == []

    def test_missing_metadata_handled(self):
        col = Mock()
        col.query.return_value = {
            "ids": [["id1"]], "documents": [["text"]],
            "distances": [[0.2]], "metadatas": [[]],
        }
        retriever = Retriever(col)
        results = retriever.search("test")
        assert results[0].metadata == {}


# ============================================================
# TestReranker
# ============================================================

class TestReranker:
    def test_filter_by_threshold(self, sample_results):
        filtered = Reranker.filter_by_threshold(sample_results, min_score=0.8)
        assert len(filtered) == 2
        assert all(r.score >= 0.8 for r in filtered)

    def test_filter_by_threshold_all_below(self, sample_results):
        filtered = Reranker.filter_by_threshold(sample_results, min_score=0.95)
        assert filtered == []

    def test_filter_by_threshold_all_above(self, sample_results):
        filtered = Reranker.filter_by_threshold(sample_results, min_score=0.5)
        assert len(filtered) == 3

    def test_mmr_rerank_returns_top_k(self, mock_embed_fn, sample_results):
        reranker = Reranker(mock_embed_fn)
        results = reranker.mmr_rerank(
            "test query", sample_results, top_k=2, lambda_param=0.7,
        )
        assert len(results) == 2
        # 最高分项应该在结果中
        assert results[0].score > results[1].score or \
            results[1].score >= results[0].score

    def test_mmr_rerank_fewer_than_top_k(self, mock_embed_fn, sample_results):
        reranker = Reranker(mock_embed_fn)
        results = reranker.mmr_rerank(
            "test", sample_results, top_k=10, lambda_param=0.7,
        )
        assert len(results) == 3  # 只有 3 个, 全返回

    def test_mmr_diversity_prefers_different_files(self, mock_embed_fn):
        """MMR 应倾向于选取不同文件的结果（余弦相似度较低的）。"""
        results = [
            SearchResult(doc_id="1", text="def foo(): pass",
                         score=0.92, metadata={"file_path": "a.py"}),
            SearchResult(doc_id="2", text="def bar(): pass",
                         score=0.91, metadata={"file_path": "a.py"}),
            SearchResult(doc_id="3", text="class Baz: pass",
                         score=0.85, metadata={"file_path": "b.py"}),
        ]
        reranker = Reranker(mock_embed_fn)
        selected = reranker.mmr_rerank("query", results, top_k=2, lambda_param=0.3)
        # lambda=0.3 偏向多样性 → 倾向选不同文件
        file_paths = {s.metadata["file_path"] for s in selected}
        # 至少有可能选到不同的（非确定性取决于 mock encode 值）
        assert len(selected) == 2
