# ============================================================
# rag_kb/tests/test_pipeline.py — Pipeline 组件测试
# ============================================================

import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock

from rag_kb.pipeline import (
    DocumentProcessor, ContextBuilder, Reranker, Chunk, SearchResult,
)


# ============================================================
# DocumentProcessor — 文件加载
# ============================================================

class TestLoadFile:
    @pytest.fixture
    def processor(self):
        return DocumentProcessor()

    def test_load_utf8(self, processor):
        p = Path(tempfile.gettempdir()) / "test_utf8.txt"
        p.write_text("Hello World\n你好世界\n", encoding="utf-8")
        text = processor.load_file(p)
        assert text is not None
        assert "Hello World" in text
        assert "你好" in text

    def test_load_gbk(self, processor):
        p = Path(tempfile.gettempdir()) / "test_gbk.txt"
        p.write_text("你好 GBK\n", encoding="gbk")
        text = processor.load_file(p)
        assert text is not None
        assert "GBK" in text

    def test_load_nonexistent(self, processor):
        with pytest.raises(FileNotFoundError):
            processor.load_file(Path("/nonexistent/file.txt"))


# ============================================================
# DocumentProcessor — 清洗
# ============================================================

class TestCleanText:
    def test_strips_and_normalizes_newlines(self):
        text = "hello\r\nworld\r\n\n\n\nfoo  bar\n"
        cleaned = DocumentProcessor.clean_text(text)
        # \r\n → \n, 3+ \n → \n\n
        assert "\r" not in cleaned
        assert cleaned.count("\n") <= 4  # hello\nworld\n\nfoo bar\n

    def test_removes_control_chars(self):
        text = "normal\x00\x08\x1f text\x7f\n"
        cleaned = DocumentProcessor.clean_text(text)
        assert "\x00" not in cleaned
        assert "\x1f" not in cleaned
        assert "normal text" in cleaned

    def test_compresses_whitespace(self):
        text = "a    b  \t  c\n"
        cleaned = DocumentProcessor.clean_text(text)
        assert "a b c" in cleaned

    def test_empty_string(self):
        assert DocumentProcessor.clean_text("") == ""


# ============================================================
# DocumentProcessor — 分块
# ============================================================

class TestRecursiveChunk:
    def test_single_short_text(self):
        chunks = DocumentProcessor.recursive_chunk(
            "Hello world", chunk_size=500, overlap=50
        )
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_paragraph_split(self):
        text = "para1\n\npara2\n\npara3"
        chunks = DocumentProcessor.recursive_chunk(
            text, chunk_size=10, overlap=2
        )
        # chunk_size=10 时 merge 逻辑将 "para1"+"para2" 合并为一个 chunk，共 2 个
        assert len(chunks) == 2

    def test_long_paragraph_force_split(self):
        text = "x" * 200
        chunks = DocumentProcessor.recursive_chunk(
            text, chunk_size=50, overlap=10
        )
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 50

    def test_empty_text(self):
        chunks = DocumentProcessor.recursive_chunk("", chunk_size=500)
        assert chunks == []

    def test_custom_separators(self):
        text = "sentence one。sentence two。sentence three"
        chunks = DocumentProcessor.recursive_chunk(
            text, chunk_size=20, overlap=2, separators=["。"]
        )
        assert len(chunks) >= 1

    def test_merge_small_chunks(self):
        text = "a\n\nb"
        chunks = DocumentProcessor.recursive_chunk(
            text, chunk_size=500, overlap=50
        )
        assert len(chunks) == 1
        assert "a" in chunks[0] and "b" in chunks[0]


# ============================================================
# DocumentProcessor — process_file
# ============================================================

class TestProcessFile:
    @pytest.fixture
    def processor(self):
        return DocumentProcessor(chunk_size=100, overlap=20)

    def test_process_single_file(self, processor):
        p = Path(tempfile.gettempdir()) / "test_doc.txt"
        p.write_text("Section 1\n\nSection 2\n\nSection 3\n", encoding="utf-8")
        chunks = processor.process_file(p)
        assert len(chunks) > 0
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.source == "test_doc.txt"
            assert c.chunk_index >= 0
            assert c.total_chunks == len(chunks)

    def test_process_empty_file(self, processor):
        p = Path(tempfile.gettempdir()) / "empty.txt"
        p.write_text("", encoding="utf-8")
        chunks = processor.process_file(p)
        assert chunks == []

    def test_to_dicts(self, processor):
        p = Path(tempfile.gettempdir()) / "dict_test.txt"
        p.write_text("Hello\n\nWorld\n", encoding="utf-8")
        chunks = processor.process_file(p)
        dicts = processor.to_dicts(chunks)
        assert len(dicts) == len(chunks)
        for d in dicts:
            assert "text" in d
            assert "source" in d
            assert "chunk_index" in d

    def test_unicode_handling(self, processor):
        p = Path(tempfile.gettempdir()) / "unicode.txt"
        p.write_text("中文段落\n\n日本語パラグラフ\n\n한국어 문단\n", encoding="utf-8")
        chunks = processor.process_file(p)
        assert len(chunks) > 0
        texts = [c.text for c in chunks]
        assert any("中文" in t for t in texts)


# ============================================================
# ContextBuilder
# ============================================================

class TestContextBuilder:
    def test_build_with_results(self):
        results = [
            SearchResult(doc_id="1", text="Document A content",
                        score=0.9, metadata={"source": "a.txt"}),
            SearchResult(doc_id="2", text="Document B content",
                        score=0.8, metadata={"source": "b.txt"}),
        ]
        system, user = ContextBuilder.build("test query", results)
        assert "test query" in user
        assert "[文档1]" in user
        assert "[文档2]" in user
        assert "a.txt" in user
        assert "b.txt" in user
        assert "Document A" in user

    def test_build_empty_results(self):
        system, user = ContextBuilder.build("query", [])
        assert "没有找到相关文档" in user
        assert "query" in user

    def test_build_custom_system_prompt(self):
        results = [SearchResult(doc_id="1", text="content",
                                score=0.5, metadata={"source": "x.txt"})]
        custom = "自定义 system prompt"
        system, user = ContextBuilder.build("q", results, system_prompt=custom)
        assert system == custom

    def test_default_system_prompt(self):
        system, _ = ContextBuilder.build("q", [])
        assert "知识库助手" in system
        assert "引用来源" in system


# ============================================================
# Reranker — filter_by_threshold
# ============================================================

class TestFilterByThreshold:
    def test_filters_below_threshold(self):
        results = [
            SearchResult("1", "a", 0.5, {}),
            SearchResult("2", "b", 0.2, {}),
            SearchResult("3", "c", 0.35, {}),
            SearchResult("4", "d", 0.1, {}),
        ]
        filtered = Reranker.filter_by_threshold(results, min_score=0.3)
        assert len(filtered) == 2
        assert all(r.score >= 0.3 for r in filtered)

    def test_all_pass(self):
        results = [SearchResult("1", "a", 0.8, {})]
        filtered = Reranker.filter_by_threshold(results, min_score=0.3)
        assert len(filtered) == 1

    def test_all_fail(self):
        results = [SearchResult("1", "a", 0.1, {})]
        filtered = Reranker.filter_by_threshold(results, min_score=0.3)
        assert filtered == []

    def test_empty_list(self):
        assert Reranker.filter_by_threshold([], 0.3) == []


# ============================================================
# Reranker — MMR (mocked)
# ============================================================

class TestMMRRerank:
    @pytest.fixture
    def reranker(self):
        import numpy as np
        # 创建 mock embedding function
        mock_embed = Mock()
        # 简单模拟: query 和 docs 用相同向量 (相关性都为 1)
        mock_embed.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0],
                                                    [0.0, 1.0], [0.5, 0.5]],
                                                   dtype=np.float64)
        return Reranker(mock_embed)

    def test_mmr_selects_diverse(self, reranker):
        results = [
            SearchResult("1", "a", 0.9, {}),
            SearchResult("2", "b", 0.9, {}),
            SearchResult("3", "c", 0.8, {}),
            SearchResult("4", "d", 0.7, {}),
        ]
        # query 设为 "test", mock 处理
        selected = reranker.mmr_rerank("test", results, top_k=2, lambda_param=0.7)
        assert len(selected) == 2

    def test_mmr_fewer_than_top_k(self, reranker):
        results = [SearchResult("1", "a", 0.9, {})]
        selected = reranker.mmr_rerank("test", results, top_k=5)
        assert len(selected) == 1

    def test_lambda_zero_returns_second_best(self, reranker):
        """λ=0 时只考虑多样性（不关心相关性），选择与被选差异最大的。"""
        results = [
            SearchResult("1", "a", 0.9, {}),
            SearchResult("2", "b", 0.9, {}),
            SearchResult("3", "c", 0.8, {}),
        ]
        selected = reranker.mmr_rerank("test", results, top_k=2, lambda_param=0.0)
        assert len(selected) == 2
        # 第一个选相关性最高的(1), 第二个选与1最不相似的(3)
        assert selected[0].doc_id == "1"
        assert selected[1].doc_id == "3"

    def test_lambda_one_returns_top_by_relevance(self, reranker):
        """λ=1 时不考虑多样性，按相关性排序。"""
        results = [
            SearchResult("1", "a", 0.9, {}),
            SearchResult("2", "b", 0.9, {}),
            SearchResult("3", "c", 0.8, {}),
        ]
        selected = reranker.mmr_rerank("test", results, top_k=3, lambda_param=1.0)
        assert len(selected) == 3
        # 按相关性降序
        assert selected[0].doc_id == "1"
