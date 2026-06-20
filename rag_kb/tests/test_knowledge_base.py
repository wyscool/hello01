# ============================================================
# rag_kb/tests/test_knowledge_base.py — KnowledgeBase CRUD 测试
# ============================================================

import tempfile
import pytest
from pathlib import Path

from rag_kb.knowledge_base import KnowledgeBase, create_knowledge_base, DocInfo
from rag_kb.pipeline import EmbeddingFunction, DocumentProcessor


# 共享的 embedding model (加载一次, 节省时间)
@pytest.fixture(scope="module")
def embed_fn():
    """模块级别 fixture, 只加载一次 embedding model。"""
    return EmbeddingFunction(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def kb(embed_fn):
    """每次测试一个新的临时 KnowledgeBase。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(
            collection_name="test_kb",
            persist_dir=tmpdir,
            embedding_function=embed_fn,
            processor=DocumentProcessor(chunk_size=200, overlap=20),
        )
        yield kb


@pytest.fixture
def test_doc1():
    """创建测试文件 1。"""
    p = Path(tempfile.gettempdir()) / "test_rag_doc1.txt"
    p.write_text(
        "Python 是一门动态类型语言。\n\n"
        "Python 的列表是可变序列，支持索引和切片。\n\n"
        "Python 的字典是键值对集合，类似 Java 的 HashMap。\n",
        encoding="utf-8",
    )
    yield p
    p.unlink(missing_ok=True)


@pytest.fixture
def test_doc2():
    """创建测试文件 2。"""
    p = Path(tempfile.gettempdir()) / "test_rag_doc2.txt"
    p.write_text(
        "Redis 支持五种数据结构: 字符串、列表、集合、有序集合、哈希。\n\n"
        "Redis 常用于缓存、消息队列、排行榜等场景。\n",
        encoding="utf-8",
    )
    yield p
    p.unlink(missing_ok=True)


# ============================================================
# 基础属性
# ============================================================

class TestKnowledgeBaseProperties:
    def test_count_starts_zero(self, kb):
        assert kb.count == 0

    def test_is_connected(self, kb):
        assert kb.is_connected is True

    def test_collection_name(self, kb):
        assert kb.collection_name == "test_kb"

    def test_collection_property(self, kb):
        col = kb.collection
        assert col is not None
        assert col.name == "test_kb"


# ============================================================
# add_file
# ============================================================

class TestAddFile:
    def test_adds_single_file(self, kb, test_doc1):
        n = kb.add_file(test_doc1)
        assert n > 0
        assert kb.count == n

    def test_reimport_replaces(self, kb, test_doc1):
        """重复导入同一文件会先删除旧数据再导入新数据。"""
        kb.add_file(test_doc1)
        first_count = kb.count
        kb.add_file(test_doc1)
        # 重新导入后总数不变 (先删后加)
        assert kb.count == first_count

    def test_add_empty_file(self, kb, test_doc1):
        empty = Path(tempfile.gettempdir()) / "empty_doc.txt"
        empty.write_text("", encoding="utf-8")
        try:
            n = kb.add_file(empty)
            assert n == 0
        finally:
            empty.unlink(missing_ok=True)

    def test_add_multiple_files(self, kb, test_doc1, test_doc2):
        n1 = kb.add_file(test_doc1)
        n2 = kb.add_file(test_doc2)
        assert kb.count == n1 + n2


# ============================================================
# search
# ============================================================

class TestSearch:
    def test_search_returns_results(self, kb, test_doc1):
        kb.add_file(test_doc1)
        results = kb.search("Python 语言", top_k=3)
        assert len(results) > 0
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "score" in r
            assert "source" in r

    def test_search_scores_between_0_and_1(self, kb, test_doc1):
        kb.add_file(test_doc1)
        results = kb.search("列表", top_k=3)
        for r in results:
            assert 0 < r["score"] <= 1.0

    def test_search_empty_kb(self, kb):
        results = kb.search("anything", top_k=5)
        assert results == []

    def test_search_results_relevant(self, kb, test_doc2):
        kb.add_file(test_doc2)
        results = kb.search("Redis 数据结构", top_k=3)
        assert len(results) > 0
        # top result 应该与 Redis 相关
        assert any("Redis" in r["text"] for r in results)


# ============================================================
# list_docs / get_doc
# ============================================================

class TestListDocs:
    def test_empty_kb(self, kb):
        docs = kb.list_docs()
        assert docs == []

    def test_lists_all_sources(self, kb, test_doc1, test_doc2):
        kb.add_file(test_doc1)
        kb.add_file(test_doc2)
        docs = kb.list_docs()
        assert len(docs) == 2
        sources = {d["source"] for d in docs}
        assert test_doc1.name in sources
        assert test_doc2.name in sources

    def test_doc_info_structure(self, kb, test_doc1):
        kb.add_file(test_doc1)
        docs = kb.list_docs()
        d = docs[0]
        assert "source" in d
        assert "chunks" in d
        assert "added_at" in d
        assert d["chunks"] > 0


class TestGetDoc:
    def test_get_existing_doc(self, kb, test_doc1):
        kb.add_file(test_doc1)
        doc = kb.get_doc(test_doc1.name)
        assert doc is not None
        assert doc["source"] == test_doc1.name
        assert doc["total_chunks"] > 0
        assert len(doc["chunks"]) == doc["total_chunks"]
        # chunks 按 chunk_index 排序
        indices = [c["chunk_index"] for c in doc["chunks"]]
        assert indices == sorted(indices)

    def test_get_nonexistent_doc(self, kb):
        assert kb.get_doc("nonexistent.txt") is None


# ============================================================
# remove_doc
# ============================================================

class TestRemoveDoc:
    def test_remove_existing(self, kb, test_doc1):
        kb.add_file(test_doc1)
        before = kb.count
        deleted = kb.remove_doc(test_doc1.name)
        assert deleted > 0
        assert kb.count == before - deleted

    def test_remove_nonexistent(self, kb):
        deleted = kb.remove_doc("not_there.txt")
        assert deleted == 0

    def test_remove_then_list_empty(self, kb, test_doc1):
        kb.add_file(test_doc1)
        kb.remove_doc(test_doc1.name)
        docs = kb.list_docs()
        assert docs == []


# ============================================================
# stats
# ============================================================

class TestStats:
    def test_stats_structure(self, kb, test_doc1):
        kb.add_file(test_doc1)
        st = kb.stats()
        assert st["collection_name"] == "test_kb"
        assert st["total_docs"] == 1
        assert st["total_chunks"] > 0
        assert test_doc1.name in st["sources"]
        assert "persist_dir" in st

    def test_empty_kb_stats(self, kb):
        st = kb.stats()
        assert st["total_docs"] == 0
        assert st["total_chunks"] == 0
        assert st["sources"] == []


# ============================================================
# add_directory
# ============================================================

class TestAddDirectory:
    def test_adds_all_matching_files(self, kb, test_doc1, test_doc2):
        # test_doc1 和 test_doc2 在同一目录
        parent = test_doc1.parent
        total = kb.add_directory(parent, patterns=("*.txt",))
        assert total > 0

    def test_no_matching_files(self, kb):
        with tempfile.TemporaryDirectory() as d:
            total = kb.add_directory(Path(d), patterns=("*.xyz",))
            assert total == 0


# ============================================================
# create_knowledge_base 工厂
# ============================================================

class TestFactory:
    def test_creates_from_config(self):
        kb = create_knowledge_base()
        assert kb is not None
        assert kb.is_connected
        assert kb.count >= 0  # 可能已有数据
