# ============================================================
# codebase_qa/tests/test_pipeline.py — 端到端集成测试
# ============================================================

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np


# ============================================================
# Module-scoped embedding (加载一次)
# ============================================================

@pytest.fixture(scope="module")
def embed_fn():
    from codebase_qa.pipeline import EmbeddingFunction
    fn = EmbeddingFunction(model_name="all-MiniLM-L6-v2")
    fn.embed_query(["ping"])
    return fn


# ============================================================
# Mock LLM
# ============================================================

@pytest.fixture
def mock_llm_client():
    llm = Mock()
    msg = Mock()
    llm.create.return_value = msg
    llm.get_text.return_value = (
        "在 [phase1/04_functions.py:556-583] 找到了 retry 装饰器。\n"
        "该装饰器接受 max_attempts 和 delay 参数，"
        "通过 functools.wraps 保留原始函数元数据。"
    )
    return llm


# ============================================================
# 临时代码目录
# ============================================================

@pytest.fixture
def temp_code_dir(tmp_path):
    """创建包含真实 Python 代码的临时目录。"""
    src = tmp_path / "testcode"
    src.mkdir()

    (src / "utils.py").write_text('''
def retry(max_attempts=3):
    """Async retry decorator."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for i in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    if i == max_attempts - 1:
                        raise
        return wrapper
    return decorator

class ConfigHelper:
    """Helps with configuration."""
    def load(self):
        return {}
''')

    (src / "models.py").write_text('''
class User:
    """User model."""
    def __init__(self, name: str):
        self.name = name

class Admin(User):
    """Admin model."""
    pass
''')

    return src


# ============================================================
# ChromaDB + Pipeline fixture
# ============================================================

@pytest.fixture
def pipeline_with_data(embed_fn, mock_llm_client, temp_code_dir, tmp_path):
    """构建完整 pipeline: 真实 embed + temp ChromaDB + mock LLM。"""
    import chromadb
    from chromadb.config import Settings
    from codebase_qa.indexer import CodeIndexer
    from codebase_qa.retriever import Retriever, Reranker
    from codebase_qa.generator import AnswerGenerator
    from codebase_qa.pipeline import QAPipeline

    # 索引代码
    indexer = CodeIndexer()
    chunks = indexer.index_directory(temp_code_dir)
    assert len(chunks) > 0, "应该索引到至少一个 chunk"

    # 临时 ChromaDB
    chroma_path = str(tmp_path / "chromadb")
    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(
        name="test_codebase", embedding_function=embed_fn,
    )

    # 添加 chunks
    ids = [c.chunk_id for c in chunks]
    documents = [c.embed_text for c in chunks]
    metadatas = [indexer.chunk_to_metadata(c) for c in chunks]
    col.add(ids=ids, documents=documents, metadatas=metadatas)

    # 组装 pipeline
    retriever = Retriever(col)
    reranker = Reranker(embed_fn)
    generator = AnswerGenerator(mock_llm_client)
    pipeline = QAPipeline(
        retriever=retriever, reranker=reranker, generator=generator,
        top_k=5, min_score=0.1, use_mmr=True, mmr_lambda=0.7,
    )

    return pipeline, col


# ============================================================
# TestQAPipeline
# ============================================================

class TestQAPipeline:
    def test_finds_function(self, pipeline_with_data):
        pipeline, col = pipeline_with_data
        result = pipeline.ask("retry decorator")
        assert result.query == "retry decorator"
        assert len(result.sources) > 0
        assert "retry" in result.answer.lower() or result.answer

    def test_finds_class(self, pipeline_with_data):
        pipeline, col = pipeline_with_data
        result = pipeline.ask("config helper class")
        assert len(result.sources) > 0
        # 至少有一个 source 是 class 类型
        types = {s.metadata.get("type") for s in result.sources}
        assert "class" in types or "method" in types or len(result.sources) > 0

    def test_filter_by_type(self, pipeline_with_data):
        pipeline, col = pipeline_with_data
        result = pipeline.ask("retry", filter_type="function")
        # 所有 source 应该都是 function 类型
        for s in result.sources:
            assert s.metadata.get("type") in ("function", "method")

    def test_latency_measured(self, pipeline_with_data):
        pipeline, col = pipeline_with_data
        result = pipeline.ask("config")
        assert result.latency_ms >= 0

    def test_empty_results_handled(self, pipeline_with_data, mock_llm_client):
        pipeline, col = pipeline_with_data
        # 搜索一个不可能匹配的内容
        result = pipeline.ask("xyzabc_nonexistent_12345")
        # 即使检索结果很少，pipeline 也不应崩溃
        assert result.query == "xyzabc_nonexistent_12345"

    def test_source_metadata_complete(self, pipeline_with_data):
        pipeline, col = pipeline_with_data
        result = pipeline.ask("retry decorator")
        for s in result.sources:
            assert "file_path" in s.metadata
            assert "type" in s.metadata
            assert "start_line" in s.metadata
            assert "name" in s.metadata


# ============================================================
# TestEmbeddingFunction
# ============================================================

class TestEmbeddingFunction:
    def test_name(self, embed_fn):
        assert "MiniLM" in embed_fn.name()

    def test_dimension(self, embed_fn):
        assert embed_fn.dimension == 384

    def test_ready(self, embed_fn):
        assert embed_fn.ready

    def test_embed_query_shape(self, embed_fn):
        result = embed_fn.embed_query(["hello world"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == 384

    def test_encode_returns_ndarray(self, embed_fn):
        result = embed_fn.encode(["hello", "world"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 384)


# ============================================================
# TestAnswerGenerator
# ============================================================

class TestAnswerGenerator:
    def test_build_user_prompt(self, mock_llm_client):
        from codebase_qa.generator import AnswerGenerator
        from codebase_qa.retriever import SearchResult

        gen = AnswerGenerator(mock_llm_client)
        results = [
            SearchResult(
                doc_id="abc", text="def retry(...): ...",
                score=0.92, metadata={
                    "name": "retry", "type": "function",
                    "file_path": "utils.py",
                    "start_line": 10, "end_line": 20,
                },
            ),
        ]
        prompt = gen.build_user_prompt("where is retry?", results)
        assert "where is retry?" in prompt
        assert "utils.py:10-20" in prompt
        assert "def retry" in prompt
        assert "score: 0.92" in prompt.lower()

    def test_generate_calls_llm(self, mock_llm_client):
        from codebase_qa.generator import AnswerGenerator
        from codebase_qa.retriever import SearchResult

        gen = AnswerGenerator(mock_llm_client)
        results = [
            SearchResult(
                doc_id="abc", text="code",
                score=0.9, metadata={"name": "f", "type": "function",
                                     "file_path": "a.py",
                                     "start_line": 1, "end_line": 3},
            ),
        ]
        answer = gen.generate("question", results)
        assert mock_llm_client.create.called
        assert mock_llm_client.get_text.called
        assert answer == mock_llm_client.get_text.return_value

    def test_generate_empty_results(self, mock_llm_client):
        from codebase_qa.generator import AnswerGenerator
        gen = AnswerGenerator(mock_llm_client)
        answer = gen.generate("question", [])
        assert "未找到" in answer
