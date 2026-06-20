# ============================================================
# mcp_server/tests/test_tools.py — 工具处理函数单元测试
# ============================================================
# 测试策略:
#   - Mock AppContext（模拟 lifespan_context）
#   - Mock QAPipeline（避免真实 LLM 调用）
#   - Mock ChromaDB Collection（避免真实向量数据库）
#   - 测试正常路径、错误路径、边界情况
#
# 类比 Java:
#   @ExtendWith(MockitoExtension.class) + @MockBean
# ============================================================

import json
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from dataclasses import dataclass

from mcp_server.lifecycle import AppContext


# ============================================================
# mock 工厂函数
# ============================================================

def _make_search_result(doc_id, name, type, file_path, score, text):
    """创建 Mock SearchResult。"""
    from codebase_qa.retriever import SearchResult
    return SearchResult(
        doc_id=doc_id, text=text, score=score,
        metadata={
            "name": name, "type": type, "file_path": file_path,
            "start_line": 10, "end_line": 20,
        },
    )


def _make_qaresponse(query, answer, sources):
    """创建 Mock QAResponse。"""
    from codebase_qa.pipeline import QAResponse
    return QAResponse(query=query, answer=answer, sources=sources, latency_ms=150.0)


# ============================================================
# fixtures
# ============================================================

@pytest.fixture
def mock_pipeline():
    pipeline = Mock()
    pipeline.top_k = 5
    pipeline.ask.return_value = _make_qaresponse(
        query="test query",
        answer="在 [utils.py:10-20] 找到了 retry 函数。",
        sources=[
            _make_search_result(
                "abc", "retry", "function", "utils.py", 0.92,
                "def retry(max_attempts=3): ...",
            ),
        ],
    )
    return pipeline


@pytest.fixture
def mock_indexer():
    from codebase_qa.indexer import CodeChunk
    indexer = Mock()
    indexer.indexed_files = 2
    indexer.index_directory.return_value = [
        CodeChunk(
            name="my_func", type="function",
            file_path="test.py", start_line=1, end_line=5,
            code_text="def my_func(): pass",
            file_hash="abc123",
        ),
    ]
    indexer.chunk_to_metadata.return_value = {
        "name": "my_func", "type": "function",
        "file_path": "test.py", "start_line": 1, "end_line": 5,
        "signature": "def my_func():",
    }
    return indexer


@pytest.fixture
def mock_collection():
    col = Mock()
    col.count.return_value = 10
    col.get.return_value = {"ids": []}
    return col


@pytest.fixture
def mock_config():
    from mcp_server.config import McpServerConfig
    from codebase_qa.config import AppConfig
    qa = AppConfig(
        collection_name="test_collection",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="claude-sonnet-4-6",
        top_k=5, min_score=0.3,
        use_mmr=True, mmr_lambda=0.7,
        cache_enabled=False,
    )
    return McpServerConfig(
        server_name="test-server",
        server_version="0.1.0",
        qa=qa,
    )


@pytest.fixture
def app_ctx(mock_pipeline, mock_indexer, mock_collection, mock_config):
    return AppContext(
        qa_pipeline=mock_pipeline,
        code_indexer=mock_indexer,
        chroma_collection=mock_collection,
        config=mock_config,
        query_cache=None,
    )


def _make_ctx(app_ctx: AppContext):
    """构造模拟的 MCP Context，注入 AppContext。"""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app_ctx
    return ctx


# ============================================================
# TestCodebaseSearch
# ============================================================

class TestCodebaseSearch:
    pytestmark = pytest.mark.asyncio

    async def test_basic_search(self, app_ctx):
        from mcp_server.tools import handle_codebase_search
        ctx = _make_ctx(app_ctx)
        result_str = await handle_codebase_search(ctx, "retry function", top_k=5)
        result = json.loads(result_str)
        assert result["question"] == "retry function"
        assert "utils.py" in result["answer"]
        assert len(result["sources"]) == 1
        assert result["sources"][0]["file_path"] == "utils.py"
        assert result["sources"][0]["name"] == "retry"
        assert result["cached"] is False

    async def test_empty_query(self, app_ctx):
        from mcp_server.tools import handle_codebase_search
        ctx = _make_ctx(app_ctx)
        result_str = await handle_codebase_search(ctx, "  ")
        result = json.loads(result_str)
        assert "error" in result
        assert "不能为空" in result["error"]

    async def test_pipeline_not_ready(self):
        from mcp_server.tools import handle_codebase_search
        ctx = _make_ctx(AppContext(qa_pipeline=None))
        result_str = await handle_codebase_search(ctx, "test")
        result = json.loads(result_str)
        assert "error" in result
        assert "未就绪" in result["error"]

    async def test_with_filter_type(self, app_ctx):
        from mcp_server.tools import handle_codebase_search
        ctx = _make_ctx(app_ctx)
        await handle_codebase_search(ctx, "retry", filter_type="function")
        # 验证 filter_type 传递到了 pipeline.ask
        app_ctx.qa_pipeline.ask.assert_called_with("retry", "function")

    async def test_latency_present(self, app_ctx):
        from mcp_server.tools import handle_codebase_search
        ctx = _make_ctx(app_ctx)
        result_str = await handle_codebase_search(ctx, "query")
        result = json.loads(result_str)
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0

    async def test_top_k_restored(self, app_ctx):
        """验证临时修改的 top_k 在调用后恢复。"""
        from mcp_server.tools import handle_codebase_search
        ctx = _make_ctx(app_ctx)
        original = app_ctx.qa_pipeline.top_k
        await handle_codebase_search(ctx, "query", top_k=3)
        assert app_ctx.qa_pipeline.top_k == original


# ============================================================
# TestCodebaseIndex
# ============================================================

class TestCodebaseIndex:
    pytestmark = pytest.mark.asyncio

    async def test_basic_index(self, app_ctx, tmp_path):
        from mcp_server.tools import handle_codebase_index
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        (test_dir / "main.py").write_text("def hello(): pass")

        ctx = _make_ctx(app_ctx)
        result_str = await handle_codebase_index(ctx, [str(test_dir)])
        result = json.loads(result_str)
        assert result["status"] == "ok"
        assert result["total_chunks"] > 0

    async def test_empty_dirs(self, app_ctx):
        from mcp_server.tools import handle_codebase_index
        ctx = _make_ctx(app_ctx)
        result_str = await handle_codebase_index(ctx, [])
        result = json.loads(result_str)
        assert "error" in result
        assert "不能为空" in result["error"]

    async def test_nonexistent_directory(self, app_ctx):
        from mcp_server.tools import handle_codebase_index
        ctx = _make_ctx(app_ctx)
        result_str = await handle_codebase_index(ctx, ["/tmp/nonexistent_path_xyz"])
        result = json.loads(result_str)
        # 如果没有目录存在, errors 列表会有记录
        if "errors" in result:
            assert any("不存在" in e for e in result["errors"])
        assert "error" in result or "status" in result

    async def test_service_not_ready(self):
        from mcp_server.tools import handle_codebase_index
        ctx = _make_ctx(AppContext(code_indexer=None, chroma_collection=None))
        result_str = await handle_codebase_index(ctx, ["./some_dir"])
        result = json.loads(result_str)
        assert "error" in result


# ============================================================
# TestCodebaseStatus
# ============================================================

class TestCodebaseStatus:
    pytestmark = pytest.mark.asyncio

    async def test_basic_status(self, app_ctx):
        from mcp_server.tools import handle_codebase_status
        ctx = _make_ctx(app_ctx)
        result_str = await handle_codebase_status(ctx)
        result = json.loads(result_str)
        assert result["service"] == "codebase-qa-mcp"
        assert result["index"]["files_indexed"] == 2
        assert result["index"]["total_chunks"] == 10
        assert "embedding_model" in result["index"]

    async def test_config_fields(self, app_ctx):
        from mcp_server.tools import handle_codebase_status
        ctx = _make_ctx(app_ctx)
        result_str = await handle_codebase_status(ctx)
        result = json.loads(result_str)
        assert "mmr_enabled" in result["config"]
        assert "top_k" in result["config"]

    async def test_no_collection(self):
        from mcp_server.tools import handle_codebase_status
        ctx = _make_ctx(AppContext(chroma_collection=None))
        result_str = await handle_codebase_status(ctx)
        result = json.loads(result_str)
        assert "error" in result
