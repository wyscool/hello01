# ============================================================
# mcp_server/tools.py — MCP 工具处理函数
# ============================================================
# 三个工具，每个对应一个 MCP Tool:
#   handle_codebase_search — 自然语言搜索代码库
#   handle_codebase_index  — 索引代码目录
#   handle_codebase_status — 查询索引状态
#
# 每个函数通过 ctx.request_context.lifespan_context 获取
# AppContext（全局依赖: qa_pipeline, code_indexer, cache 等）。
#
# 类比 Java:
#   这些函数相当于 @Service 的方法,
#   ctx.request_context.lifespan_context 相当于 @Autowired 的 Bean。
# ============================================================

import json
import time
import asyncio
from pathlib import Path

from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from mcp_server.lifecycle import AppContext


# ============================================================
# 工具 1: codebase_search
# ============================================================

async def handle_codebase_search(
    ctx: Context[ServerSession, AppContext],
    query: str,
    top_k: int = 5,
    filter_type: str = "",
) -> str:
    """自然语言搜索代码库。

    Returns JSON: { question, answer, sources, latency_ms, cached }
    """
    app: AppContext = ctx.request_context.lifespan_context
    pipeline = app.qa_pipeline

    if pipeline is None:
        return json.dumps({"error": "服务未就绪, QAPipeline 未初始化"},
                          ensure_ascii=False)

    if not query.strip():
        return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)

    # 缓存检查
    cache = app.query_cache
    if cache is not None and app.config is not None and app.config.qa.cache_enabled:
        cached = cache.get(query)
        if cached is not None:
            return json.dumps({
                "question": query,
                "answer": cached["answer"],
                "sources": cached["sources"],
                "latency_ms": 0.0,
                "cached": True,
            }, ensure_ascii=False)

    # 临时覆盖 top_k
    orig_top_k = pipeline.top_k
    pipeline.top_k = top_k
    effective_filter = filter_type if filter_type else None

    # QAPipeline.ask() 是同步阻塞操作 → 在线程池中运行
    t_start = time.time()
    response = await asyncio.to_thread(pipeline.ask, query, effective_filter)
    pipeline.top_k = orig_top_k

    total_latency = (time.time() - t_start) * 1000

    # 格式化 sources
    source_list = [
        {
            "file_path": s.metadata.get("file_path", "?"),
            "name": s.metadata.get("name", "?"),
            "type": s.metadata.get("type", "?"),
            "start_line": s.metadata.get("start_line", "?"),
            "end_line": s.metadata.get("end_line", "?"),
            "score": round(s.score, 4),
            "code_preview": s.text[:300],
        }
        for s in response.sources
    ]

    result = {
        "question": query,
        "answer": response.answer,
        "sources": source_list,
        "latency_ms": round(total_latency, 1),
        "cached": False,
    }

    # 写入缓存
    if cache is not None and app.config is not None and app.config.qa.cache_enabled:
        cache.set(query, {"answer": response.answer, "sources": source_list})

    return json.dumps(result, ensure_ascii=False)


# ============================================================
# 工具 2: codebase_index
# ============================================================

async def handle_codebase_index(
    ctx: Context[ServerSession, AppContext],
    dirs: list[str],
) -> str:
    """索引一个或多个代码目录。

    Returns JSON: { status, indexed_files, total_chunks, collection_count, errors }
    """
    app: AppContext = ctx.request_context.lifespan_context
    indexer = app.code_indexer
    collection = app.chroma_collection

    if indexer is None or collection is None:
        return json.dumps({"error": "服务未就绪"}, ensure_ascii=False)

    if not dirs:
        return json.dumps({"error": "dirs 不能为空"}, ensure_ascii=False)

    total_chunks = 0
    total_files = 0
    errors: list[str] = []

    for dir_name in dirs:
        root = Path(dir_name)
        if not root.exists():
            errors.append(f"目录不存在: {dir_name}")
            continue

        # 索引是 CPU/IO 混合操作 → 线程池
        chunks = await asyncio.to_thread(indexer.index_directory, root)

        if not chunks:
            continue

        total_files += indexer.indexed_files

        # 写入 ChromaDB
        ids = [c.chunk_id for c in chunks]
        documents = [c.embed_text for c in chunks]
        metadatas = [indexer.chunk_to_metadata(c) for c in chunks]

        # 删除同文件的旧数据（upsert 模拟）
        for c in chunks:
            try:
                existing = collection.get(
                    where={"file_path": c.file_path}
                )
                if existing and existing.get("ids"):
                    collection.delete(ids=existing["ids"])
            except Exception:
                pass

        await asyncio.to_thread(
            collection.add,
            ids=ids, documents=documents, metadatas=metadatas,
        )
        total_chunks += len(chunks)

    return json.dumps({
        "status": "ok",
        "indexed_files": total_files,
        "total_chunks": total_chunks,
        "collection_count": collection.count(),
        "errors": errors,
    }, ensure_ascii=False)


# ============================================================
# 工具 3: codebase_status
# ============================================================

async def handle_codebase_status(
    ctx: Context[ServerSession, AppContext],
) -> str:
    """查询索引统计信息。

    Returns JSON: { service, version, index: {...}, config: {...} }
    """
    app: AppContext = ctx.request_context.lifespan_context
    collection = app.chroma_collection
    indexer = app.code_indexer
    cfg = app.config

    if collection is None:
        return json.dumps({"error": "ChromaDB collection 未初始化"},
                          ensure_ascii=False)

    qa = cfg.qa
    return json.dumps({
        "service": "codebase-qa-mcp",
        "version": cfg.server_version,
        "index": {
            "files_indexed": indexer.indexed_files if indexer else 0,
            "total_chunks": collection.count(),
            "collection_name": qa.collection_name,
            "embedding_model": qa.embedding_model,
        },
        "config": {
            "llm_model": qa.llm_model,
            "mmr_enabled": qa.use_mmr,
            "top_k": qa.top_k,
            "cache_enabled": qa.cache_enabled,
        },
    }, ensure_ascii=False)
