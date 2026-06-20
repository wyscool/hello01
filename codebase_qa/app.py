# ============================================================
# codebase_qa/app.py — FastAPI 服务
# ============================================================
# 启动:
#   uvicorn codebase_qa.app:app --port 8003
#
# 端点:
#   GET  /health  — 健康检查
#   GET  /status  — 服务状态 + 索引统计
#   POST /index   — 索引代码目录
#   POST /query   — 代码库问答
# ============================================================

import os
import sys
import time
import uuid
import asyncio
import traceback
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from codebase_qa.config import AppConfig
from codebase_qa.pipeline import EmbeddingFunction, QAPipeline
from codebase_qa.indexer import CodeIndexer
from codebase_qa.retriever import Retriever, Reranker
from codebase_qa.generator import AnswerGenerator

from deploy.agent_core import LlmClient
from deploy.infrastructure import (
    RateLimiter, HealthChecker, GracefulShutdown, ServiceStats,
)
from deploy.observability import JsonLogger, Trace
from deploy.cost_control import ExactCache


# ============================================================
# 零、全局状态
# ============================================================

config = AppConfig.from_env()
logger = JsonLogger("codebase-qa", min_level="INFO")
stats = ServiceStats()
health = HealthChecker()
shutdown_mgr = GracefulShutdown()
rate_limiter = RateLimiter(config.rate_limit_per_minute)

# 在 lifespan 中初始化的组件
embed_fn: EmbeddingFunction | None = None
code_indexer: CodeIndexer | None = None
chroma_client = None
chroma_collection = None
retriever: Retriever | None = None
reranker: Reranker | None = None
generator: AnswerGenerator | None = None
qa_pipeline: QAPipeline | None = None
llm_client: LlmClient | None = None
query_cache: ExactCache | None = None

# 索引统计
_indexed_file_count: int = 0
_total_chunks: int = 0


def _check_llm() -> tuple[bool, str]:
    if llm_client is None:
        return False, "LLM client 未初始化"
    ok = llm_client.is_healthy
    return ok, "reachable" if ok else "offline"


def _check_chroma() -> tuple[bool, str]:
    if chroma_collection is None:
        return False, "ChromaDB collection 未初始化"
    try:
        count = chroma_collection.count()
        return True, f"{count} chunks"
    except Exception as e:
        return False, str(e)


def _check_embedding() -> tuple[bool, str]:
    if embed_fn is None:
        return False, "Embedding 未初始化"
    ok = embed_fn.ready
    return ok, f"dim={embed_fn.dimension}" if ok else "not loaded"


# ============================================================
# 一、中间件
# ============================================================

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get(
            "X-Request-ID", str(uuid.uuid4())[:8]
        )
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000
        logger.info("request", method=request.method,
                    path=request.url.path,
                    status=response.status_code,
                    latency_ms=round(elapsed, 1))
        return response


class ShutdownGateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            shutdown_mgr.start_request()
        except RuntimeError:
            return JSONResponse(
                status_code=503,
                content={"detail": "服务正在关闭"},
            )
        try:
            response = await call_next(request)
            return response
        finally:
            shutdown_mgr.end_request()


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not rate_limiter.allow():
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁, 请稍后重试"},
            )
        return await call_next(request)


# ============================================================
# 二、生命周期
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_fn, code_indexer, chroma_client, chroma_collection
    global retriever, reranker, generator, qa_pipeline, llm_client, query_cache

    print(f"\n  Codebase Q&A API")
    print(f"  Model: {config.llm_model}")
    print(f"  Embedding: {config.embedding_model}")
    print(f"  Port: {config.port}")

    # 1. Embedding model
    print("  Loading embedding model...")
    embed_fn = EmbeddingFunction(model_name=config.embedding_model)
    embed_fn.embed_query(["ping"])
    print(f"  Embedding: dim={embed_fn.dimension}")

    # 2. ChromaDB
    import chromadb
    from chromadb.config import Settings
    persist_dir = Path(config.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    chroma_collection = chroma_client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=embed_fn,
    )
    print(f"  ChromaDB: collection={config.collection_name}, "
          f"chunks={chroma_collection.count()}")

    # 3. CodeIndexer
    code_indexer = CodeIndexer(exclude_dirs=config.exclude_set)
    print(f"  Indexer: exclude={config.exclude_set}")

    # 4. LlmClient
    llm_client = LlmClient(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
        max_retries=config.llm_max_retries,
        timeout=config.llm_timeout_seconds,
    )
    print(f"  LLM: {'reachable' if llm_client.is_healthy else 'offline'}")

    # 5. Pipeline
    retriever = Retriever(chroma_collection)
    reranker = Reranker(embed_fn)
    generator = AnswerGenerator(llm_client)
    qa_pipeline = QAPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        top_k=config.top_k,
        min_score=config.min_score,
        use_mmr=config.use_mmr,
        mmr_lambda=config.mmr_lambda,
    )
    print(f"  Pipeline: top_k={config.top_k}, mmr={config.use_mmr}")

    # 6. Cache
    if config.cache_enabled:
        query_cache = ExactCache(
            max_size=config.cache_max_size,
            ttl_seconds=config.cache_ttl_seconds,
        )
        print(f"  Cache: max={config.cache_max_size}, "
              f"ttl={config.cache_ttl_seconds}s")

    # 7. Health checks
    health.register_check("llm", _check_llm)
    health.register_check("chroma", _check_chroma)
    health.register_check("embedding", _check_embedding)
    health.run_checks()
    health.set_ready()
    print("  Ready.\n")

    yield

    # Shutdown
    print("\n  Shutting down...")
    shutdown_mgr.initiate(health, logger)
    print("  Done.")


# ============================================================
# 三、FastAPI App
# ============================================================

app = FastAPI(
    title="Codebase Q&A",
    description="AST-based code indexing and natural language Q&A",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIdMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ShutdownGateMiddleware)
app.add_middleware(RateLimitMiddleware)


def _ensure_ready():
    if qa_pipeline is None:
        raise HTTPException(status_code=503, detail="服务未就绪")


# ============================================================
# 四、端点
# ============================================================

@app.get("/health")
async def get_health():
    health.run_checks()
    return health.status()


@app.get("/status")
async def get_status():
    _ensure_ready()
    return {
        "service": "codebase-qa",
        "version": "0.1.0",
        "stats": stats.snapshot(),
        "rate_limiter": rate_limiter.stats(),
        "cache": query_cache.stats() if query_cache else None,
        "index": {
            "files_indexed": code_indexer.indexed_files if code_indexer else 0,
            "total_chunks": chroma_collection.count() if chroma_collection else 0,
            "collection_name": config.collection_name,
            "exclude_dirs": sorted(config.exclude_set),
        },
        "config": {
            "model": config.llm_model,
            "embedding": config.embedding_model,
            "mmr": config.use_mmr,
            "top_k": config.top_k,
        },
    }


@app.post("/index")
async def post_index(req: Request):
    """索引一个或多个目录。

    Body: {"dirs": ["./phase1", "./deploy"]}
    若 dirs 为空，使用 config.project_dirs。
    """
    _ensure_ready()

    body = await req.json()
    dir_names = body.get("dirs", []) if body else []

    if not dir_names:
        dir_names = config.project_dir_list

    if not dir_names:
        raise HTTPException(status_code=400, detail="请提供 dirs 参数")

    total_chunks = 0
    total_files = 0
    errors: list[str] = []

    for dir_name in dir_names:
        root = Path(dir_name)
        if not root.exists():
            errors.append(f"目录不存在: {dir_name}")
            continue

        # 阻塞操作放入线程池
        chunks = await asyncio.to_thread(code_indexer.index_directory, root)

        if not chunks:
            continue

        total_files += code_indexer.indexed_files

        # 写入 ChromaDB
        ids = [c.chunk_id for c in chunks]
        documents = [c.embed_text for c in chunks]
        metadatas = [code_indexer.chunk_to_metadata(c) for c in chunks]

        # ChromaDB upsert: 先删除同文件的旧数据，再添加
        for c in chunks:
            # 删除同文件旧数据（按 file_path 过滤）
            try:
                existing = chroma_collection.get(
                    where={"file_path": c.file_path}
                )
                if existing and existing.get("ids"):
                    chroma_collection.delete(ids=existing["ids"])
            except Exception:
                pass  # 首次索引没有旧数据是正常的

        await asyncio.to_thread(
            chroma_collection.add,
            ids=ids, documents=documents, metadatas=metadatas,
        )
        total_chunks += len(chunks)

    return {
        "status": "ok",
        "indexed_files": total_files,
        "total_chunks": total_chunks,
        "collection_count": chroma_collection.count() if chroma_collection else 0,
        "errors": errors,
    }


@app.post("/query")
async def post_query(req: Request):
    """查询代码库。

    Body: {
        "question": "...",
        "top_k": 5 (可选),
        "filter_type": "function" (可选: function/class/method/module_level)
    }
    """
    _ensure_ready()

    body = await req.json()
    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    filter_type = body.get("filter_type")
    top_k = body.get("top_k", config.top_k)

    # 缓存检查
    cache_key = question
    if query_cache and config.cache_enabled:
        cached = query_cache.get(cache_key)
        if cached is not None:
            stats.record(status_code=200, latency_ms=0.0)
            return {
                "question": question,
                "answer": cached["answer"],
                "sources": cached["sources"],
                "latency_ms": 0.0,
                "cached": True,
            }

    # 执行查询
    t_start = time.time()

    # 临时覆盖 top_k
    orig_top_k = qa_pipeline.top_k
    qa_pipeline.top_k = top_k

    response = await asyncio.to_thread(
        qa_pipeline.ask, question, filter_type,
    )

    qa_pipeline.top_k = orig_top_k

    total_latency = (time.time() - t_start) * 1000
    stats.record(status_code=200, latency_ms=total_latency)

    source_list = [
        {
            "file_path": s.metadata.get("file_path", "?"),
            "name": s.metadata.get("name", "?"),
            "type": s.metadata.get("type", "?"),
            "start_line": s.metadata.get("start_line", "?"),
            "end_line": s.metadata.get("end_line", "?"),
            "score": s.score,
            "code": s.text[:300],
        }
        for s in response.sources
    ]

    result = {
        "question": question,
        "answer": response.answer,
        "sources": source_list,
        "latency_ms": round(total_latency, 1),
        "cached": False,
    }

    # 写入缓存
    if query_cache and config.cache_enabled:
        query_cache.set(cache_key, {"answer": response.answer, "sources": source_list})

    return result


# ============================================================
# 五、直接运行
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "codebase_qa.app:app",
        host=config.host,
        port=config.port,
        reload=False,
    )
