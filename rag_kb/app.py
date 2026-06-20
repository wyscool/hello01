# ============================================================
# rag_kb/app.py — RAG 知识库 FastAPI 服务
# ============================================================
# 启动:
#   uvicorn rag_kb.app:app --port 8002
#
# 端点:
#   GET  /              — 服务信息
#   GET  /health        — 健康检查
#   GET  /status        — 服务状态 + 统计
#   GET  /docs          — 知识库文档列表
#   GET  /docs/{source} — 文档详情
#   DELETE /docs/{source} — 删除文档
#   POST /ingest        — 导入文档
#   POST /query         — 知识库问答
# ============================================================

import os
import sys
import time
import uuid
import hashlib
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from rag_kb.config import AppConfig
from rag_kb.pipeline import (
    EmbeddingFunction, DocumentProcessor,
    Retriever, Reranker, ContextBuilder, Generator, RAGPipeline,
)
from rag_kb.knowledge_base import KnowledgeBase, create_knowledge_base

# 直接复用 deploy/ 的生产级组件
from deploy.agent_core import LlmClient
from deploy.infrastructure import (
    RateLimiter, HealthChecker, GracefulShutdown, ServiceStats,
)
from deploy.observability import JsonLogger, Trace
from deploy.cost_control import ExactCache

# ============================================================
# 〇、全局状态
# ============================================================

config = AppConfig.from_env()
logger = JsonLogger("rag-kb", min_level="INFO")
stats = ServiceStats()
health = HealthChecker()
shutdown_mgr = GracefulShutdown()
rate_limiter = RateLimiter(config.rate_limit_per_minute)

# RAG 组件 (lifespan 中初始化)
embed_fn: EmbeddingFunction | None = None
knowledge_base: KnowledgeBase | None = None
rag_pipeline: RAGPipeline | None = None
llm_client: LlmClient | None = None
query_cache: ExactCache | None = None


def _check_llm() -> tuple[bool, str]:
    if llm_client is None:
        return False, "LLM client 未初始化"
    ok = llm_client.is_healthy
    return ok, "reachable" if ok else "offline"


def _check_chroma() -> tuple[bool, str]:
    if knowledge_base is None:
        return False, "KnowledgeBase 未初始化"
    ok = knowledge_base.is_connected
    return ok, f"{knowledge_base.count} chunks" if ok else "disconnected"


def _check_embedding() -> tuple[bool, str]:
    if embed_fn is None:
        return False, "Embedding 未初始化"
    ok = embed_fn.is_ready
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
    global embed_fn, knowledge_base, rag_pipeline, llm_client, query_cache

    print(f"\n  RAG Knowledge Base API")
    print(f"  Model: {config.llm_model}")
    print(f"  Embedding: {config.embedding_model}")
    print(f"  Port: {config.port}")

    # 1. Embedding model
    print("  Loading embedding model...")
    embed_fn = EmbeddingFunction(model_name=config.embedding_model)
    embed_fn.embed_query(["ping"])  # trigger load
    print(f"  Embedding: dim={embed_fn.dimension}")

    # 2. KnowledgeBase (ChromaDB)
    knowledge_base = create_knowledge_base(config)
    print(f"  KB: collection={config.chroma_collection_name}, "
          f"chunks={knowledge_base.count}")

    # 3. LlmClient
    llm_client = LlmClient(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
        max_retries=config.llm_max_retries,
        timeout=config.llm_timeout_seconds,
    )
    print(f"  LLM: {'reachable' if llm_client.is_healthy else 'offline'}")

    # 4. RAG Pipeline
    retriever = Retriever(knowledge_base.collection)
    reranker = Reranker(embed_fn)
    context_builder = ContextBuilder()
    generator = Generator(llm_client)

    rag_pipeline = RAGPipeline(
        retriever=retriever, reranker=reranker,
        context_builder=context_builder, generator=generator,
        top_k=config.retrieval_top_k,
        min_score=config.retrieval_min_score,
        use_mmr=config.use_mmr,
        mmr_lambda=config.mmr_lambda,
    )

    # 5. Cache
    query_cache = ExactCache(
        max_size=config.cache_max_size,
        ttl_seconds=config.cache_ttl_seconds,
    ) if config.cache_enabled else None

    # 6. Health checks
    (health
     .register_check("llm_api", _check_llm)
     .register_check("chromadb", _check_chroma)
     .register_check("embedding", _check_embedding))
    health.run_checks()
    health.set_ready()

    print(f"  Health: {'pass' if health.is_healthy else 'fail'}")
    print()

    yield

    print("\n  RAG Knowledge Base API 关闭")
    shutdown_mgr.initiate(health, logger)


# ============================================================
# 三、FastAPI 应用
# ============================================================

app = FastAPI(
    title="RAG Knowledge Base API",
    version="1.0.0",
    description="RAG 知识库问答服务 — 导入文档, 自然语言检索, LLM 生成答案",
    lifespan=lifespan,
)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ShutdownGateMiddleware)
app.add_middleware(RateLimitMiddleware)


# ============================================================
# 四、辅助函数
# ============================================================

def _ensure_ready():
    if knowledge_base is None or rag_pipeline is None:
        raise HTTPException(status_code=503, detail="服务未就绪")


def _cache_key(question: str, filter_source: str | None) -> str:
    raw = question + (filter_source or "")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ============================================================
# 五、路由
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "RAG Knowledge Base API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "status": "GET /status",
            "list_docs": "GET /kb",
            "get_doc": "GET /kb/{source}",
            "delete_doc": "DELETE /kb/{source}",
            "ingest": "POST /ingest",
            "query": "POST /query",
        },
        "usage": {
            "ingest_example": (
                'curl -X POST http://localhost:8002/ingest '
                '-H "Content-Type: application/json" '
                '-d \'{"path": "./my_docs/", "type": "directory"}\''
            ),
            "query_example": (
                'curl -X POST http://localhost:8002/query '
                '-H "Content-Type: application/json" '
                '-d \'{"question": "Redis 有哪些数据结构?"}\''
            ),
        },
    }


@app.get("/health")
async def health_check():
    health.run_checks()
    return health.status()


@app.get("/status")
async def status():
    _ensure_ready()
    return {
        "service": "rag-kb",
        "stats": stats.snapshot(),
        "rate_limiter": rate_limiter.stats(),
        "kb": knowledge_base.stats(),
        "cache": query_cache.stats() if query_cache else None,
        "config": {
            "llm_model": config.llm_model,
            "embedding_model": config.embedding_model,
            "chunk_size": config.chunk_size,
            "top_k": config.retrieval_top_k,
            "mmr": config.use_mmr,
        },
    }


# --- 文档管理 ---

@app.get("/kb")
async def list_docs():
    _ensure_ready()
    return knowledge_base.list_docs()


@app.get("/kb/{source:path}")
async def get_doc(source: str):
    _ensure_ready()
    doc = knowledge_base.get_doc(source)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"文档不存在: {source}")
    return doc


@app.delete("/kb/{source:path}")
async def delete_doc(source: str):
    _ensure_ready()
    deleted = knowledge_base.remove_doc(source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"文档不存在: {source}")
    return {"source": source, "chunks_deleted": deleted}


# --- 文档导入 ---

@app.post("/ingest")
async def ingest(request: Request):
    _ensure_ready()

    body = await request.json()
    path = body.get("path", "").strip()
    ingest_type = body.get("type", "file")

    if not path:
        raise HTTPException(status_code=400, detail="path 不能为空")

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"路径不存在: {path}")

    files_processed = 0
    errors: list[str] = []

    try:
        if ingest_type == "directory":
            if not p.is_dir():
                raise HTTPException(status_code=400,
                                    detail="type=directory 但 path 不是目录")
            # 转换 .txt → *.txt 以匹配 glob 模式
            patterns = tuple(f"*{s}" for s in config.allowed_suffixes)
            total = await asyncio.to_thread(
                knowledge_base.add_directory, p, patterns
            )
            files_processed = len([
                f for f in p.glob("*")
                if f.is_file() and f.suffix in config.allowed_suffixes
            ])
            return {
                "success": True,
                "files_processed": files_processed,
                "chunks_added": total,
                "errors": errors,
            }
        else:
            if not p.is_file():
                raise HTTPException(status_code=400,
                                    detail="type=file 但 path 不是文件")
            if p.suffix not in config.allowed_suffixes:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的文件类型: {p.suffix}"
                )
            file_size_mb = p.stat().st_size / (1024 * 1024)
            if file_size_mb > config.max_ingest_file_mb:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件过大 ({file_size_mb:.1f}MB > "
                           f"{config.max_ingest_file_mb}MB)"
                )
            total = await asyncio.to_thread(knowledge_base.add_file, p)
            return {
                "success": True,
                "files_processed": 1 if total > 0 else 0,
                "chunks_added": total,
                "errors": errors,
            }
    except HTTPException:
        raise
    except Exception as e:
        errors.append(f"{p.name}: {e}")
        return {
            "success": False,
            "files_processed": 0,
            "chunks_added": 0,
            "errors": errors,
        }


# --- 问答 ---

@app.post("/query")
async def query(request: Request):
    _ensure_ready()

    body = await request.json()
    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    top_k = body.get("top_k", config.retrieval_top_k)
    filter_source = body.get("filter_source")

    req_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

    # Cache
    if query_cache:
        cached = query_cache.get(question, filter_source=filter_source)
        if cached is not None:
            cached["cached"] = True
            cached["request_id"] = req_id
            return cached

    # Where 过滤
    where_clause = None
    if filter_source:
        where_clause = {"source": filter_source}

    # 执行 RAG
    trace = Trace("rag-query", req_id)
    span = trace.start_span("pipeline", question=question[:80])

    try:
        result = await asyncio.to_thread(
            rag_pipeline.ask, question,
            where=where_clause,
        )
    except Exception as e:
        stats.record(500, 0)
        raise HTTPException(status_code=500, detail=f"查询失败: {e}")

    trace.end_span(span, "ok",
                   sources=len(result.sources),
                   latency_ms=result.latency_ms)

    stats.record(200, result.latency_ms)

    response = {
        "request_id": req_id,
        "question": question,
        "answer": result.answer,
        "sources": [{
            "source": s.metadata.get("source", ""),
            "chunk_index": s.metadata.get("chunk_index", 0),
            "score": s.score,
            "text_preview": s.text[:200],
        } for s in result.sources],
        "latency_ms": result.latency_ms,
        "cached": False,
    }

    # 写入缓存
    if query_cache:
        query_cache.set(question, response, filter_source=filter_source)

    return response


# ============================================================
# 六、直接运行
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print(f"\n  RAG Knowledge Base API")
    print(f"  http://{config.host}:{config.port}")
    print(f"  Docs: http://{config.host}:{config.port}/docs")
    uvicorn.run(
        "rag_kb.app:app",
        host=config.host, port=config.port,
        log_level="warning",
    )
