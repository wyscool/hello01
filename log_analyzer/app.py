# ============================================================
# log_analyzer/app.py — 日志分析 FastAPI 服务
# ============================================================
# 启动:
#   uvicorn log_analyzer.app:app --port 8003
#
# 端点:
#   GET  /              — 服务信息
#   GET  /health        — 健康检查
#   GET  /status        — 服务状态
#   POST /analyze       — 分析日志 (指定服务器上的文件路径)
#   POST /analyze/upload — 上传日志文件并分析
#   GET  /cache         — 缓存统计
# ============================================================

import os
import sys
import time
import uuid
import hashlib
import asyncio
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from log_analyzer.config import AppConfig
from log_analyzer.parser import LogParser
from log_analyzer.agent import LogAnalysisAgent, create_log_agent

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
logger = JsonLogger("log-analyzer", min_level="INFO")
stats = ServiceStats()
health = HealthChecker()
shutdown_mgr = GracefulShutdown()
rate_limiter = RateLimiter(config.rate_limit_per_minute)

llm_client: LlmClient | None = None
parser: LogParser | None = None
query_cache: ExactCache | None = None

# 已加载的日志文件 (session-file → entries)
_session_logs: dict[str, list] = {}


def _check_llm() -> tuple[bool, str]:
    if llm_client is None:
        return False, "LLM client 未初始化"
    ok = llm_client.is_healthy
    return ok, "reachable" if ok else "offline"


# ============================================================
# 一、中间件
# ============================================================

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
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
    global llm_client, parser, query_cache

    print(f"\n  Log Analysis Agent API")
    print(f"  Model: {config.llm_model}")
    print(f"  Port: {config.port}")
    print(f"  Max file: {config.max_file_size_mb}MB")

    llm_client = LlmClient(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
        max_retries=config.llm_max_retries,
        timeout=config.llm_timeout_seconds,
    )
    print(f"  LLM: {'reachable' if llm_client.is_healthy else 'offline'}")

    parser = LogParser(max_file_size_mb=config.max_file_size_mb)
    print(f"  Parser: ready")

    query_cache = ExactCache(
        max_size=config.cache_max_size,
        ttl_seconds=config.cache_ttl_seconds,
    ) if config.cache_enabled else None
    print(f"  Cache: {'enabled' if query_cache else 'disabled'}")

    health.register_check("llm_api", _check_llm)
    health.run_checks()
    health.set_ready()

    print(f"  Health: {'pass' if health.is_healthy else 'fail'}")
    print()

    yield

    print("\n  Log Analysis Agent API 关闭")
    shutdown_mgr.initiate(health, logger)


# ============================================================
# 三、FastAPI 应用
# ============================================================

app = FastAPI(
    title="Log Analysis Agent API",
    version="1.0.0",
    description="LLM 驱动的日志分析 Agent — 上传/指定日志文件, 自然语言分析, 根因定位",
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
    if llm_client is None or parser is None:
        raise HTTPException(status_code=503, detail="服务未就绪")


def _cache_key(question: str, file_name: str) -> str:
    raw = question + file_name
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ============================================================
# 五、路由
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "Log Analysis Agent API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "status": "GET /status",
            "analyze_path": "POST /analyze",
            "analyze_upload": "POST /analyze/upload",
            "cache": "GET /cache",
        },
        "usage": {
            "analyze_path": (
                'curl -X POST http://localhost:8003/analyze '
                '-H "Content-Type: application/json" '
                '-d \'{"path": "/var/log/app.log", "question": "分析错误日志"}\''
            ),
            "analyze_upload": (
                'curl -X POST http://localhost:8003/analyze/upload '
                '-F "file=@app.log" -F "question=分析错误日志"'
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
        "service": "log-analyzer",
        "stats": stats.snapshot(),
        "rate_limiter": rate_limiter.stats(),
        "cache": query_cache.stats() if query_cache else None,
        "config": {
            "llm_model": config.llm_model,
            "max_file_size_mb": config.max_file_size_mb,
            "agent_max_iterations": config.agent_max_iterations,
            "cache_enabled": config.cache_enabled,
        },
    }


@app.get("/cache")
async def cache_stats():
    if query_cache is None:
        return {"enabled": False}
    return {"enabled": True, **query_cache.stats()}


# --- 分析 (服务端路径) ---

@app.post("/analyze")
async def analyze_path(request: Request):
    _ensure_ready()

    body = await request.json()
    path = body.get("path", "").strip()
    question = body.get("question", "").strip()

    if not path:
        raise HTTPException(status_code=400, detail="path 不能为空")
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"文件不存在: {path}")
    if not p.is_file():
        raise HTTPException(status_code=400, detail="path 不是文件")

    req_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

    # Cache
    if query_cache:
        cached = query_cache.get(question, file_name=p.name)
        if cached is not None:
            cached["cached"] = True
            cached["request_id"] = req_id
            return cached

    # 分析
    trace = Trace("analyze", req_id)
    span = trace.start_span("analyze", question=question[:80], file=p.name)

    try:
        agent = await asyncio.to_thread(create_log_agent, p, config, logger)
        result = await asyncio.to_thread(agent.analyze, question)
    except Exception as e:
        stats.record(500, 0)
        raise HTTPException(status_code=500, detail=f"分析失败: {e}")

    trace.end_span(span, "ok", iterations=result.iterations)

    stats.record(200, result.latency_ms)

    response = {
        "request_id": req_id,
        "question": question,
        "answer": result.answer,
        "file_name": result.file_name,
        "total_entries": result.total_entries,
        "iterations": result.iterations,
        "tool_calls": result.tool_calls,
        "latency_ms": result.latency_ms,
        "cached": False,
    }

    if query_cache:
        query_cache.set(question, response, file_name=p.name)

    return response


# --- 分析 (文件上传) ---

@app.post("/analyze/upload")
async def analyze_upload(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form(...),
):
    _ensure_ready()

    req_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

    # 保存上传文件到临时目录
    suffix = Path(file.filename).suffix or ".log"
    with tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False, prefix="log_upload_"
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        p = Path(tmp_path)

        # 大小检查
        if p.stat().st_size > config.max_file_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"文件过大 (>{config.max_file_size_mb}MB)"
            )

        trace = Trace("analyze-upload", req_id)
        span = trace.start_span("analyze", question=question[:80],
                               file=file.filename)

        agent = await asyncio.to_thread(create_log_agent, p, config, logger)
        result = await asyncio.to_thread(agent.analyze, question)

        trace.end_span(span, "ok", iterations=result.iterations)
        stats.record(200, result.latency_ms)

        return {
            "request_id": req_id,
            "question": question,
            "answer": result.answer,
            "file_name": file.filename,
            "total_entries": result.total_entries,
            "iterations": result.iterations,
            "tool_calls": result.tool_calls,
            "latency_ms": result.latency_ms,
        }
    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ============================================================
# 六、直接运行
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print(f"\n  Log Analysis Agent API")
    print(f"  http://{config.host}:{config.port}")
    print(f"  Docs: http://{config.host}:{config.port}/docs")
    uvicorn.run(
        "log_analyzer.app:app",
        host=config.host, port=config.port,
        log_level="warning",
    )
