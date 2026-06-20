# ============================================================
# deploy/app.py — FastAPI 组装入口
# ============================================================
# 将所有 deploy/ 模块组装成可运行的 FastAPI 服务。
#
# 启动:
#   uvicorn deploy.app:app --host 0.0.0.0 --port 8000
#   python deploy/app.py
#
# 验证:
#   curl http://localhost:8000/health
#   curl -X POST http://localhost:8000/ask \
#        -H "Content-Type: application/json" \
#        -d '{"task": "计算 sqrt(256)", "mode": "quick"}'
# ============================================================

import os
import sys
import time
import uuid
import asyncio
import json
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# 加载 .env (从项目根目录)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from deploy.config import AppConfig
from deploy.agent_core import (
    LlmClient, MCPClient, DevAssistant, create_mcp_server,
)
from deploy.infrastructure import (
    RateLimiter, HealthChecker, GracefulShutdown, ServiceStats,
)
from deploy.observability import JsonLogger, Trace, TokenMonitor

# ============================================================
# 〇、全局状态 (app.state)
# ============================================================
# FastAPI 推荐: 共享状态存在 app.state 上
# 类比 Java: Spring 的 @Component 单例 Bean

config = AppConfig.from_env()
logger = JsonLogger("dev-assistant", min_level="INFO")
stats = ServiceStats()
health = HealthChecker()
shutdown_mgr = GracefulShutdown()
rate_limiter = RateLimiter(config.rate_limit_per_minute)
token_monitor = TokenMonitor(config.token_daily_budget)

# Agent 组件 (startup 时初始化)
llm_client: LlmClient | None = None
mcp_client: MCPClient | None = None
agent: DevAssistant | None = None

# 健康检查注册
health.register_check(
    "llm_api",
    lambda: ((llm_client.is_healthy if llm_client else False),
             "reachable" if (llm_client and llm_client.is_healthy)
             else "unreachable")
)
health.register_check(
    "rate_limiter",
    lambda: (True, f"rate={rate_limiter.current_rate}")
)
health.register_check(
    "disk",
    lambda: (True, "ok")
)


# ============================================================
# 一、FastAPI 生命周期
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的初始化与清理。

    类比 Java:
      @PostConstruct → startup
      @PreDestroy    → shutdown
    """
    global llm_client, mcp_client, agent

    # === 启动 ===
    logger.info("服务启动中",
                model=config.llm_model,
                port=config.port)

    llm_client = LlmClient(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
        max_retries=config.llm_max_retries,
        timeout=config.llm_timeout_seconds,
    )

    server = create_mcp_server(
        name="dev-assistant",
        project_root=config.project_root,
    )
    mcp_client = MCPClient()
    mcp_client.connect(server)
    agent = DevAssistant(mcp_client, llm_client)

    health.run_checks()
    health.set_ready()

    logger.info("服务已就绪",
                api_ok=llm_client.is_healthy,
                tools=server.tool_count)

    yield  # ← 服务运行中

    # === 关闭 ===
    logger.info("服务关闭中")
    shutdown_mgr.initiate(health, logger)


# ============================================================
# 二、中间件
# ============================================================

class RequestIdMiddleware(BaseHTTPMiddleware):
    """注入请求 ID (X-Request-ID)。

    每个请求分配唯一 ID，贯穿日志/追踪/响应头。
    类比 Java: Filter / OncePerRequestFilter。
    """

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get(
            "X-Request-ID",
            str(uuid.uuid4())[:8]
        )
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件。"""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000
        logger.info("request",
                    method=request.method,
                    path=request.url.path,
                    status=response.status_code,
                    duration_ms=round(elapsed, 1),
                    request_id=getattr(request.state, "request_id", "-"))
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """速率限制中间件。"""

    async def dispatch(self, request: Request, call_next):
        if not rate_limiter.allow():
            stats.record(429, 0)
            return JSONResponse(
                status_code=429,
                content={"error": "速率限制, 请稍后重试",
                         "retry_after_seconds": 60}
            )
        return await call_next(request)


class ShutdownGateMiddleware(BaseHTTPMiddleware):
    """优雅关闭门控 — 关闭中拒绝新请求。"""

    async def dispatch(self, request: Request, call_next):
        if shutdown_mgr.is_shutting_down:
            return JSONResponse(
                status_code=503,
                content={"error": "服务正在关闭"}
            )
        return await call_next(request)


# ============================================================
# 三、FastAPI 应用
# ============================================================

app = FastAPI(
    title="DevAssistant API",
    version="1.0.0",
    description="AI 开发者智能助手 — Phase 4+5 整合部署",
    lifespan=lifespan,
)

# 注册中间件 (顺序: 外层先执行)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ShutdownGateMiddleware)
app.add_middleware(RateLimitMiddleware)


# ============================================================
# 四、路由
# ============================================================

@app.get("/health")
async def get_health():
    """K8s liveness/readiness probe。

    curl http://localhost:8000/health
    """
    health.run_checks()
    return health.status()


@app.get("/status")
async def get_status():
    """服务状态详情。

    curl http://localhost:8000/status
    """
    return {
        "service": "dev-assistant",
        "version": "1.0.0",
        "config": {
            "model": config.llm_model,
            "max_concurrent": config.max_concurrent_llm,
            "rate_limit": config.rate_limit_per_minute,
        },
        "health": health.status(),
        "stats": stats.snapshot(),
        "rate_limiter": rate_limiter.stats(),
        "token_monitor": token_monitor.snapshot(),
    }


@app.get("/tools")
async def get_tools():
    """列出可用工具。

    curl http://localhost:8000/tools
    """
    if mcp_client is None:
        return {"tools": [], "count": 0}
    tools = mcp_client.list_tools()
    return {
        "tools": [
            {"name": t["name"], "description": t.get("description", "")}
            for t in tools
        ],
        "count": len(tools),
    }


@app.post("/ask")
async def ask(request: Request):
    """LLM 问答接口 (核心)。

    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"task": "计算 sqrt(256) + 100", "mode": "quick"}'

    mode: "quick" (ReAct) | "plan" (Plan-then-Act)
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    # 请求门控
    try:
        shutdown_mgr.start_request()
    except RuntimeError:
        stats.record(503, 0)
        raise HTTPException(status_code=503, detail="服务正在关闭")

    try:
        # 解析请求
        body = await request.json()
        task = body.get("task", "").strip()
        mode = body.get("mode", "quick")

        if not task:
            stats.record(400, 0)
            raise HTTPException(status_code=400, detail="task 不能为空")
        if mode not in ("quick", "plan"):
            stats.record(400, 0)
            raise HTTPException(
                status_code=400,
                detail="mode 只能是 'quick' 或 'plan'"
            )

        req_id = request.state.request_id if hasattr(
            request.state, "request_id"
        ) else str(uuid.uuid4())[:8]

        # 创建 Trace
        trace = Trace(f"ask-{mode}", request_id=req_id)

        logger.info("agent_request",
                    request_id=req_id, task=task[:80], mode=mode)

        # Agent.ask() 是同步阻塞调用, 放到线程池执行
        # 类比 Java: @Async + CompletableFuture
        start = time.time()

        span_agent = trace.start_span("agent", mode=mode, task=task[:50])
        result = await asyncio.to_thread(agent.ask, task, mode)
        trace.end_span(span_agent, status="ok")

        elapsed = (time.time() - start) * 1000
        stats.record(200, elapsed)

        # 估算 Token (粗糙)
        input_est = len(task) * 3
        output_est = len(result.get("answer", "")) * 2
        token_monitor.record(
            config.llm_model, input_est, output_est,
            agent="DevAssistant", task=task,
        )

        logger.info("agent_response",
                    request_id=req_id,
                    mode=mode,
                    iterations=result.get("iterations", 0),
                    duration_ms=round(elapsed, 1))

        return {
            "request_id": req_id,
            "answer": result.get("answer", ""),
            "mode": result.get("mode", mode),
            "iterations": result.get("iterations", 0),
            "tokens_est": input_est + output_est,
            "latency_ms": round(elapsed, 1),
            "trace": trace.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        stats.record(500, 0)
        logger.error("agent_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"内部错误: {e}")
    finally:
        shutdown_mgr.end_request()


@app.get("/")
async def root():
    return {
        "service": "DevAssistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "/status",
        "tools": "/tools",
        "ask": "POST /ask",
    }


# ============================================================
# 五、直接运行入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动 DevAssistant 服务...")
    print(f"\n  DevAssistant API")
    print(f"  {'─' * 40}")
    print(f"  地址: http://{config.host}:{config.port}")
    print(f"  文档: http://{config.host}:{config.port}/docs")
    print(f"  健康: http://{config.host}:{config.port}/health")
    print(f"  工具: http://{config.host}:{config.port}/tools")
    print(f"  问答: POST http://{config.host}:{config.port}/ask")
    print()
    uvicorn.run(
        "deploy.app:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="warning",  # 用我们自己的 JsonLogger
    )
