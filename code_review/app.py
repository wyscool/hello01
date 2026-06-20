# ============================================================
# code_review/app.py — FastAPI 服务
# ============================================================
# 启动:
#   uvicorn code_review.app:app --port 8001
#
# 用法:
#   curl -X POST http://localhost:8001/review \
#     -H "Content-Type: application/json" \
#     -d '{"code": "...", "language": "java"}'
# ============================================================

import sys
import time
import uuid
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from code_review.config import AppConfig
from code_review.agent import create_review_agent, ReviewAgent

# ============================================================
# 〇、全局状态
# ============================================================

config = AppConfig.from_env()
agent: ReviewAgent | None = None
start_time: float = 0.0
request_count = 0
error_count = 0


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


# ============================================================
# 二、生命周期
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, start_time
    start_time = time.time()
    print(f"\n  代码审查服务启动")
    print(f"  Model: {config.llm_model}")
    print(f"  Port: {config.port}")
    agent = create_review_agent(config)
    print(f"  Tools: {agent.mcp.server.tool_count if agent.mcp.server else 0}")
    print(f"  API: {'reachable' if agent.llm.is_healthy else 'offline'}")
    print()
    yield
    print("\n  代码审查服务关闭")


# ============================================================
# 三、FastAPI 应用
# ============================================================

app = FastAPI(
    title="Code Review API",
    version="1.0.0",
    description="AI 代码审查助手",
    lifespan=lifespan,
)
app.add_middleware(RequestIdMiddleware)


# ============================================================
# 四、路由
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "pass",
        "uptime_seconds": round(time.time() - start_time, 1),
        "api_ok": agent.llm.is_healthy if agent else False,
    }


@app.get("/tools")
async def tools():
    if agent is None or agent.mcp.server is None:
        return {"tools": [], "count": 0}
    tl = agent.mcp.list_tools()
    return {
        "tools": [{"name": t["name"], "description": t.get("description", "")}
                   for t in tl],
        "count": len(tl),
    }


@app.post("/review")
async def review(request: Request):
    """代码审查接口。

    Request body:
      {
        "code": "public class Foo { ... }",
        "language": "java",           // java | python
        "focus": ["security", "logic"]  // 可选, 关注领域
      }
    """
    global request_count, error_count

    if agent is None:
        raise HTTPException(status_code=503, detail="服务未就绪")

    try:
        body = await request.json()
        code = body.get("code", "").strip()
        language = body.get("language", "java")
        focus = body.get("focus")

        if not code:
            raise HTTPException(status_code=400, detail="code 不能为空")
        if language not in ("java", "python", "go", "javascript", "typescript"):
            # 不严格限制，但给提示
            pass

        req_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
        request_count += 1
        start = time.time()

        # Agent.review() 是同步阻塞调用，放线程池
        result = await asyncio.to_thread(
            agent.review, code, language, focus
        )
        elapsed = (time.time() - start) * 1000

        return {
            "request_id": req_id,
            "language": language,
            "code_length": len(code),
            "code_lines": len(code.split("\n")),
            "review": result,
            "latency_ms": round(elapsed, 1),
        }

    except HTTPException:
        error_count += 1
        raise
    except Exception as e:
        error_count += 1
        raise HTTPException(status_code=500, detail=f"审查失败: {e}")


@app.get("/")
async def root():
    return {
        "service": "Code Review API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "tools": "GET /tools",
            "review": "POST /review",
        },
        "usage": {
            "curl_example": (
                'curl -X POST http://localhost:8001/review '
                '-H "Content-Type: application/json" '
                '-d \'{"code": "public class Foo {}", "language": "java"}\''
            )
        },
    }


# ============================================================
# 五、直接运行
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print(f"\n  Code Review API")
    print(f"  http://{config.host}:{config.port}")
    print(f"  Docs: http://{config.host}:{config.port}/docs")
    uvicorn.run(
        "code_review.app:app",
        host=config.host, port=config.port,
        log_level="warning",
    )
