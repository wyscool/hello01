# ============================================================
# Phase 5, Lesson 44: 生产部署 — FastAPI、Docker、高可用
# ============================================================
#
# 本课目标:
#   把 AI 应用部署到生产环境 — API 服务化、容器化、高可用。
#
#   Java 背景映射:
#     FastAPI      ≈ Spring Boot (Web 框架)
#     uvicorn      ≈ Tomcat / Netty (应用服务器)
#     Docker       ≈ 不用类比, 一样的 Docker
#     Health Check ≈ Spring Actuator /health
#
#   核心概念:
#     1. FastAPI — 把 Python 函数变成 REST API
#     2. 健康检查 — K8s/docker-compose 知道服务是否存活
#     3. 并发控制 — Semaphore 限制 LLM 并发调用
#     4. 优雅关闭 — 收到 SIGTERM 后完成当前请求再退出
#     5. Docker 化 — 可复现的运行环境
#
#   为什么这很重要?
#     "Works on my machine" ≠ 生产可用
#     生产环境需要: 可部署、可监控、可扩缩、可恢复
#
# 预计阅读 + 实操时间: 60-70 分钟
#
# 前置: Phase 1-5 全部课程
# ============================================================

import os
import sys
import json
import time
import signal
import asyncio
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any
from collections import defaultdict

# ============================================================
# 〇、从脚本到服务
# ============================================================

print("=" * 60)
print("  Phase 5, Lesson 44: 生产部署")
print("=" * 60)
print()

print("""
  前 43 课, 我们写的都是 "脚本" — python xxx.py 运行一次。
  生产环境需要 "服务" — 持续运行, 接收 HTTP 请求, 返回响应。

  一个生产级 AI 服务需要:
    1. HTTP API      — 接收请求
    2. 健康检查       — 让负载均衡器知道服务状态
    3. 并发控制       — 避免 LLM API 被过载
    4. 优雅关闭       — 收到停止信号后安全退出
    5. 容器化         — Docker 一键部署
""")


# ============================================================
# 一、应用配置 — 12-Factor App 风格
# ============================================================
# 所有配置从环境变量读取, 而不是硬编码。

@dataclass
class AppConfig:
    """应用配置 — 统一管理所有参数。

    类比 Java:
      类似 Spring @ConfigurationProperties
    """
    # 服务
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 2

    # LLM
    llm_model: str = "claude-sonnet-4-6"
    llm_max_retries: int = 3
    llm_timeout_seconds: float = 60.0

    # 并发控制
    max_concurrent_llm: int = 5  # 同时最多 5 个 LLM 调用
    request_timeout_seconds: float = 120.0

    # 速率限制
    rate_limit_per_minute: int = 30

    # 缓存
    cache_enabled: bool = True
    cache_ttl_seconds: float = 300.0

    # Token 预算
    token_daily_budget: int = 1_000_000

    @classmethod
    def from_env(cls) -> "AppConfig":
        """从环境变量加载配置。"""
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            workers=int(os.getenv("WORKERS", "2")),
            llm_model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            llm_timeout_seconds=float(
                os.getenv("LLM_TIMEOUT", "60")
            ),
            max_concurrent_llm=int(
                os.getenv("MAX_CONCURRENT_LLM", "5")
            ),
            rate_limit_per_minute=int(
                os.getenv("RATE_LIMIT", "30")
            ),
        )


config = AppConfig.from_env()


# ============================================================
# 二、并发控制 — Semaphore
# ============================================================
# LLM API 通常有速率限制 (RPM/TPM)。
# 同时发起太多调用 → 429 Rate Limit → 调用失败。
# Semaphore 控制同时进行的 LLM 调用数。

class ConcurrencyController:
    """LLM 并发控制器。

    基于 Semaphore 控制同时进行的 LLM 调用数量。

    类比 Java:
      类似 java.util.concurrent.Semaphore
      但 Python 的 asyncio.Semaphore 支持 async/await。
    """

    def __init__(self, max_concurrent: int = 5):
        self._semaphore = threading.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self._active = 0
        self._total_waited = 0
        self._total_completed = 0

    def acquire(self) -> bool:
        """尝试获取许可 (非阻塞)。"""
        return self._semaphore.acquire(blocking=True)

    def release(self):
        self._semaphore.release()

    @property
    def available(self) -> int:
        """当前可用的许可数。"""
        # Semaphore 没有直接暴露 _value, 我们跟踪 _active 近似值
        return max(0, self.max_concurrent - self._active)

    def stats(self) -> dict:
        return {
            "max_concurrent": self.max_concurrent,
            "active": self._active,
            "available": self.available,
            "total_completed": self._total_completed,
            "total_waited": self._total_waited,
        }


# 演示
print("--- 并发控制演示 ---")
cc = ConcurrencyController(max_concurrent=3)
print(f"  最大并发: {cc.max_concurrent}, 可用: {cc.available}")

# 模拟获取和释放
cc.acquire()
cc._active = 1  # 模拟
print(f"  获取 1 个 → 可用: {cc.available}")
cc.release()
cc._active = 0
print(f"  释放 → 可用: {cc.available}")
print()


# ============================================================
# 三、速率限制 — Sliding Window
# ============================================================

class RateLimiter:
    """滑动窗口速率限制。

    限制每分钟的请求数。超过限制返回 429。
    """

    def __init__(self, max_per_minute: int = 30):
        self.max_per_minute = max_per_minute
        self._window: list[float] = []  # 请求时间戳
        self._rejected = 0

    def allow(self) -> bool:
        """检查是否允许此请求。

        Returns:
            True  → 允许
            False → 拒绝 (应返回 429)
        """
        now = time.time()
        window_start = now - 60  # 过去 60 秒

        # 清理过期的时间戳
        self._window = [t for t in self._window if t > window_start]

        if len(self._window) < self.max_per_minute:
            self._window.append(now)
            return True
        else:
            self._rejected += 1
            return False

    @property
    def current_rate(self) -> int:
        """当前窗口内的请求数。"""
        window_start = time.time() - 60
        return sum(1 for t in self._window if t > window_start)

    def stats(self) -> dict:
        return {
            "max_per_minute": self.max_per_minute,
            "current_rate": self.current_rate,
            "rejected": self._rejected,
        }


# 演示
print("--- 速率限制演示 ---")
rl = RateLimiter(max_per_minute=5)
for i in range(8):
    allowed = rl.allow()
    print(f"  请求 {i + 1}: {'✓ 允许' if allowed else '✗ 拒绝 (429)'}")
print(f"  统计: {rl.stats()}")
print()


# ============================================================
# 四、健康检查
# ============================================================
# 生产环境的关键组件:
#   Liveness Probe  — "服务还活着吗?"  (K8s 判断是否重启)
#   Readiness Probe — "可以接流量了吗?" (K8s 判断是否加入 Service)

class HealthChecker:
    """健康检查器。

    管理服务的存活状态和就绪状态。
    """

    def __init__(self):
        self._start_time = time.time()
        self._alive = True
        self._ready = False  # 初始未就绪
        self._last_check: dict[str, Any] = {}
        self._checks: dict[str, Callable] = {}  # type: ignore

    def register_check(self, name: str, fn) -> "HealthChecker":
        """注册一个健康检查函数。fn 返回 (ok: bool, detail: str)。"""
        self._checks[name] = fn
        return self

    def run_checks(self) -> dict:
        """运行所有注册的检查。"""
        results = {}
        for name, fn in self._checks.items():
            try:
                ok, detail = fn()
                results[name] = {"status": "pass" if ok else "fail",
                                 "detail": detail}
            except Exception as e:
                results[name] = {"status": "fail", "detail": str(e)}

        self._last_check = results
        return results

    @property
    def is_healthy(self) -> bool:
        """所有检查都通过? """
        return all(
            v["status"] == "pass" for v in self._last_check.values()
        ) if self._last_check else True

    def set_ready(self):
        self._ready = True

    def set_not_ready(self):
        self._ready = False

    def shutdown(self):
        self._alive = False
        self._ready = False

    def status(self) -> dict:
        uptime = time.time() - self._start_time
        return {
            "status": "pass" if self.is_healthy else "fail",
            "alive": self._alive,
            "ready": self._ready,
            "uptime_seconds": round(uptime, 1),
            "started_at": datetime.fromtimestamp(
                self._start_time, tz=timezone(timedelta(hours=8))
            ).isoformat(),
            "checks": self._last_check,
        }


# 演示
print("--- 健康检查演示 ---")
hc = HealthChecker()

# 注册检查
hc.register_check("disk", lambda: (True, "disk ok"))
hc.register_check("memory", lambda: (True, "65% used"))
hc.register_check("llm_api", lambda: (True, "reachable"))

hc.run_checks()
hc.set_ready()

status = hc.status()
print(f"  状态: {status['status']}")
print(f"  存活: {status['alive']}, 就绪: {status['ready']}")
print(f"  运行时间: {status['uptime_seconds']}s")
for name, check in status["checks"].items():
    print(f"    {name}: {check['status']} ({check['detail']})")
print()


# ============================================================
# 五、优雅关闭
# ============================================================
# 收到 SIGTERM (K8s 终止 Pod) → 停止接受新请求 → 完成当前请求 → 退出

class GracefulShutdown:
    """优雅关闭管理器。

    流程:
      1. 收到 SIGTERM/SIGINT
      2. 标记 shutting_down = True
      3. 健康检查返回 not_ready (从负载均衡摘除)
      4. 等待当前请求完成 (grace_period)
      5. 关闭资源, 退出

    类比 Java:
      类似 Spring Boot 的 graceful shutdown
      server.shutdown=graceful
      spring.lifecycle.timeout-per-shutdown-phase=30s
    """

    def __init__(self, grace_period: float = 30.0):
        self.grace_period = grace_period
        self._shutting_down = False
        self._active_requests = 0
        self._lock = threading.Lock()

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def start_request(self):
        """请求开始时调用。如果正在关闭, 抛出异常拒绝新请求。"""
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("服务正在关闭, 拒绝新请求")
            self._active_requests += 1

    def end_request(self):
        """请求结束时调用。"""
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)

    def shutdown(self, health: HealthChecker):
        """触发关闭流程。"""
        print("\n  [SHUTDOWN] 收到关闭信号...")
        self._shutting_down = True
        health.set_not_ready()
        print(f"  [SHUTDOWN] 已标记为不可达, 等待 {self._active_requests} "
              f"个活跃请求完成...")

        deadline = time.time() + self.grace_period
        while self._active_requests > 0 and time.time() < deadline:
            time.sleep(0.5)

        if self._active_requests > 0:
            print(f"  [SHUTDOWN] 超时! {self._active_requests} "
                  f"个请求未完成, 强制退出")
        else:
            print(f"  [SHUTDOWN] 所有请求已完成, 安全退出")

        health.shutdown()


# 演示
print("--- 优雅关闭演示 ---")
gs = GracefulShutdown(grace_period=5.0)
hcheck = HealthChecker()
hcheck.set_ready()

print(f"  初始: shutting_down={gs.is_shutting_down}, "
      f"active={gs._active_requests}")

# 模拟请求
gs.start_request()
print(f"  开始请求: active={gs._active_requests}")
gs.end_request()
print(f"  结束请求: active={gs._active_requests}")
print()


# ============================================================
# 六、请求上下文 — 一次请求的完整信息
# ============================================================

@dataclass
class RequestContext:
    """单次请求的上下文信息。"""
    request_id: str
    start_time: float = field(default_factory=time.time)
    user_agent: str = ""
    source_ip: str = ""

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

    def to_log_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "user_agent": self.user_agent,
        }


# ============================================================
# 七、服务统计
# ============================================================

class ServiceStats:
    """服务级别的统计信息。"""

    def __init__(self):
        self._start_time = time.time()
        self.total_requests = 0
        self.total_errors = 0
        self._latencies: list[float] = []
        self._status_codes: dict[int, int] = defaultdict(int)

    def record(self, status_code: int, latency_ms: float):
        self.total_requests += 1
        if status_code >= 400:
            self.total_errors += 1
        self._status_codes[status_code] += 1
        self._latencies.append(latency_ms)
        # 只保留最近 1000 条延迟数据
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def p99_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        sorted_lat = sorted(self._latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def snapshot(self) -> dict:
        return {
            "uptime_seconds": round(self.uptime_seconds, 1),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": round(self.error_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p99_latency_ms": round(self.p99_latency_ms, 1),
            "status_codes": dict(self._status_codes),
            "rps": round(
                self.total_requests / max(self.uptime_seconds, 1), 2
            ),
        }


# 演示
print("--- 服务统计演示 ---")
stats = ServiceStats()

# 模拟流量
import random
for _ in range(50):
    code = random.choices(
        [200, 200, 200, 200, 400, 500],  # 80% 成功
        weights=[6, 6, 3, 1, 2, 1],
    )[0]
    latency = random.gauss(200, 100)
    stats.record(code, max(10, latency))

snap = stats.snapshot()
print(f"  总请求: {snap['total_requests']}, 错误率: {snap['error_rate']:.1%}")
print(f"  平均延迟: {snap['avg_latency_ms']}ms, P99: {snap['p99_latency_ms']}ms")
print(f"  QPS: {snap['rps']}")
print(f"  状态码分布: {snap['status_codes']}")
print()


# ============================================================
# 八、组装 — 完整的服务基础架构
# ============================================================

class AIService:
    """AI 服务基础架构。

    将所有组件组装在一起, 提供:
      - 配置管理
      - 并发控制
      - 速率限制
      - 健康检查
      - 优雅关闭
      - 服务统计

    实际使用时, 配合 FastAPI 暴露 HTTP 接口。
    """

    def __init__(self, cfg: AppConfig | None = None):
        self.config = cfg or AppConfig.from_env()
        self.concurrency = ConcurrencyController(
            self.config.max_concurrent_llm
        )
        self.rate_limiter = RateLimiter(
            self.config.rate_limit_per_minute
        )
        self.health = HealthChecker()
        self.shutdown_mgr = GracefulShutdown()
        self.stats = ServiceStats()

        # 注册健康检查
        self.health.register_check(
            "concurrency",
            lambda: (self.concurrency.available >= 0,
                     f"available={self.concurrency.available}")
        )
        self.health.register_check(
            "rate_limiter",
            lambda: (True, f"rate={self.rate_limiter.current_rate}")
        )

    def start(self):
        """启动服务。"""
        self.health.set_ready()
        print(f"\n  AI 服务已启动")
        print(f"    端口: {self.config.port}")
        print(f"    模型: {self.config.llm_model}")
        print(f"    最大 LLM 并发: {self.config.max_concurrent_llm}")
        print(f"    速率限制: {self.config.rate_limit_per_minute}/min")
        print()

    def handle_request(self, request_id: str,
                       task: str, _simulate: bool = True) -> dict:
        """处理一次 LLM 请求 (模拟)。

        实际项目中, 这里调用 LLM API 或 Agent。
        这里展示的是 "基础设施层" 应该做什么。
        """
        # 1. 检查是否在关闭
        self.shutdown_mgr.start_request()

        try:
            # 2. 速率限制
            if not self.rate_limiter.allow():
                self.stats.record(429, 0)
                return {"error": "速率限制, 请稍后重试", "status": 429}

            # 3. 并发控制
            if not self.concurrency.acquire():
                self.stats.record(503, 0)
                return {"error": "服务繁忙, 请稍后重试", "status": 503}

            try:
                self.concurrency._active += 1
                start = time.time()

                # 4. 实际处理 (模拟)
                if _simulate:
                    time.sleep(random.uniform(0.05, 0.3))
                    result = f"[{self.config.llm_model}] "
                    result += f"处理 '{task[:30]}...' → 完成"
                else:
                    result = f"实际调用 LLM: {task}"

                latency = (time.time() - start) * 1000
                self.stats.record(200, latency)
                return {
                    "result": result,
                    "status": 200,
                    "latency_ms": round(latency, 1),
                    "model": self.config.llm_model,
                }
            finally:
                self.concurrency._active = max(
                    0, self.concurrency._active - 1
                )
                self.concurrency.release()
        finally:
            self.shutdown_mgr.end_request()

    def status(self) -> dict:
        """服务状态摘要。"""
        return {
            "service": "ai-service",
            "version": "1.0.0",
            "config": {
                "model": self.config.llm_model,
                "max_concurrent": self.config.max_concurrent_llm,
            },
            "health": self.health.status(),
            "concurrency": self.concurrency.stats(),
            "rate_limiter": self.rate_limiter.stats(),
            "stats": self.stats.snapshot(),
        }


# ============================================================
# 九、FastAPI 应用骨架
# ============================================================
# 以下代码用注释展示如何在 FastAPI 中使用上述组件。
# 因为 FastAPI 需要 uvicorn 运行, 本文件用模拟方式演示。

FASTAPI_SKELETON = '''
# --- 安装: pip install fastapi uvicorn ---
# --- 运行: uvicorn app:app --host 0.0.0.0 --port 8000 ---

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="AI Service", version="1.0.0")
service = AIService()

@app.on_event("startup")
async def startup():
    service.start()

@app.on_event("shutdown")
async def shutdown():
    service.shutdown_mgr.shutdown(service.health)

@app.get("/health")
async def health():
    """K8s liveness/readiness probe."""
    return service.health.status()

@app.get("/status")
async def status():
    """服务状态详情 (非敏感)。"""
    return service.status()

@app.post("/ask")
async def ask(request: Request):
    """LLM 问答接口。"""
    body = await request.json()
    task = body.get("task", "")
    if not task:
        raise HTTPException(status_code=400, detail="task 不能为空")

    req_id = request.headers.get("X-Request-ID", str(int(time.time())))
    result = service.handle_request(req_id, task)
    return JSONResponse(content=result, status_code=result.get("status", 200))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

DOCKERFILE = '''
# --- Dockerfile ---
FROM python:3.12-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 创建非 root 用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 启动
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
'''


# ============================================================
# 十、演示 — 模拟生产流量
# ============================================================

def demo():
    """模拟服务运行 — 处理多请求, 展示基础设施能力。"""
    service = AIService(config)
    service.start()

    print("--- 模拟请求流量 ---")
    import concurrent.futures

    tasks = [
        "翻译成英文: 你好世界",
        "解释什么是 RAG",
        "帮我写一个快速排序",
        "今天天气怎么样",
        "计算 123 * 456",
        "总结这段文字的主要内容",
        "把这段代码改成 async/await 风格",
        "对比 Python 和 Java 的异常处理",
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        for i, task in enumerate(tasks):
            req_id = f"req-{i + 1:03d}"
            futures.append(
                pool.submit(service.handle_request, req_id, task)
            )

        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            icon = "✓" if result.get("status") == 200 else "✗"
            print(f"  {icon} [{result.get('latency_ms', 0):.0f}ms] "
                  f"{result.get('result', result.get('error', '?'))[:60]}")

    print(f"\n--- 服务状态 ---")
    status = service.status()
    print(json.dumps(status, ensure_ascii=False, indent=2))

    # 模拟关闭
    print(f"\n--- 优雅关闭 ---")
    service.shutdown_mgr.shutdown(service.health)


# ============================================================
# 十一、入口
# ============================================================

if __name__ == "__main__":
    demo()

    print("\n" + "=" * 60)
    print("  Lesson 44 完成! 全部课程结束!")
    print("=" * 60)

    print("""
  🎉 恭喜! 你完成了 29 节课, 覆盖 5 个 Phase:

  Phase 1 (10 课) Python 基础
    ✓ 变量/类型 → 类/继承 → 异步 → 测试
    ✓ 从 "hello world" 到 "能独立写 Python 脚本"

  Phase 2 (5 课)  LLM API + Prompt
    ✓ API 调用 → Prompt 设计 → 结构化输出 → 流式 → 对话应用
    ✓ 从 "调用 API" 到 "构建 AI 应用"

  Phase 3 (5 课)  RAG + 向量数据库
    ✓ Embedding → 向量检索 → 文档处理 → 检索流水线 → 知识库
    ✓ 从 "关键词搜索" 到 "语义问答系统"

  Phase 4 (5 课)  Agent + MCP
    ✓ ReAct → 工具调用 → 多步规划 → MCP 协议 → 端到端项目
    ✓ 从 "单轮对话" 到 "自主智能体"

  Phase 5 (4 课)  AI 工程化
    ✓ 评估框架 → 可观测性 → 成本控制 → 生产部署
    ✓ 从 "能用" 到 "生产级"

  ─────────────────────────────────────────────

  从 Java 后端工程师到 AI 应用开发者,
  你学到的不仅是 Python 语法,
  更是一整套构建 AI 系统的方法论。

  下一步:
    1. 把 DevAssistant (L35) 用本课的 FastAPI 骨架部署成服务
    2. 为你的实际工作场景设计一个 AI 工具
    3. 持续跟踪 LLM 领域的新进展 (变化很快!)

  记住: AI 工程化的核心不是模型本身,
         而是如何稳定、可度量、可控地把模型集成到产品中。
  """)


# ============================================================
# 试试看 (Try This) - 解答
# ============================================================

print("\n" + "=" * 60)
print("  试试看 (Try This) - Lesson 44 练习")
print("=" * 60)
print()


# ============================================================
# 练习 1-4: 已在 deploy/app.py 中完整实现!
# ============================================================

print("--- 练习 1-4: 对应 deploy/ 项目 ---")
print("""
  练习 1 (FastAPI 启动服务):
    -> deploy/app.py (完整的 FastAPI 应用)
    启动: uvicorn deploy.app:app --host 0.0.0.0 --port 8000
    验证: curl http://localhost:8000/health
          curl -X POST http://localhost:8000/ask \\
               -H "Content-Type: application/json" \\
               -d '{"task": "计算 sqrt(256)", "mode": "quick"}'

  练习 2 (DevAssistant 集成到 /ask):
    -> deploy/app.py L265-344 (POST /ask 路由)
    - 接收 task 和 mode (L288-289)
    - 调用 agent.ask(task, mode) (L316)
    - 返回 answer / mode / iterations / trace (L336-344)
    - 使用 asyncio.to_thread 将同步 Agent 放到线程池 (L316)

  练习 3 (API Key 认证):
    deploy/app.py 已有完整的中间件体系:
    - RequestIdMiddleware (L134) - 请求 ID 注入
    - RateLimitMiddleware (L168) - 速率限制
    - ShutdownGateMiddleware (L182) - 关闭门控

    添加 API Key 认证的依赖函数 (可直接加入 app.py):

      from fastapi import Depends, HTTPException, Security
      from fastapi.security import APIKeyHeader

      API_KEY = os.getenv("API_KEY", "")
      api_key_header = APIKeyHeader(name="X-API-Key")

      def verify_api_key(key: str = Security(api_key_header)):
          if not API_KEY:
              return key  # 未配置则跳过
          if key != API_KEY:
              raise HTTPException(status_code=401, detail="无效的 API Key")
          return key

      @app.post("/ask")
      async def ask(request: Request, api_key: str = Depends(verify_api_key)):
          ...  # 原有逻辑

  练习 4 (Docker 化):
    -> deploy/Dockerfile (已完整编写)
      docker build -t ai-service:latest .
      docker run -p 8000:8000 --env-file .env ai-service:latest
    类似地, rag_kb/Dockerfile, code_review/Dockerfile,
    log_analyzer/Dockerfile 也已完成。

  小结: 练习 1-4 的核心代码已在 deploy/ 项目中完整实现,
        你可以直接运行 deploy/app.py 体验完整的生产级 AI 服务。
""")


# ============================================================
# 练习 5 (挑战): Request ID 全链路追踪
# ============================================================
# deploy/app.py 中已有 RequestIdMiddleware (L134-149),
# 这里展示如何在完整链路中使用 request_id。

def ex5_request_id_tracing():
    """演示 Request ID 全链路追踪。

    deploy/app.py 的 RequestIdMiddleware 已经:
      1. 从请求头读取或生成 request_id
      2. 存入 request.state.request_id
      3. 通过响应头 X-Request-ID 返回给客户端

    本练习: 将 request_id 贯穿到日志/Trace/LLM调用/工具调用。
    """
    print("--- 练习 5: Request ID 全链路追踪 ---")

    req_id = f"req-{int(time.time() * 1000) % 1000000:06d}"

    # 1. 轻量结构化日志 (内联版本, 避免跨文件依赖)
    from collections import defaultdict as _dd

    class _MiniJsonLogger:
        """轻量 JSON 日志器 — 仅用于演示 request_id 追踪。"""
        def __init__(self, name: str = "app"):
            self.name = name
            self._counts: dict = _dd(int)

        def _log(self, level: str, message: str, **context):
            self._counts[level] += 1
            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            _TZ = _tz(_td(hours=8))
            record = {
                "timestamp": _dt.now(_TZ).isoformat(),
                "level": level, "logger": self.name,
                "message": message, **context,
            }
            # 仅打印到 stderr (不持久化, 仅演示)
            print(f"  [{level}] {message} "
                  f"request_id={context.get('request_id', '-')}")

        def info(self, msg, **ctx): self._log("INFO", msg, **ctx)
        def debug(self, msg, **ctx): self._log("DEBUG", msg, **ctx)
        def error(self, msg, **ctx): self._log("ERROR", msg, **ctx)

        @property
        def stats(self) -> dict:
            return dict(self._counts)

    # 2. 轻量 Trace (内联版本)
    @dataclass
    class _MiniSpan:
        name: str
        span_id: str
        parent_id: str = ""
        start_time: float = 0.0
        end_time: float = 0.0
        status: str = "ok"
        metadata: dict = field(default_factory=dict)

        @property
        def duration_ms(self) -> float:
            return (self.end_time - self.start_time) * 1000

    class _MiniTrace:
        def __init__(self, name: str, request_id: str = ""):
            self.name = name
            self.request_id = request_id
            self.spans: list[_MiniSpan] = []
            self.start_time = time.time()
            self._counter = 0

        def _next_id(self) -> str:
            self._counter += 1
            return f"{self.request_id}-{self._counter}"

        def start_span(self, name: str, parent: _MiniSpan | None = None,
                       **meta) -> _MiniSpan:
            span = _MiniSpan(name=name, span_id=self._next_id(),
                            parent_id=parent.span_id if parent else "",
                            start_time=time.time(), metadata=meta)
            self.spans.append(span)
            return span

        def end_span(self, span: _MiniSpan, status: str = "ok", **attrs):
            span.end_time = time.time()
            span.status = status
            if attrs:
                span.metadata.update(attrs)

        @property
        def duration_ms(self) -> float:
            return (time.time() - self.start_time) * 1000

        def to_dict(self) -> dict:
            return {
                "trace_name": self.name, "request_id": self.request_id,
                "duration_ms": round(self.duration_ms, 1),
                "span_count": len(self.spans),
                "spans": [
                    {"name": s.name, "span_id": s.span_id,
                     "parent_id": s.parent_id,
                     "duration_ms": round(s.duration_ms, 1),
                     "status": s.status, "metadata": s.metadata}
                    for s in self.spans
                ],
            }

        def report(self) -> str:
            lines = [f"\n  Request ID 追踪: {self.name} ({self.request_id})",
                     f"  总耗时: {self.duration_ms:.0f}ms"]
            for s in self.spans:
                icon = "V" if s.status == "ok" else "X"
                indent = "    " if s.parent_id else "  "
                lines.append(f"{indent}{icon} {s.name} [{s.span_id}] "
                           f"({s.duration_ms:.0f}ms) "
                           f"{s.metadata}")
            return "\n".join(lines)

    # 3. 使用内联类演示全链路追踪
    rlog = _MiniJsonLogger("dev-assistant")

    # 注入 request_id: 通过 context 参数
    def log_with_id(level_fn, msg, **ctx):
        ctx["request_id"] = req_id
        level_fn(msg, **ctx)

    log_with_id(rlog.info, "请求开始", path="/ask", method="POST")

    trace = _MiniTrace("agent-ask", request_id=req_id)

    # LLM Think
    span_think = trace.start_span("llm_think", model="claude-sonnet-4-6")
    log_with_id(rlog.info, "llm_call_start", step="think", tokens=700)
    time.sleep(0.005)
    trace.end_span(span_think, status="ok", tokens=700)
    log_with_id(rlog.info, "llm_call_end", step="think", tokens=700)

    # 工具调用
    span_tool = trace.start_span("tool_calculate", parent=span_think,
                                tool="calculator")
    log_with_id(rlog.info, "tool_call_start", tool="calculator",
               args="sqrt(256)")
    time.sleep(0.003)
    trace.end_span(span_tool, status="ok", result="16")
    log_with_id(rlog.info, "tool_call_end", tool="calculator", result="16")

    # LLM Answer
    span_answer = trace.start_span("llm_answer", model="claude-sonnet-4-6")
    log_with_id(rlog.info, "llm_call_start", step="answer")
    time.sleep(0.005)
    trace.end_span(span_answer, status="ok", tokens=300)
    log_with_id(rlog.info, "llm_call_end", step="answer")

    log_with_id(rlog.info, "请求完成",
               total_duration_ms=round(trace.duration_ms, 1),
               span_count=len(trace.spans), status=200)

    print(trace.report())
    print(f"\n  通过 request_id='{req_id}' 可以:")
    print(f"    1. 追踪所有操作: think -> tool -> answer")
    print(f"    2. 查看 Trace JSON: {json.dumps(trace.to_dict(), ensure_ascii=False)[:100]}...")
    print(f"    3. 关联所有日志: 共 {sum(rlog.stats.values())} 条日志记录")

    print(f"""
  生产环境最佳实践 (deploy/app.py 中已实现):
    1. Middleware 层注入 request_id (L134-149)
    2. 所有 logger 调用自动带上 request_id
    3. Trace 创建时使用相同的 request_id
    4. 响应头返回 X-Request-ID, 方便客户端关联
    5. 用户投诉时: "请提供 X-Request-ID" -> 精确追查完整链路
  """)

ex5_request_id_tracing()


# ============================================================
# 练习 6 (思考): 你的第一个 AI 生产项目
# ============================================================

print("--- 练习 6: 第一个 AI 生产项目 (思考) ---")
print("""
  学完 29 节课, 覆盖 5 个 Phase 后的计划:

  +-------------------------------------------------------------+
  | 我想做的第一个 AI 生产项目:                                   |
  |                                                             |
  | 1. 项目方向: 内部工具 - 智能日志分析 + 告警助手              |
  |    已有基础: log_analyzer/ 项目                              |
  |    目标: 把日志分析从 "手动 grep + 肉眼判断" 变成             |
  |          "自动分析 -> 发现异常 -> 给出根因 -> 建议修复"       |
  |                                                             |
  | 2. 用到哪些 Phase 的知识?                                   |
  |    Phase 3 (RAG): 把历史故障知识库接入, 遇到相似异常        |
  |                   时自动检索历史解决方案                      |
  |    Phase 4 (Agent): 日志分析 Agent 自主调用 grep/jq/        |
  |                     curl 工具, 多步推理根因                  |
  |    Phase 5 (工程化): 评估 (错误检测准确率)、                  |
  |                      可观测性 (分析过程全链路追踪)、           |
  |                      成本控制 (分级模型: 简单日志用 Haiku)    |
  |                                                             |
  | 3. 第一步 (MVP, 2 周):                                      |
  |    - 用 FastAPI 包装 log_analyzer agent 成 API              |
  |    - 接入公司日志系统 (ES/Loki)                             |
  |    - 定时巡检 + 异常推送到 Slack                             |
  |                                                             |
  | 4. 第二步 (迭代):                                           |
  |    - 积累评估数据集 (什么算 "正确" 的根因分析)               |
  |    - 持续优化 Prompt 和工具设计                              |
  |    - 引入缓存: 相同错误模式的重复分析直接返回                |
  |                                                             |
  | 最重要的是: 先做出来, 有人用, 再迭代优化。                    |
  | 不要追求完美, 先追求 "有用"。                                 |
  +-------------------------------------------------------------+

  从 Java 后端工程师到 AI 应用开发者, 核心转变:
    不是学会调用 API, 而是学会:
    1. 评估 AI 输出的质量 (而不是 assertTrue/False)
    2. 观察 AI 系统的行为 (而不是查 stack trace)
    3. 控制 AI 系统的成本 (而不是优化 CPU/内存)
    4. 把不确定性变成可度量的工程问题

  这 4 个能力, 就是 Phase 5 的精髓。
""")
