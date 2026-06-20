# ============================================================
# deploy/infrastructure.py — 服务基础设施
# ============================================================
# 提取自 phase5/44_production.py
#
# 类比 Java:
#   HealthChecker     ≈ Spring Actuator /health
#   RateLimiter       ≈ Bucket4j / Resilience4j
#   GracefulShutdown  ≈ Spring Boot graceful shutdown
# ============================================================

import time
import threading
from dataclasses import dataclass, field
from typing import Callable
from collections import defaultdict


# ============================================================
# 一、速率限制
# ============================================================

class RateLimiter:
    """滑动窗口速率限制。"""

    def __init__(self, max_per_minute: int = 30):
        self.max_per_minute = max_per_minute
        self._window: list[float] = []
        self._rejected = 0

    def allow(self) -> bool:
        now = time.time()
        window_start = now - 60
        self._window = [t for t in self._window if t > window_start]
        if len(self._window) < self.max_per_minute:
            self._window.append(now)
            return True
        self._rejected += 1
        return False

    @property
    def current_rate(self) -> int:
        window_start = time.time() - 60
        return sum(1 for t in self._window if t > window_start)

    def stats(self) -> dict:
        return {
            "max_per_minute": self.max_per_minute,
            "current_rate": self.current_rate,
            "rejected": self._rejected,
        }


# ============================================================
# 二、健康检查
# ============================================================

class HealthChecker:
    """健康检查器 — liveness + readiness。"""

    def __init__(self):
        self._start_time = time.time()
        self._alive = True
        self._ready = False
        self._last_check: dict = {}
        self._checks: dict[str, Callable] = {}

    def register_check(self, name: str, fn: Callable) -> "HealthChecker":
        """注册检查函数。fn 返回 (ok: bool, detail: str)。"""
        self._checks[name] = fn
        return self

    def run_checks(self) -> dict:
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
        return all(
            v["status"] == "pass"
            for v in self._last_check.values()
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
            "checks": self._last_check,
        }


# ============================================================
# 三、优雅关闭
# ============================================================

class GracefulShutdown:
    """优雅关闭管理器。

    流程: 收到信号 → 标记关闭 → 摘除负载均衡 → 等待活跃请求 → 退出
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
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("服务正在关闭, 拒绝新请求")
            self._active_requests += 1

    def end_request(self):
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)

    def initiate(self, health: HealthChecker, logger=None):
        """触发关闭流程。"""
        msg = "收到关闭信号, 开始优雅关闭"
        if logger:
            logger.info(msg)
        else:
            print(f"[SHUTDOWN] {msg}")

        self._shutting_down = True
        health.set_not_ready()

        if logger:
            logger.info("等待活跃请求完成",
                        active_requests=self._active_requests)

        deadline = time.time() + self.grace_period
        while self._active_requests > 0 and time.time() < deadline:
            time.sleep(0.5)

        if self._active_requests > 0:
            if logger:
                logger.warn("超时! 强制退出",
                           remaining=self._active_requests)
        else:
            if logger:
                logger.info("所有请求已完成, 安全退出")

        health.shutdown()


# ============================================================
# 四、服务统计
# ============================================================

class ServiceStats:
    """服务级别统计 — 延迟、错误率、P99、状态码分布。"""

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
        s = sorted(self._latencies)
        return s[min(int(len(s) * 0.99), len(s) - 1)]

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
