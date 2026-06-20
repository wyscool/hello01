# ============================================================
# deploy/tests/test_infrastructure.py — 基础设施组件测试
# ============================================================

import time
import pytest
import threading
from unittest.mock import Mock

from deploy.infrastructure import (
    RateLimiter, HealthChecker, GracefulShutdown, ServiceStats,
)


# ============================================================
# RateLimiter
# ============================================================

class TestRateLimiter:
    def test_allow_within_limit(self):
        rl = RateLimiter(max_per_minute=100)
        for _ in range(50):
            assert rl.allow() is True

    def test_allow_exceeded(self):
        rl = RateLimiter(max_per_minute=3)
        for _ in range(3):
            assert rl.allow() is True
        assert rl.allow() is False

    def test_current_rate(self):
        rl = RateLimiter(max_per_minute=100)
        for _ in range(5):
            rl.allow()
        assert rl.current_rate == 5

    def test_allow_cleans_old_entries(self):
        """allow() 清理 60 秒窗口外的请求。"""
        rl = RateLimiter(max_per_minute=10)
        # 手动插入一个过期时间戳
        rl._window.append(time.time() - 120)
        rl.allow()
        # 过期条目被清理
        assert all(t > time.time() - 60 for t in rl._window)

    def test_stats(self):
        rl = RateLimiter(max_per_minute=30)
        rl.allow()
        rl.allow()
        # 拒绝一个
        for _ in range(30):
            rl.allow()
        s = rl.stats()
        assert s["max_per_minute"] == 30
        assert "current_rate" in s
        assert s["rejected"] >= 1

    def test_zero_max(self):
        rl = RateLimiter(max_per_minute=0)
        assert rl.allow() is False


# ============================================================
# HealthChecker
# ============================================================

class TestHealthChecker:
    @pytest.fixture
    def hc(self):
        return HealthChecker()

    def test_initial_state(self, hc):
        assert hc.is_healthy is True  # 无检查时默认健康
        assert hc._ready is False

    def test_set_ready(self, hc):
        hc.set_ready()
        assert hc._ready is True

    def test_set_not_ready(self, hc):
        hc.set_ready()
        hc.set_not_ready()
        assert hc._ready is False

    def test_register_and_run_check_pass(self, hc):
        hc.register_check("db", lambda: (True, "ok"))
        results = hc.run_checks()
        assert results["db"]["status"] == "pass"
        assert results["db"]["detail"] == "ok"

    def test_register_and_run_check_fail(self, hc):
        hc.register_check("db", lambda: (False, "timeout"))
        results = hc.run_checks()
        assert results["db"]["status"] == "fail"

    def test_check_exception_caught(self, hc):
        hc.register_check("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        results = hc.run_checks()
        assert results["bad"]["status"] == "fail"
        assert "boom" in results["bad"]["detail"]

    def test_is_healthy_after_checks(self, hc):
        hc.register_check("a", lambda: (True, "ok"))
        hc.register_check("b", lambda: (False, "bad"))
        hc.run_checks()
        assert hc.is_healthy is False

    def test_is_healthy_multiple_fail(self, hc):
        hc.register_check("a", lambda: (True, "ok"))
        hc.register_check("b", lambda: (True, "ok"))
        hc.run_checks()
        assert hc.is_healthy is True

    def test_shutdown(self, hc):
        hc.set_ready()
        hc.shutdown()
        assert hc._alive is False
        assert hc._ready is False

    def test_status_structure(self, hc):
        hc.set_ready()
        hc.register_check("disk", lambda: (True, "ok"))
        hc.run_checks()
        st = hc.status()
        assert st["status"] == "pass"
        assert st["alive"] is True
        assert st["ready"] is True
        assert st["uptime_seconds"] >= 0
        assert "checks" in st

    def test_fluent_register(self, hc):
        """register_check 返回 self 支持链式调用。"""
        result = hc.register_check("a", lambda: (True, "ok"))
        assert result is hc


# ============================================================
# GracefulShutdown
# ============================================================

class TestGracefulShutdown:
    @pytest.fixture
    def gs(self):
        return GracefulShutdown(grace_period=1.0)

    def test_initial_not_shutting_down(self, gs):
        assert gs.is_shutting_down is False

    def test_start_request_when_normal(self, gs):
        gs.start_request()
        assert gs._active_requests == 1
        gs.end_request()

    def test_start_request_when_shutting_down(self, gs):
        gs._shutting_down = True
        with pytest.raises(RuntimeError, match="拒绝新请求"):
            gs.start_request()

    def test_end_request_decrements(self, gs):
        gs._active_requests = 3
        gs.end_request()
        assert gs._active_requests == 2

    def test_end_request_not_negative(self, gs):
        gs.end_request()
        assert gs._active_requests == 0

    def test_initiate_sets_state(self, gs):
        health = HealthChecker()
        health.set_ready()
        gs.initiate(health)
        assert gs.is_shutting_down is True
        assert health._ready is False

    def test_initiate_with_logger(self, gs):
        health = HealthChecker()
        mock_log = Mock()
        gs.initiate(health, logger=mock_log)
        assert mock_log.info.call_count >= 2
        # 第一条日志: "收到关闭信号"
        first_call_msg = mock_log.info.call_args_list[0][0][0]
        assert "关闭" in first_call_msg

    def test_initiate_waits_for_requests(self, gs):
        """有活跃请求时应等待完成。"""
        gs.grace_period = 5.0
        health = HealthChecker()

        def delayed_end():
            time.sleep(0.3)
            gs.end_request()

        gs._active_requests = 1
        t = threading.Thread(target=delayed_end)
        t.start()
        gs.initiate(health)
        t.join()
        assert gs._active_requests == 0

    def test_thread_safety(self, gs):
        """并发 start/end 不应丢失计数。"""
        gs._active_requests = 0
        errors = []

        def worker():
            try:
                for _ in range(100):
                    gs.start_request()
                    gs.end_request()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert gs._active_requests == 0
        assert len(errors) == 0


# ============================================================
# ServiceStats
# ============================================================

class TestServiceStats:
    @pytest.fixture
    def ss(self):
        return ServiceStats()

    def test_initial_snapshot(self, ss):
        snap = ss.snapshot()
        assert snap["total_requests"] == 0
        assert snap["total_errors"] == 0
        assert snap["error_rate"] == 0.0

    def test_record_success(self, ss):
        ss.record(200, 45.0)
        assert ss.total_requests == 1
        assert ss.total_errors == 0

    def test_record_error(self, ss):
        ss.record(500, 100.0)
        assert ss.total_requests == 1
        assert ss.total_errors == 1

    def test_record_4xx_is_error(self, ss):
        ss.record(404, 10.0)
        assert ss.total_errors == 1

    def test_error_rate(self, ss):
        ss.record(200, 10)
        ss.record(500, 20)
        ss.record(200, 30)
        assert ss.error_rate == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_avg_latency(self, ss):
        for lat in [10.0, 20.0, 30.0]:
            ss.record(200, lat)
        assert ss.avg_latency_ms == pytest.approx(20.0)

    def test_p99_latency(self, ss):
        # 100 个 10ms + 1 个 100ms: p99 index = int(101*0.99) = 99 → s[99] = 10.0
        for _ in range(100):
            ss.record(200, 10.0)
        ss.record(200, 100.0)
        assert ss.p99_latency_ms == pytest.approx(10.0)

    def test_p99_single_value(self, ss):
        ss.record(200, 42.0)
        assert ss.p99_latency_ms == 42.0

    def test_uptime(self, ss):
        assert ss.uptime_seconds >= 0

    def test_status_codes_distribution(self, ss):
        ss.record(200, 10)
        ss.record(200, 20)
        ss.record(404, 15)
        snap = ss.snapshot()
        assert snap["status_codes"][200] == 2
        assert snap["status_codes"][404] == 1

    def test_latency_buffer_capped(self, ss):
        """超过 1000 条时只保留最新 1000。"""
        for i in range(1500):
            ss.record(200, float(i))
        assert len(ss._latencies) == 1000
        # 最新的是 1499, 最早的是 500
        assert ss._latencies[-1] == 1499.0
        assert ss._latencies[0] == 500.0

    def test_rps_in_snapshot(self, ss):
        ss.record(200, 10)
        snap = ss.snapshot()
        assert "rps" in snap
        assert snap["rps"] >= 0
