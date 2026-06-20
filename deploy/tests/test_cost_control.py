# ============================================================
# deploy/tests/test_cost_control.py — 成本控制组件测试
# ============================================================

import time
import pytest
from unittest.mock import patch, Mock

from deploy.cost_control import (
    calc_cost, MODEL_PRICES, ExactCache, ModelRouter, SmartClient,
)


# ============================================================
# calc_cost
# ============================================================

class TestCalcCost:
    def test_sonnet_pricing(self):
        # $3/M input, $15/M output
        cost = calc_cost("claude-sonnet-4-6", 1000000, 1000000)
        assert cost == pytest.approx(18.0)

    def test_haiku_pricing(self):
        cost = calc_cost("claude-haiku-4-5", 1000000, 1000000)
        assert cost == pytest.approx(4.8)

    def test_opus_pricing(self):
        cost = calc_cost("claude-opus-4", 1000000, 1000000)
        assert cost == pytest.approx(90.0)

    def test_deepseek_pricing(self):
        cost = calc_cost("deepseek-v3", 1000000, 1000000)
        assert cost == pytest.approx(1.37)

    def test_zero_tokens(self):
        cost = calc_cost("claude-sonnet-4-6", 0, 0)
        assert cost == 0.0

    def test_unknown_model(self):
        cost = calc_cost("unknown-model", 1000000, 1000000)
        assert cost == 0.0

    def test_partial_tokens(self):
        cost = calc_cost("claude-sonnet-4-6", 500, 200)
        assert cost == pytest.approx(500 / 1e6 * 3.0 + 200 / 1e6 * 15.0)


# ============================================================
# ExactCache
# ============================================================

class TestExactCache:
    @pytest.fixture
    def cache(self):
        return ExactCache(max_size=10, ttl_seconds=60)

    def test_get_miss_returns_none(self, cache):
        assert cache.get("unknown") is None

    def test_set_and_get(self, cache):
        cache.set("prompt", "result")
        assert cache.get("prompt") == "result"

    def test_get_after_ttl_expiry(self, cache):
        cache.ttl_seconds = 0.1
        cache.set("prompt", "value")
        time.sleep(0.15)
        assert cache.get("prompt") is None

    def test_key_different_params(self, cache):
        cache.set("prompt", "r1", model="sonnet")
        cache.set("prompt", "r2", model="haiku")
        assert cache.get("prompt", model="sonnet") == "r1"
        assert cache.get("prompt", model="haiku") == "r2"

    def test_key_same_content_same_hash(self, cache):
        k1 = cache._key("hello", a=1)
        k2 = cache._key("hello", a=1)
        assert k1 == k2

    def test_hit_rate(self, cache):
        cache.set("p", "v")
        cache.get("p")   # hit
        cache.get("p2")  # miss
        assert cache.hit_rate == pytest.approx(0.5)

    def test_hit_rate_zero_on_empty(self, cache):
        assert cache.hit_rate == 0.0

    def test_lru_eviction(self, cache):
        cache.max_size = 3
        for i in range(5):
            cache.set(f"p{i}", f"v{i}")
        assert len(cache._store) == 3
        # 最早插入的 p0, p1 被淘汰
        assert cache.get("p0") is None
        assert cache.get("p1") is None
        # 最近插入的还在
        assert cache.get("p4") == "v4"

    def test_get_moves_to_end(self, cache):
        cache.max_size = 3
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        cache.get("a")  # 将 a 移到末尾
        cache.set("d", "4")  # 淘汰最旧的 b
        assert cache.get("a") == "1"
        assert cache.get("b") is None

    def test_hit_count_increments(self, cache):
        cache.set("p", "v")
        cache.get("p")
        cache.get("p")
        entry = cache._store[cache._key("p")]
        assert entry.hit_count == 3  # 1(set) + 2(get)

    def test_stats(self, cache):
        cache.set("a", "1")
        cache.get("a")
        cache.get("b")
        s = cache.stats()
        assert s["size"] == 1
        assert s["hits"] == 1
        assert s["misses"] == 1


# ============================================================
# ModelRouter
# ============================================================

class TestModelRouter:
    @pytest.fixture
    def router(self):
        return ModelRouter(
            cheap="claude-haiku-4-5",
            normal="claude-sonnet-4-6",
            premium="claude-opus-4",
        )

    def test_complex_task_routes_to_premium(self, router):
        for task in ["分析代码架构", "多步推理任务", "规划系统设计"]:
            assert router.route(task) == "claude-opus-4"

    def test_simple_task_routes_to_cheap(self, router):
        for task in ["翻译一段文字", "摘要总结", "分类提取"]:
            assert router.route(task) == "claude-haiku-4-5"

    def test_short_task_routes_to_cheap(self, router):
        assert router.route("hi") == "claude-haiku-4-5"

    def test_normal_task_routes_to_normal(self, router):
        # 不包含简单/复杂信号字，且长度 >= 20 的普通任务
        assert router.route("请帮我写一段完整的Python代码来实现数据排序功能") == "claude-sonnet-4-6"

    def test_high_context_routes_to_premium(self, router):
        assert router.route("写代码", context_tokens=60000) == "claude-opus-4"

    def test_context_50001_is_premium(self, router):
        assert router.route("随便什么", context_tokens=50001) == "claude-opus-4"

    def test_context_50000_not_premium(self, router):
        assert router.route("随便什么", context_tokens=50000) != "claude-opus-4"

    def test_simple_signal_substring_match(self, router):
        """信号检测是子串匹配，不是分词。"""
        assert router.route("请帮我翻译这段") == "claude-haiku-4-5"


# ============================================================
# SmartClient
# ============================================================

class TestSmartClient:
    @pytest.fixture
    def client(self):
        return SmartClient(sem_threshold=0.85)

    def test_exact_cache_hit(self, client):
        # 首次调用 miss, 存入精确缓存
        result1, meta1 = client.call("test task")
        assert meta1["cache"] == "miss"
        # 第二次调用 hit 精确缓存
        result2, meta2 = client.call("test task")
        assert meta2["cache"] == "exact"
        assert result1 == result2

    def test_cache_hit_count(self, client):
        client.call("task1")
        client.call("task1")
        client.call("task1")
        assert client.cache_hits == 2
        assert client.llm_calls == 1

    def test_stats(self, client):
        client.call("a")
        client.call("a")
        s = client.stats()
        assert s["total_requests"] == 2
        assert s["cache_hits"] == 1
        assert s["llm_calls"] == 1

    def test_llm_fn_called(self, client):
        mock_fn = Mock(return_value=("result", 100, 50))
        result, meta = client.call("task", llm_fn=mock_fn)
        assert meta["cache"] == "miss"
        assert mock_fn.call_count == 1

    def test_cost_tracking(self, client):
        # 使用真实 cost 计算 (sonnet: $3/M in, $15/M out)
        client.call("task", llm_fn=Mock(return_value=("result", 100, 50)))
        assert client.total_cost > 0
