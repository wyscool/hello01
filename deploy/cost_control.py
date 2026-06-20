# ============================================================
# deploy/cost_control.py — 成本控制
# ============================================================
# 提取自 phase5/43_cost_control.py
#
# 类比 Java:
#   ExactCache    ≈ Caffeine Cache
#   SemanticCache ≈ Redis + 向量搜索
#   ModelRouter   ≈ 策略模式 (Strategy Pattern)
# ============================================================

import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any
from collections import OrderedDict


# ============================================================
# 〇、模型定价
# ============================================================

MODEL_PRICES = {
    "claude-opus-4": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 0.8, "output": 4.0},
    "deepseek-v3": {"input": 0.27, "output": 1.10},
}


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICES.get(model)
    if not prices:
        return 0.0
    return (input_tokens / 1_000_000 * prices["input"]
            + output_tokens / 1_000_000 * prices["output"])


# ============================================================
# 一、精确缓存 — LRU + TTL
# ============================================================

@dataclass
class CacheEntry:
    value: Any
    created_at: float = field(default_factory=time.time)
    hit_count: int = 1


class ExactCache:
    """LRU + TTL 精确缓存。"""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _key(self, prompt: str, **params) -> str:
        raw = json.dumps({"prompt": prompt, **params},
                         sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, prompt: str, **params) -> Any | None:
        key = self._key(prompt, **params)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        if time.time() - entry.created_at > self.ttl_seconds:
            del self._store[key]
            self._misses += 1
            return None
        self._store.move_to_end(key)
        entry.hit_count += 1
        self._hits += 1
        return entry.value

    def set(self, prompt: str, value: Any, **params):
        key = self._key(prompt, **params)
        while len(self._store) >= self.max_size:
            self._store.popitem(last=False)
        self._store[key] = CacheEntry(value=value)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "size": len(self._store), "max_size": self.max_size,
            "hits": self._hits, "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
        }


# ============================================================
# 二、语义缓存 — Embedding 相似匹配
# ============================================================

class SemanticCache:
    """语义缓存 — 相似查询命中 (lazy load 模型)。"""

    def __init__(self, threshold: float = 0.85, max_size: int = 500):
        self.threshold = threshold
        self.max_size = max_size
        self._queries: list[str] = []
        self._embeddings: list = []
        self._results: list[Any] = []
        self._hits = 0
        self._misses = 0
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._model = False  # 标记不可用

    def get(self, query: str) -> tuple[Any | None, float]:
        self._ensure_model()
        if self._model is False or not self._queries:
            self._misses += 1
            return None, 0.0

        from sentence_transformers import util
        import numpy as np

        query_emb = self._model.encode([query], convert_to_numpy=True)[0]
        best_idx, best_sim = -1, 0.0
        for i, cached_emb in enumerate(self._embeddings):
            sim = float(util.cos_sim(query_emb, cached_emb)[0][0])
            if sim > best_sim:
                best_sim, best_idx = sim, i

        if best_sim >= self.threshold and best_idx >= 0:
            self._hits += 1
            return self._results[best_idx], round(best_sim, 4)
        self._misses += 1
        return None, 0.0

    def set(self, query: str, result: Any):
        self._ensure_model()
        if self._model is False:
            return
        import numpy as np
        if len(self._queries) >= self.max_size:
            self._queries.pop(0)
            self._embeddings.pop(0)
            self._results.pop(0)
        self._queries.append(query)
        self._embeddings.append(
            self._model.encode([query], convert_to_numpy=True)[0]
        )
        self._results.append(result)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "size": len(self._queries), "max_size": self.max_size,
            "hits": self._hits, "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
            "threshold": self.threshold,
        }


# ============================================================
# 三、模型分级路由
# ============================================================

class ModelRouter:
    """根据任务复杂度选择模型。"""

    SIMPLE_SIGNALS = ["翻译", "translate", "摘要", "summarize",
                      "分类", "classify", "提取", "extract"]
    COMPLEX_SIGNALS = ["规划", "plan", "分析", "analyze",
                       "多步", "推理", "reason", "架构", "architect"]

    def __init__(self, cheap: str = "claude-haiku-4-5",
                 normal: str = "claude-sonnet-4-6",
                 premium: str = "claude-opus-4"):
        self.cheap = cheap
        self.normal = normal
        self.premium = premium

    def route(self, task: str, context_tokens: int = 0) -> str:
        if context_tokens > 50000:
            return self.premium
        for sig in self.COMPLEX_SIGNALS:
            if sig in task:
                return self.premium
        for sig in self.SIMPLE_SIGNALS:
            if sig in task:
                return self.cheap
        if len(task) < 20:
            return self.cheap
        return self.normal


# ============================================================
# 四、SmartClient — 缓存 + 路由组合
# ============================================================

class SmartClient:
    """智能 LLM 客户端: 精确缓存 → 语义缓存 → 模型路由 → LLM。

    使用方法:
        client = SmartClient()
        result, meta = client.call("翻译: hello")
    """

    def __init__(self, sem_threshold: float = 0.85):
        self.exact_cache = ExactCache()
        self.sem_cache = SemanticCache(threshold=sem_threshold)
        self.router = ModelRouter()
        self.total_cost = 0.0
        self.cache_hits = 0
        self.llm_calls = 0

    def call(self, task: str, llm_fn=None, context_tokens: int = 0
             ) -> tuple[str, dict]:
        """执行请求，自动处理缓存和路由。

        Args:
            task: 用户任务
            llm_fn: 实际 LLM 调用函数 (input_tk, output_tk) → result
            context_tokens: 上下文 token 数

        Returns:
            (result_text, metadata)
        """
        # 第一层: 精确缓存
        cached = self.exact_cache.get(task)
        if cached is not None:
            self.cache_hits += 1
            return cached, {"cache": "exact", "cost": 0.0}

        # 第二层: 语义缓存
        cached, sim = self.sem_cache.get(task)
        if cached is not None:
            self.cache_hits += 1
            return cached, {"cache": "semantic", "cost": 0.0,
                           "similarity": sim}

        # 第三层: 实际调用
        model = self.router.route(task, context_tokens)
        self.llm_calls += 1

        # 模拟调用 (如果没有提供 llm_fn)
        if llm_fn is None:
            result = f"[{model}] 关于 '{task[:30]}...' 的回复"
            input_tk = len(task) * 2 + context_tokens
            output_tk = len(result) * 2
        else:
            result, input_tk, output_tk = llm_fn(task, model)

        cost = calc_cost(model, input_tk, output_tk)
        self.total_cost += cost

        # 存入缓存
        self.exact_cache.set(task, result)
        self.sem_cache.set(task, result)

        return result, {
            "cache": "miss", "model": model, "cost": cost,
            "input_tokens": input_tk, "output_tokens": output_tk,
        }

    def stats(self) -> dict:
        total = self.cache_hits + self.llm_calls
        return {
            "total_requests": total,
            "cache_hits": self.cache_hits,
            "llm_calls": self.llm_calls,
            "cache_hit_rate": round(
                self.cache_hits / max(total, 1), 3
            ),
            "total_cost": round(self.total_cost, 6),
            "exact_cache": self.exact_cache.stats(),
            "sem_cache": self.sem_cache.stats(),
        }
