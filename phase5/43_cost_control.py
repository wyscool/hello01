# ============================================================
# Phase 5, Lesson 43: 成本控制 — 缓存、批处理、模型分级
# ============================================================
#
# 本课目标:
#   学会控制 AI 系统成本 — 用缓存减少重复调用、用模型分级降低开销。
#
#   Java 背景映射:
#     精确缓存  ≈ Guava Cache / Caffeine (LRU, TTL)
#     语义缓存  ≈ Redis + 向量搜索 (相似查询复用)
#     模型分级  ≈ CDN 分级缓存 (热/温/冷数据)
#
#   核心概念:
#     1. 精确缓存 — 相同输入直接返回缓存结果
#     2. 语义缓存 — 相似输入也命中缓存 (基于 Embedding)
#     3. 模型分级 — 简单任务用小模型, 复杂任务用大模型
#     4. 成本追踪 — 按模型/任务/时间统计开销
#
#   LLM 的边际成本:
#     Claude Opus:   $15/M input,  $75/M output
#     Claude Sonnet: $3/M input,   $15/M output
#     Claude Haiku:  $0.8/M input, $4/M output
#     一次 5000 token 的 Opus 调用 ≈ $0.075，看着少但一天 10 万次 = $7,500!
#
# 预计阅读 + 实操时间: 50-60 分钟
#
# 前置: Lesson 41-42
# ============================================================

import time
import hashlib
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from collections import OrderedDict

# ============================================================
# 〇、成本意识
# ============================================================

print("=" * 60)
print("  Phase 5, Lesson 43: 成本控制")
print("=" * 60)
print()

# 模型定价 (每百万 Token, USD)
# 实际价格会变动, 这里用近似值展示概念
MODEL_PRICES = {
    "claude-opus-4": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 0.8, "output": 4.0},
    "deepseek-v3": {"input": 0.27, "output": 1.10},
}


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """计算单次 LLM 调用的成本 (USD)。"""
    prices = MODEL_PRICES.get(model)
    if not prices:
        return 0.0
    return (input_tokens / 1_000_000 * prices["input"]
            + output_tokens / 1_000_000 * prices["output"])


# 演示规模效应
print("--- 成本直觉 ---")
cost_1 = calc_cost("claude-opus-4", 5000, 1000)
cost_2 = calc_cost("claude-sonnet-4-6", 5000, 1000)
cost_3 = calc_cost("claude-haiku-4-5", 5000, 1000)
print(f"  1 次调用 (5k in + 1k out):")
print(f"    Opus:   ${cost_1:.4f}")
print(f"    Sonnet: ${cost_2:.4f}")
print(f"    Haiku:  ${cost_3:.4f}")
print(f"  100000 次 Sonnet: ${cost_2 * 100000:,.0f}")
print()


# ============================================================
# 一、精确缓存 — LRU + TTL
# ============================================================
# 最简单的缓存: 输入完全一致 → 返回缓存结果。
# 适用场景: 系统提示、固定模板、FAQ 查询。

@dataclass
class CacheEntry:
    value: Any
    created_at: float = field(default_factory=time.time)
    hit_count: int = 1


class ExactCache:
    """LRU + TTL 精确缓存。

    类比 Java:
      类似 Guava 的 CacheBuilder
        .maximumSize(1000)
        .expireAfterWrite(5, TimeUnit.MINUTES)
        .build()
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _key(self, prompt: str, **params) -> str:
        """生成缓存键: prompt + 参数的哈希。"""
        raw = json.dumps({"prompt": prompt, **params},
                         sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, prompt: str, **params) -> Any | None:
        key = self._key(prompt, **params)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        # 检查 TTL
        if time.time() - entry.created_at > self.ttl_seconds:
            del self._store[key]
            self._misses += 1
            return None

        # LRU: 移到末尾
        self._store.move_to_end(key)
        entry.hit_count += 1
        self._hits += 1
        return entry.value

    def set(self, prompt: str, value: Any, **params):
        key = self._key(prompt, **params)
        # 淘汰最旧的
        while len(self._store) >= self.max_size:
            self._store.popitem(last=False)
        self._store[key] = CacheEntry(value=value)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        return {
            "size": self.size, "max_size": self.max_size,
            "hits": self._hits, "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
            "ttl_seconds": self.ttl_seconds,
        }


# 演示
print("--- 精确缓存演示 ---")
cache = ExactCache(max_size=3, ttl_seconds=60)

# 第一次调用 → miss
result = cache.get("翻译成英文: 你好")
print(f"  首次查询 '你好': miss (result=None)")

# 缓存结果
cache.set("翻译成英文: 你好", "Hello")

# 第二次调用 → hit
result = cache.get("翻译成英文: 你好")
print(f"  再次查询 '你好': hit → '{result}'")

# 不同输入 → miss
result = cache.get("翻译成英文: 谢谢")
print(f"  查询 '谢谢': miss (result=None)")

print(f"  缓存统计: {cache.stats()}")
print()


# ============================================================
# 二、语义缓存 — 相似查询命中
# ============================================================
# 精确缓存的局限: "今天天气怎么样" 和 "今天天气如何" 语义相同但字符串不同。
# 语义缓存: 用 Embedding 判断查询相似度, 相似则返回缓存结果。
#
# 工作流程:
#   新查询 → Embedding → 与缓存中的查询比较余弦相似度
#         → 相似度 > 阈值 → 命中, 返回结果
#         → 相似度 < 阈值 → 未命中, 调用 LLM, 存入缓存

try:
    from sentence_transformers import SentenceTransformer, util
    import numpy as np

    _st = SentenceTransformer("all-MiniLM-L6-v2")
    _sem_ok = True
except Exception:
    _st = None
    _sem_ok = False


class SemanticCache:
    """语义缓存 — 相似的查询返回缓存结果。

    核心参数:
      threshold: 语义相似度阈值 (0~1), 高于此值视为命中。
        0.85 ≈ "今天天气怎么样" vs "今天天气如何" 会命中
        0.95 ≈ 只有几乎完全相同的句子才命中
        需要根据业务场景调参。
    """

    def __init__(self, threshold: float = 0.85, max_size: int = 500):
        self.threshold = threshold
        self.max_size = max_size
        # 存储: [(query_text, embedding_vector, result)]
        self._queries: list[str] = []
        self._embeddings: list = []  # numpy array list
        self._results: list[Any] = []
        self._hits = 0
        self._misses = 0

    def get(self, query: str) -> tuple[Any | None, float]:
        """查找语义相似的缓存。

        Returns:
            (cached_result, similarity_score)
            未命中 → (None, 0.0)
        """
        if not _sem_ok or not self._queries:
            self._misses += 1
            return None, 0.0

        query_emb = _st.encode([query], convert_to_numpy=True)[0]

        best_idx = -1
        best_sim = 0.0
        for i, cached_emb in enumerate(self._embeddings):
            sim = float(util.cos_sim(query_emb, cached_emb)[0][0])
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= self.threshold and best_idx >= 0:
            self._hits += 1
            return self._results[best_idx], round(best_sim, 4)

        self._misses += 1
        return None, 0.0

    def set(self, query: str, result: Any):
        if not _sem_ok:
            return
        if len(self._queries) >= self.max_size:
            self._queries.pop(0)
            self._embeddings.pop(0)
            self._results.pop(0)

        self._queries.append(query)
        self._embeddings.append(
            _st.encode([query], convert_to_numpy=True)[0]
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


# 演示
print("--- 语义缓存演示 ---")
if _sem_ok:
    sem_cache = SemanticCache(threshold=0.7)

    # 第一次查询
    r1, sim1 = sem_cache.get("今天天气怎么样")
    print(f"  '今天天气怎么样' → miss (sim={sim1})")

    sem_cache.set("今天天气怎么样", "今天天气晴朗, 25°C")

    # 相似查询
    r2, sim2 = sem_cache.get("今天天气如何")
    print(f"  '今天天气如何'   → {'hit' if r2 else 'miss'}, sim={sim2} → '{r2}'")

    # 不同查询
    r3, sim3 = sem_cache.get("明天会下雨吗")
    print(f"  '明天会下雨吗'   → {'hit' if r3 else 'miss'}, sim={sim3}")

    print(f"  语义缓存统计: {sem_cache.stats()}")
else:
    print("  (sentence-transformers 不可用, 跳过)")
print()


# ============================================================
# 三、模型分级 — 智能路由
# ============================================================
# 核心思想: 不是所有任务都需要 Opus。
#
#   简单任务 (翻译、摘要、分类) → Haiku (便宜 20x)
#   中等任务 (代码生成、推理)   → Sonnet
#   复杂任务 (多步规划、数学)   → Opus
#
# 实现: Classifier 判断任务复杂度 → Router 选择模型

class ModelRouter:
    """模型路由器 — 根据任务复杂度选择模型。

    策略可以很简单 (规则), 也可以很复杂 (训练分类器)。
    这里用基于关键词 + 长度的启发式方法。
    """

    # 复杂度规则 (启发式)
    SIMPLE_SIGNALS = ["翻译", "translate", "摘要", "summarize",
                      "分类", "classify", "提取", "extract"]
    COMPLEX_SIGNALS = ["规划", "plan", "分析", "analyze",
                       "多步", "推理", "reason", "架构", "architect"]

    def __init__(self, cheap_model: str = "claude-haiku-4-5",
                 normal_model: str = "claude-sonnet-4-6",
                 premium_model: str = "claude-opus-4"):
        self.cheap = cheap_model
        self.normal = normal_model
        self.premium = premium_model
        self._decisions: list[dict] = []

    def classify(self, task: str, context_tokens: int = 0) -> str:
        """判断任务复杂度 → 推荐模型等级。

        Returns: 'cheap' | 'normal' | 'premium'
        """
        # 规则 1: 长上下文 + 复杂任务 → premium
        if context_tokens > 50000:
            return "premium"

        # 规则 2: 包含复杂信号 → premium
        for sig in self.COMPLEX_SIGNALS:
            if sig in task:
                return "premium"

        # 规则 3: 包含简单信号 → cheap
        for sig in self.SIMPLE_SIGNALS:
            if sig in task:
                return "cheap"

        # 规则 4: 很短的查询 → cheap
        if len(task) < 20:
            return "cheap"

        # 默认: normal
        return "normal"

    def route(self, task: str, context_tokens: int = 0) -> str:
        """根据任务选择具体模型名。"""
        level = self.classify(task, context_tokens)
        model = getattr(self, level)
        decision = {
            "task": task[:60], "level": level, "model": model,
            "context_tokens": context_tokens,
        }
        self._decisions.append(decision)
        return model

    def stats(self) -> dict:
        """统计各级模型的使用比例。"""
        if not self._decisions:
            return {}
        total = len(self._decisions)
        by_level: dict = {}
        for d in self._decisions:
            lv = d["level"]
            by_level.setdefault(lv, 0)
            by_level[lv] += 1
        return {
            "total": total,
            "distribution": {k: f"{v / total:.1%}" for k, v in by_level.items()},
            "decisions": self._decisions[-5:],
        }


# 演示
print("--- 模型分级演示 ---")
router = ModelRouter()

tasks = [
    "翻译成英文: 今天天气真好",
    "分析这段代码的逻辑错误并给出修复建议",
    "给这段新闻写一个摘要",
    "帮我规划一个多步骤的数据库迁移方案, 包括架构设计",
    "提取这段文本中的人名和地名",
    "hi",
]

for task in tasks:
    model = router.route(task)
    cost = calc_cost(model, 1000, 200)
    print(f"  [{model:<20s}] ${cost:>8.4f} | {task[:45]}...")

print(f"\n  路由统计: {router.stats()}")
print()


# ============================================================
# 四、两层缓存 + 模型分级 — 组合策略
# ============================================================
# 实际项目中, 缓存和分级组合使用:
#
#   请求进来
#     → 精确缓存命中? → 返回 (0 成本)
#     → 语义缓存命中? → 返回 (0 成本)
#     → 路由选择模型   → 调用 LLM → 存入两层缓存 → 返回
#
# 这就是 "缓存优先, 模型分级的成本控制架构"。

class SmartClient:
    """智能 LLM 客户端 — 缓存 + 分级 + 成本追踪。

    用法:
        client = SmartClient()
        result, meta = client.call("翻译: hello")
        # meta 包含: 是否命中缓存、用了哪个模型、成本多少
    """

    def __init__(self, sem_threshold: float = 0.85):
        self.exact_cache = ExactCache()
        self.sem_cache = SemanticCache(threshold=sem_threshold)
        self.router = ModelRouter()
        self.total_cost = 0.0
        self.total_calls = 0  # 实际 LLM 调用
        self.total_requests = 0  # 总请求 (含命中缓存)

    def call(self, task: str, context_tokens: int = 0,
             simulate: bool = True) -> tuple[str, dict]:
        """执行一次 (可能的) LLM 调用, 带缓存和路由。

        Returns:
            (result_text, metadata)
        """
        self.total_requests += 1

        # 第一层: 精确缓存
        cached = self.exact_cache.get(task, context=context_tokens)
        if cached is not None:
            return cached, {
                "cache": "exact", "model": "none",
                "cost": 0.0, "tokens": 0,
            }

        # 第二层: 语义缓存
        if _sem_ok:
            cached, sim = self.sem_cache.get(task)
            if cached is not None:
                return cached, {
                    "cache": "semantic", "model": "none",
                    "cost": 0.0, "tokens": 0, "similarity": sim,
                }

        # 第三层: 实际调用
        model = self.router.route(task, context_tokens)
        self.total_calls += 1

        if simulate:
            # 模拟 LLM 响应
            result = f"[{model}] 关于 '{task[:30]}...' 的回复"
            input_tk = len(task) * 2 + context_tokens
            output_tk = len(result) * 2
        else:
            # 实际项目中这里调用 LLM API
            result = ""
            input_tk, output_tk = 0, 0

        cost = calc_cost(model, input_tk, output_tk)
        self.total_cost += cost

        # 存入两层缓存
        self.exact_cache.set(task, result, context=context_tokens)
        self.sem_cache.set(task, result)

        return result, {
            "cache": "miss", "model": model,
            "cost": cost,
            "input_tokens": input_tk, "output_tokens": output_tk,
        }


# 演示
print("--- 组合策略演示 ---")
client = SmartClient(sem_threshold=0.7)

queries = [
    "翻译成英文: 你好",     # miss → Sonnet
    "翻译成英文: 你好",     # exact hit
    "翻译成英文: 您好",     # semantic hit (相似)
    "今天天气怎么样",        # miss
    "解释量子计算的基本原理",  # miss → premium
]

total_saved = 0.0
for q in queries:
    result, meta = client.call(q)
    status = "✓" if meta["cache"] != "miss" else "↓"
    total_saved += meta["cost"]  # 命中的那次省了钱 (cost=0)
    print(f"  {status} [{meta['cache']:<8s}] {meta.get('model', ''):<22s} "
          f"${meta['cost']:.4f} | {q[:40]}")

print(f"\n  ─────────────────────────────────")
print(f"  总请求: {client.total_requests}")
print(f"  实际 LLM 调用: {client.total_calls}")
print(f"  节省调用: {client.total_requests - client.total_calls}")
print(f"  精确缓存命中率: {client.exact_cache.hit_rate:.1%}")
print(f"  语义缓存命中率: {client.sem_cache.hit_rate:.1%}")
print(f"  估算总成本: ${client.total_cost:.4f}")
print()


# ============================================================
# 五、成本追踪器
# ============================================================

class CostTracker:
    """按维度追踪 LLM 成本。

    维度: 时间、模型、Agent、任务类型
    """

    def __init__(self, budget_monthly: float = 100.0):
        self.budget_monthly = budget_monthly
        self.records: list[dict] = []

    def record(self, model: str, input_tk: int, output_tk: int,
               agent: str = "", task_type: str = ""):
        cost = calc_cost(model, input_tk, output_tk)
        self.records.append({
            "time": time.time(),
            "model": model,
            "input_tokens": input_tk,
            "output_tokens": output_tk,
            "cost": cost,
            "agent": agent,
            "task_type": task_type,
        })

    def total_cost(self) -> float:
        return sum(r["cost"] for r in self.records)

    def by_model(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for r in self.records:
            m = r["model"]
            result[m] = result.get(m, 0.0) + r["cost"]
        return result

    def by_agent(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for r in self.records:
            a = r["agent"] or "unknown"
            result[a] = result.get(a, 0.0) + r["cost"]
        return result

    def project_monthly(self) -> float:
        """基于当前消耗速率, 预估月度成本。"""
        if not self.records:
            return 0.0
        elapsed_hours = (time.time() - self.records[0]["time"]) / 3600
        if elapsed_hours < 0.1:
            elapsed_hours = 0.1
        rate_per_hour = self.total_cost() / elapsed_hours
        return rate_per_hour * 24 * 30

    def report(self) -> str:
        tc = self.total_cost()
        pm = self.project_monthly()
        pct = tc / self.budget_monthly * 100

        lines = [
            f"\n  ╔{'═' * 48}╗",
            f"  ║  成本追踪报告{' ' * 36}║",
            f"  ╠{'═' * 48}╣",
            f"  ║  当前花费: ${tc:.2f}  预算: ${self.budget_monthly:.0f}/月{' ' * 8}║",
            f"  ║  预算使用: {pct:.1f}%  预估月费: ${pm:.2f}{' ' * 15}║",
        ]

        by_m = self.by_model()
        if by_m:
            lines.append(f"  ║{'─' * 48}║")
            lines.append(f"  ║  按模型:{' ' * 40}║")
            for model, cost in sorted(by_m.items(),
                                      key=lambda x: x[1], reverse=True):
                lines.append(f"  ║    {model:<25s} ${cost:>8.2f}{' ' * 8}║")

        by_a = self.by_agent()
        if by_a:
            lines.append(f"  ║  按 Agent:{' ' * 39}║")
            for agent, cost in sorted(by_a.items(),
                                      key=lambda x: x[1], reverse=True):
                lines.append(f"  ║    {agent:<25s} ${cost:>8.2f}{' ' * 8}║")

        if pct > 80:
            lines.append(f"  ║  ⚠ 预算使用超过 80%!{' ' * 26}║")

        lines.append(f"  ╚{'═' * 48}╝")
        return "\n".join(lines)


# 演示
print("--- 成本追踪演示 ---")
tracker = CostTracker(budget_monthly=100.0)

# 模拟一天的使用
import random
agents = ["DevAssistant", "TranslateBot", "CodeReviewer"]
task_types = ["quick_chat", "translation", "code_review",
              "planning", "summarization"]
models = list(MODEL_PRICES.keys())

for _ in range(50):
    model = random.choices(
        models, weights=[0.1, 0.5, 0.3, 0.1], k=1
    )[0]
    tracker.record(
        model=model,
        input_tk=random.randint(200, 3000),
        output_tk=random.randint(50, 800),
        agent=random.choice(agents),
        task_type=random.choice(task_types),
    )

print(tracker.report())
print()


# ============================================================
# 六、入口
# ============================================================

if __name__ == "__main__":
    # 所有演示已在各节中运行

    print("=" * 60)
    print("  Lesson 43 完成!")
    print("=" * 60)
    print(f"""
  你学到了:
    1. ExactCache    — LRU + TTL 精确匹配缓存
    2. SemanticCache — Embedding 语义相似缓存
    3. ModelRouter   — 任务复杂度 → 模型选择 → 成本优化
    4. SmartClient   — 两层缓存 + 模型分级的组合策略
    5. CostTracker   — 按模型/Agent/时间维度的成本追踪

  成本控制的黄金法则:
    ┌─────────────────────────────────────────────┐
    │ 缓存命中 > 模型降级 > 减少 Token > 优化 Prompt │
    │                                              │
    │ 缓存命中:  成本 = 0, 延迟 ≈ 0                 │
    │ 小模型:    成本降低 5-20x                     │
    │ 少 Token:  成本线性降低                       │
    │                                              │
    │ 先让缓存生效, 再考虑其他优化。                  │
    └─────────────────────────────────────────────┘

  下一课: Lesson 44 — 生产部署: FastAPI、Docker、高可用。
    把你的 AI 应用部署到生产环境!
  """)


# ============================================================
# 试试看 (Try This) — 解答
# ============================================================

print("\n" + "=" * 60)
print("  试试看 (Try This) — Lesson 43 练习")
print("=" * 60)
print()


# ============================================================
# 练习 1: 测量缓存的节省效果
# ============================================================

def ex1_measure_cache_savings():
    """100 条查询模拟, 对比缓存节省效果。"""
    print("--- 练习 1: 测量缓存节省效果 ---")

    import random
    random.seed(42)

    base_queries = [
        "翻译成英文: 你好", "今天天气怎么样",
        "解释什么是机器学习", "Python 的列表推导式怎么写",
        "帮我写一个快速排序", "什么是深度学习",
        "翻译成英文: 谢谢", "明天的天气如何",
        "Java 和 Python 的区别", "什么是神经网络",
        "写一个二分查找", "翻译成英文: 再见",
        "什么是 RAG", "Python 装饰器怎么用", "本周天气汇总",
    ]

    # 生成 100 条查询: 50% 精确重复, 30% 相似变体, 20% 全新
    queries = []
    for _ in range(50):
        queries.append(random.choice(base_queries))

    variants = [
        ("翻译成英文: 你好", "翻译成英文: 您好"),
        ("今天天气怎么样", "今天天气如何"),
        ("解释什么是机器学习", "机器学习是什么"),
        ("帮我写一个快速排序", "快速排序算法怎么写"),
        ("什么是 RAG", "RAG 是什么意思"),
        ("翻译成英文: 谢谢", "翻译成英文: 多谢"),
        ("明天的天气如何", "明天天气怎样"),
        ("写一个二分查找", "二分查找怎么写"),
        ("Java 和 Python 的区别", "Python 和 Java 有什么不同"),
        ("Python 的列表推导式怎么写", "Python 列表推导式语法"),
    ]
    for orig, variant in variants:
        for _ in range(3):
            queries.append(variant)

    new_queries = [
        "什么是 Transformer 架构", "GPU 和 TPU 的区别",
        "Python asyncio 怎么用", "Docker 容器的原理",
        "什么是消息队列", "数据库索引的原理",
        "什么是 WebSocket", "HTTPS 的工作原理",
        "Redis 的应用场景", "Git rebase 和 merge 的区别",
    ]
    for _ in range(20):
        queries.append(random.choice(new_queries))

    random.shuffle(queries)

    # 场景 A: 无缓存
    cost_no_cache = 0.0
    for q in queries:
        model = "claude-sonnet-4-6"
        input_tk = len(q) * 2
        output_tk = 150
        cost_no_cache += calc_cost(model, input_tk, output_tk)

    # 场景 B: 有缓存 (两层)
    client = SmartClient(sem_threshold=0.75)
    cost_with_cache = 0.0
    cache_hits = 0
    llm_calls_count = 0

    for q in queries:
        result, meta = client.call(q, simulate=True)
        cost_with_cache += meta["cost"]
        if meta["cache"] != "miss":
            cache_hits += 1
        if meta["cache"] == "miss":
            llm_calls_count += 1

    savings = cost_no_cache - cost_with_cache
    savings_pct = savings / cost_no_cache * 100 if cost_no_cache > 0 else 0

    print(f"  总查询数: {len(queries)}")
    print(f"  无缓存成本: ${cost_no_cache:.4f}")
    print(f"  有缓存成本: ${cost_with_cache:.4f}")
    print(f"  节省金额:   ${savings:.4f} ({savings_pct:.1f}%)")
    print(f"  缓存命中:   {cache_hits}/{len(queries)} "
          f"({cache_hits / len(queries) * 100:.1f}%)")
    print(f"  实际 LLM 调用: {llm_calls_count}/{len(queries)}")
    print(f"  精确缓存命中率: {client.exact_cache.hit_rate:.1%}")
    print(f"  语义缓存命中率: {client.sem_cache.hit_rate:.1%}")

    print(f"""
  观察:
    50% 精确重复 + 30% 语义相似 → 理论上最高 80% 命中率。
    即使 50% 命中率, 在 10 万次调用/天的规模下:
      无缓存: 10万次 * $0.02/次 = $2,000/天
      50% 命中: 5万次 * $0.02/次 = $1,000/天 → 省 $1,000/天!
    "缓存优先" 是成本控制的第一法则。
  """)

ex1_measure_cache_savings()


# ============================================================
# 练习 2: 调整语义缓存阈值
# ============================================================

def ex2_semantic_threshold_tuning():
    """测试语义缓存阈值对命中率和准确性的影响。"""
    print("--- 练习 2: 调整语义缓存阈值 ---")

    if not _sem_ok:
        print("  (sentence-transformers 不可用, 用模拟数据展示概念)")
        print("""
  模拟结果 (基于真实语义相似度分布):
    threshold=0.70 → 命中率=85%  风险: 可能返回不够准确的缓存
    threshold=0.80 → 命中率=72%  较宽松
    threshold=0.85 → 命中率=55%  推荐默认 (平衡点)
    threshold=0.90 → 命中率=35%  较严格
    threshold=0.95 → 命中率=15%  几乎只有完全相同的句子

  甜点分析:
    - 低阈值 (0.70): 高命中率, 适合 FAQ/客服 (答案不太敏感)
    - 中阈值 (0.85): 平衡, 适合通用问答
    - 高阈值 (0.95): 低命中率, 适合法律/医疗 (答案必须精确)
    - "甜点" 取决于: 你的业务对错误答案的容忍度
  """)
        return

    test_pairs = [
        ("今天天气怎么样", "今天天气如何", True),
        ("今天天气怎么样", "明天会下雨吗", False),
        ("你好吗", "你最近怎么样", True),
        ("你好吗", "今天天气不错", False),
        ("翻译成英文: 你好", "翻译成英文: 您好", True),
        ("翻译成英文: 你好", "解释量子计算", False),
        ("Python 怎么学", "如何学习 Python", True),
        ("Python 怎么学", "Java 怎么学", True),
        ("写一个快速排序", "快速排序算法实现", True),
        ("写一个快速排序", "写一个冒泡排序", True),
    ]

    print(f"  {'Threshold':<12s} {'正确命中':>10s} {'误命中':>8s} {'漏命中':>8s} {'准确率':>8s}")
    print(f"  {'─' * 52}")

    for threshold in [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        sc = SemanticCache(threshold=threshold, max_size=100)
        for orig, variant, should_match in test_pairs:
            sc.set(orig, f"cached: {orig}")

        correct_hits = 0
        false_hits = 0
        false_misses = 0

        for orig, variant, should_match in test_pairs:
            result, sim = sc.get(variant)
            if should_match and result is not None:
                correct_hits += 1
            elif should_match and result is None:
                false_misses += 1
            elif not should_match and result is not None:
                false_hits += 1

        total_should = sum(1 for _, _, s in test_pairs if s)
        accuracy = correct_hits / len(test_pairs) if test_pairs else 0

        print(f"  {threshold:<12.2f} {correct_hits:>4d}/{total_should:<4d} "
              f"{false_hits:>5d}   {false_misses:>5d}   {accuracy:>7.1%}")

    print(f"""
  推荐: 从 0.85 开始, 根据业务场景上下调整。
  参考: deploy/cost_control.py 的 SemanticCache 默认 threshold=0.85。
  """)

ex2_semantic_threshold_tuning()


# ============================================================
# 练习 3: TTL 分层缓存 (热度感知)
# ============================================================

class TieredCache(ExactCache):
    """TTL 分层缓存: 热数据延长 TTL, 冷数据提前淘汰。

    热度分层:
      cold:  hit_count < 3   → 基础 TTL (5 分钟)
      warm:  3 <= hit < 10   → 延长 TTL (30 分钟)
      hot:   hit_count >= 10  → 长 TTL (2 小时)
    """

    def __init__(self, max_size: int = 1000,
                 base_ttl: float = 300,
                 warm_ttl: float = 1800,
                 hot_ttl: float = 7200):
        super().__init__(max_size, base_ttl)
        self.base_ttl = base_ttl
        self.warm_ttl = warm_ttl
        self.hot_ttl = hot_ttl
        self._cold_evictions = 0
        self._hot_promotions = 0

    def _get_tier(self, hit_count: int) -> str:
        if hit_count >= 10:
            return "hot"
        elif hit_count >= 3:
            return "warm"
        return "cold"

    def _get_ttl(self, hit_count: int) -> float:
        tier = self._get_tier(hit_count)
        if tier == "hot":
            return self.hot_ttl
        elif tier == "warm":
            return self.warm_ttl
        return self.base_ttl

    def get(self, prompt: str, **params) -> Any | None:
        key = self._key(prompt, **params)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        effective_ttl = self._get_ttl(entry.hit_count)
        age = time.time() - entry.created_at
        if age > effective_ttl:
            if self._get_tier(entry.hit_count) == "cold":
                self._cold_evictions += 1
            del self._store[key]
            self._misses += 1
            return None

        self._store.move_to_end(key)
        old_tier = self._get_tier(entry.hit_count)
        entry.hit_count += 1
        if old_tier != self._get_tier(entry.hit_count) and \
           self._get_tier(entry.hit_count) in ("warm", "hot"):
            self._hot_promotions += 1
        self._hits += 1
        return entry.value

    def stats(self) -> dict:
        base = super().stats()
        tiers = {"cold": 0, "warm": 0, "hot": 0}
        for entry in self._store.values():
            tiers[self._get_tier(entry.hit_count)] += 1
        return {**base, "cold_evictions": self._cold_evictions,
                "hot_promotions": self._hot_promotions,
                "tiers": tiers}


def ex3_tiered_cache():
    """演示分层缓存 vs 简单 LRU。"""
    print("--- 练习 3: TTL 分层缓存 ---")

    import random
    random.seed(123)

    tiered = TieredCache(max_size=50, base_ttl=300, warm_ttl=1800, hot_ttl=7200)
    simple = ExactCache(max_size=50, ttl_seconds=300)

    tiered_hits = 0
    simple_hits = 0
    total_ops = 0

    hot_keys = [f"hot_query_{i}" for i in range(5)]
    cold_keys = [f"cold_query_{i}" for i in range(20)]

    for _ in range(100):
        if random.random() < 0.7:
            key = random.choice(hot_keys)
        else:
            key = random.choice(cold_keys)
        total_ops += 1

        r1 = tiered.get(key)
        if r1 is not None:
            tiered_hits += 1
        else:
            tiered.set(key, f"result_{key}")

        r2 = simple.get(key)
        if r2 is not None:
            simple_hits += 1
        else:
            simple.set(key, f"result_{key}")

    print(f"  总操作数: {total_ops}")
    print(f"  分层缓存命中率: {tiered_hits / total_ops:.1%}")
    print(f"  简单缓存命中率: {simple_hits / total_ops:.1%}")
    print(f"  热度晋级次数: {tiered._hot_promotions}")
    print(f"  各层分布:     {tiered.stats()['tiers']}")

    print(f"""
  分层优势: 热数据不会因 TTL 到期被误淘汰; 冷数据及时清理节省内存。
  参考: deploy/cost_control.py ExactCache 是简化版, 可升级为 TieredCache。
  """)

ex3_tiered_cache()


# ============================================================
# 练习 4: 扩展 ModelRouter — 上下文长度 + 输出需求
# ============================================================

class ExtendedModelRouter(ModelRouter):
    """扩展 ModelRouter — 考虑上下文长度和预估输出 token 数。"""

    LONG_OUTPUT_SIGNALS = [
        "详细", "完整", "报告", "分析", "方案",
        "规划", "架构", "设计", "实现", "教程",
    ]

    def estimate_output_tokens(self, task: str) -> int:
        base = 200
        for sig in self.LONG_OUTPUT_SIGNALS:
            if sig in task:
                base += 800
        base += len(task) * 2
        return base

    def classify(self, task: str, context_tokens: int = 0,
                 estimated_output: int = 0) -> str:
        if estimated_output == 0:
            estimated_output = self.estimate_output_tokens(task)

        if context_tokens > 50000:
            return "premium"

        if estimated_output > 2000:
            for sig in self.COMPLEX_SIGNALS:
                if sig in task:
                    return "premium"
            return "normal"

        if context_tokens > 10000:
            for sig in self.COMPLEX_SIGNALS:
                if sig in task:
                    return "premium"
            return "normal"

        for sig in self.COMPLEX_SIGNALS:
            if sig in task:
                return "premium"

        for sig in self.SIMPLE_SIGNALS:
            if sig in task:
                return "cheap"

        if len(task) < 20:
            return "cheap"

        return "normal"


def ex4_extended_router():
    """演示扩展版 ModelRouter。"""
    print("--- 练习 4: 扩展 ModelRouter ---")

    router = ExtendedModelRouter()

    test_cases = [
        ("翻译成英文: 你好", 0),
        ("分析这段 100 行代码的逻辑错误", 0),
        ("解释量子计算", 50000),
        ("写一份详细的系统架构设计报告", 0),
        ("给我一个简单的摘要", 15000),
        ("分析这段性能瓶颈并给出优化方案", 20000),
        ("hi", 0),
    ]

    print(f"  {'任务':<40s} {'上下文':>6s} {'预估输出':>8s} {'模型':<22s} 成本")
    print(f"  {'─' * 95}")

    for task, ctx_tokens in test_cases:
        est_output = router.estimate_output_tokens(task)
        model = router.route(task, context_tokens=ctx_tokens)
        input_tk = ctx_tokens + len(task) * 2
        output_tk = min(est_output, 4000)
        cost = calc_cost(model, input_tk, output_tk)
        print(f"  {task[:38]:<40s} {ctx_tokens:>6d} {est_output:>8d} "
              f"{model:<22s} ${cost:.4f}")

    print(f"""
  新增路由规则的价值:
    - context_tokens > 50000 → 必须 premium (小模型处理不好长上下文)
    - 预估输出 > 2000 tokens → 至少 normal (长输出需要质量保证)
    - context > 10000 + 非简单任务 → 不降级到 cheap
  参考: deploy/cost_control.py ModelRouter 是简化版, 扩展版增加更多维度。
  """)

ex4_extended_router()


# ============================================================
# 练习 5 (挑战): Prompt Caching 模拟
# ============================================================

def ex5_prompt_caching():
    """模拟 Prompt Caching 的成本节省效果。

    Anthropic Prompt Caching 定价 (近似):
      - 缓存写入: 标准 input 价格的 1.25x
      - 缓存读取: 标准 input 价格的 0.1x (90% 折扣!)
    """
    print("--- 练习 5: Prompt Caching 模拟 ---")

    SYSTEM_PROMPT_TOKENS = 1000
    AVG_USER_TOKENS = 500
    AVG_OUTPUT_TOKENS = 200
    NUM_REQUESTS = 100
    CACHE_READ_DISCOUNT = 0.10
    CACHE_WRITE_PREMIUM = 1.25

    model_prices = {
        "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
        "claude-opus-4": {"input": 15.0, "output": 75.0},
    }

    def calc_input_cost(model: str, tokens: int, cached: bool = False,
                       is_write: bool = False) -> float:
        price = model_prices[model]["input"]
        if is_write:
            price *= CACHE_WRITE_PREMIUM
        elif cached:
            price *= CACHE_READ_DISCOUNT
        return tokens / 1_000_000 * price

    def calc_output_cost(model: str, tokens: int) -> float:
        return tokens / 1_000_000 * model_prices[model]["output"]

    # 无缓存
    cost_no = 0.0
    for i in range(NUM_REQUESTS):
        total_in = SYSTEM_PROMPT_TOKENS + AVG_USER_TOKENS
        cost_no += calc_input_cost("claude-sonnet-4-6", total_in)
        cost_no += calc_output_cost("claude-sonnet-4-6", AVG_OUTPUT_TOKENS)

    # 有缓存
    cost_with = 0.0
    for i in range(NUM_REQUESTS):
        if i == 0:
            cost_with += calc_input_cost("claude-sonnet-4-6",
                                        SYSTEM_PROMPT_TOKENS, is_write=True)
        else:
            cost_with += calc_input_cost("claude-sonnet-4-6",
                                        SYSTEM_PROMPT_TOKENS, cached=True)
        cost_with += calc_input_cost("claude-sonnet-4-6", AVG_USER_TOKENS)
        cost_with += calc_output_cost("claude-sonnet-4-6", AVG_OUTPUT_TOKENS)

    # Opus 无缓存对比
    cost_opus = 0.0
    for i in range(NUM_REQUESTS):
        total_in = SYSTEM_PROMPT_TOKENS + AVG_USER_TOKENS
        cost_opus += calc_input_cost("claude-opus-4", total_in)
        cost_opus += calc_output_cost("claude-opus-4", AVG_OUTPUT_TOKENS)

    print(f"  Scenario: {SYSTEM_PROMPT_TOKENS}tk system + "
          f"{AVG_USER_TOKENS}tk user * {NUM_REQUESTS} 次")
    print(f"  Sonnet 无缓存:        ${cost_no:.4f}")
    print(f"  Sonnet Prompt Caching: ${cost_with:.4f}")
    print(f"  Opus 无缓存:          ${cost_opus:.4f}")
    savings = cost_no - cost_with
    print(f"  Sonnet 节省: ${savings:.4f} ({savings / cost_no * 100:.1f}%)")
    print(f"  Opus vs Cached Sonnet: Opus 贵 {cost_opus / cost_with:.1f}x")

    # 缓存命中率敏感性
    import random
    random.seed(42)
    print(f"\n  缓存命中率敏感性:")
    for hit_rate in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:
        cost = 0.0
        for i in range(NUM_REQUESTS):
            if i == 0:
                cost += calc_input_cost("claude-sonnet-4-6",
                                       SYSTEM_PROMPT_TOKENS, is_write=True)
            elif random.random() < hit_rate:
                cost += calc_input_cost("claude-sonnet-4-6",
                                       SYSTEM_PROMPT_TOKENS, cached=True)
            else:
                cost += calc_input_cost("claude-sonnet-4-6",
                                       SYSTEM_PROMPT_TOKENS)
            cost += calc_input_cost("claude-sonnet-4-6", AVG_USER_TOKENS)
            cost += calc_output_cost("claude-sonnet-4-6", AVG_OUTPUT_TOKENS)
        pct = (cost_no - cost) / cost_no * 100
        print(f"    {hit_rate:>6.0%} → ${cost:.4f} (节省 {pct:.1f}%)")

    print(f"""
  关键洞察:
    1. System Prompt 越大, 缓存收益越高
    2. 把静态内容 (system prompt, 工具定义) 放前面做缓存候选
    3. 把动态内容 (user message, 上下文) 放后面
    4. Prompt Caching 是将调用成本降低 10-25% 的 "免费午餐"
  """)

ex5_prompt_caching()


# ============================================================
# 练习 6 (思考): 成本 vs 质量权衡
# ============================================================

print("--- 练习 6: 成本 vs 质量权衡 (思考) ---")
print("""

  Haiku (评分 0.8, 便宜 20x) vs Opus (评分 0.95)

  场景 1: 代码格式化/变量命名建议 (内部工具)
    → 选 Haiku。用户可以立即验证结果, 省下的钱可服务更多开发者。

  场景 2: FAQ 自动回复 (客服系统)
    → 选 Haiku。答案相对固定, 答不好会升级到人工, 不会丢失用户。

  场景 3: 医疗咨询/用药建议
    → 必须 Opus。0.8 正确率 = 20% 错误率, 在医疗场景完全不可接受。

  决策框架:
    1. 错误的代价有多大?  高代价 → premium / 低代价 → cheap
    2. 用户能否验证结果?   可验证 → cheap / 不可验证 → premium
    3. 是否有时效性要求?   低延迟 → cheap / 准确性优先 → premium

  实用技巧: 先用 Opus 建立 baseline, 再尝试降级看质量下降可接受度。
""")


# ============================================================
# 课后反思
# ============================================================

print("--- 课后反思 ---")
print("""
  Q: 你的 SmartClient 缓存命中率是多少?
  A: 取决于查询分布。FAQ 类可达 80-90%, 开放式问答 20-40%。
     目标不是 100%, 而是找到缓存和成本的平衡点。

  Q: 如果预算减半, 先砍哪个模型?
  A: 砍 Opus (最贵)。策略: 收紧路由 → 降低语义阈值 → 缩短输出 →
     最后才考虑减少服务量。先砍最贵的, 优化策略优于减少服务。
""")
