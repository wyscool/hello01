# ============================================================
# Phase 5, Lesson 41: 评估框架与指标设计
# ============================================================
#
# 本课目标:
#   学会度量 AI 系统的质量 — 评估框架、指标设计、自动化评测。
#
#   Java 背景映射:
#     EvalSuite    ≈ JUnit TestSuite (但评估的是 AI 输出, 不是代码)
#     TestCase     ≈ @Test 方法 (定义输入、期望、实际、评分)
#     Metric       ≈ Assertion (断言 "答案应该正确")
#
#   核心概念:
#     1. 为什么评估是 AI 工程的基石
#     2. 传统指标: 准确率、精确率、召回率、F1
#     3. 文本生成指标: BLEU、ROUGE-L
#     4. 语义评估: 基于 Embedding 的相似度
#     5. 构建可复用的评估框架
#
# 预计阅读 + 实操时间: 50-60 分钟
#
# 前置: Phase 1-4 完成
# ============================================================

import math
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Any
from collections import Counter

# ============================================================
# 〇、为什么评估很重要
# ============================================================
#
# 你有没有想过这些问题:
#   "改了 Prompt, 效果到底变好还是变差了?"
#   "用 Claude Opus vs Sonnet, 对任务的影响有多大?"
#   "80% 的正确率够不够? 剩下 20% 的错是什么样的?"
#
# 没有评估 = 盲目开发。类比 Java:
#   - 写代码 → 写 JUnit 测试 → CI 自动跑
#   - 调 Prompt → 写评估用例 → 自动评分
#   本质上是一样的: 用可度量的方式验证质量。
#
# AI 系统的评估比传统软件测试更难:
#   传统: assert result.equals(expected)   ← 确定性的
#   AI:   "这个回答好吗?"                    ← 主观的、多维度的
#
# 所以需要: 指标 + 框架 + 自动化

print("=" * 60)
print("  Phase 5, Lesson 41: 评估框架与指标设计")
print("=" * 60)
print()


# ============================================================
# 一、传统分类指标
# ============================================================
# 这些指标来自信息检索和机器学习, 是评估的基石。

@dataclass
class ConfusionMatrix:
    """混淆矩阵 — 分类问题的基础统计。

    类比 Java:
      这是一个 POJO, 统计 TP/FP/FN/TN 四种计数。
    """
    tp: int = 0  # True Positive  — 预测对, 实际对
    fp: int = 0  # False Positive — 预测对, 实际错 (误报)
    fn: int = 0  # False Negative — 预测错, 实际对 (漏报)
    tn: int = 0  # True Negative  — 预测错, 实际错

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        """准确率 = (TP + TN) / Total
        所有预测中, 正确的比例。
        陷阱: 类别不均衡时, 高准确率可能没意义。
        例如: 99% 的邮件不是垃圾邮件, 全部预测 "不是" = 99% 准确率。
        """
        if self.total == 0:
            return 0.0
        return (self.tp + self.tn) / self.total

    @property
    def precision(self) -> float:
        """精确率 = TP / (TP + FP)
        预测为 "正确" 的结果中, 真正正确的比例。
        高精确率 = 不乱报。宁可漏过, 不可错杀。
        """
        predicted_positive = self.tp + self.fp
        return self.tp / predicted_positive if predicted_positive > 0 else 0.0

    @property
    def recall(self) -> float:
        """召回率 = TP / (TP + FN)
        所有真正正确的样本中, 被找出来的比例。
        高召回率 = 少遗漏。宁可错杀, 不可放过。
        """
        actual_positive = self.tp + self.fn
        return self.tp / actual_positive if actual_positive > 0 else 0.0

    @property
    def f1(self) -> float:
        """F1 = 2 × (Precision × Recall) / (Precision + Recall)
        精确率和召回率的调和平均。
        为什么用调和平均而不是算术平均?
          P=1.0, R=0.0 → 算术=0.5, 调和=0.0
          调和平均惩罚极端值, 要求两者都高。
        """
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def report(self) -> str:
        return (
            f"TP={self.tp} FP={self.fp} FN={self.fn} TN={self.tn} | "
            f"Acc={self.accuracy:.3f} P={self.precision:.3f} "
            f"R={self.recall:.3f} F1={self.f1:.3f}"
        )


# 演示
print("--- 分类指标演示 ---")
# 场景: 检索系统返回 10 条结果, 其中 6 条相关, 4 条不相关
#       实际有 8 条相关文档, 系统找回了 6 条
cm = ConfusionMatrix(tp=6, fp=4, fn=2, tn=0)
print(f"  检索结果: {cm.report()}")
print(f"  解读: 精确率={cm.precision:.1%} (返回的结果中 60% 相关)")
print(f"        召回率={cm.recall:.1%} (相关文档中 75% 被找到)")
print()


# ============================================================
# 二、文本生成指标 — BLEU
# ============================================================
# BLEU (Bilingual Evaluation Understudy):
#   衡量生成文本和参考文本的 n-gram 重叠程度。
#   原始用途: 机器翻译评估。如今广泛用于 LLM 输出评估。
#
# 核心思想:
#   1. 统计生成文本中有多少 n-gram 出现在参考文本中
#   2. 对过短的生成施加长度惩罚 (避免 "只要输出高频词就得高分")
#
# 局限性 (重要!):
#   BLEU 只看表面匹配, 不懂语义。
#   "今天天气真好" vs "今天真是个美好的日子" → BLEU 可能很低
#   但语义几乎相同。这就是为什么还需要语义评估 (第三节)。

def _count_ngrams(text: str, n: int) -> Counter:
    """统计文本中所有 n-gram 的出现次数。"""
    # 简单分词: 按字符切分 (英文按空格, 中文按字)
    # 生产环境应用更专业的分词器
    tokens = text.split() if " " in text else list(text)
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)


def _clipped_count(candidate_ngrams: Counter, reference_ngrams: Counter) -> int:
    """截断计数: 每个 n-gram 的计数不超过参考中的出现次数。

    假设 candidate 里 "the" 出现了 5 次, reference 里只有 3 次
    → 截断到 3。防止模型反复输出同一个词刷分。
    """
    total = 0
    for ngram, count in candidate_ngrams.items():
        total += min(count, reference_ngrams.get(ngram, 0))
    return total


def bleu_score(candidate: str, references: list[str],
               max_n: int = 4, smooth: bool = True) -> dict:
    """计算 BLEU 分数。

    Args:
        candidate: 模型生成的文本
        references: 参考文本列表 (可以多个)
        max_n: 最大 n-gram 长度, 默认 4
        smooth: 是否使用平滑 (避免 0 分)

    Returns:
        dict: 包含 BLEU, precisions, brevity_penalty 等

    类比 Java:
      这是 BLEU 算法的纯 Python 实现。
      nltk 有现成的, 但自己实现一遍才能理解细节。
    """
    cand_tokens = candidate.split() if " " in candidate else list(candidate)

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = _count_ngrams(candidate, n)
        # 合并所有 reference 的 n-gram (取并集的最大计数)
        ref_ngrams: Counter = Counter()
        for ref in references:
            ref_counter = _count_ngrams(ref, n)
            for ng, count in ref_counter.items():
                if count > ref_ngrams.get(ng, 0):
                    ref_ngrams[ng] = count

        clipped = _clipped_count(cand_ngrams, ref_ngrams)
        total = sum(cand_ngrams.values())

        if total == 0:
            prec = 0.0
        else:
            prec = clipped / total

        # 平滑: 如果 n-gram 精度为 0, 设为一个小值
        if smooth and prec == 0.0:
            prec = 1.0 / (2 ** n * 100)  # 随 n 增大而减小
        precisions.append(prec)

    # 几何平均
    if all(p == 0.0 for p in precisions):
        geo_mean = 0.0
    else:
        log_sum = sum(math.log(p) for p in precisions if p > 0)
        geo_mean = math.exp(log_sum / len(precisions))

    # 长度惩罚 (Brevity Penalty)
    cand_len = len(cand_tokens)
    ref_lens = [len(r.split() if " " in r else list(r)) for r in references]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - cand_len))
    if cand_len >= closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / max(cand_len, 1))

    bleu = bp * geo_mean

    return {
        "bleu": round(bleu, 4),
        "precisions": [round(p, 4) for p in precisions],
        "bp": round(bp, 4),
        "cand_len": cand_len,
        "ref_len": closest_ref_len,
    }


# 演示
print("--- BLEU 演示 ---")
cand = "今天天气真好 阳光明媚"
refs = ["今天天气非常好 阳光明媚"]
result = bleu_score(cand, refs)
print(f"  Candidate: {cand}")
print(f"  Reference: {refs[0]}")
print(f"  BLEU={result['bleu']}  P1={result['precisions'][0]}  "
      f"BP={result['bp']}")

# 对比: 完全匹配
cand2 = "今天天气非常好 阳光明媚"
result2 = bleu_score(cand2, refs)
print(f"  完全匹配: BLEU={result2['bleu']}")
print()


# ============================================================
# 三、ROUGE-L — 最长公共子序列
# ============================================================
# ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
#   侧重召回率, 关注参考文本中的内容有多少被生成文本覆盖。
#   ROUGE-L 基于最长公共子序列 (Longest Common Subsequence)。
#
# BLEU vs ROUGE:
#   BLEU  侧重精确率 (生成了多少正确的 n-gram)
#   ROUGE 侧重召回率 (参考中有多少被覆盖)
#   两者互补, 通常一起使用。

def _lcs_len(a: list, b: list) -> int:
    """最长公共子序列长度 (DP 解法)。

    示例: a="abcd", b="acbd" → LCS="abd", 长度=3
    类比 Java: 和 LeetCode 1143 一样。
    """
    m, n = len(a), len(b)
    # 用两行优化空间, 只需要上一行
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


def rouge_l(candidate: str, reference: str) -> dict:
    """计算 ROUGE-L 分数。"""
    cand_tokens = candidate.split() if " " in candidate else list(candidate)
    ref_tokens = reference.split() if " " in reference else list(reference)
    lcs = _lcs_len(cand_tokens, ref_tokens)

    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    precision = lcs / len(cand_tokens) if cand_tokens else 0.0
    f1 = (2 * recall * precision / (recall + precision)
          if (recall + precision) > 0 else 0.0)

    return {"rouge_l": round(f1, 4), "recall": round(recall, 4),
            "precision": round(precision, 4), "lcs_len": lcs}


# 演示
print("--- ROUGE-L 演示 ---")
r = rouge_l("今天天气真好 阳光明媚", "今天天气非常好 阳光明媚")
print(f"  ROUGE-L={r['rouge_l']}  R={r['recall']}  P={r['precision']}  LCS={r['lcs_len']}")
print()


# ============================================================
# 四、语义评估 — 基于 Embedding 的相似度
# ============================================================
# BLEU/ROUGE 的问题: 只看字面, 不懂意思。
# "你好吗" vs "你最近怎么样" → BLEU=0 但语义高度相似。
#
# 语义评估: 用 Embedding 模型把文本变成向量, 计算余弦相似度。
# 这在 Phase 3 (Lesson 21) 中已经学过, 现在用于评估。

try:
    from sentence_transformers import SentenceTransformer, util

    _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    _st_available = True
except Exception:
    _st_model = None
    _st_available = False


def semantic_similarity(candidate: str, reference: str) -> float:
    """计算两段文本的语义相似度 (余弦相似度, 0~1)。"""
    if not _st_available:
        return -1.0  # 不可用时返回 -1
    emb = _st_model.encode([candidate, reference], convert_to_tensor=True)
    sim = util.cos_sim(emb[0], emb[1]).item()
    return round(max(0.0, sim), 4)


# 演示
print("--- 语义相似度演示 ---")
if _st_available:
    sim1 = semantic_similarity("你好吗", "你最近怎么样")
    sim2 = semantic_similarity("你好吗", "今天天气不错")
    sim3 = semantic_similarity("今天天气真好 阳光明媚",
                               "今天天气特别好 阳光灿烂")
    print(f"  '你好吗' vs '你最近怎么样'  → 语义相似度 = {sim1}")
    print(f"  '你好吗' vs '今天天气不错'    → 语义相似度 = {sim2}")
    print(f"  近义句对                          → 语义相似度 = {sim3}")
    print(f"  解读: 语义相似 >0.7 通常认为意思相近。")
else:
    print("  (sentence-transformers 不可用, 跳过语义评估)")
print()


# ============================================================
# 五、评估框架 — EvalSuite
# ============================================================
# 把上面的指标整合成一个可复用的框架。
#
# 设计原则:
#   1. TestCase 定义单个评估用例 (输入、期望、评分函数)
#   2. EvalSuite 管理一组 TestCase, 批量运行, 生成报告
#   3. 多种评分策略 (精确匹配、BLEU、语义相似度、自定义)
#
# 类比 Java:
#   EvalSuite ≈ JUnit TestSuite
#   TestCase  ≈ @Test
#   run()     ≈ mvn test

@dataclass
class TestCase:
    """单个评估用例。

    类比 Java: 相当于一个 @Test 方法。
    """
    id: str
    input: str  # 给模型/系统的输入
    expected: str  # 期望的正确输出
    actual: str = ""  # 实际输出 (由被测系统生成)
    metadata: dict = field(default_factory=dict)

    # 评分结果
    exact_match: bool = False
    bleu: float = 0.0
    rouge_l: float = 0.0
    semantic: float = -1.0
    custom_score: float | None = None
    passed: bool = False


class EvalSuite:
    """评估套件 — 批量运行和报告。

    用法:
        suite = EvalSuite("Prompt v2 评估")
        suite.add(TestCase("01", ...))
        suite.run(model_fn)   # model_fn 是被测函数
        suite.report()
    """

    def __init__(self, name: str, pass_threshold: float = 0.6):
        self.name = name
        self.pass_threshold = pass_threshold  # 综合分数 ≥ 此值即通过
        self.cases: list[TestCase] = []
        self.created_at = time.time()

    def add(self, case: TestCase) -> "EvalSuite":
        self.cases.append(case)
        return self

    def add_batch(self, cases: list[TestCase]) -> "EvalSuite":
        self.cases.extend(cases)
        return self

    def run(self, system_fn: Callable[[str], str]) -> list[TestCase]:
        """用被测系统运行所有用例。

        system_fn: 输入字符串 → 输出字符串 (被测系统本身)
        """
        print(f"\n{'=' * 60}")
        print(f"  评估: {self.name}")
        print(f"  用例数: {len(self.cases)}")
        print(f"{'=' * 60}")

        for i, case in enumerate(self.cases):
            print(f"\n  [{i + 1}/{len(self.cases)}] {case.id}: {case.input[:50]}...")

            # 1. 调用被测系统
            try:
                case.actual = system_fn(case.input)
            except Exception as e:
                case.actual = f"[ERROR] {e}"

            # 2. 精确匹配
            case.exact_match = (case.actual.strip().lower()
                                == case.expected.strip().lower())

            # 3. BLEU
            bleu_result = bleu_score(case.actual, [case.expected])
            case.bleu = bleu_result["bleu"]

            # 4. ROUGE-L
            rouge_result = rouge_l(case.actual, case.expected)
            case.rouge_l = rouge_result["rouge_l"]

            # 5. 语义相似度
            case.semantic = semantic_similarity(case.actual, case.expected)

            # 6. 综合评分 (加权)
            composite = (
                0.15 * (1.0 if case.exact_match else 0.0)
                + 0.15 * case.bleu
                + 0.15 * case.rouge_l
                + 0.55 * max(0.0, case.semantic)  # 语义权重最高
            )
            case.custom_score = round(composite, 4)
            case.passed = composite >= self.pass_threshold

            icon = "✓" if case.passed else "✗"
            print(f"    {icon} exact={case.exact_match} bleu={case.bleu:.3f} "
                  f"rouge={case.rouge_l:.3f} sem={case.semantic:.3f} "
                  f"→ score={case.custom_score:.3f}")

        return self.cases

    def report(self) -> str:
        """生成评估报告。"""
        if not self.cases:
            return "无评估用例。"

        passed = sum(1 for c in self.cases if c.passed)
        total = len(self.cases)

        # 聚合指标
        avg_bleu = sum(c.bleu for c in self.cases) / total
        avg_rouge = sum(c.rouge_l for c in self.cases) / total
        avg_semantic = sum(max(0, c.semantic) for c in self.cases) / total
        avg_score = sum(c.custom_score or 0 for c in self.cases) / total
        exact_rate = sum(1 for c in self.cases if c.exact_match) / total

        lines = [
            f"\n{'=' * 60}",
            f"  评估报告: {self.name}",
            f"{'=' * 60}",
            f"  通过: {passed}/{total} ({passed / total * 100:.1f}%)",
            f"  精确匹配率: {exact_rate:.1%}",
            f"  平均 BLEU:    {avg_bleu:.4f}",
            f"  平均 ROUGE-L: {avg_rouge:.4f}",
            f"  平均语义相似度: {avg_semantic:.4f}",
            f"  平均综合分:   {avg_score:.4f}",
            f"  {'─' * 40}",
        ]

        # 列出失败的用例
        failed = [c for c in self.cases if not c.passed]
        if failed:
            lines.append(f"  失败用例 ({len(failed)}):")
            for c in failed:
                lines.append(f"    ✗ {c.id}: {c.input[:40]}...")
                lines.append(f"      期望: {c.expected[:60]}...")
                lines.append(f"      实际: {c.actual[:60]}...")
        else:
            lines.append(f"  全部通过!")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def compare(self, other: "EvalSuite") -> str:
        """对比两次评估 (例如 Prompt A vs Prompt B)。"""
        lines = [
            f"\n{'=' * 60}",
            f"  对比: {self.name} vs {other.name}",
            f"{'=' * 60}",
        ]
        s_pass = sum(1 for c in self.cases if c.passed)
        o_pass = sum(1 for c in other.cases if c.passed)
        s_score = sum(c.custom_score or 0 for c in self.cases) / max(len(self.cases), 1)
        o_score = sum(c.custom_score or 0 for c in other.cases) / max(len(other.cases), 1)

        lines.append(f"  {'指标':<20} {'A':>10} {'B':>10} {'变化':>10}")
        lines.append(f"  {'-' * 50}")
        lines.append(f"  {'通过率':<20} {s_pass / max(len(self.cases), 1):>9.1%} "
                     f"{o_pass / max(len(other.cases), 1):>9.1%} "
                     f"{o_pass / max(len(other.cases), 1) - s_pass / max(len(self.cases), 1):>+9.1%}")
        lines.append(f"  {'综合分':<20} {s_score:>10.4f} {o_score:>10.4f} "
                     f"{o_score - s_score:>+10.4f}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


# ============================================================
# 六、实战演示
# ============================================================

def demo():
    """构建一个模拟评估场景。"""
    print("--- 实战: 评估一个 '翻译助手' 系统 ---")
    print()

    # 模拟被测系统 — 实际项目中这是你的 LLM 调用函数
    def translate_system_v1(text: str) -> str:
        """版本 1: 简单的字典翻译 (模拟)。"""
        mapping = {
            "你好": "Hello",
            "今天天气真好": "The weather is nice today",
            "谢谢你的帮助": "Thank you for your help",
            "我喜欢写代码": "I like writing code",
            "这本书很有趣": "This book is interesting",
        }
        return mapping.get(text, f"[翻译] {text}")

    def translate_system_v2(text: str) -> str:
        """版本 2: 改进的翻译 (模拟更好的 Prompt/模型)。"""
        mapping = {
            "你好": "Hello there",
            "今天天气真好": "What a beautiful day today",
            "谢谢你的帮助": "Thank you for your kind help",
            "我喜欢写代码": "I enjoy writing code",
            "这本书很有趣": "This book is very interesting",
        }
        return mapping.get(text, f"[翻译] {text}")

    # 构建评估数据集
    test_cases = [
        TestCase("t1", "你好", "Hello"),
        TestCase("t2", "今天天气真好", "The weather is great today"),
        TestCase("t3", "谢谢你的帮助", "Thank you for your help"),
        TestCase("t4", "我喜欢写代码", "I like coding"),
        TestCase("t5", "这本书很有趣", "This book is very interesting"),
    ]

    # 评估版本 1
    suite_v1 = EvalSuite("翻译系统 v1 (基础版)")
    suite_v1.add_batch(test_cases)
    suite_v1.run(translate_system_v1)
    print(suite_v1.report())

    # 重置 (复用相同的用例定义)
    cases_v2 = [
        TestCase("t1", "你好", "Hello"),
        TestCase("t2", "今天天气真好", "The weather is great today"),
        TestCase("t3", "谢谢你的帮助", "Thank you for your help"),
        TestCase("t4", "我喜欢写代码", "I like coding"),
        TestCase("t5", "这本书很有趣", "This book is very interesting"),
    ]

    suite_v2 = EvalSuite("翻译系统 v2 (改进版)")
    suite_v2.add_batch(cases_v2)
    suite_v2.run(translate_system_v2)
    print(suite_v2.report())

    # 对比
    print(suite_v1.compare(suite_v2))

    # 关键洞察
    print("""
  关键洞察:
    ┌─────────────────────────────────────────────┐
    │ 评估不是一次性的, 而是持续的:                 │
    │                                              │
    │ 每次改 Prompt → 跑评估 → 看指标变化          │
    │ 每次换模型   → 跑评估 → 对比分数              │
    │ 每次加数据   → 跑评估 → 确认没有退化          │
    │                                              │
    │ 这就是 "AI 工程化" 的核心:                    │
    │ 用工程方法管理 AI 的不确定性。                │
    └─────────────────────────────────────────────┘
  """)


# ============================================================
# 七、入口
# ============================================================

if __name__ == "__main__":
    demo()

    print("=" * 60)
    print("  Lesson 41 完成!")
    print("=" * 60)
    print(f"""
  你学到了:
    1. 混淆矩阵 — TP/FP/FN/TN → 精确率/召回率/F1
    2. BLEU     — n-gram 精度 + 长度惩罚
    3. ROUGE-L  — 最长公共子序列, 侧重召回率
    4. 语义评估  — Embedding 向量 + 余弦相似度
    5. EvalSuite — 可复用的评估框架

  这些是 Phase 5 的基础能力。
  下一课: Lesson 42 — 可观测性: 日志、追踪、监控。
    如何知道你的 AI 系统在生产环境中发生了什么?
  """)


# ============================================================
# 试试看 (Try This) — 解答
# ============================================================

print("\n" + "=" * 60)
print("  试试看 (Try This) — Lesson 41 练习")
print("=" * 60)
print()


# ============================================================
# 练习 1: 为 DevAssistant 设计评估用例
# ============================================================
# 从 Lesson 35 的 DevAssistant 中选 3 个典型问题,
# 定义 input / expected, 用 EvalSuite 评分。
#
# 注意: 真正的 DevAssistant 需要 LLM API, 这里用模拟函数演示。
#   你可以将下面的 system_fn 替换为真实的 DevAssistant.ask() 调用。

def ex1_devassistant_eval():
    """为 DevAssistant 设计 3 个评估用例, 用 EvalSuite 评分。"""

    # 模拟 DevAssistant 行为 (替换为真实调用即可)
    def mock_devassistant(task: str) -> str:
        """模拟 DevAssistant — 基于简单规则的响应。"""
        if "计算" in task or "sqrt" in task or "数学" in task:
            return "计算结果: sqrt(256) = 16, 加上 100 等于 116"
        if "翻译" in task or "translate" in task:
            if "你好" in task:
                return "Hello"
            if "天气" in task:
                return "The weather is nice today"
            return "Translation result"
        if "解释" in task or "RAG" in task or "什么是" in task:
            return "RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术, "
            "它先从知识库中检索相关文档, 再让 LLM 基于文档生成回答。"
        if "代码" in task or "排序" in task or "写" in task:
            return "以下是快速排序的 Python 实现: def quicksort(arr): ..."
        return f"关于 '{task[:30]}...' 的回答"

    # 定义 3 个典型评估用例
    #   input:     用户可能问的问题
    #   expected:  期望回答中至少包含的关键信息
    test_cases = [
        TestCase(
            id="dev_calc",
            input="帮我计算 sqrt(256) + 100 等于多少",
            expected="计算结果: sqrt(256) = 16",
        ),
        TestCase(
            id="dev_translate",
            input="翻译成英文: 你好",
            expected="Hello",
        ),
        TestCase(
            id="dev_explain",
            input="解释什么是 RAG",
            expected="RAG 是一种结合检索和生成的技术",
        ),
    ]

    suite = EvalSuite("DevAssistant 基础功能评估")
    suite.add_batch(test_cases)
    suite.run(mock_devassistant)
    print(suite.report())

ex1_devassistant_eval()


# ============================================================
# 练习 2: 设计 "分类评估"
# ============================================================
# 创建意图分类器, 设计 10 个测试用例, 用 ConfusionMatrix 评估。

def ex2_classification_eval():
    """意图分类评估 — 用 ConfusionMatrix 评估分类器。"""

    # 意图分类器: 判断用户意图是 "代码" 还是 "闲聊"
    def classify(text: str) -> str:
        """简单规则分类器。"""
        code_keywords = ["bug", "代码", "排序", "函数", "算法", "编程",
                        "python", "java", "debug", "错误", "修复"]
        for kw in code_keywords:
            if kw in text.lower():
                return "代码"
        return "闲聊"

    # 设计 10 个测试用例: (输入文本, 真实标签)
    test_data = [
        ("帮我修一个 bug", "代码"),
        ("今天天气真好", "闲聊"),
        ("快速排序怎么写", "代码"),
        ("你喜欢吃什么", "闲聊"),
        ("Python 的函数装饰器怎么用", "代码"),
        ("周末去哪玩了", "闲聊"),
        ("这段代码有错误", "代码"),
        ("推荐一本好书", "闲聊"),
        ("Java 的多线程编程", "代码"),
        ("你好, 最近怎么样", "闲聊"),
    ]

    # 构建混淆矩阵
    cm = ConfusionMatrix()
    for text, true_label in test_data:
        predicted = classify(text)
        if true_label == "代码" and predicted == "代码":
            cm.tp += 1  # 正确识别为"代码"
        elif true_label == "闲聊" and predicted == "代码":
            cm.fp += 1  # 误判为"代码"
        elif true_label == "代码" and predicted == "闲聊":
            cm.fn += 1  # 漏掉了"代码"
        elif true_label == "闲聊" and predicted == "闲聊":
            cm.tn += 1  # 正确识别为"闲聊"

    print("--- 练习 2: 意图分类评估 ---")
    print(f"  分类器: 基于关键词的简单规则")
    print(f"  测试用例数: {len(test_data)}")
    print(f"  混淆矩阵: {cm.report()}")

    # 逐条分析错误
    print(f"\n  误判分析:")
    for text, true_label in test_data:
        predicted = classify(text)
        if predicted != true_label:
            print(f"    ✗ '{text}' → 预测={predicted}, 实际={true_label}")

    print(f"""
  思考: 如何提高召回率?
    当前分类器使用关键词精确匹配, 容易遗漏同义词。
    提高召回率的方法:
    1. 扩展关键词列表 (如加入 "开发", "写一个", "实现")
       → 代价: 可能增加误报 (FP), 降低精确率
    2. 使用语义分类: Embedding + 相似度阈值
       → "编程" 和 "写代码" 语义相近, 都能识别
    3. 使用轻量 LLM (如 Haiku) 做分类
       → 最准确但成本最高
    权衡: 精确率 vs 召回率 永远在博弈, 取决于业务场景:
      - 代码审查场景: 宁可多召回 (高 R), 人工再过滤
      - 客服分流场景: 宁可少误判 (高 P), 避免闲聊进技术通道
  """)

ex2_classification_eval()


# ============================================================
# 练习 3: 扩展 BLEU — 中文分词支持
# ============================================================
# 改进 BLEU: 用字符级 n-gram (字粒度) 处理中文,
# 对比改进前后的差异。

def ex3_bleu_chinese():
    """扩展 BLEU 支持中文 — 字粒度 n-gram。"""

    # 原始 BLEU 的分词逻辑:
    #   text.split() if " " in text else list(text)
    # 对中文来说, list(text) 是按字拆分, 这是合理的字粒度方案。

    # 改进: 支持中英混合, 中文按字, 英文按空格
    def smart_tokenize(text: str) -> list[str]:
        """智能分词: 中文按字切分, 英文按空格切分。"""
        import re
        tokens = []
        # 分离中文字符和英文单词
        for segment in re.split(r'(\s+|[a-zA-Z]+)', text):
            segment = segment.strip()
            if not segment:
                continue
            if re.match(r'^[a-zA-Z]+$', segment):
                tokens.append(segment)  # 英文单词作为一个 token
            else:
                tokens.extend(list(segment))  # 中文按字拆分
        return tokens

    def bleu_score_v2(candidate: str, references: list[str],
                      max_n: int = 4, smooth: bool = True) -> dict:
        """改进版 BLEU — 使用 smart_tokenize。"""
        # 复用原有的 _count_ngrams, _clipped_count, bleu_score 逻辑
        # 但使用 smart_tokenize 替换原有分词
        cand_tokens = smart_tokenize(candidate)
        precisions = []
        for n in range(1, max_n + 1):
            # 统计 candidate 的 n-gram
            cand_ngrams = {}
            for i in range(len(cand_tokens) - n + 1):
                ng = tuple(cand_tokens[i:i + n])
                cand_ngrams[ng] = cand_ngrams.get(ng, 0) + 1

            # 统计 reference 的 n-gram (取最大计数)
            ref_ngrams = {}
            for ref in references:
                ref_tokens = smart_tokenize(ref)
                for i in range(len(ref_tokens) - n + 1):
                    ng = tuple(ref_tokens[i:i + n])
                    ref_ngrams[ng] = max(ref_ngrams.get(ng, 0),
                                        sum(1 for j in range(
                                            len(ref_tokens) - n + 1
                                        ) if tuple(ref_tokens[j:j + n]) == ng))

            clipped = sum(min(cand_ngrams.get(ng, 0), ref_ngrams.get(ng, 0))
                         for ng in cand_ngrams)
            total = sum(cand_ngrams.values())
            prec = clipped / total if total > 0 else 0.0
            if smooth and prec == 0.0:
                prec = 1.0 / (2 ** n * 100)
            precisions.append(prec)

        if all(p == 0.0 for p in precisions):
            geo_mean = 0.0
        else:
            log_sum = sum(math.log(p) for p in precisions if p > 0)
            geo_mean = math.exp(log_sum / len(precisions))

        cand_len = len(cand_tokens)
        ref_lens = [len(smart_tokenize(r)) for r in references]
        closest_ref_len = min(ref_lens, key=lambda x: abs(x - cand_len))
        bp = (1.0 if cand_len >= closest_ref_len
              else math.exp(1 - closest_ref_len / max(cand_len, 1)))

        return {
            "bleu": round(bp * geo_mean, 4),
            "precisions": [round(p, 4) for p in precisions],
            "tokens_cand": cand_tokens,
        }

    # 对比测试
    print("--- 练习 3: BLEU 中文扩展 ---")

    test_pairs = [
        # (candidate, reference)
        ("今天天气真好 阳光明媚", ["今天天气非常好 阳光灿烂"]),
        ("Hello world 欢迎来到 Python", ["Hello world 欢迎学习 Python"]),
        ("这段代码有 bug 需要修复", ["这段程序有错误需要修改"]),
        ("RAG 结合检索和生成", ["RAG 融合了搜索和创建"]),
    ]

    print(f"  {'Candidate':<30s} {'原 BLEU':>8s} {'改进 BLEU':>8s} {'提升':>8s}")
    print(f"  {'─' * 56}")
    for cand, refs in test_pairs:
        r1 = bleu_score(cand, refs)
        r2 = bleu_score_v2(cand, refs)
        diff = r2["bleu"] - r1["bleu"]
        print(f"  {cand[:28]:<30s} {r1['bleu']:>8.4f} {r2['bleu']:>8.4f} {diff:>+8.4f}")
        print(f"    改进版 tokens: {r2['tokens_cand']}")

    print(f"""
  观察:
    原版 BLEU 对中文按字切分 (list), 改进版按字切分 + 保留英文单词。
    对于纯中文, 改进版结果和原版近似 (因为都是字粒度)。
    对于中英混合, 改进版更合理: 英文单词不拆散, n-gram 匹配更有意义。
    本质上, BLEU 对中文始终有局限: 中文的语义单位是 "词" 而非 "字"。
    生产环境建议用专业分词器 (jieba/THULAC), 或直接使用语义评估。
  """)

ex3_bleu_chinese()


# ============================================================
# 练习 4: 人工评估模式 (Human-in-the-Loop)
# ============================================================
# 修改 EvalSuite.run(), 添加 interactive=True 参数,
# 让用户打分。这就是 RLHF 中 Human Feedback 的基础!

class HumanEvalSuite(EvalSuite):
    """支持人工评估的 EvalSuite 扩展。

    usage:
        suite = HumanEvalSuite("人工评估")
        suite.add(TestCase(...))
        suite.run(system_fn, interactive=True)  # 交互模式
    """

    def run(self, system_fn: Callable[[str], str],
            interactive: bool = False) -> list[TestCase]:
        """扩展 run(), 添加交互式评分模式。"""
        print(f"\n{'=' * 60}")
        print(f"  评估: {self.name} {'(人工交互模式)' if interactive else '(自动模式)'}")
        print(f"  用例数: {len(self.cases)}")
        print(f"{'=' * 60}")

        for i, case in enumerate(self.cases):
            print(f"\n  [{i + 1}/{len(self.cases)}] {case.id}: {case.input[:60]}...")

            # 1. 调用被测系统
            try:
                case.actual = system_fn(case.input)
            except Exception as e:
                case.actual = f"[ERROR] {e}"

            # 2. 自动指标 (始终计算)
            case.exact_match = (case.actual.strip().lower()
                               == case.expected.strip().lower())
            bleu_result = bleu_score(case.actual, [case.expected])
            case.bleu = bleu_result["bleu"]
            rouge_result = rouge_l(case.actual, case.expected)
            case.rouge_l = rouge_result["rouge_l"]
            case.semantic = semantic_similarity(case.actual, case.expected)

            if interactive:
                # 交互模式: 展示所有信息, 让用户打分
                print(f"    ┌─ 输入 (Input) ─────────────────────")
                print(f"    │ {case.input[:80]}")
                print(f"    ├─ 期望 (Expected) ──────────────────")
                print(f"    │ {case.expected[:80]}")
                print(f"    ├─ 实际 (Actual) ────────────────────")
                print(f"    │ {case.actual[:80]}")
                print(f"    ├─ 自动指标 ─────────────────────────")
                print(f"    │ BLEU={case.bleu:.3f}  "
                      f"ROUGE-L={case.rouge_l:.3f}  "
                      f"语义={case.semantic:.3f}")
                print(f"    └────────────────────────────────────")

                try:
                    score = float(input(f"    请打分 (1-5, Enter 跳过): ") or "0")
                    if 1 <= score <= 5:
                        case.custom_score = score
                        case.passed = score >= self.pass_threshold * 5
                    else:
                        print(f"    无效分数, 使用自动评分")
                        case.custom_score = None
                except (ValueError, EOFError):
                    print(f"    使用自动评分 (非交互环境)")
                    case.custom_score = None

            # 综合评分
            if case.custom_score is None:
                composite = (
                    0.15 * (1.0 if case.exact_match else 0.0)
                    + 0.15 * case.bleu
                    + 0.15 * case.rouge_l
                    + 0.55 * max(0.0, case.semantic)
                )
                case.custom_score = round(composite, 4)
                case.passed = composite >= self.pass_threshold
            else:
                # 人工评分已设置 passed
                pass

            icon = "✓" if case.passed else "✗"
            print(f"    {icon} score={case.custom_score}")

        return self.cases


def ex4_human_eval():
    """演示人工评估模式 (非交互式自动回退)。"""
    print("--- 练习 4: 人工评估模式 ---")

    def mock_system(text: str) -> str:
        mapping = {
            "你好": "Hello",
            "今天天气真好": "The weather is nice today",
            "谢谢": "Thank you",
        }
        return mapping.get(text, f"Response to: {text}")

    suite = HumanEvalSuite("人工评估测试", pass_threshold=0.6)
    suite.add(TestCase("h1", "你好", "Hello there"))
    suite.add(TestCase("h2", "今天天气真好", "The weather is great"))
    suite.add(TestCase("h3", "谢谢", "Thank you"))

    # interactive=True 但 stdin 非 tty → 自动回退到自动评分
    print("  (非交互环境, 自动回退到自动评分)")
    suite.run(mock_system, interactive=False)
    print(suite.report())

    print(f"""
  关键理解:
    人工评估 (Human-in-the-Loop) 是 RLHF 的基础:
    1. 模型生成回答 → 人工评分 (1-5)
    2. 收集 (prompt, response, score) 三元组
    3. 用这些数据训练 Reward Model
    4. Reward Model 指导 LLM 微调 (PPO/DPO)

    EvalSuite.interactive=True 就是这个流程的第一步。
    实际项目中, 人工评分通常通过标注平台 (LabelStudio/Argilla) 完成,
    而不是命令行交互, 但原理相同。
  """)

ex4_human_eval()


# ============================================================
# 练习 5 (挑战): 基于 LLM 的评估 (LLM-as-Judge)
# ============================================================
# 用 LLM 作为裁判, 对比 LLM 评分和自动指标的相关性。
# 因为本课不依赖外部 API, 这里用启发式模拟 LLM 评分 + 真实指标对比。

def ex5_llm_judge():
    """LLM-as-Judge: 模拟 LLM 评分, 对比与自动指标的相关性。

    在真实项目中:
      prompt = f"请评估以下回答的质量 (1-5分)\\n问题: {q}\\n回答: {a}"
      score = llm_call(prompt)  # 调用 Claude/GPT API
    """

    # 模拟 LLM-as-Judge 的评分逻辑:
    #   真实 LLM 会综合判断: 准确性、完整性、流畅性、有用性
    #   这里用启发式模拟:
    def simulated_llm_judge(question: str, answer: str,
                            expected: str) -> float:
        """模拟 LLM 评分 (基于语义相似度 + 长度 + 关键词覆盖)。"""
        # 语义相似度 (权重 0.5)
        sem = semantic_similarity(answer, expected)
        if sem < 0:
            sem = 0.5  # sentence-transformers 不可用时的默认值

        # 长度惩罚 (权重 0.2) — 太短或太长都不好
        len_ratio = min(len(answer), len(expected)) / max(len(answer), len(expected), 1)
        len_score = 1.0 - abs(len_ratio - 0.85) * 2  # 最佳长度比 ≈0.85
        len_score = max(0, min(1, len_score))

        # 关键词覆盖 (权重 0.3)
        import re
        # 提取 expected 中的关键词 (长度 >= 2 的汉字词)
        exp_words = set(re.findall(r'[一-鿿]{2,}', expected))
        ans_words = set(re.findall(r'[一-鿿]{2,}', answer))
        kw_score = (len(exp_words & ans_words) / max(len(exp_words), 1)
                    if exp_words else 0.5)

        raw = 0.5 * sem + 0.2 * len_score + 0.3 * kw_score
        # 映射到 1-5 分
        return round(1 + 4 * raw, 1)

    # 构建测试集: 不同质量等级的回答
    eval_pairs = [
        # (问题, 期望回答, 实际回答)
        ("什么是 RAG",
         "RAG 是检索增强生成, 结合信息检索和文本生成的技术",
         "RAG 是检索增强生成, 结合检索和生成的 AI 技术"),  # 好
        ("什么是 RAG",
         "RAG 是检索增强生成, 结合信息检索和文本生成的技术",
         "RAG 是一种方法"),  # 太简略
        ("什么是 RAG",
         "RAG 是检索增强生成, 结合信息检索和文本生成的技术",
         "量子计算是利用量子力学原理进行计算的技术, 与经典计算不同"),  # 跑题
        ("翻译: 你好",
         "Hello",
         "Hello there"),  # 稍多
        ("翻译: 你好",
         "Hello",
         "Hi"),  # 近似
    ]

    print("--- 练习 5: LLM-as-Judge 评估 ---")
    print(f"  {'问题':<18s} {'BLEU':>7s} {'ROUGE':>7s} {'语义':>7s} {'LLM评分':>7s} {'评估'}")
    print(f"  {'─' * 58}")

    for question, expected, actual in eval_pairs:
        b = bleu_score(actual, [expected])["bleu"]
        r = rouge_l(actual, expected)["rouge_l"]
        s = semantic_similarity(actual, expected)
        llm_score = simulated_llm_judge(question, actual, expected)

        # 判断质量
        if llm_score >= 3.5:
            quality = "优秀"
        elif llm_score >= 2.5:
            quality = "合格"
        else:
            quality = "不合格"

        print(f"  {question[:16]:<18s} {b:>7.3f} {r:>7.3f} {s:>7.3f} "
              f"{llm_score:>7.1f} {quality}")

    print(f"""
  关键发现:
    1. LLM-as-Judge 可以捕捉到 BLEU/ROUGE 无法感知的维度:
       - 跑题 vs 简略: BLEU 可能都很低, 但 LLM 能区分
       - 准确性 vs 流畅性: 自动指标只能看表面匹配
    2. LLM 评分和语义相似度有较高相关性 (0.6-0.8),
       但 LLM 能给出更 "人性化" 的判断。
    3. LLM-as-Judge 的成本:
       - 每次评估也是一次 LLM 调用 (花费 Token)
       - 评估 1000 条数据 ≈ 评估本身的成本可能超过被评估的系统
    4. 最佳实践:
       - 开发和实验阶段: 自动指标 (免费, 快速)
       - 里程碑/上线前: LLM-as-Judge (细致, 但贵)
       - 持续监控: 自动指标 + 异常时触发 LLM 复评
  """)

ex5_llm_judge()


# ============================================================
# 练习 6 (思考): 评估的 "天花板"
# ============================================================

print("--- 练习 6: 评估的天花板 (思考) ---")
print("""
  如果所有指标都在 0.9 以上, 这个系统是不是就没问题了?

  答案: 不一定。以下是几个常见的陷阱:

  1. Goodhart's Law (古德哈特定律):
     "当一个指标成为目标, 它就不再是一个好的指标。"
     如果你只优化 BLEU, 模型会学会生成高频 n-gram 凑分数,
     而不是真正好的回答。这就是 "指标作弊"。

  2. 评估数据泄露 (Data Leakage):
     如果评估数据集和训练数据有重叠, 高分只是 "背诵" 而非 "理解"。
     模型记住了答案, 但换个问法就答不上来。

  3. 评估数据的代表性:
     你的 100 条评估用例能代表所有用户的 100 万种提问方式吗?
     边缘 case (罕见问题、长尾场景、对抗输入) 往往没被覆盖。

  4. "Vibes-based evaluation" (基于直觉的评估):
     这是 2024-2025 年 AI 圈流行的做法:
     - 不只看自动指标, 更多依赖 "用起来感觉怎么样?"
     - 开发者自己频繁使用自己的产品, 凭经验判断质量
     - 优点: 能发现指标发现不了的问题 (语气、偏见、幻觉)
     - 缺点: 不可重复、不可扩展、主观性强
     - 为什么流行? 因为当前自动指标还不能完全替代人的判断,
       就像代码审查不只看测试覆盖率, 还要看代码质量。

  5. 多维度评估的必要性:
     - 准确性:   答案对不对
     - 完整性:   有没有遗漏关键信息
     - 安全性:   有没有有害内容
     - 流畅性:   读起来通不通顺
     - 有用性:   是否真正解决了用户的问题
     - 延迟:     响应快不快
     没有一个指标能覆盖所有维度。

  结论: 评估是一个持续的过程, 不是一次性的任务。
        高分是必要条件, 但不是充分条件。
        就像软件测试: 100% 覆盖率不代表没有 bug。
""")


# ============================================================
# 课后反思
# ============================================================

print("--- 课后反思 ---")
print("""
  Q: BLEU 和语义相似度, 哪个更接近你的直觉判断?
  A: 语义相似度更接近直觉。因为:
     - "今天天气真好" vs "今天真是个美好的日子" — BLEU 很低但语义高度相似
     - 人对 "意思相近" 的判断基于语义, 不是基于字面
     - 但语义相似度也有盲区: "我不喜欢" vs "我喜欢" 语义相似度可能不低,
       但表达了相反的观点。所以需要多指标互补。

  Q: 如果你只能选一个指标监控生产系统, 你选哪个?
  A: 语义相似度 (semantic similarity)。
     理由:
     - BLEU/ROUGE 只看表面, 对中文尤其不准
     - 语义相似度能捕捉到 "意思对不对" 这个核心需求
     - 配合一个 "基准答案库" (Golden Set), 持续监控回答质量是否退化
     - 但是, 还需要一个补充机制: 异常检测 (当语义相似度突然下降时告警)
""")
