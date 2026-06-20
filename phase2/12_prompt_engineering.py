# ============================================================
# Phase 2, Lesson 12: Prompt 工程 —— 设计模式、角色、少样本
# ============================================================
#
# 本课目标:
#   1. 理解 Prompt 工程是什么 —— 类比 SQL 调优
#   2. 角色设定 (Role Prompting)
#   3. 少样本学习 (Few-shot Prompting)
#   4. 思维链 (Chain of Thought)
#   5. 结构化 Prompt (XML / Markdown 分隔)
#   6. Prompt 模板化 —— 变量替换
#   7. 迭代优化 —— 从烂 Prompt 到好 Prompt
#   8. 常见任务模式: 摘要、提取、分类、翻译、问答
#   9. Prompt 长度 vs 质量权衡
#   10. 实战: 构建一个 Prompt 调试工具
#
# 预计阅读 + 实操时间: 40-50 分钟
#
# 前置: 已完成 Lesson 11, API 可用
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from anthropic import Anthropic
from anthropic.types import Message


# ------------------------------------------------------------
# 〇、环境准备 —— 复用 Lesson 11 的客户端
# ------------------------------------------------------------

def _get_text(response: Message) -> str:
    parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts)


api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")

client_kwargs = {"api_key": api_key} if api_key else {}
if base_url:
    client_kwargs["base_url"] = base_url
client = Anthropic(**client_kwargs)


def ask(prompt: str, system: str | None = None, model: str = "claude-sonnet-4-6",
        max_tokens: int = 300, temperature: float = 0.0) -> str:
    """本课的通用问询函数。默认 temperature=0 以保持输出确定性。"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        return _get_text(response)
    except Exception as e:
        return f"[调用失败: {e}]"


# 连接测试
try:
    ask("ping")
    api_ok = True
    print("✅ API 连接正常\n")
except Exception:
    api_ok = False
    print("⚠️  API 不可用, 将以模拟模式运行 (仍可学习 Prompt 设计思路)\n")


# ============================================================
# 一、Prompt 工程是什么? —— 类比 SQL 调优
# ============================================================
# 同一个问题, 不同的问法, 结果天差地别。
# Prompt 工程 = 精心设计输入, 让 LLM 输出你想要的。
#
# 类比 SQL:
#   SELECT * FROM users WHERE name LIKE '%张%'
#   vs
#   SELECT id, name FROM users WHERE name LIKE '%张%' ORDER BY created_at LIMIT 10
#
#   两条 SQL 都查同张表, 但列子集、排序、分页让结果完全不同。
#   Prompt 也一样——同样的 LLM, 不同的 prompt → 不同的输出质量。
#
# Prompt 工程的核心问题:
#   1. 怎么说清楚你要什么?        → 角色设定、指令细化
#   2. 怎么给模型正确的示范?      → 少样本学习
#   3. 怎么让模型一步步推理?      → 思维链
#   4. 怎么组织信息让模型理解?    → 结构化 Prompt


# ------------------------------------------------------------
# 二、角色设定 (Role Prompting) —— 告诉 AI "你是谁"
# ------------------------------------------------------------
# 在 system prompt 中定义角色, 模型会"入戏"。
# 这是成本最低、效果最显著的 Prompt 技巧。
#
# 对比: 不设角色 vs 设角色

print("=" * 60)
print("角色设定对比")
print("=" * 60)

QUESTION = "解释一下 Python 的装饰器"

# 无角色
answer_raw = ask(QUESTION)
print(f"\n📝 无角色:\n{answer_raw[:120]}...")

# 有角色: Python 专家对 Java 开发者
SYS_EXPERT = """你是一位资深 Python 培训师, 专门培训 Java 开发者。
用 Java 的注解 (Annotation) 和 AOP 做类比来解释 Python 概念。
回答控制在 100 字以内。"""

answer_expert = ask(QUESTION, system=SYS_EXPERT)
print(f"\n🎓 Python 培训师:\n{answer_expert}")

# 有角色: 5 岁小孩的老师
SYS_KID = """用 5 岁小孩能听懂的语言解释技术概念。用比喻, 不用术语。"""

answer_kid = ask(QUESTION, system=SYS_KID)
print(f"\n🧒 给 5 岁小孩:\n{answer_kid}")


# ------------------------------------------------------------
# 三、角色设定的工程原则
# ------------------------------------------------------------
# 好的 system prompt 包含 3 个要素:
#   你是谁 (Role)        → "你是 Python 代码审查专家"
#   你要做什么 (Task)     → "审查代码, 找出安全漏洞和性能问题"
#   怎么做 (Constraint)  → "用中文回复, 按严重程度排序, 给出修复建议"
#
# 反例 vs 正例:

BAD_SYSTEM = "你是一个助手"
GOOD_SYSTEM = """你是一个 Python 代码审查专家。
任务: 审查用户提供的代码。
输出格式:
  1. 严重问题 (安全漏洞、逻辑错误)
  2. 一般问题 (性能、可读性)
每个问题包含: 代码位置 | 问题描述 | 修复建议"""

CODE_TO_REVIEW = '''
def get_user(uid):
    sql = f"SELECT * FROM users WHERE id = {uid}"
    return db.execute(sql)
'''

print("\n" + "=" * 60)
print("Prompt 质量对比: 模糊 vs 精准")
print("=" * 60)

if api_ok:
    print(f"\n❌ 模糊 system prompt:\n{ask(CODE_TO_REVIEW, system=BAD_SYSTEM)[:150]}...")
    print(f"\n✅ 精准 system prompt:\n{ask(CODE_TO_REVIEW, system=GOOD_SYSTEM)}")


# ------------------------------------------------------------
# 四、少样本学习 (Few-shot Prompting) —— 给模型看例子
# ------------------------------------------------------------
# 不用改模型参数, 直接在 prompt 里给几个输入→输出的例子,
# 模型会模仿例子的格式和风格回复。
#
# 类比 Java: 给新人看代码范例 → 新人按范例风格写代码。

print("\n" + "=" * 60)
print("少样本学习 (Few-shot)")
print("=" * 60)

# 任务: 情感分类 (正面 / 负面 / 中性)
# Zero-shot (不给例子):
ZERO_SHOT = """判断以下评论的情感倾向 (正面 / 负面 / 中性):
评论: "快递太慢了, 包装也破了"
情感:"""

answer_zero = ask(ZERO_SHOT)
print(f"\nZero-shot: {answer_zero}")

# Few-shot (给 3 个例子):
FEW_SHOT = """判断以下评论的情感倾向 (正面 / 负面 / 中性)。

示例:
评论: "质量很好, 下次还来买"
情感: 正面

评论: "一般般吧, 没什么惊喜"
情感: 中性

评论: "太差了, 用了两天就坏了"
情感: 负面

现在判断这条:
评论: "快递太慢了, 包装也破了"
情感:"""

answer_few = ask(FEW_SHOT)
print(f"Few-shot:  {answer_few}")

# 关键: 例子要覆盖所有类别, 格式要一致。
# 如果只有一个类别的例子, 模型会偏斜。


# ------------------------------------------------------------
# 五、思维链 (Chain of Thought) —— "让我们一步一步想"
# ------------------------------------------------------------
# 让模型展示推理过程, 能显著提升复杂任务的准确率。
# 最简单的做法: 加一句 "让我们一步一步分析"。
#
# 类比: 考试时要求"写出计算过程"——逼着学生不跳步。

print("\n" + "=" * 60)
print("思维链 (Chain of Thought)")
print("=" * 60)

MATH_PROBLEM = """一个商店有以下促销:
- 满 200 减 30
- 会员打 9 折
- 两种优惠可以叠加 (先满减再打折)

小明是会员, 买了以下商品:
- 运动鞋 159 元
- T恤 89 元
- 袜子 29 元

小明最终要付多少钱?"""

# 直接给答案
answer_direct = ask(MATH_PROBLEM)
print(f"\n直接回答:\n{answer_direct[:200]}...")

# 引导推理过程
COT_SYSTEM = "你是一个数学老师。回答时先写出每一步的计算过程, 最后给出答案。"

answer_cot = ask(MATH_PROBLEM, system=COT_SYSTEM, max_tokens=500)
print(f"\n思维链回答:\n{answer_cot}")


# ------------------------------------------------------------
# 六、Few-shot + CoT 组合 —— 最强大的单轮 Prompt 技巧
# ------------------------------------------------------------
# 例子里的回复也展示推理过程, 模型会模仿"先推理再回答"的模式。

FEW_SHOT_COT = """判断邮件是否紧急。先分析内容, 再给出结论。

示例 1:
邮件: "下周二的会议改到周三下午 3 点"
分析: 这是会议时间调整, 不是紧急事务, 可以稍后处理
结论: 不紧急

示例 2:
邮件: "服务器宕机了! 客户无法下单! 请立刻处理!!!"
分析: 服务器故障导致客户无法下单, 直接影响收入, 需要立即响应
结论: 紧急

现在判断:
邮件: "这个季度的报销截止日是本周五, 请尽快提交"
分析:"""

print("\n" + "=" * 60)
print("Few-shot + CoT 组合")
print("=" * 60)

answer_fs_cot = ask(FEW_SHOT_COT)
print(f"\n{answer_fs_cot}")


# ------------------------------------------------------------
# 七、结构化 Prompt —— 用分隔符组织信息
# ------------------------------------------------------------
# 当 prompt 包含多种信息 (指令、上下文、数据、格式要求),
# 用 XML 标签或 Markdown 分隔符清晰标记每部分。
#
# 类比: 代码中用注释分隔不同逻辑块。

print("\n" + "=" * 60)
print("结构化 Prompt")
print("=" * 60)

# 混乱版:
MESSY = """总结这篇文章。文章是关于 Python 3.12 的新特性的。
Python 3.12 于 2023 年 10 月发布，带来了多项重要更新。
最引人注目的是更详细的错误信息，现在解释器能更精确地指出错误位置。
类型提示语法也得到了改进，支持更简洁的泛型写法 (PEP 695)。
f-string 表达式现在更灵活了，支持嵌套引号。
性能方面，整体速度提升了大约 5%。
此外还引入了对 Linux perf 分析器的支持，方便性能调试。
请用 3 个要点总结，中文回复。"""

# 结构化版:
STRUCTURED = """<task>用 3 个要点总结以下文章, 中文回复</task>

<article>
Python 3.12 于 2023 年 10 月发布，带来了多项重要更新。
最引人注目的是更详细的错误信息，现在解释器能更精确地指出错误位置。
类型提示语法也得到了改进，支持更简洁的泛型写法 (PEP 695)。
f-string 表达式现在更灵活了，支持嵌套引号。
性能方面，整体速度提升了大约 5%。
此外还引入了对 Linux perf 分析器的支持，方便性能调试。
</article>

<format>
- 要点 1
- 要点 2
- 要点 3
</format>"""

if api_ok:
    print(f"\n混乱版:\n{ask(MESSY)}")
    print(f"\n───")
    print(f"\n结构化版:\n{ask(STRUCTURED)}")

print("""
结构化技巧:
  <task>...</task>    — 要做什么
  <context>...</context> — 背景信息
  <data>...</data>    — 输入数据
  <format>...</format> — 输出格式
  <example>...</example> — 示例

用 ``` 代码块也有效 (Markdown 风格), XML 更灵活。""")


# ------------------------------------------------------------
# 八、Prompt 模板化 —— 可复用的 Prompt 工厂
# ------------------------------------------------------------

print("=" * 60)
print("Prompt 模板")
print("=" * 60)


class PromptTemplate:
    """
    Prompt 模板: 把可变数据填入固定的 Prompt 框架。
    类比: Java 的 String.format() / MessageFormat。

    提示工程中, 模板 = 系统指令 + 占位符 + 示例。
    """

    def __init__(self, system_prompt: str, template: str):
        self.system = system_prompt
        self.template = template

    def format(self, **kwargs) -> tuple[str, str]:
        """填入变量, 返回 (system_prompt, user_prompt)。"""
        return self.system, self.template.format(**kwargs)

    def ask(self, **kwargs) -> str:
        system, user = self.format(**kwargs)
        return ask(user, system=system)


# 模板 1: 代码审查
code_review_tmpl = PromptTemplate(
    system_prompt="""你是一个资深代码审查员。审查以下代码, 输出:
1. 问题位置 (行号或函数名)
2. 问题描述
3. 严重程度 (🔴 严重 / 🟡 一般 / 🟢 建议)
4. 修复方案""",
    template="""审查以下 {language} 代码:
```{language}
{code}
```"""
)

# 模板 2: 文本摘要
summarize_tmpl = PromptTemplate(
    system_prompt="用 {style} 风格总结以下内容, {length} 字以内。",
    template="{text}"
)

print("\n📋 模板示例: 代码审查")
review_result = code_review_tmpl.ask(
    language="python",
    code='''
def calc(l):
    r = []
    for i in range(len(l)):
        r.append(l[i] * 2)
    return r
'''
)
print(review_result)

print("\n📋 模板示例: 文本摘要")
content = "Python 3.12 带来了更好的错误信息、更简洁的泛型语法、更快的性能。f-string 现在支持嵌套引号。Linux perf 支持让性能调试更简单。"
print(summarize_tmpl.ask(style="简洁", length="30", text=content))


# ------------------------------------------------------------
# 九、迭代优化 —— 从烂 Prompt 到好 Prompt
# ------------------------------------------------------------
# Prompt 工程不是一次性写对的。和写代码一样, 需要迭代。
#
# 优化循环:
#   1. 写一个初始 Prompt
#   2. 测试 → 观察输出哪不好
#   3. 修改 Prompt (加角色/约束/示例/思维链)
#   4. 再测试 → 直到满意

print("\n" + "=" * 60)
print("迭代优化演示")
print("=" * 60)

# V1: 太模糊
PROMPT_V1 = "给我写个 Python 排序函数"

# V2: 加了角色和约束
PROMPT_V2 = """你是一个 Python 算法专家。
写一个排序函数, 要求:
- 函数名 sort_by_field
- 对 list[dict] 按指定字段排序
- 支持升序/降序参数
- 包含 type hints 和 docstring"""

# V3: 加了 Few-shot
PROMPT_V3 = """你是一个 Python 算法专家。
写一个排序函数, 要求:
- 函数名 sort_by_field
- 对 list[dict] 按指定字段排序
- 支持升序/降序参数
- 包含 type hints 和 docstring

示例输入:
  data = [{"name": "Bob", "age": 30}, {"name": "Alice", "age": 25}]
  sort_by_field(data, "age", reverse=False)
  → [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]

请输出完整函数。"""

print(f"\nV1 (模糊):\n{ask(PROMPT_V1)}")
print(f"\n───")
print(f"\nV2 (+角色 +约束):\n{ask(PROMPT_V2)}")
print(f"\n───")
print(f"\nV3 (+Few-shot):\n{ask(PROMPT_V3)}")


# ------------------------------------------------------------
# 十、常见任务模式的 Prompt 范式
# ------------------------------------------------------------
# 工程中 80% 的 LLM 需求落入以下 5 类。

print("\n" + "=" * 60)
print("五种任务模式的 Prompt 范式")
print("=" * 60)

PATTERNS = """
┌──────────┬──────────────────────────────────────────────┐
│ 任务类型   │ Prompt 范式                                   │
├──────────┼──────────────────────────────────────────────┤
│ 分类      │ "将以下文本分类为: A / B / C。只输出类别名。"      │
│ 提取      │ "从以下文本中提取 {字段1}、{字段2}。JSON 格式。"   │
│ 摘要      │ "用 N 句话总结以下内容。保留关键数字和结论。"       │
│ 翻译      │ "将以下 {源语言} 翻译成 {目标语言}, 保持原意。"     │
│ 生成      │ "根据以下要求生成 {内容类型}。风格: {风格描述}。"   │
└──────────┴──────────────────────────────────────────────┘

关键原则:
  1. 分类/提取: temperature=0 (需要确定性)
  2. 摘要/翻译: temperature=0~0.3 (基本确定, 允许措辞变化)
  3. 创意生成: temperature=0.7~1.0 (需要多样性)
"""
print(PATTERNS)


# ------------------------------------------------------------
# 综合实战: Prompt 调试器
# ------------------------------------------------------------
# 构建一个工具, 方便对比不同 Prompt 的效果。

print("=" * 60)
print("综合实战: Prompt 调试器")
print("=" * 60)


class PromptDebugger:
    """
    Prompt A/B 测试工具。
    同一输入, 对比不同 Prompt 的输出。
    """

    def __init__(self, input_text: str):
        self.input_text = input_text
        self.variants: dict[str, tuple[str, str]] = {}  # name → (system, template)

    def add_variant(self, name: str, system: str, template: str):
        self.variants[name] = (system, template)

    def run(self) -> list[dict]:
        results = []
        for name, (system, template) in self.variants.items():
            prompt_text = template.format(input=self.input_text)
            print(f"\n{'─' * 40}")
            print(f"🔬 变体: {name}")
            print(f"   System: {system[:60]}...")
            output = ask(prompt_text, system=system)
            print(f"   输出: {output[:200]}")
            results.append({
                "name": name,
                "system": system,
                "output": output,
            })
        return results

    def compare_length(self) -> None:
        """对比各变体的输出长度。"""
        print(f"\n📊 输出长度对比:")
        for name, (system, _) in self.variants.items():
            output = ask(self.input_text, system=system)
            print(f"  {name}: {len(output)} chars")


# 演示: 对比不同的代码解释 Prompt
debugger = PromptDebugger(
    "from functools import lru_cache\n\n@lru_cache(maxsize=128)\ndef fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)"
)

debugger.add_variant(
    "默认",
    "",
    "{input}\n\n解释这段代码",
)

debugger.add_variant(
    "Java专家",
    "你培训 Java 开发者学 Python。用 Java 类比解释。",
    "{input}\n\n解释这段代码",
)

debugger.add_variant(
    "调试模式",
    "你是 Python 运行时调试器。逐行解释代码执行过程, 包括内存变化。",
    "{input}\n\n解释这段代码",
)

debugger.add_variant(
    "面试答案",
    "你是技术面试候选人。给出简洁、专业、1 分钟说完的解释。",
    "{input}\n\n解释这段代码",
)

if api_ok:
    debugger.run()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Lesson 12 完成! Prompt 工程的核心已掌握。")
    print("=" * 60)
    print(f"""
  回顾:
  1. 角色设定 = System Prompt — 决定 AI "是谁"
  2. Few-shot = 给例子 — AI 模仿格式和风格
  3. Chain of Thought = "一步步想" — 提升复杂推理
  4. 结构化 Prompt = XML/标签分隔 — 信息更清晰
  5. 模板化 = Prompt + 占位符 — 可复用
  6. 迭代优化 — 从 V1 改到 V3, 质量飞跃
  7. Temperature = 0 要准确, 1 要创意

  一句话总结:
    好的 Prompt = 清晰的角色 + 具体的指令 + 合理的示例 + 明确的格式
""")


# ============================================================
# 试试看 (Try This) —— 练习实现
# ============================================================

print("\n" + "=" * 60)
print("试试看: 练习 1 — 三版本 Prompt 设计对比")
print("=" * 60)

# 练习 1: 选一个工作场景 (日志分析), 设计 3 个版本的 Prompt
# 场景: 从一段应用日志中分析错误原因

LOG_SAMPLE = """
2026-06-19 10:23:45 ERROR [OrderService] - 订单创建失败: order_id=ORD-8912
2026-06-19 10:23:45 DEBUG [OrderService] - 库存检查返回: stock_available=false, sku=SKU-441
2026-06-19 10:23:46 ERROR [PaymentGateway] - 支付回调超时: gateway=Alipay, timeout_ms=5000
2026-06-19 10:23:47 WARN [OrderService] - 降级处理: 使用缓存库存数据
"""

# V1: 一句话描述 (模糊)
PROMPT_V1 = f"分析这段日志:\n{LOG_SAMPLE}"

# V2: 加角色 + 输出格式约束
PROMPT_V2_SYSTEM = """你是一个资深 SRE (Site Reliability Engineer)。
分析日志时按以下格式输出:
1. 错误摘要 (一句话)
2. 根因分析 (最可能的根本原因)
3. 影响范围 (哪些服务/功能受影响)
4. 修复建议 (按优先级排列)"""

PROMPT_V2 = f"分析以下应用日志:\n{LOG_SAMPLE}"

# V3: 加 Few-shot 示例
PROMPT_V3_SYSTEM = """你是 SRE 专家。分析日志, 输出格式参考以下示例:
---
错误摘要: 支付服务调用超时导致订单创建失败
根因分析: 支付宝网关响应超过 5 秒超时阈值, 导致支付验证失败, 进而阻塞订单创建流程
影响范围: OrderService (订单创建), PaymentGateway (支付处理)
修复建议:
  [高] 增加支付网关超时时间到 8s, 或改为异步回调模式
  [中] 库存服务添加缓存预热, 避免缓存未命中
---
现在分析新日志。"""

PROMPT_V3 = f"分析日志:\n{LOG_SAMPLE}"

print("\n  📋 场景: 应用日志分析")
print(f"  日志内容:\n{LOG_SAMPLE}")

if api_ok:
    for ver_name, sys_prompt, user_prompt in [
        ("V1 (模糊)", None, PROMPT_V1),
        ("V2 (+角色 +格式)", PROMPT_V2_SYSTEM, PROMPT_V2),
        ("V3 (+Few-shot)", PROMPT_V3_SYSTEM, PROMPT_V3),
    ]:
        print(f"\n  {'─' * 40}")
        print(f"  {ver_name}:")
        result = ask(user_prompt, system=sys_prompt, max_tokens=300)
        print(f"  {result[:200]}...")
else:
    print("\n  (模拟)")
    for ver_name, desc in [
        ("V1", "一句话回复, 无结构"),
        ("V2", "包含错误摘要、根因分析、影响范围、修复建议"),
        ("V3", "格式整洁, 按优先级排列, 和示例结构一致"),
    ]:
        print(f"  {ver_name}: {desc}")

print("\n" + "=" * 60)
print("试试看: 练习 2 — 反向 Prompt 工程")
print("=" * 60)

# 练习 2: 给 AI 看输出, 让它反推 Prompt
SAMPLE_OUTPUT = """
分析结果:
- 函数名: process_orders, 输入: list[dict], 输出: list[str]
- 逻辑: 遍历订单列表, 筛选有效订单 (price>0 且 quantity>0),
  计算折扣后总价, 格式化输出 "{商品名}: ${总价:.2f}"
- 代码质量: 使用了 list comprehension, type hints 完整
- 改进建议: 可用 NamedTuple 代替 dict 提高类型安全
"""

REVERSE_PROMPT = f"""以下是一个 AI 的输出文本。请反推原始的 Prompt (用户给 AI 的指令)。

AI 输出:
{SAMPLE_OUTPUT}

请反推出最可能的原始 Prompt。包括:
1. AI 被赋予的角色
2. 具体的任务指令
3. 输出格式要求
4. 可能的约束条件"""

print(f"\n  给 AI 的输出样本:\n{SAMPLE_OUTPUT}")
print(f"\n  反向 Prompt 工程 — 让 AI 反推原始 Prompt:")

if api_ok:
    reversed_result = ask(REVERSE_PROMPT, max_tokens=400)
    print(f"\n  🤖 反推结果:\n{reversed_result}")
else:
    print("""
  (模拟) AI 反推结果:
  1. 角色: 代码审查专家 / 代码分析助手
  2. 任务: 分析一段 Python 函数, 解释其功能和结构
  3. 格式要求: 分点列出 (函数签名、逻辑、代码质量、改进建议)
  4. 约束: 关注代码可读性和类型安全

  💡 这是研究 LLM 行为的有趣方法——通过输出反推输入,
     可以帮我们理解为什么 AI 会"误解"某些指令。""")

print("\n" + "=" * 60)
print("试试看: 练习 3 — PromptTemplate 实用模板")
print("=" * 60)

# 练习 3: 创建 SQL 生成器、单元测试生成器、Git commit 生成器

# 模板 A: SQL 生成器
sql_generator = PromptTemplate(
    system_prompt="你是一个 SQL 专家。只输出 SQL 语句, 不要额外解释。使用标准 SQL 语法。",
    template="""生成 SQL 查询:
表名: {table}
字段: {columns}
条件: {condition}
排序: {order_by}
行数限制: {limit}"""
)

# 模板 B: 单元测试生成器
test_generator = PromptTemplate(
    system_prompt="""你是 Python 测试专家。用 pytest 风格生成测试代码。
每个测试函数以 test_ 开头。包含:
- 正常情况测试
- 边界情况测试
- 异常情况测试
只输出测试代码, 不要额外解释。""",
    template="""为以下函数生成 pytest 单元测试:
```python
{code}
```"""
)

# 模板 C: Git commit 消息生成器
commit_generator = PromptTemplate(
    system_prompt="""你是 Git 提交消息专家。生成 Conventional Commits 格式:
<type>(<scope>): <subject>
type: feat/fix/docs/refactor/test/chore
subject 用中文, 不超过 50 字""",
    template="""根据以下代码变更生成 commit message:
{changes}"""
)

print("\n  📋 模板 A — SQL 生成器:")
sql_result = sql_generator.ask(
    table="orders",
    columns="order_id, customer_name, total_amount, created_at",
    condition="total_amount > 100 AND created_at >= '2026-01-01'",
    order_by="created_at DESC",
    limit="20"
)
print(f"  {sql_result}")

print("\n  📋 模板 B — 单元测试生成器:")
test_result = test_generator.ask(
    code="""def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b"""
)
print(f"  {test_result[:300]}...")

print("\n  📋 模板 C — Git Commit 生成器:")
commit_result = commit_generator.ask(
    changes="添加了用户登录的 JWT token 验证中间件, 修改了 auth.py, 新增 jwt_utils.py"
)
print(f"  {commit_result}")

print("\n" + "=" * 60)
print("试试看: 练习 4 — Temperature 一致性实验")
print("=" * 60)

# 练习 4: 同一个 Prompt 跑 5 次, 观察分类任务 vs 创意任务的差异

CLASSIFY_PROMPT = """判断以下评论的情感倾向 (正面 / 负面 / 中性)。只输出一个词。

评论: "物流很快, 包装完好, 商品质量也不错"
情感:"""

CREATIVE_PROMPT = """用一句话给一家 Python 培训机构写一句广告语, 要有创意和感染力。"""

print("\n  🔬 实验: temperature=1.0, 每种任务跑 5 次")

if api_ok:
    print("\n  ── 分类任务 (预期: 结果一致) ──")
    classify_results = []
    for i in range(5):
        result = ask(CLASSIFY_PROMPT, temperature=1.0, max_tokens=20)
        classify_results.append(result.strip())
        print(f"    第 {i+1} 次: {result.strip()}")

    unique_classify = set(classify_results)
    print(f"\n  分类一致性: {len(unique_classify)} 种不同结果 / 5 次")
    print(f"  结果: {'✅ 高度一致' if len(unique_classify) <= 2 else '⚠️ 有波动'}")

    print("\n  ── 创意任务 (预期: 每次不同) ──")
    creative_results = []
    for i in range(5):
        result = ask(CREATIVE_PROMPT, temperature=1.0, max_tokens=80)
        creative_results.append(result.strip())
        print(f"    第 {i+1} 次: {result.strip()[:60]}...")

    unique_creative = set(creative_results)
    print(f"\n  创意多样性: {len(unique_creative)} 种不同结果 / 5 次")
    print(f"  结果: {'✅ 多样化' if len(unique_creative) >= 3 else '⚠️ 较单一'}")

else:
    print("""
  (模拟)
  ── 分类任务 ──
    第 1 次: 正面
    第 2 次: 正面
    第 3 次: 正面
    第 4 次: 正面
    第 5 次: 正面
  分类一致性: 1 种不同结果 / 5 次 ✅

  ── 创意任务 ──
    第 1 次: "Python 学习, 从入门到升职加薪!"
    第 2 次: "用 Python 解放双手, 让代码为你工作"
    第 3 次: "三天入门, 三月精通 — Python 改变你的职业生涯"
    第 4 次: "Python: 让复杂变简单, 让不可能变可能"
    第 5 次: "写更少的代码, 做更多的事 — Python 之道"
  创意多样性: 5 种不同结果 / 5 次 ✅

  💡 结论:
  - 分类/提取任务: 即使 temperature=1.0, 结果也高度一致
  - 创意生成任务: temperature=1.0 带来丰富的多样性
  - 工程建议: 生产环境的数据提取一定要用 temperature=0""")

print("\n" + "=" * 60)
print("试试看: 练习 5 — 自动 Prompt 优化器 (挑战)")
print("=" * 60)

# 练习 5: 自动 Prompt 优化器
class AutoPromptOptimizer:
    """
    自动 Prompt 优化器 —— 让 AI 自己迭代改进 Prompt。

    流程:
      1. 用当前 Prompt 生成输出
      2. 让 AI 评判输出质量 (打分 1-10)
      3. 让 AI 提出 Prompt 改进建议
      4. 应用改进, 回到步骤 1
      5. 直到分数满意或达到最大迭代次数

    类比 Java: 类似遗传算法中的适应度函数 + 变异操作。
    """

    def __init__(self, task_description: str, expected_output_format: str,
                 max_iterations: int = 3):
        self.task = task_description
        self.expected_format = expected_output_format
        self.max_iterations = max_iterations
        self.history: list[dict] = []

    def run(self, initial_prompt: str) -> dict:
        """运行优化循环。"""
        current_prompt = initial_prompt

        for iteration in range(self.max_iterations):
            print(f"\n  ── 迭代 {iteration + 1}/{self.max_iterations} ──")

            # Step 1: 用当前 Prompt 生成
            print(f"  📝 当前 Prompt: {current_prompt[:80]}...")
            output = ask(current_prompt, temperature=0.0, max_tokens=200)

            # Step 2: 让 AI 评分
            score_prompt = f"""评判以下 AI 输出。任务要求: {self.task}
期望格式: {self.expected_format}

AI 输出:
{output}

请打分 (1-10), 并说明扣分原因。格式:
分数: X/10
扣分原因: ..."""
            score_result = ask(score_prompt, temperature=0.0, max_tokens=200)

            # 尝试解析分数
            score = 5  # 默认
            for line in score_result.split("\n"):
                if "分数" in line and "/" in line:
                    try:
                        score = int(line.split(":")[1].strip().split("/")[0])
                    except (ValueError, IndexError):
                        pass

            print(f"  📊 评分: {score}/10")

            record = {
                "iteration": iteration + 1,
                "prompt": current_prompt,
                "output": output[:100],
                "score": score,
            }
            self.history.append(record)

            if score >= 8:
                print(f"  ✅ 达到满意分数, 优化结束!")
                break

            # Step 3: 让 AI 提出改进建议
            if iteration < self.max_iterations - 1:
                improve_prompt = f"""你是一个 Prompt 工程专家。以下 Prompt 的输出评分为 {score}/10。
任务: {self.task}
当前 Prompt: {current_prompt}
评分反馈: {score_result}

请改进这个 Prompt, 让它能达到 8 分以上。只输出改进后的 Prompt, 不要额外说明。"""
                current_prompt = ask(improve_prompt, temperature=0.3, max_tokens=300)
                print(f"  🔄 优化后 Prompt: {current_prompt[:80]}...")

        # 返回最佳结果
        best = max(self.history, key=lambda r: r["score"]) if self.history else None
        return {"best_prompt": best["prompt"] if best else current_prompt,
                "best_score": best["score"] if best else 0,
                "history": self.history}


# 演示: 优化一个分类 Prompt
print("\n  🔬 优化一个'日志分类' Prompt:")
optimizer = AutoPromptOptimizer(
    task_description="将日志分为 ERROR/WARN/INFO 三类",
    expected_output_format="分类: ERROR/WARN/INFO, 原因: 一句话",
    max_iterations=2,  # 演示用, 实际可以更多
)

initial = "分析这段日志并分类"
if api_ok:
    result = optimizer.run(initial)
    print(f"\n  🏆 最佳 Prompt (评分 {result['best_score']}/10):")
    print(f"  {result['best_prompt']}")
else:
    print("""
  (模拟优化过程)
  ── 迭代 1/2 ──
    📝 Prompt: "分析这段日志并分类"
    📊 评分: 4/10
    🔄 优化: "你是日志分析专家。将日志分为 ERROR/WARN/INFO 三类。只输出类别名。"

  ── 迭代 2/2 ──
    📝 Prompt: "你是日志分析专家..."
    📊 评分: 8/10
  🏆 最佳 Prompt 评分: 8/10

  💡 这个练习展示了 Prompt 优化的核心思想:
    Prompt 不是一次写对的, 而是迭代出来的。
    和写代码一样: 先跑通, 再优化。""")

print("\n" + "=" * 60)
print("试试看: 练习 6 — Anthropic Prompt 工程指南探索")
print("=" * 60)

# 练习 6: Anthropic 官方 Prompt 工程指南
print("""
  📖 Anthropic 官方 Prompt 工程指南:
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering

  核心技巧 (本课覆盖 + 官方补充):

  ┌─────────────────────────┬──────────────────────────────────────┐
  │ 技巧                     │ 说明                                  │
  ├─────────────────────────┼──────────────────────────────────────┤
  │ 角色设定 (已学)          │ 在 system prompt 定义角色               │
  │ Few-shot (已学)          │ 给示例引导格式                         │
  │ Chain of Thought (已学)  │ "让我们一步步分析"                     │
  │ 结构化分隔 (已学)        │ XML 标签分隔不同信息                    │
  │ Prompt 模板 (已学)       │ 占位符 + 工厂模式                      │
  ├─────────────────────────┼──────────────────────────────────────┤
  │ 🔑 Prompt Caching       │ 缓存重复的 system prompt, 降低成本      │
  │ 🔑 清晰指令 > 模糊指令   │ "用 3 句话总结" > "总结一下"            │
  │ 🔑 输出格式前置          │ 先告诉模型你想什么格式, 再给输入数据      │
  │ 🔑 正面引导 > 负面禁止   │ "请用专业语气" > "不要太随意"            │
  │ 🔑 长 Prompt 分段        │ 用标题和编号组织长指令                   │
  │ 🔑 避免歧义              │ "回答是/否" > "给出你的判断"              │
  │ 🔑 示例覆盖边界          │ 给出正例和反例, 覆盖所有类别              │
  │ 🔑 迭代式开发            │ Prompt 像代码一样需要测试和优化           │
  └─────────────────────────┴──────────────────────────────────────┘

  核心补充 (本课未深入):
  1. Prompt Caching: 如果 system prompt 在多轮对话中不变,
     Anthropic 会自动缓存它, 降低 input token 费用。
     标记需要缓存的断点: cache_control={"type": "ephemeral"}

  2. 避免"幻觉"的三板斧:
     - 明确要求"如果不知道就说不知道"
     - 要求引用具体来源
     - 用 Tool Use 强制外部验证

  3. 多模态 Prompt: Claude 支持图片输入 (vision),
     可以在 content 中添加 image 类型的 block。

  建议: 有空时通读一遍官方指南, 尤其注意"常见错误"章节。""")

print("\n" + "=" * 60)
print("  Lesson 12 试试看练习全部完成!")
print("=" * 60)
