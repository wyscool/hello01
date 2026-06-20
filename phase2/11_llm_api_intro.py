# ============================================================
# Phase 2, Lesson 11: LLM API 入门 —— Claude / OpenAI 基础调用
# ============================================================
#
# 本课目标:
#   1. 理解 LLM API 是什么 (本质就是 HTTP 请求)
#   2. 获取 API Key、配置环境变量
#   3. 安装 anthropic SDK、发出第一个请求
#   4. 理解 Messages API: system / user / assistant 三种角色
#   5. 解析响应对象
#   6. 核心参数: model、max_tokens、temperature
#   7. 流式输出 (streaming)
#   8. OpenAI SDK 对照 (一次学会两个平台)
#   9. 多轮对话
#   10. 费用意识 & 最佳实践
#
# 预计阅读 + 实操时间: 40-50 分钟
#
# ⚠️ 课前准备 (重要!):
#   1. 注册 Anthropic Console: https://console.anthropic.com
#   2. 获取 API Key (Settings → API Keys)
#   3. 在项目根目录创建 .env 文件:
#        ANTHROPIC_API_KEY=sk-ant-xxxxx
#        ANTHROPIC_BASE_URL=https://api.anthropic.com  (可选, 代理/中转时修改)
#   4. OpenAI Key (可选):
#        OPENAI_API_KEY=sk-xxxxx
#
#   API 调用会收费, 但本节课全部请求合计不超过 $0.01。
# ============================================================

import os
from pathlib import Path

# 加载 .env 文件
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


# ------------------------------------------------------------
# 一、LLM API 的本质 —— 就是一个 HTTP 请求
# ------------------------------------------------------------
# 不要被"大模型"这个词吓到。从工程角度看, LLM API 就是一个 HTTP POST:
#
#   POST https://api.anthropic.com/v1/messages
#   Headers: x-api-key: sk-ant-xxxxx
#   Body: {
#     "model": "claude-sonnet-4-6",
#     "max_tokens": 1024,
#     "messages": [
#       {"role": "user", "content": "你好"}
#     ]
#   }
#
# 响应就是一个 JSON。
# SDK 只是封装了这个过程, 让你不用手写 HTTP 调用。
#
# Java 类比: Spring RestTemplate / Retrofit 的接口代理。
# Anthropic SDK ≈ 带类型的 HTTP Client。
#
# 关于 base_url:
#   - 默认请求地址是 https://api.anthropic.com
#   - 如果你用的是 API 代理/中转服务 (如国内访问需要代理),
#     或者自建的兼容接口 (如 LiteLLM、OpenRouter),
#     可以通过 ANTHROPIC_BASE_URL 修改目标地址。
#   - 本质就是 HTTP 的 base URL, SDK 所有请求都发到这里。
#
# 你也可以用 curl 直接调——本质完全一样。
# 试试看 (把 YOUR_KEY 换成你的 API Key):
#   curl https://api.anthropic.com/v1/messages \
#     -H "x-api-key: $ANTHROPIC_API_KEY" \
#     -H "anthropic-version: 2023-06-01" \
#     -H "content-type: application/json" \
#     -d '{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}'


# ------------------------------------------------------------
# 二、检查环境 —— API Key 是否配置好了
# ------------------------------------------------------------
from anthropic import Anthropic  # type: ignore
from anthropic.types import Message


def _get_text(response: Message) -> str:
    """从响应中提取纯文本 (跳过 thinking block)。"""
    parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts)


api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")  # 代理/中转服务地址
openai_key = os.getenv("OPENAI_API_KEY")

# 标记 API 实际是否可用 (而非只看环境变量, 因为可能有系统代理)
api_available = False

print("=" * 60)
print("环境检查")
print("=" * 60)

if api_key:
    masked = api_key[:10] + "..." + api_key[-4:]
    print(f"  ✅ ANTHROPIC_API_KEY: {masked}")
else:
    print("  ⚠️  未配置 ANTHROPIC_API_KEY (将尝试系统级配置)")

if base_url:
    print(f"  ✅ ANTHROPIC_BASE_URL: {base_url}")
    print(f"     (自定义 API 地址, 可能是代理或中转服务)")
else:
    print(f"  ℹ️  ANTHROPIC_BASE_URL: 使用默认 https://api.anthropic.com")

if openai_key:
    print(f"  ✅ OPENAI_API_KEY (可选): {openai_key[:7]}...{openai_key[-4:]}")
else:
    print("  ℹ️  OPENAI_API_KEY 未配置 (可选, 不影响本节学习)")


# ------------------------------------------------------------
# 三、第一个 API 调用 —— "Hello, LLM!"
# ------------------------------------------------------------
# 类比 Java 写第一个 REST 调用: new RestTemplate().getForObject(...)
#
# 三个基本对象:
#   Anthropic() — 客户端, 持有 API Key
#   Messages.create() — 发送消息, 获取回复
#   model — 指定用哪个模型

print("\n" + "=" * 60)
print("第一个 API 调用")
print("=" * 60)

try:
    # 构建客户端: 支持自定义 base_url (代理/中转)
    client_kwargs = {"api_key": api_key} if api_key else {}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = Anthropic(**client_kwargs)

    response = client.messages.create(
        model="claude-sonnet-4-6",     # 模型名称
        max_tokens=200,                  # 最大输出 token 数
        messages=[
            {
                "role": "user",
                "content": "用一句话介绍 Python 语言的特点",
            }
        ],
    )

    # response 是一个结构化对象, 不是字符串
    print(f"  响应类型: {type(response).__name__}")
    print(f"  模型: {response.model}")
    print(f"  ID: {response.id}")

    # 最重要的字段: content — 这是模型的回复
    # content 是一个列表, 每项是 TextBlock 或 ThinkingBlock。
    # TextBlock: 模型的文字回复
    # ThinkingBlock: 模型的思考过程 (部分模型支持, 如 Claude Opus、DeepSeek)
    print(f"\n  回复内容:")
    for block in response.content:
        if block.type == "text":
            print(f"  {block.text}")
        elif block.type == "thinking":
            print(f"  [思考: {block.thinking[:80]}...]")

    # 用量信息 —— 关注费用!
    print(f"\n  用量: input={response.usage.input_tokens}, "
          f"output={response.usage.output_tokens} tokens")

    api_available = True  # API 调用成功, 后续演示可以运行

except Exception as e:
    import sys
    print(f"  ❌ 调用失败: {e}")
    print(f"\n  可能的原因:")
    print(f"  1. API Key 未配置或已过期")
    print(f"  2. 网络无法连接 api.anthropic.com (需要科学上网)")
    print(f"  3. 账户余额不足")
    print(f"\n  💡 我们降级为演示模式, 后续示例会模拟响应内容。")
    print(f"     请解决网络问题后重新运行。")


# ------------------------------------------------------------
# 四、理解 Messages API 的三种角色
# ------------------------------------------------------------
# Claude API 使用 Messages 格式, 每条消息有 role + content:
#
#   role: "system"   → 系统提示词 (设定 AI 的行为和角色)
#   role: "user"     → 用户说的话
#   role: "assistant" → AI 的回复 (多轮对话时放入上下文)
#
# 类比:
#   system   ≈ 设置游戏 NPC 的角色描述
#   user     ≈ 玩家说的话
#   assistant ≈ NPC 的回复
#
# ⚠️ 规则: 每轮对话必须以 user 消息开始!
#   不能 system → assistant (中间必须有 user)

SYSTEM_PROMPT = """你是一位 Python 学习助手, 专门帮助 Java 开发者学习 Python。
回答时要:
- 用中文回复
- 用 Java 概念做类比
- 代码示例要有对比
- 回答简洁, 不超过 100 字"""


def ask_claude(user_message: str, system: str | None = None) -> str:
    """
    封装一个简单的问询函数。
    后续课程中会逐步完善这个函数, 加入流式输出、工具调用等。
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=messages,
        )
        return _get_text(response)
    except Exception as e:
        return f"[调用失败: {e}]"


if api_available:
    print("\n" + "=" * 60)
    print("系统提示词 + 角色演示")
    print("=" * 60)

    # 普通问询 (没有 system prompt)
    q = "Python 的 list 和 Java 的 ArrayList 有什么区别?"
    print(f"\n  用户: {q}")

    answer = ask_claude(q, system=SYSTEM_PROMPT)
    print(f"\n  AI:\n{answer}")

    # 换一个 topic, system prompt 保持一致
    q2 = "解释一下 with 语句"
    print(f"\n  用户: {q2}")

    answer2 = ask_claude(q2, system=SYSTEM_PROMPT)
    print(f"\n  AI:\n{answer2}")

else:
    print("\n" + "=" * 60)
    print("系统提示词 + 角色演示 (模拟)")
    print("=" * 60)
    print("  (API 不可用, 跳过真实调用)")
    print()
    print("  system prompt 示例:")
    print(f"  {SYSTEM_PROMPT[:80]}...")


# ------------------------------------------------------------
# 五、核心参数详解
# ------------------------------------------------------------
# model:      选择模型 (claude-sonnet-4-6 / claude-haiku-4-5 / claude-opus-4-7)
# max_tokens: 回复的最大 token 数 (1 token ≈ 0.75 个英文词, ≈ 1-1.5 个中文字)
# temperature: 0.0-1.0, 控制随机性。
#              0 = 确定性 (适合分类、抽取), 1 = 创造性 (适合写作、头脑风暴)
# stop_sequences: 遇到这些字符串就停止生成

print("\n" + "=" * 60)
print("核心参数: temperature 效果对比")
print("=" * 60)

if api_available:
    PROMPT = "写一个有创意的 Python 变量名"

    for temp in [0.0, 0.5, 1.0]:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=50,
            temperature=temp,
            messages=[{"role": "user", "content": PROMPT}],
        )
        print(f"  temperature={temp}: {_get_text(response).strip()}")

else:
    print("  (API 不可用, 跳过演示)")
    print()
    print("  temperature 的直观理解:")
    print("    0.0 → 「1 + 1 = 2」  (每次一样)")
    print("    0.5 → 「中午吃面还是饭?」(有选择空间)")
    print("    1.0 → 「讲个笑话」    (每次不同)")

print("\n  📊 模型速查:")
print("  ┌─────────────────────┬──────┬──────────────────────────┐")
print("  │ 模型                 │ 速度  │ 适用场景                  │")
print("  ├─────────────────────┼──────┼──────────────────────────┤")
print("  │ claude-haiku-4-5    │ 最快  │ 分类、提取、简单总结       │")
print("  │ claude-sonnet-4-6   │ 中等  │ 日常对话、代码生成 (默认)   │")
print("  │ claude-opus-4-7     │ 最慢  │ 复杂推理、架构设计          │")
print("  └─────────────────────┴──────┴──────────────────────────┘")


# ------------------------------------------------------------
# 六、流式输出 (Streaming) —— 实时打字效果
# ------------------------------------------------------------
# 默认的 create() 是同步调用: 发送请求 → 等待 → 收到完整回复。
# streaming=True: 边生成边返回, 类似打字效果。
#
# 为什么用 streaming:
#   1. 用户体验好 (不用等完整回复)
#   2. 可以在 token 到达时实时处理
#   3. 长回复不会超时

print("\n" + "=" * 60)
print("流式输出 (Streaming)")
print("=" * 60)

if api_available:
    print("  实时输出:")
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": "用 Python 写一个 斐波那契数列 函数, 并解释",
            }
        ],
    ) as stream:
        for text_chunk in stream.text_stream:
            print(text_chunk, end="", flush=True)

    print()  # 换行

    # 流结束后获取完整消息
    final_message = stream.get_final_message()
    total_tokens = final_message.usage.input_tokens + final_message.usage.output_tokens
    print(f"\n  总 tokens: {total_tokens}")

else:
    print("  (API 不可用, 跳过)")
    print()
    print("  streaming 本质:")
    print("    HTTP Server-Sent Events (SSE)")
    print("    每个 chunk 是一个 JSON: {\"type\": \"content_block_delta\", ...}")
    print()
    print("  Java 类比: WebFlux 的 Flux<Data> / Spring 的 StreamingResponseBody")


# ------------------------------------------------------------
# 七、多轮对话 —— 用 messages 数组维护上下文
# ------------------------------------------------------------
# LLM 是无状态的。每次调用都是独立请求。
# 多轮对话 = 把历史消息放进 messages 数组, 每次越放越多。
#
# 类比:
#   你是 LLM
#   每次对话, 我给你一张纸, 上面写着之前的所有聊天记录
#   你根据这张纸来回复
#   下一次对话, 我给你一张更长的纸 (加上你的新回复)

print("\n" + "=" * 60)
print("多轮对话")
print("=" * 60)


class SimpleChat:
    """
    最简单的多轮对话实现。
    本质就是维护一个 messages 列表。

    后续课程会扩展这个类, 添加:
      - token 计数
      - 上下文窗口管理
      - 工具调用
    """

    def __init__(self, system_prompt: str | None = None, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.messages: list[dict] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def send(self, user_input: str) -> str:
        """发送用户消息, 返回 AI 回复。"""
        self.messages.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=self.messages,
        )

        reply = _get_text(response)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def message_count(self) -> int:
        return len(self.messages)

    def estimated_tokens(self) -> int:
        """粗略估算总 token 数。"""
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            total += len(content) * 1.5  # 中英文混合的粗略估算
        return int(total)


if api_available:
    chat = SimpleChat(
        system_prompt="你是一个 Python 技术面试官, 正在考察候选人的 Python 基础。"
    )

    print("\n  💬 对话开始:")
    print()

    exchanges = [
        "请介绍一下我的背景: 我是 Java 后端工程师",
        "好的, 第一个问题: Python 的 GIL 是什么?",
        "我用 asyncio 可以绕过 GIL 吗?",
    ]

    for user_msg in exchanges:
        print(f"  🧑 面试者: {user_msg}")
        reply = chat.send(user_msg)
        print(f"  🤖 面试官: {reply[:150]}...")
        if len(reply) > 150:
            print(f"         (回复太长, 截断显示)")
        print()

    print(f"  对话轮次: {chat.message_count()} 条消息")
    print(f"  估算 tokens: ~{chat.estimated_tokens()}")

else:
    print("  (API 不可用, 跳过)")
    print()
    print("  多轮对话的数据结构:")
    print("  messages = [")
    print('    {"role": "system", "content": "你是..."},')
    print('    {"role": "user", "content": "你好"},')
    print('    {"role": "assistant", "content": "你好! 有什么可以帮你?"},')
    print('    {"role": "user", "content": "解释 GIL"},')
    print('    {"role": "assistant", "content": "GIL 是..."},')
    print("  ]")
    print()
    print("  ⚠️ 注意: 每条消息都包含之前的所有对话 → token 消耗递增")


# ------------------------------------------------------------
# 八、OpenAI SDK 对照 —— 一次学两个平台
# ------------------------------------------------------------
# Anthropic 和 OpenAI 的 API 非常相似。
# 核心差异:
#   - OpenAI 用 ChatCompletion API
#   - Anthropic 用 Messages API
#   - 参数名略有不同
#   - OpenAI 不支持 system → assistant 的严格顺序

print("\n" + "=" * 60)
print("OpenAI SDK 对照")
print("=" * 60)

print("""
  Anthropic SDK:
  ┌────────────────────────────────────────────────────┐
  │ client = Anthropic(api_key=...)                    │
  │ response = client.messages.create(                 │
  │     model="claude-sonnet-4-6",                     │
  │     max_tokens=500,                                 │
  │     system="你是助手",       ← system 单独参数       │
  │     messages=[                                     │
  │         {"role": "user", "content": "你好"}         │
  │     ]                                              │
  │ )                                                  │
  │ print(response.content[0].text)   ← content 是列表  │
  └────────────────────────────────────────────────────┘

  OpenAI SDK (pip install openai):
  ┌────────────────────────────────────────────────────┐
  │ client = OpenAI(api_key=...)                       │
  │ response = client.chat.completions.create(          │
  │     model="gpt-4o",                                │
  │     max_completion_tokens=500,    ← 参数名不同!      │
  │     messages=[                                     │
  │         {"role": "system", "content": "你是助手"},  │
  │         {"role": "user", "content": "你好"}         │
  │     ]                                              │
  │ )                                                  │
  │ print(response.choices[0].message.content)         │
  └────────────────────────────────────────────────────┘

  差异速查:
    Anthropic                   OpenAI
    ──────────────────────────────────────────────
    max_tokens                  max_completion_tokens
    messages (system 可独立)    messages (system 是 message 的一条)
    response.content[0].text    response.choices[0].message.content
    response.stop_reason        response.choices[0].finish_reason
""")


# ------------------------------------------------------------
# 九、费用意识 —— 每个 token 都是钱
# ------------------------------------------------------------
# API 是按 token 计费的, 不是按"次"计费。
#
# 大致价格 (2026 年):
#   Haiku:  $0.80  / 百万 input tokens,  $4.00  / 百万 output tokens
#   Sonnet: $3.00  / 百万 input tokens,  $15.00 / 百万 output tokens
#   Opus:   $15.00 / 百万 input tokens,  $75.00 / 百万 output tokens
#
# 一个典型对话 (2000 input tokens, 500 output tokens) 用 Sonnet:
#   input:  2000 × $3.00 / 1,000,000  = $0.006
#   output:  500 × $15.00 / 1,000,000 = $0.0075
#   总计: ~$0.013 (约 1 美分)
#
# 省钱技巧:
#   1. 简单任务用 Haiku (分类、提取、格式化)
#   2. system prompt 精简, 每次调用都会算进去
#   3. 多轮对话时, 合理裁剪历史消息
#   4. 开启 prompt caching (Lesson 14 会讲到)


# ------------------------------------------------------------
# 综合实战: 构建一个"代码翻译官"
# ------------------------------------------------------------
# 把 Java 代码翻译成 Python —— 对 Java 开发者最实用的 AI 功能。

print("\n" + "=" * 60)
print("综合实战: Java → Python 代码翻译官")
print("=" * 60)

JAVA_TO_PYTHON_SYSTEM = """你是一个代码翻译专家, 专门把 Java 代码翻译成 Python。
翻译规则:
1. 结果只输出 Python 代码, 不要额外的解释
2. 使用 Python 惯用写法 (list comprehension, zip, context manager...)
3. 如果有 Java 独有特性, 写出等效的 Python 替代方案
4. 保持缩进和代码风格整洁
5. 用 type hints"""

JAVA_CODE = '''
public List<String> processOrders(List<Map<String, Object>> orders) {
    List<String> result = new ArrayList<>();
    for (Map<String, Object> order : orders) {
        double price = (Double) order.get("price");
        int quantity = (Integer) order.get("quantity");
        if (price > 0 && quantity > 0) {
            double total = price * quantity * (price > 100 ? 0.9 : 1.0);
            result.add(String.format("%s: $%.2f", order.get("name"), total));
        }
    }
    return result;
}
'''

if api_available:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=JAVA_TO_PYTHON_SYSTEM,
        messages=[{"role": "user", "content": f"翻译这段 Java 代码:\n{JAVA_CODE}"}],
    )

    print("\n  Java 代码 (输入):")
    print(JAVA_CODE)
    print("\n  Python 代码 (翻译结果):")
    print(_get_text(response))

    print(f"\n  用量: {response.usage.input_tokens} + {response.usage.output_tokens} tokens")
    cost_est = (
        response.usage.input_tokens * 3.0 / 1_000_000
        + response.usage.output_tokens * 15.0 / 1_000_000
    )
    print(f"  估算费用: ${cost_est:.6f}")

else:
    print("\n  (API 不可用, 跳过代码翻译演示)")
    print()
    print("  -- 预期翻译结果示意 --")
    print('''
    def process_orders(orders: list[dict[str, object]]) -> list[str]:
        result: list[str] = []
        for order in orders:
            price = float(order["price"])
            quantity = int(order["quantity"])
            if price > 0 and quantity > 0:
                discount = 0.9 if price > 100 else 1.0
                total = price * quantity * discount
                name = order["name"]
                result.append(f"{name}: ${total:.2f}")
        return result
''')


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🎉 Lesson 11 完成! 你已经发出了第一个 LLM 请求。")
    print("=" * 60)
    print(f"""
  回顾: 你学会了什么?
  1. LLM API 本质 = 带 text generation 的 HTTP POST
  2. Messages API 三种角色: system / user / assistant
  3. 用 system prompt 控制 AI 行为 ← 最重要!
  4. temperature 控制创造性
  5. streaming = 实时打字效果
  6. 多轮对话 = 把历史消息全部放进 messages
  7. 按 token 计费 → 每次调用都要考虑成本
""")


# ============================================================
# 试试看 (Try This) —— 练习实现
# ============================================================

print("\n" + "=" * 60)
print("试试看: 练习 1 — 角色对比实验")
print("=" * 60)

# 练习 1: 用三种不同的 System Prompt 测试 AI 的回复风格差异
REVIEWER_SYSTEM = """你是一位严格的代码审查员。对于用户提供的任何代码, 你必须:
1. 找出至少 3 个问题 (即使代码看起来完美)
2. 按严重程度排序: 安全 > 逻辑 > 性能 > 可读性
3. 每个问题附上具体的修改建议
4. 用中文回复, 专业但不严厉"""

HUMOR_SYSTEM = """你是一位幽默的技术作家。解释任何技术概念时:
1. 必须用一个笑话或有趣的比喻开头
2. 用日常生活中的事物做类比
3. 让人在笑的同时理解技术要点
4. 用中文回复, 回答控制在 100 字以内"""

SOCRATES_SYSTEM = """你是苏格拉底式的导师。你的教学方法是:
1. 永远不直接给出答案
2. 用反问引导对方自己发现答案
3. 每次回复只问 2-3 个关键问题
4. 用中文回复, 语气友好但坚持原则"""

# 一段有改进空间的代码
TEST_CODE = '''
def process(data):
    result = []
    for i in range(len(data)):
        val = data[i]
        if val != None:
            result.append(str(val))
    return result
'''

print("\n  测试代码:")
print(TEST_CODE)

if api_available:
    for role_name, sys_prompt in [
        ("严格审查员", REVIEWER_SYSTEM),
        ("幽默作家", HUMOR_SYSTEM),
        ("苏格拉底导师", SOCRATES_SYSTEM),
    ]:
        print(f"\n  {'─' * 40}")
        print(f"  角色: {role_name}")
        print(f"  AI 回复: ", end="")
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=200,
            temperature=0.7,
            system=sys_prompt,
            messages=[{"role": "user", "content": f"请分析这段代码:\n{TEST_CODE}"}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
        print()
else:
    print("\n  (API 不可用, 模拟展示)")
    print("""
  角色: 严格审查员
    1. [安全] val != None 应该用 val is not None (PEP 8)
    2. [可读性] range(len(data)) 应改为 enumerate(data)
    3. [性能] 可用 list comprehension: [str(v) for v in data if v is not None]

  角色: 幽默作家
    "这段代码就像把所有衣服塞进洗衣机却不分类——虽然能跑, 但总有一天会染色!
    val != None 就像问'你是不是不是空'——Python 更爱 is not None 这种直白表白。"

  角色: 苏格拉底导师
    "如果我告诉你这段代码有 3 处可以改进, 你会先从哪里开始看?
    你觉得 Python 中检查 None 的最好方式是什么?
    为什么我们有时会习惯性地用 range(len()) 模式?""")

print("\n" + "=" * 60)
print("试试看: 练习 2 — SimpleChat 添加 token_limit")
print("=" * 60)


# 练习 2: 给 SimpleChat 添加 token_limit, 自动裁剪历史消息
class BoundedChat:
    """
    带上下文窗口限制的 SimpleChat 升级版。

    类比 Java: 类似 LinkedHashMap 的 removeEldestEntry 机制,
    当容量超过阈值时自动淘汰最旧的消息。
    """

    def __init__(self, system_prompt: str | None = None,
                 model: str = "claude-sonnet-4-6",
                 token_limit: int = 4000):
        self.model = model
        self.token_limit = token_limit
        self.messages: list[dict] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def estimated_tokens(self) -> int:
        """粗略估算总 token 数。"""
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                # 中文约 1 char/token, 英文约 4 char/token
                total += len(content) * 1.5
            elif isinstance(content, list):
                total += len(str(content)) * 1.5
        return int(total)

    def _trim(self) -> None:
        """
        裁剪最早的非 system 消息, 直到 token 总数在限制内。
        保留 system prompt (index 0) 和最近的对话。
        """
        while self.estimated_tokens() > self.token_limit and len(self.messages) > 2:
            # 跳过 system (index 0), 移除 index 1
            removed = self.messages.pop(1)
            # 如果移除的是 assistant 消息, 前面的 user 消息也移除 (保持配对)
            if removed.get("role") == "assistant" and len(self.messages) > 1:
                self.messages.pop(1)
            print(f"  ⚠️  裁剪历史消息 (当前估算: {self.estimated_tokens()} tokens)")

    def send(self, user_input: str) -> str:
        """发送用户消息, 返回 AI 回复。超过 token_limit 时自动裁剪。"""
        self.messages.append({"role": "user", "content": user_input})
        self._trim()

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=self.messages,
            )
            reply = _get_text(response)
            self.messages.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"[调用失败: {e}]"

    def token_usage_bar(self, width: int = 30) -> str:
        """显示上下文用量的进度条。"""
        cur = self.estimated_tokens()
        ratio = min(cur / self.token_limit, 1.0)
        filled = int(ratio * width)
        bar = "█" * filled + "░" * (width - filled)
        pct = int(ratio * 100)
        return f"[{bar}] {pct}% ({cur} / {self.token_limit} tokens)"


# 演示: 用较小的 token_limit 观察裁剪效果
demo_chat = BoundedChat(
    system_prompt="你是一个简洁的 Python 助手。用中文回复, 不超过 50 字。",
    token_limit=1000,  # 故意设小, 方便观察裁剪
)

print(f"\n  初始状态: {demo_chat.token_usage_bar()}")

# 模拟多轮对话, 触发裁剪
test_msgs = [
    "Python 的 list 和 Java 的 ArrayList 有什么区别? (请详细解释)",
    "那 dict 和 HashMap 呢? 也请详细对比",
    "再解释一下 set 和 tuple",
    "Python 的 decorator 是什么? 请用 200 字详细说明",
    "解释 GIL 是什么, 300 字",
]

if api_available:
    for msg in test_msgs:
        print(f"\n  🧑: {msg[:40]}...")
        reply = demo_chat.send(msg)
        print(f"  🤖: {reply[:100]}...")
        print(f"  {demo_chat.token_usage_bar()}")
else:
    print("\n  (模拟)")
    for i, msg in enumerate(test_msgs, 1):
        print(f"  🧑: {msg[:40]}...")
        print(f"  🤖: [模拟回复 # {i}]")
        print(f"  {demo_chat.token_usage_bar().replace('0', str(i * 200))}")
        if i >= 3:
            print(f"  ⚠️  触发裁剪! 移除最早的消息对...")

print("\n" + "=" * 60)
print("试试看: 练习 3 — 探索 Streaming 事件类型")
print("=" * 60)

# 练习 3: 遍历 stream 本身, 观察所有事件类型
print("\n  遍历 stream 对象, 观察事件类型:")

if api_available:
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=50,
        temperature=0.0,
        messages=[{"role": "user", "content": "说 Hello World"}],
    ) as stream:
        for event in stream:
            etype = type(event).__name__
            print(f"  事件: {etype}", end="")
            # 显示每种事件的额外信息
            if "StartEvent" in etype and "Message" in etype:
                print(f"  id={event.message.id}, model={event.message.model}")
            elif "ContentBlockStart" in etype:
                cb = event.content_block
                print(f"  index={event.index}, content_type={cb.type}")
            elif "ContentBlockDelta" in etype:
                delta = event.delta
                if hasattr(delta, "text"):
                    print(f"  text='{delta.text}'")
                elif hasattr(delta, "partial_json"):
                    print(f"  json='{delta.partial_json}'")
                elif hasattr(delta, "thinking"):
                    print(f"  thinking='{delta.thinking[:30]}...'")
                else:
                    print(f"  delta_type={delta.type}")
            elif "ContentBlockStop" in etype:
                print()
            elif "MessageDelta" in etype:
                print(f"  stop_reason={event.delta.stop_reason}, usage={event.usage}")
            elif "MessageStop" in etype:
                print()
            else:
                print()

    print(f"\n  📊 总结:")
    print(f"    - message_start: 流的开始, 包含消息 ID 和模型名")
    print(f"    - content_block_start: 新内容块开始 (text / thinking / tool_use)")
    print(f"    - content_block_delta: 增量 token (text_delta / thinking_delta / input_json_delta)")
    print(f"    - content_block_stop: 内容块结束")
    print(f"    - message_delta: 消息级别更新 (stop_reason, usage)")
    print(f"    - message_stop: 流结束")
else:
    print("""
  (模拟) 事件序列:
  事件: RawMessageStartEvent  id=msg_xxx, model=claude-sonnet-4-6
  事件: RawContentBlockStartEvent  index=0, content_type=text
  事件: RawContentBlockDeltaEvent  text='Hello'
  事件: RawContentBlockDeltaEvent  text=' World'
  事件: RawContentBlockStopEvent
  事件: RawMessageDeltaEvent  stop_reason=end_turn, usage={...}
  事件: RawMessageStopEvent""")

print("\n" + "=" * 60)
print("试试看: 练习 4 — OpenAI SDK 对照 (模拟)")
print("=" * 60)

# 练习 4: OpenAI SDK 对照
print("""
  由于本课聚焦 Anthropic SDK, 这里展示 OpenAI SDK 的等效调用代码。
  如果你有 OpenAI API Key, 可以 pip install openai 后运行:

  from openai import OpenAI
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

  response = client.chat.completions.create(
      model="gpt-4o",
      max_completion_tokens=300,
      messages=[
          {"role": "system", "content": "你是 Python 助手"},
          {"role": "user", "content": "解释装饰器"},
      ],
  )
  print(response.choices[0].message.content)

  关键差异速查:
  ┌────────────────────────────────────────────────┐
  │ Anthropic              │ OpenAI                │
  │────────────────────────┼───────────────────────│
  │ client.messages.create │ client.chat.completions.create │
  │ max_tokens             │ max_completion_tokens │
  │ system 参数 (独立)      │ messages[0] role=system │
  │ response.content[0].text│ response.choices[0].message.content │
  │ stop_reason            │ finish_reason         │
  │ tools (Anthropic 格式) │ tools (含 type:function 包装) │
  └────────────────────────────────────────────────┘

  学习建议: 先精通一个 SDK, 另一个只需查文档对比字段名即可。""")

print("\n" + "=" * 60)
print("试试看: 练习 5 — Temperature 实验")
print("=" * 60)

# 练习 5: 不同 temperature 对同一问题的输出变化
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
TEST_QUESTION = "写一个 Python 类来管理一个简单的待办事项列表 (TodoList), 包含添加、完成、列出功能"

print(f"\n  问题: {TEST_QUESTION[:50]}...")
print()

if api_available:
    for temp in TEMPERATURES:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            temperature=temp,
            messages=[{"role": "user", "content": TEST_QUESTION}],
        )
        reply = _get_text(response)
        # 分析输出特征
        lines = reply.strip().split("\n")
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
        print(f"  temperature={temp}:")
        print(f"    输出长度: {len(reply)} chars, {len(lines)} 行")
        print(f"    代码风格: {'包含注释/解释' if any('#' in l for l in lines) else '纯代码'}")
        print(f"    类名数量: {reply.count('class ')}")
        print(f"    --- 前 80 字 ---")
        print(f"    {reply[:80].strip()}...")
        print()
else:
    print("  (模拟 — 实际 API 不可用)")
    print("""
  temperature=0.0: 每次输出几乎相同, 代码结构固定
  temperature=0.3: 基本结构相同, 变量名/注释偶有变化
  temperature=0.7: 代码风格有差异 (有人用 dataclass, 有人用普通类)
  temperature=1.0: 每次输出都可能不同, 结构、命名、设计模式都有变化

  关键理解:
  - temp=0 适合需要确定性的任务: 分类、提取、格式化
  - temp=0.7~1.0 适合需要创造性的任务: 写作、头脑风暴、创意代码
  - 代码生成通常用 0.0~0.3, 因为代码需要确定性""")

print("\n" + "=" * 60)
print("试试看: 练习 6 — 探索 Response 完整 JSON 结构")
print("=" * 60)

# 练习 6: 查看 response 的完整 JSON 结构
if api_available:
    print("\n  Response 完整 JSON 结构:")
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=50,
            messages=[{"role": "user", "content": "Hi"}],
        )
        # model_dump_json 是 Pydantic 模型的方法, 输出完整 JSON
        full_json = response.model_dump_json(indent=2)
        # 只显示前 800 字符, 避免太长
        print(f"  (总长度: {len(full_json)} chars, 显示前 800 chars)")
        print(full_json[:800])
        print("  ...")
        print(f"""
  关键字段说明:
  - id:              消息唯一 ID (如 msg_xxx)
  - type:            固定为 "message"
  - role:            固定为 "assistant"
  - content:         回复内容列表 (TextBlock / ThinkingBlock / ToolUseBlock)
  - model:           使用的模型名
  - stop_reason:     停止原因 (end_turn / tool_use / max_tokens / stop_sequence)
  - stop_sequence:   触发的停止序列 (如果有)
  - usage:           {{input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens}}

  content 列表中的每项:
  - TextBlock:       {{type: "text", text: "..."}}
  - ThinkingBlock:   {{type: "thinking", thinking: "...", signature: "..."}}
  - ToolUseBlock:    {{type: "tool_use", id: "tool_xxx", name: "...", input: {{...}}}}

  Java 类比: response.model_dump_json() ≈ Jackson ObjectMapper.writeValueAsString()
  """)
    except Exception as e:
        print(f"  ❌ 获取失败: {e}")
else:
    print("""
  (模拟) Response JSON 结构:
  {
    "id": "msg_01ABC123...",
    "type": "message",
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "Hello! How can I help you today?"
      }
    ],
    "model": "claude-sonnet-4-6",
    "stop_reason": "end_turn",
    "stop_sequence": null,
    "usage": {
      "input_tokens": 10,
      "output_tokens": 8,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 0
    }
  }

  实际项目中, response.model_dump_json(indent=2) 可以用于调试日志。
  所有字段都是类型安全的 Pydantic 模型, IDE 有完整的代码补全。""")

print("\n" + "=" * 60)
print("  Lesson 11 试试看练习全部完成!")
print("=" * 60)
