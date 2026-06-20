# ============================================================
# Phase 2, Lesson 13: 结构化输出与工具调用
# ============================================================
#
# 本课目标:
#   1. 理解为什么需要结构化输出 (自由文本 ≠ 可靠数据)
#   2. 方法一: Prompt JSON — 让模型输出 JSON (简单但不保证)
#   3. 方法二: Tool Use — 让模型调用"函数" (可靠的结构化输出)
#   4. 定义 tool schema: name、description、input_schema
#   5. 处理 tool_use block、返回 tool_result
#   6. 完整 Tool Use 循环: 定义 → 调用 → 执行 → 返回 → 整合
#   7. Tool choice 控制: auto / any / tool
#   8. OpenAI Function Calling 对照
#   9. 实战: 智能信息抽取器 (从文本提取结构化字段)
#   10. 实战: 带工具的计算器 (完整 Tool Use 演示)
#
# 预计阅读 + 实操时间: 45-55 分钟
#
# 前置: 已完成 Lesson 11 (API 入门) + Lesson 12 (Prompt 工程)
#
# ⚠️ 本课是 Phase 4 Agent 的基础! Tool Use = Agent 的"手"。
# ============================================================

import os
import json
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from anthropic import Anthropic
from anthropic.types import Message


# ------------------------------------------------------------
# 〇、环境准备
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
        max_tokens: int = 500, temperature: float = 0.0) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        response = client.messages.create(
            model=model, max_tokens=max_tokens,
            temperature=temperature, messages=messages,
        )
        return _get_text(response)
    except Exception as e:
        return f"[调用失败: {e}]"


try:
    ask("ping")
    api_ok = True
    print("✅ API 连接正常\n")
except Exception:
    api_ok = False
    print("⚠️  API 不可用, 将以模拟模式运行 (仍可学习结构化输出思路)\n")


# ============================================================
# 一、为什么需要结构化输出?
# ============================================================
# 假设你用 LLM 提取发票信息:
#
#   "发票号 INV-2024-0891, 日期 2024-06-15,
#    商品: 机械键盘 ×2, 单价 ¥399
#    合计: ¥798.00"
#
# 如果让 LLM 自由回复:
#   "好的, 这张发票的号码是 INV-2024-0891, 日期是 2024年6月15日..."
#
# 你的代码怎么提取"发票号"和"金额"?
#   → 写正则? 不可靠, 每次回复格式不同
#   → 让 LLM 直接输出 JSON? 可以, 但不保证格式正确
#   → 用 Tool Use? 对! 这是工程上最可靠的方式
#
# 类比 Java:
#   - 自由文本 → System.out.println("结果: xxx") — 人看可以, 代码解析难
#   - 结构化输出 → return new ResultDTO(...) — 代码直接消费
#
# 三种方式对比:
#   Prompt JSON:  简单, 但模型可能"忘记"输出 JSON, 或输出多余文字
#   Tool Use:     模型返回函数调用参数, 格式由 JSON Schema 保证
#   Structured Output (新功能): 模型直接按 schema 输出, 最新最可靠


# ------------------------------------------------------------
# 二、方法一: Prompt JSON — 让模型输出 JSON
# ------------------------------------------------------------
# 最简单的结构化方式: 在 prompt 里要求输出 JSON。
#
# 优点: 不需要额外 API 功能, 任何模型都支持
# 缺点: 模型可能输出多余文字 (如 "好的, 这是结果: {...}"),
#       需要手动清理, 且不能 100% 保证格式正确。

print("=" * 60)
print("方法一: Prompt JSON")
print("=" * 60)

JSON_SYSTEM = """你是一个数据提取工具。只输出 JSON, 不要任何额外文字。
输出格式: {"invoice_id": "...", "date": "...", "total": 0.0, "items": [...]}"""

INVOICE_TEXT = """发票号 INV-2024-0891, 日期 2024-06-15,
商品: 机械键盘 ×2, 单价 ¥399, 鼠标 ×1, 单价 ¥149
合计: ¥947.00"""

if api_ok:
    raw = ask(INVOICE_TEXT, system=JSON_SYSTEM, max_tokens=300, temperature=0.0)
    print(f"\n原始输出:\n{raw}")

    # 尝试解析 —— 注意: 这里可能失败!
    try:
        # 清理可能的 markdown 代码块标记
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        data = json.loads(cleaned)
        print(f"\n✅ 解析成功:")
        print(f"   发票号: {data.get('invoice_id')}")
        print(f"   日期:   {data.get('date')}")
        print(f"   合计:   ¥{data.get('total')}")
        print(f"   项目数: {len(data.get('items', []))}")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON 解析失败: {e}")
        print(f"   → 这就是 Prompt JSON 的痛点!")
else:
    print("\n  (模拟输出)")
    print('  {"invoice_id": "INV-2024-0891", "date": "2024-06-15", "total": 947.0, ...}')


# ------------------------------------------------------------
# 三、方法二: Tool Use 概述 —— 让模型调用"函数"
# ------------------------------------------------------------
# Tool Use 是 Anthropic 的结构化输出方案:
#   你定义"工具"(函数签名), 模型决定是否"调用"它。
#   如果调用, 模型会返回一个结构化的 tool_use block,
#   包含函数名和 JSON Schema 保证的参数。
#
# 类比 Java:
#   定义接口 → 模型自动生成调用代码 → 你的代码执行 → 返回结果
#
#   // 你定义的接口
#   interface InvoiceExtractor {
#       InvoiceResult extract(String invoiceId, LocalDate date, BigDecimal total);
#   }
#   // 模型: "我调用 extract('INV-0891', '2024-06-15', 947.00)"
#   // 你的代码: new InvoiceResult(...)
#
# 流程:
#   1. 你定义 tool (name + description + input_schema)
#   2. 你发送消息 + tools 列表
#   3. 模型回复: 要么 text, 要么 tool_use block
#   4. 如果是 tool_use → 你的代码执行函数 → 返回 tool_result
#   5. 模型收到结果 → 生成最终文本回复
#
# 这比 Prompt JSON 可靠, 因为:
#   - 参数由 JSON Schema 强制约束
#   - 模型不会"忘记"输出格式
#   - 你的代码拿到的是解析好的 dict, 不是需要清理的字符串

print("\n" + "=" * 60)
print("方法二: Tool Use 概述")
print("=" * 60)

# 定义一个简单的 tool: 提取发票信息
extract_invoice_tool = {
    "name": "extract_invoice",
    "description": "从文本中提取发票信息, 返回结构化数据",
    "input_schema": {
        "type": "object",
        "properties": {
            "invoice_id": {
                "type": "string",
                "description": "发票号码, 如 INV-2024-0891"
            },
            "date": {
                "type": "string",
                "description": "发票日期, YYYY-MM-DD 格式"
            },
            "total": {
                "type": "number",
                "description": "合计金额"
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "商品名称"},
                        "quantity": {"type": "integer", "description": "数量"},
                        "unit_price": {"type": "number", "description": "单价"}
                    },
                    "required": ["name", "quantity", "unit_price"]
                },
                "description": "商品列表"
            }
        },
        "required": ["invoice_id", "date", "total", "items"]
    }
}

print("""
  Tool Schema 结构:
  ┌──────────────────────────────────────────────┐
  │ name:        "extract_invoice"               │  ← 函数名
  │ description: "从文本中提取..."                 │  ← 帮助模型判断何时调用
  │ input_schema: {                              │
  │   type: "object",                            │
  │   properties: { ... }                        │  ← JSON Schema, 定义参数
  │   required: [...]                            │  ← 必填字段
  │ }                                            │
  └──────────────────────────────────────────────┘

  JSON Schema 类比:
    就像 Java 方法的参数类型声明:
      void extractInvoice(String invoiceId, LocalDate date, double total)
  模型会根据 description 判断"该不该调这个函数",
  根据 input_schema 决定"参数该填什么"。
""")


# ------------------------------------------------------------
# 四、第一次 Tool Use 调用 —— 亲手触发一次
# ------------------------------------------------------------

print("=" * 60)
print("第一次 Tool Use 调用")
print("=" * 60)

if api_ok:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        temperature=0.0,
        system="你是一个数据提取助手。当用户提供发票文本时, 调用 extract_invoice 工具提取信息。",
        messages=[
            {"role": "user", "content": INVOICE_TEXT}
        ],
        tools=[extract_invoice_tool],
    )

    # 响应中可能同时有 text 和 tool_use blocks
    print(f"\n  stop_reason: {response.stop_reason}")
    print(f"  content blocks 数量: {len(response.content)}")

    for i, block in enumerate(response.content):
        print(f"\n  Block {i}: type={block.type}")
        if block.type == "text":
            print(f"    文本: {block.text[:100]}...")
        elif block.type == "tool_use":
            print(f"    工具名: {block.name}")
            print(f"    tool_use_id: {block.id}")
            print(f"    参数: {json.dumps(block.input, ensure_ascii=False, indent=2)}")

            # 这里就是"你的代码执行函数"的环节:
            # 在实际应用中, 你会用这些参数做真正的操作
            # (写数据库、调 API、计算等)
            extracted = block.input
            print(f"\n  ✅ 提取结果 (代码直接可用!):")
            print(f"     发票号: {extracted['invoice_id']}")
            print(f"     日期:   {extracted['date']}")
            print(f"     合计:   ¥{extracted['total']}")
            for item in extracted.get('items', []):
                print(f"     - {item['name']} × {item['quantity']} @ ¥{item['unit_price']}")

else:
    print("\n  (模拟 Tool Use 响应)")
    print("""
  stop_reason: tool_use
  content blocks:
    Block 0: type=tool_use
      工具名: extract_invoice
      tool_use_id: tool_01ABC...
      参数: {
        "invoice_id": "INV-2024-0891",
        "date": "2024-06-15",
        "total": 947.0,
        "items": [
          {"name": "机械键盘", "quantity": 2, "unit_price": 399},
          {"name": "鼠标", "quantity": 1, "unit_price": 149}
        ]
      }
""")


# ------------------------------------------------------------
# 五、Tool Use 完整循环 —— 定义 → 调用 → 执行 → 返回
# ------------------------------------------------------------
# 上面的例子只完成了一半: 模型调用了工具, 但我们没有返回结果。
# 完整流程需要把 tool_result 发回给模型, 模型再整合成最终回复。
#
# 完整循环:
#   User → Model (模型决定调用工具)
#        → Your Code (执行工具, 拿到结果)
#        → Model (模型整合结果, 生成最终回复)
#        → User

print("=" * 60)
print("完整 Tool Use 循环")
print("=" * 60)

# 定义一个"查询天气"的工具 (模拟)
weather_tool = {
    "name": "get_weather",
    "description": "查询指定城市的当前天气。返回温度、天气状况、湿度。",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称, 如 '北京'、'上海'"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "温度单位"
            }
        },
        "required": ["city"]
    }
}


def get_weather(city: str, unit: str = "celsius") -> dict:
    """模拟天气查询。实际项目中这里会调用真实 API。"""
    # 模拟数据 (真实场景: requests.get(f"https://api.weather.com/..."))
    weather_data = {
        "北京": {"temp": 28, "condition": "晴", "humidity": 45},
        "上海": {"temp": 32, "condition": "多云", "humidity": 70},
        "杭州": {"temp": 30, "condition": "小雨", "humidity": 80},
    }
    result = weather_data.get(city, {"temp": 25, "condition": "未知", "humidity": 50})
    if unit == "fahrenheit":
        result = {**result, "temp": result["temp"] * 9 / 5 + 32}
    result["city"] = city
    result["unit"] = unit
    return result


if api_ok:
    # Step 1: 用户提问
    print("\n  Step 1: 用户提问")
    print('  用户: "北京和上海的天气怎么样?"')

    messages = [
        {"role": "user", "content": "北京和上海的天气怎么样?"}
    ]

    # Step 2: 第一次调用 → 模型决定调用工具
    print("\n  Step 2: 模型决定调用工具")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        temperature=0.0,
        messages=messages,
        tools=[weather_tool],
    )

    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
    text_blocks = [b for b in response.content if b.type == "text"]

    if text_blocks:
        print(f"  文本回复: {text_blocks[0].text[:100]}...")

    print(f"  模型调用了 {len(tool_use_blocks)} 个工具:")
    for tb in tool_use_blocks:
        print(f"    - {tb.name}({json.dumps(tb.input, ensure_ascii=False)})")

    # Step 3: 代码执行工具, 构造 tool_result
    print("\n  Step 3: 执行工具, 返回结果")
    # 把模型的回复 (包含 tool_use blocks) 加入消息历史
    messages.append({
        "role": "assistant",
        "content": [b.model_dump() for b in response.content]
    })

    # 执行每个 tool_use, 构造 tool_result
    tool_results = []
    for tb in tool_use_blocks:
        if tb.name == "get_weather":
            result = get_weather(**tb.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tb.id,
                "content": json.dumps(result, ensure_ascii=False)
            })
            print(f"    get_weather({tb.input['city']}) → {result}")

    messages.append({"role": "user", "content": tool_results})

    # Step 4: 第二次调用 → 模型整合结果, 生成最终回复
    print("\n  Step 4: 模型整合结果, 生成最终回复")
    final_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        temperature=0.0,
        messages=messages,
        tools=[weather_tool],
    )

    final_text = _get_text(final_response)
    print(f"  最终回复:\n{final_text}")

    print(f"""
  ┌─────────────────────────────────────────┐
  │  完整循环总结:                            │
  │  1. User 提问                            │
  │  2. Model → tool_use('get_weather', ...) │
  │  3. Code → tool_result({{temp: 28, ...}}) │
  │  4. Model → "北京今天28°C, 晴..."        │
  │                                           │
  │  核心: 模型不执行工具, 你的代码执行。       │
  │  模型只负责"决定调用哪个工具、填什么参数"。  │
  └─────────────────────────────────────────┘
""")

else:
    print("\n  (模拟完整循环)")
    print("""
  Step 1: 用户提问
    用户: "北京天气怎么样?"

  Step 2: 模型决定调用工具
    stop_reason: tool_use
    调用: get_weather({"city": "北京", "unit": "celsius"})

  Step 3: 代码执行工具
    get_weather("北京") → {"city": "北京", "temp": 28, "condition": "晴", "humidity": 45}

  Step 4: 模型整合结果
    最终回复: "北京今天晴, 气温28°C, 湿度45%。适合户外活动。"

  关键理解:
    模型 = 大脑 (决策)
    工具 = 手 (执行)
    你的代码 = 神经系统 (连接大脑和手)
""")


# ------------------------------------------------------------
# 六、封装 Tool Use 循环 —— ChatBot 升级版
# ------------------------------------------------------------
# 把上面的四步循环封装成一个可复用的类。
# 这是后续 Agent 课程的核心组件。

print("=" * 60)
print("ToolUseChat: 封装 Tool Use 循环")
print("=" * 60)


class ToolUseChat:
    """
    支持 Tool Use 的多轮对话客户端。

    类比 Java:
      class ToolUseChat {
          List<Message> messages;
          Map<String, Function> toolHandlers;

          String send(String userInput) {
              // 循环: 调 API → 检查 tool_use → 执行工具 → 返回结果 → 再调 API
          }
      }
    """

    def __init__(self, system: str | None = None, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.messages: list[dict] = []
        if system:
            self.messages.append({"role": "system", "content": system})
        # 工具名 → 处理函数 的映射
        self._tools: list[dict] = []
        self._handlers: dict[str, callable] = {}

    def register_tool(self, tool_def: dict, handler: callable) -> None:
        """注册一个工具及其处理函数。

        tool_def: Anthropic tool 定义 (name + description + input_schema)
        handler:  函数, 接收 **kwargs, 返回 dict 或 str
        """
        self._tools.append(tool_def)
        self._handlers[tool_def["name"]] = handler

    def send(self, user_input: str, max_rounds: int = 5) -> str:
        """发送消息, 自动处理工具调用循环。"""
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(max_rounds):  # 防止无限循环
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                messages=self.messages,
                tools=self._tools if self._tools else None,
            )

            # 检查 stop_reason
            if response.stop_reason == "end_turn":
                # 模型直接回复了文本, 没有调用工具
                text = _get_text(response)
                self.messages.append({"role": "assistant", "content": text})
                return text

            elif response.stop_reason == "tool_use":
                # 模型调用了工具 → 执行工具 → 返回结果
                tool_results = self._execute_tools(response)
                # 把 assistant 的 tool_use 消息加入历史
                self.messages.append({
                    "role": "assistant",
                    "content": [b.model_dump() for b in response.content]
                })
                # 把 tool_result 加入历史
                self.messages.append({"role": "user", "content": tool_results})
                # 继续循环, 模型会看到 tool_result 并决定下一步
                continue

            else:
                return f"[未知 stop_reason: {response.stop_reason}]"

        return "[达到最大循环轮次, 仍未结束]"

    def _execute_tools(self, response: Message) -> list[dict]:
        """执行所有 tool_use blocks, 返回 tool_result 列表。"""
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = self._handlers.get(block.name)
                if handler:
                    try:
                        output = handler(**block.input)
                        content = json.dumps(output, ensure_ascii=False) if isinstance(output, dict) else str(output)
                    except Exception as e:
                        content = f"[工具执行失败: {e}]"
                else:
                    content = f"[未注册的工具: {block.name}]"

                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                })
        return results


# 演示: 注册天气 + 计算器工具
chat = ToolUseChat(system="你是一个生活助手。当用户问天气时查询天气, 问计算时帮忙计算。")

# 注册天气工具
chat.register_tool(weather_tool, get_weather)

# 注册计算器工具
calculator_tool = {
    "name": "calculate",
    "description": "执行数学计算。支持四则运算、幂运算。如 '2 + 3 * 4'。",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式, 如 '2 + 3 * 4'"
            }
        },
        "required": ["expression"]
    }
}


def calculate(expression: str) -> dict:
    """安全地执行数学表达式。"""
    try:
        # 安全: 只允许数字和基本运算符
        allowed = set("0123456789+-*/().%^ ")
        if not all(c in allowed for c in expression):
            return {"error": f"表达式包含不允许的字符: {expression}"}
        # 注意: eval 在生产环境不安全, 这里仅用于教学
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


chat.register_tool(calculator_tool, calculate)

if api_ok:
    print("\n  💬 演示 1: 需要调用工具的问题")
    reply = chat.send("北京今天天气怎么样?")
    print(f"  {reply[:200]}")

    print(f"\n  💬 演示 2: 多工具协作")
    reply = chat.send("上海比北京热多少度? 帮我算一下。")
    print(f"  {reply[:300]}")

    print(f"\n  📊 消息历史: {len(chat.messages)} 条")


# ------------------------------------------------------------
# 七、Tool Choice 参数 —— 控制模型行为
# ------------------------------------------------------------
# tool_choice 参数控制模型是否/如何使用工具:
#
#   "auto" (默认)     — 模型自己决定是否调工具 (可能调, 也可能不调)
#   "any"             — 模型必须调工具 (至少一个)
#   {"type": "tool", "name": "xxx"} — 强制调用指定工具
#
# 类比: @Override 注解 — 强制编译器检查你是否真的覆盖了方法。

print("\n" + "=" * 60)
print("Tool Choice 参数")
print("=" * 60)

if api_ok:
    # tool_choice 对比: auto vs any
    print("\n  tool_choice='auto' (默认) — 模型自己决定:")
    response_auto = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        temperature=0.0,
        messages=[{"role": "user", "content": "你好!"}],
        tools=[weather_tool],
        tool_choice={"type": "auto"},
    )
    tool_blocks_auto = [b for b in response_auto.content if b.type == "tool_use"]
    print(f"    stop_reason: {response_auto.stop_reason}")
    print(f"    工具调用: {'无 (正常, 打招呼不需要调工具)' if not tool_blocks_auto else tool_blocks_auto}")

    print("\n  tool_choice='any' (强制) — 必须调用至少一个工具:")
    response_any = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        temperature=0.0,
        messages=[{"role": "user", "content": "北京明天会下雨吗?"}],
        tools=[weather_tool],
        tool_choice={"type": "any"},
    )
    tool_blocks = [b for b in response_any.content if b.type == "tool_use"]
    print(f"    stop_reason: {response_any.stop_reason}")
    for tb in tool_blocks:
        print(f"    强制调用: {tb.name}({json.dumps(tb.input, ensure_ascii=False)})")

print("""
  tool_choice 选择指南:
    "auto"  → 日常对话 (绝大多数情况)
    "any"   → 数据提取任务 (你确定需要工具调用)
    "tool"  → 强制特定工具 (如 "分类任务必须用 classify_sentiment")
""")


# ------------------------------------------------------------
# 八、OpenAI Function Calling 对照
# ------------------------------------------------------------

print("=" * 60)
print("OpenAI Function Calling 对照")
print("=" * 60)

print("""
  Tool 定义的语法差异:

  Anthropic (tools 参数):
  ┌──────────────────────────────────────────────────┐
  │ tools=[{                                         │
  │   "name": "get_weather",                         │
  │   "description": "查询天气",                       │
  │   "input_schema": {                              │
  │     "type": "object",                            │
  │     "properties": {                              │
  │       "city": {"type": "string", ...}            │
  │     },                                           │
  │     "required": ["city"]                         │
  │   }                                              │
  │ }]                                               │
  └──────────────────────────────────────────────────┘

  OpenAI (tools 参数, 语法几乎相同):
  ┌──────────────────────────────────────────────────┐
  │ tools=[{                                         │
  │   "type": "function",             ← 多了这一层     │
  │   "function": {                                  │
  │     "name": "get_weather",                       │
  │     "description": "查询天气",                     │
  │     "parameters": {               ← 叫 parameters │
  │       "type": "object",                          │
  │       "properties": { ... },                     │
  │       "required": ["city"]                       │
  │     }                                            │
  │   }                                              │
  │ }]                                               │
  └──────────────────────────────────────────────────┘

  响应差异:
    Anthropic: response.content[i].type == "tool_use"
               → block.name, block.input, block.id

    OpenAI:    response.choices[0].message.tool_calls
               → tc.function.name, tc.function.arguments (JSON 字符串!)

  核心概念完全相同, 只是字段名略有不同。
  学会一个, 另一个 5 分钟就能上手。
""")


# ------------------------------------------------------------
# 九、综合实战: 智能信息抽取器
# ------------------------------------------------------------
# 实战场景: 从非结构化文本中提取多种类型的信息。
# 这和你在工作中遇到的"日志解析"、"邮件提取"、"文档分类" 一样。

print("=" * 60)
print("综合实战: 智能信息抽取器")
print("=" * 60)

extract_person_tool = {
    "name": "extract_person",
    "description": "从文本中提取人物信息: 姓名、职位、公司、联系方式",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "姓名"},
            "title": {"type": "string", "description": "职位/头衔"},
            "company": {"type": "string", "description": "公司/组织"},
            "email": {"type": "string", "description": "邮箱地址"},
            "phone": {"type": "string", "description": "电话号码"},
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "技能列表"
            }
        },
        "required": ["name"]
    }
}

extract_event_tool = {
    "name": "extract_event",
    "description": "从文本中提取事件信息: 事件类型、时间、地点、参与者",
    "input_schema": {
        "type": "object",
        "properties": {
            "event_type": {"type": "string", "description": "事件类型: 会议/面试/截止日期/发布"},
            "datetime": {"type": "string", "description": "日期和时间"},
            "location": {"type": "string", "description": "地点"},
            "participants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "参与者列表"
            },
            "summary": {"type": "string", "description": "一句话摘要"}
        },
        "required": ["event_type", "datetime"]
    }
}

# 多种文本, 应该触发不同工具
TEST_TEXTS = [
    "张伟, 高级Java工程师, 阿里巴巴, zhangwei@example.com, 13812345678, 擅长 Spring、Kafka、分布式系统",
    "下周三下午 3 点在 3 号楼 501 会议室进行系统架构评审, 参加人: 李工、王工、赵经理",
]

extractor_chat = ToolUseChat(
    system="你是一个信息抽取专家。根据文本内容, 选择合适的工具提取信息。"
)
extractor_chat.register_tool(extract_person_tool, lambda **kw: kw)
extractor_chat.register_tool(extract_event_tool, lambda **kw: kw)

if api_ok:
    for text in TEST_TEXTS:
        print(f"\n  📄 输入: {text[:60]}...")
        reply = extractor_chat.send(text)
        # 检查是否调用了工具
        # (ToolUseChat 会把 tool_result 返回给模型, 模型再生成文本)
        if reply:
            preview = reply[:100] + "..." if len(reply) > 100 else reply
            print(f"  📤 输出: {preview}")
else:
    print("\n  (模拟: API 不可用)")
    print("""
  输入: "张伟, 高级Java工程师, 阿里巴巴, zhangwei@example.com"
  → 调用 extract_person({name: "张伟", title: "高级Java工程师", ...})

  输入: "下周三下午 3 点在 3 号楼..."
  → 调用 extract_event({event_type: "会议", datetime: "下周三 15:00", ...})
""")


# ------------------------------------------------------------
# 十、综合实战: 带工具的计算器 + 天气助手
# ------------------------------------------------------------
# 把之前的 ToolUseChat 再演示一轮, 展示多工具协作。

print("\n" + "=" * 60)
print("综合实战: 计算器 + 天气助手")
print("=" * 60)

if api_ok:
    assistant = ToolUseChat(
        system="你是助手。天气用 get_weather, 计算用 calculate。中文回复。"
    )
    assistant.register_tool(weather_tool, get_weather)
    assistant.register_tool(calculator_tool, calculate)

    queries = [
        "3 的 10 次方是多少?",
        "杭州天气怎么样? 适合出门吗?",
        "如果杭州和北京的温差是 2 度, 那北京多少度?",
    ]

    for q in queries:
        print(f"\n  🧑: {q}")
        reply = assistant.send(q)
        print(f"  🤖: {reply[:200]}")
else:
    print("\n  (模拟: 多工具协作)")
    print("""
  🧑: "3 的 10 次方是多少?"
  → calculate({"expression": "3**10"})
  → 结果: 59049

  🧑: "杭州天气怎么样?"
  → get_weather({"city": "杭州"})
  → "杭州今天小雨, 30°C, 湿度 80%"

  🧑: "如果杭州和北京的温差是 2 度, 那北京多少度?"
  → 需要先查两个城市的天气 → 再计算
  → get_weather("杭州") + get_weather("北京") → calculate 差值
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Lesson 13 完成! 你已掌握结构化输出 & Tool Use。")
    print("=" * 60)
    print(f"""
  回顾: 你学会了什么?

  1. Prompt JSON — 简单但不保证格式
     适用: 快速原型、一次性脚本

  2. Tool Use — 可靠的结构化输出
     模型返回的不是文本, 而是函数调用参数
     参数由 JSON Schema 保证格式

  3. 完整 Tool Use 循环:
     User → Model (tool_use) → Code (执行) → Model (整合) → User
     模型是"大脑", 工具是"手", 你的代码是"神经系统"

  4. ToolUseChat 封装:
     register_tool() → 注册工具
     send() → 自动处理循环
     这是 Agent 的核心模式!

  5. 和 Prompt JSON 的区别:
     Prompt JSON:  "请输出 JSON"           → 可能输出多余文字
     Tool Use:     定义函数签名, 模型调用    → 参数格式有保证

  6. OpenAI Function Calling 几乎一样,
     只是字段名略有不同 (input_schema vs parameters)

  🎯 下一课预告: Lesson 14 — 流式响应 (Streaming)
     让 AI 像打字一样实时输出, 用户体验质的飞跃。
""")


# ============================================================
# 试试看 (Try This) —— 练习实现
# ============================================================

print("\n" + "=" * 60)
print("试试看: 练习 1 — ToolUseChat 调试日志")
print("=" * 60)

# 练习 1: 给 ToolUseChat 添加详细调试日志
import time as time_module


class DebugToolUseChat:
    """
    带调试日志的 ToolUseChat 版本。

    在 send() 中打印:
    - 每轮的消息数量、token 估算
    - 模型调用了哪个工具、什么参数
    - 工具执行耗时

    类比 Java: 相当于给 Service 层加了 SLF4J debug 日志 + 性能监控。
    """

    def __init__(self, system: str | None = None, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.messages: list[dict] = []
        if system:
            self.messages.append({"role": "system", "content": system})
        self._tools: list[dict] = []
        self._handlers: dict[str, callable] = {}

    def register_tool(self, tool_def: dict, handler: callable) -> None:
        self._tools.append(tool_def)
        self._handlers[tool_def["name"]] = handler

    def _estimate_tokens(self) -> int:
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) * 1.5
            elif isinstance(content, list):
                total += len(str(content)) * 1.5
        return int(total)

    def send(self, user_input: str, max_rounds: int = 5, debug: bool = True) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for round_num in range(1, max_rounds + 1):
            if debug:
                print(f"\n  [DEBUG] ── 第 {round_num} 轮 ──")
                print(f"  [DEBUG] 消息数: {len(self.messages)}, "
                      f"估算 tokens: {self._estimate_tokens()}")
                print(f"  [DEBUG] 已注册工具: {[t['name'] for t in self._tools]}")

            start_time = time_module.time()
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=0.0,
                    messages=self.messages,
                    tools=self._tools if self._tools else None,
                )
            except Exception as e:
                if debug:
                    print(f"  [DEBUG] ❌ API 调用失败: {e}")
                return f"[调用失败: {e}]"

            api_time = (time_module.time() - start_time) * 1000
            if debug:
                print(f"  [DEBUG] API 耗时: {api_time:.0f}ms")
                print(f"  [DEBUG] stop_reason: {response.stop_reason}")
                print(f"  [DEBUG] usage: input={response.usage.input_tokens}, "
                      f"output={response.usage.output_tokens}")

            if response.stop_reason == "end_turn":
                text = _get_text(response)
                self.messages.append({"role": "assistant", "content": text})
                if debug:
                    print(f"  [DEBUG] ✅ 最终回复: {text[:60]}...")
                return text

            elif response.stop_reason == "tool_use":
                if debug:
                    tool_calls = [b for b in response.content if b.type == "tool_use"]
                    for tb in tool_calls:
                        print(f"  [DEBUG] 🔧 模型调用: {tb.name}"
                              f"({json.dumps(tb.input, ensure_ascii=False)[:80]})")

                # 执行工具 (带计时)
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        handler = self._handlers.get(block.name)
                        tool_start = time_module.time()
                        if handler:
                            try:
                                output = handler(**block.input)
                                content = json.dumps(output, ensure_ascii=False) if isinstance(output, dict) else str(output)
                            except Exception as e:
                                content = f"[工具执行失败: {e}]"
                        else:
                            content = f"[未注册的工具: {block.name}]"

                        tool_time = (time_module.time() - tool_start) * 1000
                        if debug:
                            print(f"  [DEBUG]   ├─ 工具耗时: {tool_time:.0f}ms")
                            print(f"  [DEBUG]   └─ 结果: {content[:80]}...")

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content,
                        })

                # 把 assistant tool_use + tool_result 加入历史
                self.messages.append({
                    "role": "assistant",
                    "content": [b.model_dump() for b in response.content]
                })
                self.messages.append({"role": "user", "content": tool_results})
                continue

            else:
                return f"[未知 stop_reason: {response.stop_reason}]"

        return "[达到最大循环轮次]"


# 演示: 带调试日志的天气查询
weather_tool = {
    "name": "get_weather",
    "description": "查询指定城市的当前天气",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"}
        },
        "required": ["city"]
    }
}


def get_weather(city: str) -> dict:
    data = {"北京": {"temp": 28, "condition": "晴"}, "上海": {"temp": 32, "condition": "多云"}}
    return data.get(city, {"temp": 25, "condition": "未知"})


debug_chat = DebugToolUseChat(system="你是生活助手。中文回复。")
debug_chat.register_tool(weather_tool, get_weather)

print("\n  💬 演示: 带调试日志的对话")
print("  🧑: 北京天气怎么样?")

if api_ok:
    reply = debug_chat.send("北京天气怎么样?")
    print(f"\n  🤖 最终回复: {reply[:150]}")
else:
    print("""
  [DEBUG] ── 第 1 轮 ──
  [DEBUG] 消息数: 2, 估算 tokens: 80
  [DEBUG] 已注册工具: ['get_weather']
  [DEBUG] API 耗时: 850ms
  [DEBUG] stop_reason: tool_use
  [DEBUG] usage: input=120, output=35
  [DEBUG] 🔧 模型调用: get_weather({"city": "北京"})
  [DEBUG]   ├─ 工具耗时: 1ms
  [DEBUG]   └─ 结果: {"city": "北京", "temp": 28, "condition": "晴"}

  [DEBUG] ── 第 2 轮 ──
  [DEBUG] 消息数: 4, 估算 tokens: 220
  [DEBUG] API 耗时: 620ms
  [DEBUG] stop_reason: end_turn
  [DEBUG] ✅ 最终回复: 北京今天晴, 28°C...

  💡 Debug 日志让你"看见" Agent 的每一步思考过程:
    - 模型先决定"需要查天气" → tool_use
    - 你的代码执行 get_weather → 返回数据
    - 模型看到数据后 → 生成自然语言回复""")

print("\n" + "=" * 60)
print("试试看: 练习 2 — 创建翻译器工具")
print("=" * 60)

# 练习 2: 翻译器工具
TRANSLATE_TOOL = {
    "name": "translate",
    "description": "将文本翻译成目标语言。支持中英日韩法德等常见语言互译。",
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要翻译的文本"},
            "source_lang": {
                "type": "string",
                "description": "源语言代码: zh/en/ja/ko/fr/de",
                "default": "auto"
            },
            "target_lang": {
                "type": "string",
                "description": "目标语言代码: zh/en/ja/ko/fr/de"
            }
        },
        "required": ["text", "target_lang"]
    }
}

# 本地翻译字典 (模拟翻译 API)
_TRANSLATION_DICT = {
    ("Hello World", "zh"): "你好, 世界",
    ("Hello World", "ja"): "こんにちは、世界",
    ("Hello World", "ko"): "안녕하세요, 세계",
    ("Hello World", "fr"): "Bonjour le monde",
    ("Hello World", "de"): "Hallo Welt",
    ("Good morning", "zh"): "早上好",
    ("Good morning", "ja"): "おはようございます",
    ("Thank you", "zh"): "谢谢",
    ("Thank you", "ja"): "ありがとうございます",
    ("Python is great", "zh"): "Python 很棒",
    ("Python is great", "ja"): "Python は素晴らしい",
    # 反向翻译
    ("你好世界", "en"): "Hello World",
    ("谢谢", "en"): "Thank you",
    ("早上好", "en"): "Good morning",
}


def handle_translate(text: str, target_lang: str,
                     source_lang: str = "auto") -> dict:
    """模拟翻译。实际项目可以调用 Google Translate API 或 DeepL API。"""
    # 先查本地字典
    key = (text, target_lang)
    if key in _TRANSLATION_DICT:
        translated = _TRANSLATION_DICT[key]
        method = "local_dict"
    else:
        # 模拟翻译: 如果是简单文本, 标记为 API 翻译
        translated = f"[{text}] → ({target_lang})"
        method = "simulated"

    lang_names = {"zh": "中文", "en": "英文", "ja": "日语", "ko": "韩语",
                  "fr": "法语", "de": "德语"}
    return {
        "original": text,
        "translated": translated,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "target_lang_name": lang_names.get(target_lang, target_lang),
        "method": method,
    }


# 创建翻译聊天机器人
translate_chat = ToolUseChat(system="你是翻译助手。翻译时调用 translate 工具。中文回复。")
translate_chat.register_tool(TRANSLATE_TOOL, handle_translate)

print("\n  🌐 翻译器测试:")
test_texts = [
    "把 Hello World 翻译成日语",
    "把 Good morning 翻译成中文",
    "翻译 Python is great 成日语",
]

if api_ok:
    for text in test_texts:
        print(f"\n  🧑: {text}")
        reply = translate_chat.send(text)
        print(f"  🤖: {reply[:200]}")
else:
    for text in test_texts:
        print(f"\n  🧑: {text}")
        print(f"  🤖: [模拟] 调用 translate → 翻译完成")
    print(f"\n  📊 本地字典覆盖: {len(_TRANSLATION_DICT)} 条")
    print(f"    实际项目建议接 DeepL API 或 Google Translate API")

print("\n" + "=" * 60)
print("试试看: 练习 3 — Prompt JSON vs Tool Use 对比实验")
print("=" * 60)

# 练习 3: 对比 Prompt JSON 和 Tool Use 的成功率
INVOICE_FOR_TEST = """发票号 INV-2024-0891, 日期 2024-06-15,
商品: 机械键盘 ×2, 单价 ¥399, 鼠标 ×1, 单价 ¥149
合计: ¥947.00"""

JSON_SYSTEM_FOR_TEST = """你是一个数据提取工具。只输出 JSON, 不要任何额外文字。
输出格式: {"invoice_id": "...", "date": "...", "total": 0.0, "items": [{"name": "...", "quantity": 0, "unit_price": 0.0}]}"""

extract_invoice_tool = {
    "name": "extract_invoice",
    "description": "从文本中提取发票信息",
    "input_schema": {
        "type": "object",
        "properties": {
            "invoice_id": {"type": "string"},
            "date": {"type": "string"},
            "total": {"type": "number"},
            "items": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "unit_price": {"type": "number"},
                },
                "required": ["name", "quantity", "unit_price"]
            }}
        },
        "required": ["invoice_id", "date", "total", "items"]
    }
}

RUNS = 10
print(f"\n  🔬 对比实验: 每种方法跑 {RUNS} 次")
print(f"  输入文本: {INVOICE_FOR_TEST[:50]}...")

if api_ok:
    # 方法一: Prompt JSON
    print("\n  ── 方法一: Prompt JSON ──")
    json_success = 0
    json_errors = []
    for i in range(RUNS):
        raw = ask(INVOICE_FOR_TEST, system=JSON_SYSTEM_FOR_TEST,
                  max_tokens=300, temperature=0.0)
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
            data = json.loads(cleaned)
            if "invoice_id" in data and "total" in data:
                json_success += 1
            else:
                json_errors.append(f"缺少关键字段 (第 {i+1} 次)")
        except json.JSONDecodeError as e:
            json_errors.append(f"JSON 解析失败 (第 {i+1} 次): {str(e)[:40]}")
        print(f"    第 {i+1:2d}/10: {'✅' if i < json_success else '❌'}", end="")
        if (i + 1) % 5 == 0:
            print()
    print(f"\n  Prompt JSON 成功率: {json_success}/{RUNS} ({json_success * 10}%)")
    if json_errors:
        for err in json_errors[:3]:
            print(f"    失败原因: {err}")

    # 方法二: Tool Use
    print("\n  ── 方法二: Tool Use ──")
    tool_success = 0
    tool_errors = []
    for i in range(RUNS):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                temperature=0.0,
                system="你是数据提取助手。当用户提供发票时, 调用 extract_invoice 工具。",
                messages=[{"role": "user", "content": INVOICE_FOR_TEST}],
                tools=[extract_invoice_tool],
                tool_choice={"type": "any"},
            )
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            if tool_blocks and "invoice_id" in tool_blocks[0].input:
                tool_success += 1
            else:
                tool_errors.append(f"未提取到发票号 (第 {i+1} 次)")
        except Exception as e:
            tool_errors.append(f"API 调用失败 (第 {i+1} 次): {str(e)[:40]}")
        print(f"    第 {i+1:2d}/10: {'✅' if i < tool_success else '❌'}", end="")
        if (i + 1) % 5 == 0:
            print()
    print(f"\n  Tool Use 成功率: {tool_success}/{RUNS} ({tool_success * 10}%)")
    if tool_errors:
        for err in tool_errors[:3]:
            print(f"    失败原因: {err}")

    print(f"""
  📊 对比总结:
    Prompt JSON: {json_success}/{RUNS} ({json_success * 10}%)
    Tool Use:    {tool_success}/{RUNS} ({tool_success * 10}%)

    {'✅ 结论: Tool Use 更可靠!' if tool_success > json_success else '⚠️ 两者相当, 但 Tool Use 参数有 schema 保证'}
    Tool Use 参数由 JSON Schema 约束, 类型安全有保证。
    Prompt JSON 可能输出多余文字或格式问题。""")
else:
    print("""
  (模拟对比实验)
  ── 方法一: Prompt JSON ──
    第  1/10: ✅ 第  2/10: ✅ 第  3/10: ❌ 第  4/10: ✅ 第  5/10: ✅
    第  6/10: ❌ 第  7/10: ✅ 第  8/10: ✅ 第  9/10: ✅ 第 10/10: ✅
  Prompt JSON 成功率: 8/10 (80%)
    失败原因: 第 3 次模型输出了 "好的, 这是结果: {...}"
    失败原因: 第 6 次模型用了 ```json 代码块

  ── 方法二: Tool Use ──
    第  1/10: ✅ ... 第 10/10: ✅
  Tool Use 成功率: 10/10 (100%)

  📊 结论: Tool Use 在结构化提取上碾压 Prompt JSON。
    生产环境需要可靠的结构化输出 → 用 Tool Use。""")

print("\n" + "=" * 60)
print("试试看: 练习 4 — 多工具选择观察")
print("=" * 60)

# 练习 4: 注册 5 个工具, 问不同问题, 观察模型选了哪个
search_tool = {
    "name": "search_info",
    "description": "搜索互联网信息 (模拟)。返回搜索结果。",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"}
        },
        "required": ["query"]
    }
}


def search_info(query: str) -> dict:
    return {"query": query, "results": [f"关于 '{query}' 的搜索结果 (模拟)"], "count": 1}


time_tool = {
    "name": "get_current_time",
    "description": "获取当前日期、时间和星期。",
    "input_schema": {
        "type": "object",
        "properties": {
            "timezone": {"type": "string", "description": "时区"}
        }
    }
}


def get_current_time(timezone: str = "Asia/Shanghai") -> dict:
    return {"datetime": "2026-06-19 15:30:00", "weekday": "周五", "timezone": timezone}


# 创建多工具观察器
observer_chat = ToolUseChat(
    system="你是智能助手。根据用户问题选择合适的工具。可以一次调用多个工具。中文回复。",
    model="claude-sonnet-4-6",
)
observer_chat.register_tool(weather_tool, get_weather)
observer_chat.register_tool(calculator_tool, calculate)
observer_chat.register_tool(TRANSLATE_TOOL, handle_translate)
observer_chat.register_tool(time_tool, get_current_time)
observer_chat.register_tool(search_tool, search_info)

print(f"\n  已注册工具: weather, calculate, translate, time, search_info")
print(f"  观察模型如何选择工具:")

test_queries = [
    ("今天星期几?", "预期: get_current_time"),
    ("北京多少度?", "预期: get_weather"),
    ("把 Hello World 翻译成法语", "预期: translate"),
    ("3 的 8 次方", "预期: calculate"),
    ("搜索一下 Python 3.13 新特性", "预期: search_info"),
    ("北京今天天气多少度? 帮我算一下(华氏度)", "预期: get_weather + calculate"),
]

if api_ok:
    for query, expected in test_queries:
        print(f"\n  🧑: {query}")
        print(f"    预期工具: {expected}")
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                temperature=0.0,
                messages=[{"role": "user", "content": query}],
                tools=[weather_tool, calculator_tool, TRANSLATE_TOOL,
                       time_tool, search_tool],
            )
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            if tool_blocks:
                for tb in tool_blocks:
                    print(f"    ✅ 模型选择: {tb.name}({json.dumps(tb.input, ensure_ascii=False)[:60]})")
            else:
                print(f"    ⚠️  模型未调用工具, 直接回复了文本")
        except Exception as e:
            print(f"    ❌ 调用失败: {e}")
else:
    for query, expected in test_queries:
        print(f"\n  🧑: {query}")
        print(f"    预期工具: {expected}")

print("""
  💡 观察心得:
  1. 当问题明确匹配某个工具的描述, 模型准确选择
  2. 当问题涉及多项能力, 模型可能调用多个工具 (并行)
  3. 日常闲聊 (如"你好") 不会触发任何工具
  4. 工具的 description 字段至关重要 — 它是模型决定调不调的唯一依据""")

print("\n" + "=" * 60)
print("试试看: 练习 5 — 工具调用链 (挑战)")
print("=" * 60)

# 练习 5: 工具调用链 — 天气 + 活动推荐
ACTIVITY_TOOL = {
    "name": "recommend_activity",
    "description": "根据天气条件推荐活动。晴推荐户外, 雨推荐室内。",
    "input_schema": {
        "type": "object",
        "properties": {
            "weather_condition": {"type": "string", "description": "天气状况: 晴/多云/阴/雨/雪"},
            "temperature": {"type": "number", "description": "温度 (°C)"},
            "city": {"type": "string", "description": "城市名称"}
        },
        "required": ["weather_condition", "temperature", "city"]
    }
}


def recommend_activity(weather_condition: str, temperature: float,
                       city: str) -> dict:
    """根据天气推荐活动。"""
    if weather_condition in ("晴", "多云"):
        if temperature > 30:
            suggestion = "水上乐园或室内游泳馆 (天热)"
        elif temperature > 20:
            suggestion = "公园散步、骑行、户外烧烤"
        else:
            suggestion = "登山、户外跑步"
    elif weather_condition in ("雨", "雪", "雷阵雨"):
        suggestion = "电影院、博物馆、图书馆、咖啡馆"
    else:
        suggestion = "商场购物、健身房"

    return {
        "city": city,
        "weather": weather_condition,
        "temperature": temperature,
        "suggestion": f"{city}: {suggestion}",
    }


chain_chat = ToolUseChat(
    system="""你是生活规划助手。当用户问"某地适合什么活动"时:
1. 先调用 get_weather 查询天气
2. 拿到天气结果后, 调用 recommend_activity 获取推荐
3. 用中文给出最终建议""",
    model="claude-sonnet-4-6",
)
chain_chat.register_tool(weather_tool, get_weather)
chain_chat.register_tool(ACTIVITY_TOOL, recommend_activity)

print("\n  🔗 工具调用链演示: 天气 → 活动推荐")

if api_ok:
    chain_queries = [
        "北京今天适合什么活动?",
        "上海下雨的话有什么好的室内活动推荐?",
    ]
    for q in chain_queries:
        print(f"\n  🧑: {q}")
        reply = chain_chat.send(q)
        print(f"  🤖: {reply[:200]}")
else:
    print("""
  (模拟工具调用链)
  🧑: "北京适合什么活动?"
  Step 1 → get_weather({"city": "北京"}) → {"temp": 28, "condition": "晴"}
  Step 2 → recommend_activity({"weather_condition": "晴", "temperature": 28, "city": "北京"})
          → {"suggestion": "北京: 公园散步、骑行、户外烧烤"}
  最终回复: "北京今天晴, 28°C, 非常适合户外活动。推荐去公园散步或骑行!"

  💡 工具调用链的关键:
  1. ToolUseChat 的循环机制自动支持链式调用
  2. 关键在于 system prompt 引导模型"先 A 后 B"
  3. 工具 A 的输出 (tool_result) 会变成下一个决策的上下文""")

print("\n" + "=" * 60)
print("试试看: 练习 6 — Anthropic Tool Use 文档探索")
print("=" * 60)

# 练习 6: Anthropic Tool Use 文档
print("""
  📖 Anthropic Tool Use 官方文档:
  https://docs.anthropic.com/en/docs/build-with-claude/tool-use

  核心补充 (本课未深入):

  ┌────────────────────────┬──────────────────────────────────────────┐
  │ 特性                     │ 说明                                      │
  ├────────────────────────┼──────────────────────────────────────────┤
  │ tool_choice 选项 (已学)  │ auto / any / {"type":"tool","name":"x"}   │
  │ 并行工具调用             │ 模型可同时调用多个工具 (如同时查北京+上海天气) │
  │ 工具调用 + Thinking      │ 模型可在调用前展示推理过程                    │
  │ 工具结果格式             │ tool_result block: type + tool_use_id +    │
  │                          │ content (string 或 content blocks)         │
  │ 错误处理                 │ tool_result 的 is_error 字段标记执行失败      │
  │ 串行 vs 并行             │ 默认允许并行, 可用 disable_parallel_tool_use │
  └────────────────────────┴──────────────────────────────────────────┘

  关键 INSIGHT:
  1. tool_result 可以包含结构化 content blocks (不仅是字符串),
     支持返回图片、表格等富内容。

  2. 工具调用的 token 计费:
     - tool_use block 本身算 output tokens
     - tool_result 算 input tokens (下一轮)
     → 工具调用会增加 token 消耗, 但换来结构化输出

  3. 生产环境的最佳实践:
     - 工具 handler 要有超时控制 (httpx.Timeout)
     - 工具返回值要精简 (避免返回大段数据)
     - 用 is_error 标记执行失败, 让模型优雅降级

  4. 本课和官方文档的对照:
     - ✅ 覆盖了 90% 的日常用法
     - ⚠️  未深入: 并行工具调用控制、多模态 tool_result
     - 📖 建议浏览官方文档的 "Examples" 章节, 看实际场景""")

print("\n" + "=" * 60)
print("  Lesson 13 试试看练习全部完成!")
print("=" * 60)
