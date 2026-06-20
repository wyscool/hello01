# ============================================================
# Phase 2, Lesson 14: 流式响应 —— SSE、事件类型、实时输出
# ============================================================
#
# 本课目标:
#   1. 理解流式响应的底层: SSE (Server-Sent Events)
#   2. stream.text_stream — 最简单的文本流
#   3. stream 事件类型: message_start / content_block_delta / message_stop ...
#   4. 流式 + Tool Use — 工具调用在流中的样子
#   5. 构建实时终端聊天 (带打字效果)
#   6. OpenAI streaming 对照
#   7. 实战: 流式代码生成器
#   8. 实战: 带工具调用的流式聊天机器人
#
# 预计阅读 + 实操时间: 40-50 分钟
#
# 前置: 已完成 Lesson 11 (API 基础) + Lesson 13 (Tool Use)
# ============================================================

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from anthropic import Anthropic
from anthropic.types import Message
from anthropic.types import (
    RawMessageStartEvent,
    RawMessageDeltaEvent,
    RawMessageStopEvent,
    RawContentBlockStartEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStopEvent,
)


# ------------------------------------------------------------
# 〇、环境准备
# ------------------------------------------------------------

api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")

client_kwargs = {"api_key": api_key} if api_key else {}
if base_url:
    client_kwargs["base_url"] = base_url
client = Anthropic(**client_kwargs)


def _get_text(response: Message) -> str:
    parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts)


try:
    client.messages.create(
        model="claude-sonnet-4-6", max_tokens=10,
        messages=[{"role": "user", "content": "ping"}],
    )
    api_ok = True
    print("✅ API 连接正常\n")
except Exception:
    api_ok = False
    print("⚠️  API 不可用, 将以模拟模式运行\n")


# ============================================================
# 一、流式输出回顾 —— 从 Lesson 11 的基础说起
# ============================================================
# Lesson 11 演示了最简单的流式输出:
#
#   with client.messages.stream(...) as stream:
#       for text_chunk in stream.text_stream:
#           print(text_chunk, end="", flush=True)
#
# 这节课我们深入底层, 理解每一行背后的机制。
#
# 同步 vs 流式:
#   create()      → 发送请求 → 等 2 秒 → 收到完整 JSON → 解析 → 返回
#   stream()      → 发送请求 → 0.1 秒后收到第一个 chunk → 边收边处理
#
# 类比 Java:
#   create()  = RestTemplate.postForObject() — 等完整响应
#   stream()  = WebClient + Flux<DataBuffer> — 流式消费
#   SSE       = Spring 的 SseEmitter / Flux<ServerSentEvent>


# ------------------------------------------------------------
# 二、SSE 原理 —— 流式响应的底层
# ------------------------------------------------------------
# SSE (Server-Sent Events) 是一种 HTTP 长连接协议:
#
#   GET /v1/messages (stream=true)
#   →
#   event: message_start
#   data: {"type": "message_start", "message": {...}}
#
#   event: content_block_start
#   data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", ...}}
#
#   event: content_block_delta
#   data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "你好"}}
#
#   event: content_block_delta
#   data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}}
#
#   event: content_block_stop
#   data: {"type": "content_block_stop", "index": 0}
#
#   event: message_delta
#   data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {...}}
#
#   event: message_stop
#   data: {"type": "message_stop"}
#
# 每个事件是一行 JSON。SDK 帮你解析了, 但你也可以手动处理。

print("=" * 60)
print("SSE 事件类型一览")
print("=" * 60)

print("""
  ┌─────────────────────┬──────────────────────────────────────┐
  │ 事件类型              │ 含义                                  │
  ├─────────────────────┼──────────────────────────────────────┤
  │ message_start       │ 消息开始 (包含 message.id, model)      │
  │ content_block_start │ 内容块开始 (文本块 / 工具调用块)        │
  │ content_block_delta │ 内容增量 (一个 token 的文本 / JSON 片段)│
  │ content_block_stop  │ 内容块结束                             │
  │ message_delta       │ 消息增量 (stop_reason, usage)          │
  │ message_stop        │ 流结束                                 │
  └─────────────────────┴──────────────────────────────────────┘

  一个简单回复 ("你好!") 的事件序列:
    message_start → content_block_start → content_block_delta × N
    → content_block_stop → message_delta → message_stop
""")


# ------------------------------------------------------------
# 三、stream.text_stream —— 最简单的文本流
# ------------------------------------------------------------
# SDK 提供了 text_stream 迭代器, 屏蔽了所有事件细节,
# 每次 yield 一个文本 token。

print("=" * 60)
print("text_stream: 简单文本流")
print("=" * 60)

if api_ok:
    print("  实时输出: ", end="", flush=True)
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=150,
        temperature=0.7,
        messages=[{
            "role": "user",
            "content": "用一句话介绍 Python 的异步编程, 50 字以内"
        }],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            time.sleep(0.02)  # 模拟打字机效果, 实际项目不需要

    print()
    final = stream.get_final_message()
    print(f"\n  用量: {final.usage.input_tokens} + {final.usage.output_tokens} tokens")
else:
    print("  (模拟) 实时输出: Python 的 async/await 让你用同步写法写异步代码...")
    print("  text_stream 本质: 一个 Generator[str], 每次 yield 一个文本片段")


# ------------------------------------------------------------
# 四、遍历底层事件 —— 看到流式响应的全貌
# ------------------------------------------------------------
# text_stream 跳过了所有非文本事件。
# 如果你需要处理 tool_use、获取 usage 信息、监控进度,
# 你需要直接遍历 stream 本身。

print("\n" + "=" * 60)
print("底层事件遍历")
print("=" * 60)

if api_ok:
    print("  事件序列:")
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=50,
        temperature=0.0,
        messages=[{"role": "user", "content": "说 'Hello'"}],
    ) as stream:
        for event in stream:
            event_type = type(event).__name__
            if event_type == "RawMessageStartEvent":
                print(f"  🔵 message_start: id={event.message.id}, model={event.message.model}")
            elif event_type == "RawContentBlockStartEvent":
                cb = event.content_block
                print(f"  🟢 content_block_start: index={event.index}, type={cb.type}")
            elif event_type == "RawContentBlockDeltaEvent":
                delta = event.delta
                if delta.type == "text_delta":
                    print(f"  📝 delta: '{delta.text}'")
                elif delta.type == "input_json_delta":
                    print(f"  📝 delta (json): '{delta.partial_json}'")
            elif event_type == "RawContentBlockStopEvent":
                print(f"  🔴 content_block_stop: index={event.index}")
            elif event_type == "RawMessageDeltaEvent":
                print(f"  🟡 message_delta: stop_reason={event.delta.stop_reason}")
            elif event_type == "RawMessageStopEvent":
                print(f"  ⚫ message_stop")

    print(f"\n  总 input tokens: {stream.get_final_message().usage.input_tokens}")
else:
    print("""
  (模拟) 事件序列:
  🔵 message_start: id=msg_xxx, model=claude-sonnet-4-6
  🟢 content_block_start: index=0, type=text
  📝 delta: 'Hello'
  📝 delta: '!'
  🔴 content_block_stop: index=0
  🟡 message_delta: stop_reason=end_turn
  ⚫ message_stop
""")


# ------------------------------------------------------------
# 五、流式 + Tool Use —— 工具调用在流中的形态
# ------------------------------------------------------------
# 当模型决定调用工具时, 流中的事件序列会有所不同:
#   可能会有 thinking block (思考过程)
#   content_block_start (type=tool_use)
#   content_block_delta (type=input_json_delta) × N  — 参数逐个 token 到达
#   content_block_stop
#
# 这就带来一个问题: 你需要等 tool_use 的 JSON 完整到达,
# 才能解析参数、执行工具。

print("=" * 60)
print("流式 Tool Use")
print("=" * 60)

weather_tool = {
    "name": "get_weather",
    "description": "查询指定城市的天气",
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


if api_ok:
    print("\n  观察流中 tool_use 的事件形态:")
    print("  (注意 content_block 的 type 变化)")
    print()

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=300,
        temperature=0.0,
        messages=[{"role": "user", "content": "查一下北京天气"}],
        tools=[weather_tool],
    ) as stream:
        for event in stream:
            etype = type(event).__name__
            if etype == "RawContentBlockStartEvent":
                cb = event.content_block
                print(f"  🟢 block_start index={event.index} type={cb.type}")
                if cb.type == "tool_use":
                    print(f"      工具名: {cb.name}")
            elif etype == "RawContentBlockDeltaEvent":
                delta = event.delta
                if delta.type == "text_delta":
                    print(f"  📝 text: '{delta.text[:40]}'")
                elif delta.type == "input_json_delta":
                    print(f"  🔧 json_delta: '{delta.partial_json}'")
            elif etype == "RawContentBlockStopEvent":
                print(f"  🔴 block_stop index={event.index}")
            elif etype == "RawMessageDeltaEvent":
                print(f"  🟡 stop_reason: {event.delta.stop_reason}")

    # 获取最终的 tool_use 参数
    final_msg = stream.get_final_message()
    for block in final_msg.content:
        if block.type == "tool_use":
            print(f"\n  完整 tool_use 参数: {json.dumps(block.input, ensure_ascii=False)}")
else:
    print("""
  (模拟) 流式 Tool Use 事件:
  🟢 block_start index=0 type=text
  📝 text: '好的, 让我查一下...'
  🔴 block_stop index=0
  🟢 block_start index=1 type=tool_use
      工具名: get_weather
  🔧 json_delta: '{"city":'
  🔧 json_delta: '"北京"}'
  🔴 block_stop index=1
  🟡 stop_reason: tool_use
""")


# ------------------------------------------------------------
# 六、流式 Tool Use 完整循环 —— 边想边做
# ------------------------------------------------------------
# 把流式输出和 Tool Use 循环组合起来。
# 关键挑战: 流是不完整的, 需要 get_final_message() 拿到完整数据。

print("\n" + "=" * 60)
print("流式 Tool Use 完整循环")
print("=" * 60)


class StreamingToolChat:
    """
    结合流式输出和工具调用的聊天客户端。

    核心流程:
      1. stream() 创建流
      2. text_stream 实时打印文本
      3. get_final_message() 检查是否 tool_use
      4. 如果是 tool_use → 执行工具 → 把结果加入 messages → 回到步骤 1
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

    def send(self, user_input: str, max_rounds: int = 5) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(max_rounds):
            with client.messages.stream(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                messages=self.messages,
                tools=self._tools if self._tools else None,
            ) as stream:
                # 实时打印文本
                for text in stream.text_stream:
                    print(text, end="", flush=True)

                final = stream.get_final_message()

            if final.stop_reason == "end_turn":
                text = _get_text(final)
                self.messages.append({"role": "assistant", "content": text})
                return text

            elif final.stop_reason == "tool_use":
                # 收集 tool_use blocks, 执行工具
                tool_results = []
                assistant_content = []
                for block in final.content:
                    assistant_content.append(block.model_dump())
                    if block.type == "tool_use":
                        handler = self._handlers.get(block.name)
                        if handler:
                            try:
                                output = handler(**block.input)
                                content = json.dumps(output, ensure_ascii=False) if isinstance(output, dict) else str(output)
                            except Exception as e:
                                content = f"[工具执行失败: {e}]"
                        else:
                            content = f"[未注册工具: {block.name}]"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content,
                        })
                        print(f"\n  🔧 [{block.name}] → {content[:60]}...")

                self.messages.append({"role": "assistant", "content": assistant_content})
                self.messages.append({"role": "user", "content": tool_results})
                print()  # 换行
                continue

            else:
                return f"[未知 stop_reason: {final.stop_reason}]"

        return "[达到最大循环轮次]"


if api_ok:
    chat = StreamingToolChat(system="你是生活助手。天气用 get_weather, 计算用 calculate。中文回复。")
    chat.register_tool(weather_tool, get_weather)

    print("\n  💬 演示:")
    print("  🧑: 北京天气如何?")
    print("  🤖: ", end="", flush=True)
    chat.send("北京天气如何?")

else:
    print("""
  (模拟) 流式 Tool Use 循环:
  🧑: "北京天气如何?"
  🤖: 让我查一下...
  🔧 [get_weather] → {"temp": 28, "condition": "晴"}
  🤖: 北京今天晴, 28°C, 适合外出。
""")


# ------------------------------------------------------------
# 七、流式代码生成器 —— 实战
# ------------------------------------------------------------
# 一个经典场景: 流式输出代码, 像 Cursor/Copilot 那样。

print("=" * 60)
print("实战: 流式代码生成器")
print("=" * 60)

CODE_SYSTEM = """你是一个 Python 代码生成器。只输出代码, 不要解释。
代码要包含 type hints、docstring、关键注释。"""

REQUESTS = {
    "快速排序": "实现快速排序算法, 函数名 quicksort, 对 list[int] 排序",
    "单例装饰器": "写一个 singleton 装饰器, 确保类只有一个实例",
    "重试函数": "写一个 retry 装饰器, 支持最大重试次数和指数退避",
}

if api_ok:
    for name, prompt in REQUESTS.items():
        print(f"\n  📝 {name}:")
        print(f"  {'─' * 40}")
        print("  ", end="", flush=True)
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=2000,  # 给足 token, thinking 和输出各不耽误
            temperature=0.0,
            system=CODE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for event in stream:
                etype = type(event).__name__
                if etype == "RawContentBlockDeltaEvent":
                    delta = event.delta
                    if delta.type == "text_delta":
                        print(delta.text, end="", flush=True)
                    elif delta.type == "thinking_delta":
                        # 思考过程不打印, 但知道它在"想"
                        pass
                elif etype == "RawContentBlockStartEvent":
                    if event.content_block.type == "thinking":
                        print("[思考中...] ", end="", flush=True)
        print()
else:
    print("\n  (模拟) 流式代码生成")
    print("""
  📝 快速排序:
    def quicksort(arr: list[int]) -> list[int]:
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
""")


# ------------------------------------------------------------
# 八、带工具的流式代码助手 —— 综合实战
# ------------------------------------------------------------
# 结合 streaming + tool use: 一个能查文档、能执行代码的助手。

print("=" * 60)
print("综合实战: 流式代码助手")
print("=" * 60)

# 工具 1: 代码审查
code_review_tool = {
    "name": "review_code",
    "description": "审查 Python 代码, 检查潜在问题",
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "要审查的 Python 代码"},
            "focus": {
                "type": "string",
                "enum": ["security", "performance", "style", "all"],
                "description": "审查重点"
            }
        },
        "required": ["code"]
    }
}


def review_code(code: str, focus: str = "all") -> dict:
    """模拟代码审查。实际项目可以调 linter / SAST 工具。"""
    issues = []
    if "eval(" in code or "exec(" in code:
        issues.append({"severity": "high", "type": "security", "msg": "使用了 eval/exec, 存在代码注入风险"})
    if "except:" in code or "except Exception:" in code:
        if "raise" not in code:
            issues.append({"severity": "medium", "type": "style", "msg": "裸 except 吞掉了异常, 建议至少记录日志"})
    if "range(len(" in code:
        issues.append({"severity": "low", "type": "style", "msg": "建议用 enumerate() 代替 range(len())"})
    if not issues:
        issues.append({"severity": "info", "type": "all", "msg": "未发现明显问题"})
    return {"focus": focus, "issues": issues}


# 工具 2: 执行 Python 代码 (安全沙箱)
run_python_tool = {
    "name": "run_python",
    "description": "在沙箱中执行 Python 代码, 返回输出结果",
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "要执行的 Python 代码"},
        },
        "required": ["code"]
    }
}


def run_python(code: str) -> dict:
    """安全执行 Python 代码。"""
    import io
    allowed_builtins = {
        "print": print, "len": len, "range": range, "list": list,
        "dict": dict, "set": set, "tuple": tuple, "str": str,
        "int": int, "float": float, "bool": bool, "sum": sum,
        "min": min, "max": max, "sorted": sorted, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter, "abs": abs,
        "round": round, "isinstance": isinstance, "type": type,
    }
    try:
        buf = io.StringIO()
        _print = lambda *a, **kw: print(*a, **kw, file=buf)
        safe_builtins = {**allowed_builtins, "print": _print}
        exec(code, {"__builtins__": safe_builtins}, {})
        output = buf.getvalue()
        return {"success": True, "output": output.strip() or "(无输出)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if api_ok:
    assistant = StreamingToolChat(
        system="你是 Python 编程助手。审查代码用 review_code, 运行代码用 run_python。中文回复。",
        model="claude-sonnet-4-6",
    )
    assistant.register_tool(code_review_tool, review_code)
    assistant.register_tool(run_python_tool, run_python)

    tasks = [
        "审查这段代码: \ndef double_items(lst):\n    result = []\n    for i in range(len(lst)):\n        result.append(lst[i] * 2)\n    return result",
        "执行以下代码并告诉我结果: \nprint(sum(range(1, 101)))",
    ]

    for task in tasks:
        print(f"\n  🧑: {task[:50]}...")
        print("  🤖: ", end="", flush=True)
        assistant.send(task)
        print()

else:
    print("""
  (模拟) 流式代码助手对话:

  🧑: 审查这段代码...
  🤖: 让我审查一下...
  🔧 [review_code] → {"issues": [{"severity": "low", ...}]}
  🤖: 发现以下问题:
  1. 第 3 行: 建议用 enumerate() 代替 range(len())
  2. 代码风格良好, 无安全问题。

  🧑: 执行代码 print(sum(range(1, 101)))
  🤖: 让我运行...
  🔧 [run_python] → {"success": true, "output": "5050"}
  🤖: 执行成功! 结果是 5050 (1 到 100 的和)。
""")


# ------------------------------------------------------------
# 九、OpenAI Streaming 对照
# ------------------------------------------------------------

print("=" * 60)
print("OpenAI Streaming 对照")
print("=" * 60)

print("""
  Anthropic (stream):
  ┌──────────────────────────────────────────────┐
  │ with client.messages.stream(...) as stream:   │
  │     for text in stream.text_stream:           │
  │         print(text, end="")                   │
  │                                               │
  │     final = stream.get_final_message()        │
  └──────────────────────────────────────────────┘

  OpenAI (stream):
  ┌──────────────────────────────────────────────┐
  │ stream = client.chat.completions.create(      │
  │     ..., stream=True                          │
  │ )                                             │
  │ for chunk in stream:                          │
  │     if chunk.choices[0].delta.content:        │
  │         print(chunk.choices[0].delta.content, │
  │               end="")                         │
  └──────────────────────────────────────────────┘

  核心差异:
    Anthropic: stream.text_stream (生成器) → 最方便
               stream 本身 → 完整事件 (用于 tool_use)
    OpenAI:    for chunk in stream → 每个 chunk 包含 choices[0].delta
               delta.content → 文本 token
               delta.tool_calls → 工具调用参数 (也是增量)

  Anthropic 的 text_stream 更方便,
  OpenAI 需要手动判断 delta 类型。
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Lesson 14 完成! 流式响应已掌握。")
    print("=" * 60)
    print(f"""
  回顾: 你学会了什么?

  1. SSE 原理 — HTTP 长连接, 每个 chunk 是一行 JSON
  2. stream.text_stream — 最简单的流式文本消费
  3. 事件类型 — 6 种事件, 覆盖流式响应的完整生命周期
  4. 流式 + Tool Use — 工具参数以 input_json_delta 形式到达
  5. StreamingToolChat — 流式输出 + 自动工具调用循环
  6. 流式代码生成 — 像 Cursor/Copilot 那样边想边输出

  关键理解:
    非流式 = 等菜上齐再吃
    流式   = 边做边上菜

    text_stream 够用 80% 场景。
    剩下 20% 需要处理 tool_use, 才需要遍历底层事件。

  🎯 下一课预告: Lesson 15 — 构建一个对话应用 (Phase 2 收官项目)
     融合 Prompt + 工具调用 + 流式输出, 做出第一个完整的 AI 应用!
""")


# ============================================================
# 试试看 (Try This) —— 练习实现
# ============================================================

print("\n" + "=" * 60)
print("试试看: 练习 1 — 打字速度控制")
print("=" * 60)

# 练习 1: 给 StreamingToolChat 添加打字速度控制
# 直接定义一个速度可配置的流式输出函数来演示


def stream_with_speed(prompt: str, char_delay: float = 0.0,
                      model: str = "claude-sonnet-4-6") -> None:
    """
    带速度控制的流式输出。

    char_delay=0:    瞬间输出 (无延迟)
    char_delay=0.02: 舒适阅读速度
    char_delay=0.05: 慢速打字效果

    实际项目中应该加延迟吗?
    - 不建议! API 返回速度已经足够快, 额外延迟会降低用户体验
    - 这个功能主要用于演示和教学, 帮你理解"流"的感觉
    - 唯一的例外: 如果输出速度太快 (如代码生成), 可以加 0.01s
      但更好的做法是让前端/终端自己控制渲染节奏
    """
    print(f"  延迟={char_delay}s: ", end="", flush=True)
    if api_ok:
        with client.messages.stream(
            model=model,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            char_count = 0
            for text in stream.text_stream:
                print(text, end="", flush=True)
                if char_delay > 0:
                    time.sleep(char_delay)
                char_count += len(text)
        print()
        print(f"  输出 {char_count} 字符")
    else:
        # 模拟不同速度
        demo_text = "Python 的流式输出让你可以边生成边阅读, 就像有人在屏幕那头打字一样。"
        for speed, label in [(0.0, "无延迟"), (0.02, "正常速度"), (0.05, "慢速")]:
            print(f"\n  {label}: ", end="", flush=True)
            for ch in demo_text:
                print(ch, end="", flush=True)
                time.sleep(speed)
            print()


print("\n  对比三种打字速度:")
print("  (API 不可用时用模拟文本演示)")
stream_with_speed("用一句话介绍 Python 异步编程", char_delay=0.0)
stream_with_speed("用一句话介绍 Python 异步编程", char_delay=0.02)
stream_with_speed("用一句话介绍 Python 异步编程", char_delay=0.05)

print("""
  💡 思考: 实际项目中应该加延迟吗?
  答案: 不应该! 原因:
  1. API 的返回速度就是用户的期望速度 — 用户想看多快, API 就有多快
  2. 人为加延迟 = 故意让用户体验变差
  3. 真正的优化方向是: 优化 first-token latency, 而不是添加延迟
  4. 唯一例外: 教学演示、打字机效果展示""")

print("\n" + "=" * 60)
print("试试看: 练习 2 — 流式翻译器")
print("=" * 60)

# 练习 2: 流式翻译器


def stream_translate(text: str, target_lang: str = "英文",
                     model: str = "claude-sonnet-4-6") -> str:
    """
    流式翻译器: 输入文本, 流式输出翻译结果。

    用户体验对比:
    - 非流式: 等待 2-5 秒 → 一次性显示全部翻译
    - 流式:   立即看到第一个词 → 逐词呈现 → 感受"翻译正在进行"
    """
    system = f"你是一个翻译器。将用户输入翻译成{target_lang}。只输出翻译结果, 不要额外解释。"

    full_text = ""
    if api_ok:
        print(f"  🤖 翻译 ({target_lang}): ", end="", flush=True)
        start = time.time()
        with client.messages.stream(
            model=model,
            max_tokens=500,
            temperature=0.3,
            system=system,
            messages=[{"role": "user", "content": text}],
        ) as stream:
            for text_chunk in stream.text_stream:
                print(text_chunk, end="", flush=True)
                full_text += text_chunk
        elapsed = time.time() - start
        print(f"\n  翻译耗时: {elapsed:.1f}s")
    else:
        # 模拟流式翻译
        mock_translation = {
            "Python 是一种解释型、面向对象的高级编程语言, 以其简洁的语法和强大的标准库而闻名。":
            "Python is an interpreted, object-oriented, high-level programming language, "
            "known for its concise syntax and powerful standard library.",
            "今天天气真好, 适合出去走走。":
            "The weather is really nice today, perfect for going out for a walk.",
        }
        translated = mock_translation.get(text, f"[Translation of: {text[:30]}...]")
        print(f"  🤖 翻译 ({target_lang}): ", end="", flush=True)
        for ch in translated:
            print(ch, end="", flush=True)
            time.sleep(0.02)
        full_text = translated
        print()
    return full_text


print("\n  🌐 流式翻译演示:")
texts = [
    "Python 是一种解释型、面向对象的高级编程语言, 以其简洁的语法和强大的标准库而闻名。",
    "今天天气真好, 适合出去走走。",
]

for t in texts:
    print(f"\n  📝 原文: {t}")
    stream_translate(t, "英文")
    time.sleep(0.3)

# 扩展: 交互式流式翻译
print("\n  💡 扩展思路——交互式流式翻译器:")
print("""
  while True:
      text = input("输入中文 (q 退出): ")
      if text == 'q':
          break
      stream_translate(text, "英文")

  适用场景:
  - 快速翻译长段落, 边看边理解
  - 翻译内容不确定时, 流式输出让你更快判断"翻译方向对不对\"""")

print("\n" + "=" * 60)
print("试试看: 练习 3 — 首字延迟 (First-Token Latency) 对比")
print("=" * 60)

# 练习 3: 测量流式 vs 非流式的首字延迟
PROMPT_FOR_LATENCY = "用 50 个字解释 Python 的装饰器"

print(f"\n  🔬 测量: \"{PROMPT_FOR_LATENCY}\"")
print(f"  跑 3 次, 记录首字延迟")

if api_ok:
    print("\n  ── 非流式 (create) ──")
    non_stream_times = []
    for i in range(3):
        start = time.time()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            temperature=0.0,
            messages=[{"role": "user", "content": PROMPT_FOR_LATENCY}],
        )
        elapsed = time.time() - start
        non_stream_times.append(elapsed)
        text = _get_text(response)
        print(f"    第 {i+1} 次: {elapsed:.3f}s (首个词: '{text[:15]}...')")

    print(f"  非流式平均总延迟: {sum(non_stream_times) / len(non_stream_times):.3f}s")

    print("\n  ── 流式 (stream) ──")
    stream_first_token_times = []
    for i in range(3):
        first_token_time = None
        start = time.time()
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=100,
            temperature=0.0,
            messages=[{"role": "user", "content": PROMPT_FOR_LATENCY}],
        ) as stream:
            for text in stream.text_stream:
                if first_token_time is None:
                    first_token_time = time.time() - start
                    first_token_text = text
                # 不打印, 只测量
        stream_first_token_times.append(first_token_time or 0)
        print(f"    第 {i+1} 次: 首 token {first_token_time:.3f}s "
              f"(第一个词: '{first_token_text[:15] if first_token_text else 'N/A'}...')")

    avg_ft = sum(stream_first_token_times) / len(stream_first_token_times)
    print(f"  流式平均首 token 延迟: {avg_ft:.3f}s")

    total_stream_times = []
    for i in range(3):
        start = time.time()
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=100,
            temperature=0.0,
            messages=[{"role": "user", "content": PROMPT_FOR_LATENCY}],
        ) as stream:
            for _ in stream.text_stream:
                pass
            stream.get_final_message()
        total_stream_times.append(time.time() - start)

    avg_total_stream = sum(total_stream_times) / len(total_stream_times)
    avg_total_nonstream = sum(non_stream_times) / len(non_stream_times)
    print(f"  流式平均总耗时: {avg_total_stream:.3f}s")
    print(f"""
  📊 延迟对比总结:
    非流式首次可读: {avg_total_nonstream:.3f}s (必须等完整回复)
    流式首 token:   {avg_ft:.3f}s (立刻看到第一个词!)
    流式完整回复:   {avg_total_stream:.3f}s

    用户体验差异:
    - 非流式: 用户盯着空白等 {avg_total_nonstream:.1f}s
    - 流式:   用户 {avg_ft:.1f}s 后就开始阅读理解
    - 心理感受: 流式"感觉快"是因为人在阅读时 AI 在持续生成

    💡 这就像:
    - 非流式 = 等服务员把所有菜做好一次性上桌
    - 流式   = 边做边上菜, 做完一道上一道""")
else:
    print("""
  (模拟测量结果)
  ── 非流式 ──
    第 1 次: 2.350s (首个词: 'Python 装饰器是...')
    第 2 次: 2.180s
    第 3 次: 2.420s
  平均总延迟: 2.317s

  ── 流式 ──
    第 1 次: 首 token 0.620s (第一个词: 'Python')
    第 2 次: 首 token 0.580s
    第 3 次: 首 token 0.610s
  平均首 token 延迟: 0.603s
  平均总耗时: 2.350s

  📊 结论:
  - 首 token 延迟是总量延迟的 ~25%
  - 流式让用户在 0.6s 后就能看到内容
  - 这是流式最大的 UX 优势: 感知性能 >> 实际性能""")

print("\n" + "=" * 60)
print("试试看: 练习 4 — 思考过程可视化")
print("=" * 60)

# 练习 4: 可视化 AI 的 "思考过程"
print("\n  🧠 演示: 区分 thinking 和正式回复")

if api_ok:
    print("  提问: Python 的 GIL 是什么?")
    print("  ", end="", flush=True)

    thinking_text = ""
    reply_text = ""

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": "Python 的 GIL 是什么? 用 100 字解释"
        }],
    ) as stream:
        for event in stream:
            etype = type(event).__name__
            if etype == "RawContentBlockStartEvent":
                if event.content_block.type == "thinking":
                    print("\n  ┌─ 思考过程 ──────────────────────┐")
                    print("  │ ", end="", flush=True)
            elif etype == "RawContentBlockDeltaEvent":
                delta = event.delta
                if delta.type == "thinking_delta":
                    thinking_text += delta.thinking
                    # 用灰色缩进显示 thinking
                    print(delta.thinking, end="", flush=True)
                elif delta.type == "text_delta":
                    if thinking_text and not reply_text:
                        print("\n  └──────────────────────────────────┘")
                        print("  ┌─ 正式回复 ──────────────────────┐")
                        print("  │ ", end="", flush=True)
                    reply_text += delta.text
                    print(delta.text, end="", flush=True)
            elif etype == "RawContentBlockStopEvent":
                if reply_text:
                    print("\n  └──────────────────────────────────┘")
    print()
    if thinking_text:
        print(f"  📊 思考过程: {len(thinking_text)} 字符")
    print(f"  📊 正式回复: {len(reply_text)} 字符")
else:
    print("""
  (模拟 thinking 可视化)
  提问: Python 的 GIL 是什么?

  ┌─ 思考过程 ──────────────────────┐
  │ 用户问 GIL, 我需要:
  │ 1. 解释 GIL 全称 (Global Interpreter Lock)
  │ 2. 说明它为什么存在 (CPython 内存管理)
  │ 3. 对比 Java 的线程模型
  │ 4. 控制在 100 字以内
  │ 5. 用中文回复, 技术准确但通俗
  └──────────────────────────────────┘
  ┌─ 正式回复 ──────────────────────┐
  │ GIL (全局解释器锁) 是 CPython 的机制,
  │ 确保同一时刻只有一个线程执行 Python 字节码。
  │ 类比 Java: Java 支持真正的多线程并行,
  │ 而 Python 的多线程受 GIL 限制, 适合 IO 密集型任务。
  └──────────────────────────────────┘

  💡 Thinking 的应用场景:
  - 复杂推理任务, 让用户看到 AI 的思考路径
  - 教学: 学生可以看到"专家是如何思考的"
  - 调试: 当 AI 输出不满意时, 查看 thinking 定位问题
  - 注意: thinking 计费但不直接展示给用户 (控制权在开发者)""")

print("\n" + "=" * 60)
print("试试看: 练习 5 — 可中断的流式输出 (挑战)")
print("=" * 60)

# 练习 5: 可中断的流式输出


def interruptible_stream(prompt: str, model: str = "claude-sonnet-4-6") -> None:
    """
    可中断的流式输出演示。

    虽然 stream 上下文管理器不支持真正的中断 (因为 HTTP 请求已发出),
    但我们可以:
    1. 捕获 KeyboardInterrupt, 停止消费剩余的 stream
    2. 打印 "[已中断]", 让用户知道部分回复已被丢弃
    3. 实际 project 中可以用 httpx 的 cancel token 发 HTTP 取消请求

    模拟方案: 在流式循环中检测一个"中断标志",
    用 signal 或 threading.Event 模拟用户中断。
    """
    import threading
    import signal

    interrupted = threading.Event()

    # 模拟: 设置一个定时器在 1.5 秒后"中断"
    def simulate_interrupt():
        time.sleep(1.5)
        interrupted.set()

    print("  🧪 模拟中断: 1.5 秒后自动中断")
    print(f"  🤖: ", end="", flush=True)

    if api_ok:
        response_chars = 0
        try:
            threading.Thread(target=simulate_interrupt, daemon=True).start()
            with client.messages.stream(
                model=model,
                max_tokens=300,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    if interrupted.is_set():
                        print(f"\n  ⚡ [用户中断] (已输出 {response_chars} 字符)")
                        # 不消费剩余的 stream → 相当于丢弃后续 token
                        # 注意: API 已经生成了全部 token, 费用已经产生
                        print(f"  ⚠️  提示: 虽中断显示, 但 API 已消费全部生成 token")
                        return
                    print(text, end="", flush=True)
                    response_chars += len(text)
            print()
        except Exception as e:
            print(f"\n  ❌ 流错误: {e}")
    else:
        # 模拟流式输出 + 中断
        demo_response = ("Python 的流式输出非常有用的特性之一。"
                         "它让你可以在收到完整响应之前就开始处理数据。"
                         "这在构建聊天应用时特别重要, 因为用户不希望等待 5 秒才能看到回复。"
                         "流式输出通过 SSE (Server-Sent Events) 实现, "
                         "每个事件包含一个增量 token。")
        char_count = 0
        for ch in demo_response:
            if interrupted.is_set():
                print(f"\n  ⚡ [用户中断] (已输出 {char_count} 字符)")
                print(f"  ⚠️  提示: 虽中断显示, 但 API 已消费全部生成 token")
                return
            print(ch, end="", flush=True)
            time.sleep(0.03)
            char_count += 1

        # 模拟 KeyboardInterrupt
        if char_count > len(demo_response) * 0.4:
            print(f"\n  ⚡ [用户中断] (已输出 {char_count} 字符)")
            print(f"  ⚠️  提示: 虽中断显示, 但 API 已消费全部生成 token")
            return
        print()


print()
interruptible_stream("详细解释 Python 流式输出的优点, 200 字")

print("""
  💡 中断流式输出的关键技术点:
  1. 在消费 text_stream 的循环中检测中断条件
  2. 中断后, 已消费的 token 仍然计费
  3. 真正取消 HTTP 请求需要 httpx 的取消机制 (signal/handle)
  4. 实际生产中, 推荐用 asyncio + httpx 实现真正的取消
  5. KeyboardInterrupt 在主线程可以被捕获, 流式循环中也能响应""")

print("\n" + "=" * 60)
print("试试看: 练习 6 — Anthropic Streaming 文档探索")
print("=" * 60)

# 练习 6: Anthropic Streaming 文档
print("""
  📖 Anthropic Streaming 官方文档:
  https://docs.anthropic.com/en/docs/build-with-claude/streaming

  核心补充 (本课未深入):

  ┌────────────────────────┬──────────────────────────────────────────┐
  │ 特性                     │ 说明                                      │
  ├────────────────────────┼──────────────────────────────────────────┤
  │ text_stream (已学)       │ 最简单的文本流, 跳过所有非文本事件          │
  │ 底层事件遍历 (已学)      │ for event in stream → 6 种事件类型         │
  │ 流式 Tool Use (已学)     │ input_json_delta 增量到达                  │
  │ get_final_message (已学) │ 流结束后获取完整的 Message 对象            │
  ├────────────────────────┼──────────────────────────────────────────┤
  │ 🔑 错误处理              │ stream 中的异常通过 event 传递, 不是 throw  │
  │ 🔑 重连机制              │ 流断开后无法恢复, 需重新请求                  │
  │ 🔑 SDK 内存管理           │ text_stream 是懒加载的 Generator, 不缓存全文│
  │ 🔑 超时控制              │ stream(timeout=...) 设置流超时              │
  │ 🔑 中途取消              │ 建议用 asyncio + httpx 的 cancel 机制       │
  └────────────────────────┴──────────────────────────────────────────┘

  补充的最佳实践:
  1. 流式 + Tool Use 的模式:
     用 text_stream 实时展示, 用 get_final_message() 检查 tool_use。
     不要在流中尝试"实时"解析 tool_use (JSON 不完整)。

  2. 错误处理:
     流式调用可能在中途断开 (网络问题)。
     生产代码应捕获 stream 中的异常, 并使用 get_final_message()
     或自己维护的 partial_text 做降级处理。

  3. 性能考虑:
     text_stream 每次 yield 的文本长度不固定 (1 到几十个字符)。
     前端渲染时注意批量更新, 避免每个 token 触发一次 DOM 重绘。

  4. 与普通 create() 的选择:
     - 聊天应用 / 代码生成: 必须用 stream
     - 后台批处理 / 数据提取: 用 create 更简单
     - Tool Use 单一调用: create 足够

  建议: 浏览官方文档的 "Error handling" 和 "Timeouts" 章节。""")

print("\n" + "=" * 60)
print("  Lesson 14 试试看练习全部完成!")
print("=" * 60)
