# ============================================================
# Phase 2, Lesson 15: 构建对话应用 —— Phase 2 收官项目
# ============================================================
#
# 本课目标:
#   融合 Phase 2 全部技能，构建一个完整的终端对话应用 —— PyChat。
#   你能实际使用它，也能继续扩展它。
#
#   融合的技能:
#     Lesson 11: API 基础调用 + 多轮对话
#     Lesson 12: System Prompt 设计 + 模板化
#     Lesson 13: Tool Use + 结构化输出
#     Lesson 14: 流式响应 + 事件处理
#
#   新增知识:
#     1. 交互式 CLI 循环 (input() + 信号处理)
#     2. 对话历史管理 + token 估算 + 自动裁剪
#     3. 命令系统 (/help, /clear, /system, /tools ...)
#     4. 多工具注册与调度
#     5. 终端颜色与格式化输出
#     6. 应用架构: 单一入口, 分层设计
#
# 预计阅读 + 实操时间: 50-60 分钟
#
# 前置: 已完成 Lesson 11-14
# ============================================================

import os
import sys
import json
import textwrap
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from anthropic import Anthropic


# ============================================================
# 〇、终端颜色工具 —— 让输出好看一点
# ============================================================
# 用 ANSI 转义码给终端输出加颜色。不依赖第三方库。

class Color:
    """ANSI 颜色码。类比: Java 的 ANSI 常量类。"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    @staticmethod
    def style(text: str, *codes: str) -> str:
        return "".join(codes) + text + Color.RESET


# ============================================================
# 一、工具库 —— 定义 PyChat 可用的工具
# ============================================================
# 每个工具由两部分组成:
#   1. tool_def   — Anthropic tool schema (告诉模型怎么调用)
#   2. handler    — 实际执行的函数 (你的代码)

# --- 工具 1: 天气查询 ---

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "查询指定城市的当前天气。返回温度、天气状况、湿度。",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称, 如 北京、上海"}
        },
        "required": ["city"]
    }
}

_WEATHER_DATA = {
    "北京":   {"temp": 28, "condition": "晴", "humidity": 45},
    "上海":   {"temp": 32, "condition": "多云", "humidity": 70},
    "广州":   {"temp": 33, "condition": "雷阵雨", "humidity": 85},
    "深圳":   {"temp": 31, "condition": "阴", "humidity": 78},
    "杭州":   {"temp": 30, "condition": "小雨", "humidity": 80},
    "成都":   {"temp": 26, "condition": "阴", "humidity": 65},
    "东京":   {"temp": 24, "condition": "晴", "humidity": 55},
    "新加坡": {"temp": 31, "condition": "雷阵雨", "humidity": 90},
}


def handle_weather(city: str) -> dict:
    result = _WEATHER_DATA.get(city, {"temp": 22, "condition": "未知", "humidity": 50})
    return {"city": city, **result, "unit": "celsius"}


# --- 工具 2: 计算器 ---

CALC_TOOL = {
    "name": "calculate",
    "description": "执行数学计算。支持 +、-、*、/、** (幂)、% (取余)、// (整除)。例如 '2 + 3 * 4'。",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式"}
        },
        "required": ["expression"]
    }
}

_ALLOWED_CHARS = set("0123456789+-*/().%^ eEjpiPI")


def handle_calculate(expression: str) -> dict:
    if not all(c in _ALLOWED_CHARS for c in expression):
        return {"error": f"表达式包含不允许的字符。允许: 数字、运算符、括号"}
    try:
        import math
        safe = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "int": int, "float": float,
            "math": math,
        }
        result = eval(expression, {"__builtins__": {}}, safe)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


# --- 工具 3: 当前时间 ---

TIME_TOOL = {
    "name": "get_current_time",
    "description": "获取当前日期和时间, 以及星期几。不需要参数。",
    "input_schema": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "时区, 如 Asia/Shanghai、America/New_York"
            }
        },
        "required": []
    }
}

_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def handle_time(timezone: str = "Asia/Shanghai") -> dict:
    now = datetime.now()
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": _WEEKDAYS[now.weekday()],
        "timezone": timezone,
    }


# --- 工具 4: 执行 Python (沙箱) ---

RUN_TOOL = {
    "name": "run_python",
    "description": "在安全沙箱中执行 Python 代码, 返回输出结果。用于验证代码逻辑、计算结果。",
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python 代码"},
        },
        "required": ["code"]
    }
}


def handle_run(code: str) -> dict:
    import io
    allowed = {
        "print": print, "len": len, "range": range, "list": list,
        "dict": dict, "set": set, "tuple": tuple, "str": str,
        "int": int, "float": float, "bool": bool, "sum": sum,
        "min": min, "max": max, "sorted": sorted, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter, "abs": abs,
        "round": round, "isinstance": isinstance, "type": type,
        "reversed": reversed, "any": any, "all": all,
        "bin": bin, "hex": hex, "oct": oct, "ord": ord, "chr": chr,
        "format": format, "divmod": divmod, "pow": pow,
        "complex": complex, "slice": slice, "iter": iter, "next": next,
    }
    try:
        buf = io.StringIO()
        safe_builtins = {**allowed, "print": lambda *a, **kw: print(*a, **kw, file=buf)}
        exec(code, {"__builtins__": safe_builtins}, {})
        output = buf.getvalue()
        return {"success": True, "output": output.strip() or "(无输出)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# 工具注册表: name → (tool_def, handler)
TOOL_REGISTRY: dict[str, tuple[dict, callable]] = {
    "get_weather":     (WEATHER_TOOL, handle_weather),
    "calculate":       (CALC_TOOL, handle_calculate),
    "get_current_time": (TIME_TOOL, handle_time),
    "run_python":      (RUN_TOOL, handle_run),
}

# --- 工具 5: 翻译器 ---
TRANSLATE_TOOL = {
    "name": "translate",
    "description": "将文本翻译成目标语言。支持中英日韩法德等常见语言互译。",
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要翻译的文本"},
            "target_lang": {
                "type": "string",
                "enum": ["zh", "en", "ja", "ko", "fr", "de"],
                "description": "目标语言代码: zh(中文)/en(英文)/ja(日语)/ko(韩语)/fr(法语)/de(德语)"
            }
        },
        "required": ["text", "target_lang"]
    }
}

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
    ("你好世界", "en"): "Hello World",
    ("谢谢", "en"): "Thank you",
    ("早上好", "en"): "Good morning",
}


def handle_translate(text: str, target_lang: str) -> dict:
    """模拟翻译。实际项目可接入 Google Translate / DeepL API。"""
    key = (text, target_lang)
    lang_names = {"zh": "中文", "en": "英文", "ja": "日语", "ko": "韩语",
                  "fr": "法语", "de": "德语"}
    if key in _TRANSLATION_DICT:
        translated = _TRANSLATION_DICT[key]
        method = "local_dict"
    else:
        translated = f"[{text}] → ({target_lang})"
        method = "simulated"
    return {
        "original": text,
        "translated": translated,
        "target_lang": target_lang,
        "target_lang_name": lang_names.get(target_lang, target_lang),
        "method": method,
    }

TOOL_REGISTRY["translate"] = (TRANSLATE_TOOL, handle_translate)

# 预置角色 (用于 /role 切换)
PRESET_ROLES: dict[str, str] = {
    "代码审查专家": """你是一位资深的代码审查专家。
对于任何代码, 你需要:
1. 找到至少 3 个潜在问题 (安全、逻辑、性能、可读性)
2. 按严重程度排序
3. 每个问题给出具体的修改建议
4. 用中文回复, 专业但友善""",

    "技术面试官": """你是一位资深 Python 技术面试官, 正在面试一个 Java 后端工程师。
面试规则:
1. 每次问一个 Python 问题, 由浅入深
2. 用 Java 做类比辅助理解
3. 根据候选人的回答, 调整下一个问题的难度
4. 用中文对话, 保持面试氛围 (严肃但不压迫)""",

    "创意写作助手": """你是一位创意写作助手, 擅长技术博客和文档写作。
写作风格:
1. 生动有趣, 善用比喻和故事
2. 技术内容准确但不枯燥
3. 适合技术博客和公众号文章
4. 用中文写作, 可以适当幽默""",
}


# ============================================================
# 二、PyChat —— 对话应用核心类
# ============================================================
# 融合了 Lesson 11-14 的所有概念:
#   - messages 数组维护对话历史 (L11)
#   - system prompt 控制行为 (L12)
#   - tool registry + tool_use 循环 (L13)
#   - stream + text_stream 实时输出 (L14)
#
# 架构层次:
#   PyChat.send()           ← 入口: 发消息, 返回回复
#     → _stream_reply()     ← 流式调用 API
#     → _execute_tools()    ← 处理 tool_use blocks
#     → _trim_history()     ← 上下文过长时裁剪
#
# 类比 Java:
#   PyChat ≈ Spring Service + Session State
#   TOOL_REGISTRY ≈ @Component Map<String, Function>
#   send() ≈ @PostMapping("/chat") 的 handler

class PyChat:
    """终端对话应用核心。"""

    DEFAULT_SYSTEM = """你是一个友好、知识渊博的 AI 助手, 叫 PyChat。
你运行在用户的终端里, 由 Python + Anthropic API 驱动。

你的能力:
- 回答编程问题 (尤其擅长 Python + Java 对比)
- 查询天气 (用 get_weather)
- 执行计算 (用 calculate)
- 查看当前时间 (用 get_current_time)
- 运行 Python 代码验证结果 (用 run_python)

回复风格:
- 中文回复
- 简洁但不敷衍
- 涉及代码时, 给出可运行的示例
- 如果用户是 Java 背景, 用 Java 做类比"""

    def __init__(self, model: str = "claude-sonnet-4-6",
                 system: str | None = None,
                 max_context_tokens: int = 8000):
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.messages: list[dict] = []

        # 设置 system prompt
        self.system = system if system is not None else self.DEFAULT_SYSTEM
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

        # 客户端
        api_key = os.getenv("ANTHROPIC_API_KEY")
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        kwargs = {"api_key": api_key} if api_key else {}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = Anthropic(**kwargs)

        # 启用的工具 (默认全部启用)
        self.enabled_tools: set[str] = set(TOOL_REGISTRY.keys())

        # 统计
        self.total_tokens = 0
        self.turn_count = 0

    # ── Token 估算 ──────────────────────────────────

    def estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数。中文 ≈ 1 char/token, 英文 ≈ 4 char/token。"""
        chinese = sum(1 for c in text if '一' <= c <= '鿿')
        other = len(text) - chinese
        return int(chinese * 1.2 + other / 3.5)

    def estimate_total_tokens(self) -> int:
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.estimate_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += self.estimate_tokens(json.dumps(block, ensure_ascii=False))
        return total

    # ── 上下文管理 ──────────────────────────────────

    def _trim_history(self) -> None:
        """
        当上下文超过 max_context_tokens 时, 裁剪最早的非 system 消息。
        保留 system prompt + 最近的消息。
        """
        while self.estimate_total_tokens() > self.max_context_tokens and len(self.messages) > 2:
            # 跳过 system prompt (index 0)
            removed = self.messages.pop(1)  # 移除最早的 user/assistant 消息
            role = removed.get("role", "?")
            # 如果是 (user, assistant) 对, 再移除上一条
            if role == "assistant" and len(self.messages) > 1 and self.messages[1].get("role") == "user":
                self.messages.pop(1)

    # ── 工具执行 ──────────────────────────────────

    def _execute_tools(self, content_blocks: list) -> list[dict]:
        """执行所有 tool_use blocks, 返回 tool_result 列表。"""
        results = []
        for block in content_blocks:
            if isinstance(block, dict):
                block_type = block.get("type", "")
            else:
                block_type = getattr(block, "type", "")

            if block_type == "tool_use":
                if isinstance(block, dict):
                    name, tool_id, inp = block["name"], block["id"], block["input"]
                else:
                    name, tool_id, inp = block.name, block.id, block.input

                entry = TOOL_REGISTRY.get(name)
                if entry:
                    _, handler = entry
                    try:
                        output = handler(**inp)
                        content = json.dumps(output, ensure_ascii=False) if isinstance(output, dict) else str(output)
                        print(f"\n  {Color.style(f'🔧 {name}', Color.DIM)} → {content[:60]}...")
                    except Exception as e:
                        content = f"[工具执行失败: {e}]"
                        print(f"\n  {Color.style(f'❌ {name} 失败', Color.RED)}: {e}")
                else:
                    content = f"[未启用的工具: {name}]"

                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                })
        return results

    # ── 核心: 发送消息 ──────────────────────────────

    def send(self, user_input: str, max_rounds: int = 5) -> str:
        """发送消息, 返回完整回复文本。自动处理工具调用循环。"""
        self.messages.append({"role": "user", "content": user_input})
        self.turn_count += 1

        for _ in range(max_rounds):
            tools = [TOOL_REGISTRY[t][0] for t in self.enabled_tools] if self.enabled_tools else None

            # 流式调用
            with self.client.messages.stream(
                model=self.model,
                max_tokens=2048,
                temperature=0.7,
                messages=self.messages,
                tools=tools,
            ) as stream:
                # 实时打印文本, 同时累积部分文本 (用于中断恢复)
                self._partial_text = ""
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    self._partial_text += text

                final = stream.get_final_message()
                self._partial_text = ""  # 正常完成, 清除部分文本

            self.total_tokens += final.usage.input_tokens + final.usage.output_tokens

            if final.stop_reason == "end_turn":
                text = self._extract_text(final)
                self.messages.append({"role": "assistant", "content": text})
                self._trim_history()
                return text

            elif final.stop_reason == "tool_use":
                # 把 assistant 消息 (含 tool_use blocks) 加入历史
                assistant_content = []
                for block in final.content:
                    if hasattr(block, "model_dump"):
                        assistant_content.append(block.model_dump())
                    elif isinstance(block, dict):
                        assistant_content.append(block)
                self.messages.append({"role": "assistant", "content": assistant_content})

                # 执行工具, 返回结果
                tool_results = self._execute_tools(assistant_content)
                print()  # 工具执行后的换行
                self.messages.append({"role": "user", "content": tool_results})
                # 继续循环 → 模型看到工具结果后生成最终回复
                continue

            else:
                return f"[未知 stop_reason: {final.stop_reason}]"

        return "[达到最大工具调用轮次]"

    @staticmethod
    def _extract_text(message) -> str:
        parts = []
        for block in message.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)

    # ── 流式中断支持 ──────────────────────────────

    def get_partial_response(self) -> str:
        """获取流式中断时的部分文本。"""
        return getattr(self, "_partial_text", "")

    def _commit_partial(self, text: str) -> None:
        """将中断时的部分回复保存到消息历史。"""
        if text.strip():
            self.messages.append({"role": "assistant", "content": text + " [已中断]"})
            self.turn_count += 1

    # ── 系统提示词管理 ──────────────────────────────

    def set_system(self, system: str) -> None:
        self.system = system
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = system
        else:
            self.messages.insert(0, {"role": "system", "content": system})

    def clear(self) -> None:
        """清空对话历史, 保留 system prompt。"""
        self.messages = [{"role": "system", "content": self.system}] if self.system else []
        self.turn_count = 0
        print(f"{Color.style('✅ 对话已清空', Color.GREEN)}")

    # ── 对话保存/加载 ──────────────────────────────

    def save(self, filepath: str) -> None:
        """保存对话历史为 JSON 文件。"""
        data = {
            "model": self.model,
            "system": self.system,
            "messages": self.messages,
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{Color.style(f'✅ 对话已保存到 {filepath}', Color.GREEN)}")
        print(f"   消息: {len(self.messages)} 条 | 轮次: {self.turn_count}")

    def load(self, filepath: str) -> None:
        """从 JSON 文件恢复对话历史。"""
        if not os.path.exists(filepath):
            print(f"{Color.style(f'❌ 文件不存在: {filepath}', Color.RED)}")
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.model = data.get("model", self.model)
        self.system = data.get("system", self.system)
        self.messages = data.get("messages", [])
        self.turn_count = data.get("turn_count", 0)
        self.total_tokens = data.get("total_tokens", 0)
        print(f"{Color.style(f'✅ 对话已从 {filepath} 恢复', Color.GREEN)}")
        print(f"   消息: {len(self.messages)} 条 | 轮次: {self.turn_count}")

    # ── 上下文窗口可视化 ───────────────────────────

    def context_bar(self, width: int = 30) -> str:
        """返回上下文用量进度条。超过 90% 时红色提醒。"""
        cur = self.estimate_total_tokens()
        ratio = min(cur / self.max_context_tokens, 1.0)
        filled = int(ratio * width)
        bar = "█" * filled + "░" * (width - filled)
        pct = int(ratio * 100)
        if pct > 90:
            bar_color = Color.RED
        elif pct > 70:
            bar_color = Color.YELLOW
        else:
            bar_color = Color.GREEN
        return f"{Color.style(f'[{bar}]', bar_color)} {pct}% ({cur:,} / {self.max_context_tokens:,} tokens)"

    # ── 状态信息 ──────────────────────────────────

    @property
    def info(self) -> dict:
        return {
            "model": self.model,
            "messages": len(self.messages),
            "turns": self.turn_count,
            "estimated_tokens": self.estimate_total_tokens(),
            "total_api_tokens": self.total_tokens,
            "enabled_tools": sorted(self.enabled_tools),
        }


# ============================================================
# 三、CLI 交互界面 —— 命令解析 + 主循环
# ============================================================
# 把 PyChat 包装成交互式命令行应用。
#
# 命令系统:
#   /help           — 显示帮助
#   /clear          — 清空对话
#   /system         — 查看当前 system prompt
#   /system <text>  — 设置 system prompt
#   /tools          — 列出工具
#   /tool <on|off> <name> — 启用/禁用工具
#   /info           — 查看会话信息
#   /quit 或 /exit  — 退出

class PyChatCLI:
    """PyChat 的命令行界面。"""

    WELCOME = f"""
{Color.style('╔══════════════════════════════════════╗', Color.CYAN)}
{Color.style('║        🐍 PyChat — AI 对话终端       ║', Color.CYAN + Color.BOLD)}
{Color.style('║    Phase 2 收官项目 | Python + Claude ║', Color.CYAN)}
{Color.style('╚══════════════════════════════════════╝', Color.CYAN)}

  输入消息开始对话, /help 查看命令, /quit 退出
"""

    def __init__(self):
        self.chat = PyChat()

    def run(self):
        print(self.WELCOME)

        # 检查 API 连通性
        try:
            self.chat.client.messages.create(
                model=self.chat.model, max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
            )
            print(f"  {Color.style('✅ API 连接正常', Color.GREEN)}")
            print(f"  模型: {self.chat.model}")
            print(f"  工具: {', '.join(sorted(self.chat.enabled_tools))}")
            print(f"  输入消息开始, 或 /help 查看帮助\n")
        except Exception as e:
            print(f"  {Color.style(f'❌ API 不可用: {e}', Color.RED)}")
            print(f"  请检查 .env 中的 API Key 和网络连接\n")
            return

        # 主循环
        while True:
            try:
                user_input = input(f"{Color.style('🧑 你', Color.GREEN + Color.BOLD)}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{Color.style('👋 再见!', Color.CYAN)}")
                break

            if not user_input:
                continue

            # 处理命令
            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    break  # /quit 返回 True
                continue

            # 正常对话
            print(f"{Color.style('🤖 PyChat', Color.BLUE + Color.BOLD)}: ", end="", flush=True)
            try:
                self.chat.send(user_input)
                print()  # 回复后的换行
            except KeyboardInterrupt:
                # 流式中断: 用户按 Ctrl+C
                partial = self.chat.get_partial_response()
                print(f"\n  {Color.style('⚡ [已中断]', Color.YELLOW)}", end="")
                if partial:
                    print(f" (部分回复已保留: {len(partial)} chars)")
                    self.chat._commit_partial(partial)
                else:
                    print()
            except Exception as e:
                print(f"\n  {Color.style(f'❌ 错误: {e}', Color.RED)}")
            print()  # 空行分隔

    def _handle_command(self, raw: str) -> bool:
        """处理 / 命令。返回 True 表示退出。"""
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit", "/q"):
            print(f"{Color.style('👋 再见!', Color.CYAN)}")
            return True

        elif cmd == "/help":
            print(f"""
  {Color.style('命令列表', Color.BOLD)}
  {'─' * 40}
  /help           显示此帮助
  /clear          清空对话历史
  /system         查看当前 system prompt
  /system <text>  设置新的 system prompt
  /role <name>    切换预置角色 (代码审查专家/技术面试官/创意写作助手)
  /tools          列出可用工具
  /tool on <name> 启用工具
  /tool off <name> 禁用工具
  /save <file>    保存对话到 JSON 文件
  /load <file>    从 JSON 文件恢复对话
  /history [N]    显示最近 N 条对话 (默认 10)
  /info           查看会话信息
  /quit, /exit    退出

  {Color.style('对话技巧', Color.BOLD)}
  {'─' * 40}
  - 直接输入文字开始对话
  - 问我天气、计算、翻译、运行代码, 我会自动调用工具
  - 用 /role 切换角色 (如 "代码审查专家")
  - 用 /save 保存重要对话, /load 恢复
  - 用 /clear 开启新话题 (节省 token)
""")

        elif cmd == "/save":
            if not arg:
                arg = f"pychat_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                self.chat.save(arg)
            except Exception as e:
                print(f"{Color.style(f'❌ 保存失败: {e}', Color.RED)}")

        elif cmd == "/load":
            if not arg:
                print(f"{Color.style('用法: /load <文件名>', Color.YELLOW)}")
                return False
            try:
                self.chat.load(arg)
            except json.JSONDecodeError as e:
                print(f"{Color.style(f'❌ JSON 解析失败: {e}', Color.RED)}")
            except Exception as e:
                print(f"{Color.style(f'❌ 加载失败: {e}', Color.RED)}")

        elif cmd == "/role":
            if not arg:
                print(f"\n  {Color.style('预置角色', Color.BOLD)}")
                print(f"  {'─' * 40}")
                for name, prompt in PRESET_ROLES.items():
                    preview = prompt[:60].replace("\n", " ")
                    print(f"  {Color.style(name, Color.YELLOW)}: {preview}...")
                print(f"\n  用法: /role <角色名>")
            elif arg in PRESET_ROLES:
                self.chat.set_system(PRESET_ROLES[arg])
                self.chat.clear()
                print(f"{Color.style(f'✅ 已切换为: {arg}', Color.GREEN)}")
            else:
                names = ", ".join(PRESET_ROLES.keys())
                print(f"{Color.style(f'未知角色: {arg}', Color.YELLOW)}")
                print(f"  可用: {names}")

        elif cmd == "/history":
            try:
                n = int(arg) if arg else 10
            except ValueError:
                n = 10

            messages = self.chat.messages
            # 跳过 system prompt
            display_msgs = [m for m in messages if m.get("role") != "system"]
            recent = display_msgs[-n:] if len(display_msgs) > n else display_msgs

            if not recent:
                print(f"{Color.style(' (暂无对话历史)', Color.DIM)}")
                return False

            print(f"\n  {Color.style(f'对话历史 (最近 {len(recent)} 条)', Color.BOLD)}")
            print(f"  {'─' * 50}")
            for i, msg in enumerate(recent, 1):
                role = msg.get("role", "?")
                content = msg.get("content", "")
                # 处理嵌套的 tool 内容
                if isinstance(content, list):
                    # 提取文本或工具调用信息
                    tool_names = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "tool_use":
                                tool_names.append(block.get("name", "?"))
                            elif block.get("type") == "text":
                                tool_names.append(f"text: {block.get('text', '')[:30]}...")
                            elif block.get("type") == "tool_result":
                                tool_names.append(f"result: {block.get('content', '')[:30]}...")
                    content = " | ".join(tool_names) if tool_names else "[工具消息]"
                elif len(str(content)) > 80:
                    content = str(content)[:80] + "..."

                if role == "user":
                    prefix = Color.style("🧑 你", Color.GREEN)
                elif role == "assistant":
                    prefix = Color.style("🤖 AI", Color.BLUE)
                else:
                    prefix = role
                print(f"  {i:3d}. {prefix}: {content}")
            print()

        elif cmd == "/info":
            info = self.chat.info
            context_bar = self.chat.context_bar()
            print(f"""
  {Color.style('会话信息', Color.BOLD)}
  {'─' * 40}
  模型:       {info['model']}
  对话轮次:   {info['turns']}
  消息条数:   {info['messages']}
  估算 tokens: {info['estimated_tokens']:,}
  API tokens:  {info['total_api_tokens']:,}
  启用工具:   {', '.join(info['enabled_tools']) if info['enabled_tools'] else '(无)'}
  上下文用量: {context_bar}
""")

        elif cmd == "/clear":
            self.chat.clear()

        elif cmd == "/system":
            if arg:
                self.chat.set_system(arg)
                self.chat.clear()
                print(f"{Color.style('✅ System prompt 已更新 (对话已清空)', Color.GREEN)}")
            else:
                print(f"\n  {Color.style('当前 System Prompt:', Color.BOLD)}")
                print(f"  {'─' * 40}")
                for line in self.chat.system.split("\n"):
                    print(f"  {Color.style(line, Color.DIM)}")
                print()

        elif cmd == "/tools":
            print(f"\n  {Color.style('可用工具', Color.BOLD)}")
            print(f"  {'─' * 40}")
            for name, (tool_def, _) in TOOL_REGISTRY.items():
                status = "✅" if name in self.chat.enabled_tools else "⛔"
                desc = tool_def["description"]
                print(f"  {status} {Color.style(name, Color.YELLOW)}: {desc}")
            print()

        elif cmd == "/tool":
            if not arg:
                print(f"{Color.style('用法: /tool <on|off> <工具名>', Color.YELLOW)}")
                return False
            sub_parts = arg.split(maxsplit=1)
            action = sub_parts[0].lower()
            tool_name = sub_parts[1] if len(sub_parts) > 1 else ""
            if action == "on" and tool_name in TOOL_REGISTRY:
                self.chat.enabled_tools.add(tool_name)
                print(f"{Color.style(f'✅ {tool_name} 已启用', Color.GREEN)}")
            elif action == "off" and tool_name in TOOL_REGISTRY:
                self.chat.enabled_tools.discard(tool_name)
                print(f"{Color.style(f'⛔ {tool_name} 已禁用', Color.YELLOW)}")
            else:
                print(f"{Color.style(f'未知工具: {tool_name}', Color.RED)}")

        else:
            print(f"{Color.style(f'未知命令: {cmd}。输入 /help 查看帮助。', Color.YELLOW)}")

        return False


# ============================================================
# 四、快速演示模式 —— 非交互式, 方便调试
# ============================================================
# 如果不想每次都手动交互, 可以用这个函数跑预设对话。

def demo():
    """非交互式演示, 展示 PyChat 的核心能力。"""
    print("=" * 60)
    print("  PyChat 演示模式 (非交互)")
    print("=" * 60)

    try:
        chat = PyChat()
    except Exception as e:
        print(f"  ❌ 初始化失败: {e}")
        return

    demos = [
        "帮我算一下 2 的 20 次方是多少?",
        "北京今天天气怎么样? 适合跑步吗?",
        "用 Python 写一个检查回文字符串的函数, 然后帮我测试 'racecar' 和 'hello'",
    ]

    for i, msg in enumerate(demos, 1):
        print(f"\n{'─' * 50}")
        print(f"  演示 {i}/{len(demos)}")
        print(f"  🧑: {msg}")
        print(f"  🤖: ", end="", flush=True)
        try:
            chat.send(msg)
            print()
        except Exception as e:
            print(f"\n  ❌ 错误: {e}")
            break

    print(f"\n{'─' * 50}")
    info = chat.info
    print(f"  轮次: {info['turns']} | 消息: {info['messages']} | API tokens: {info['total_api_tokens']}")


# ============================================================
# 五、架构总览 —— 你学会了什么
# ============================================================

ARCHITECTURE = f"""
  {Color.style('PyChat 架构', Color.BOLD)}
  {'─' * 50}

  ┌─────────────────────────────────────────┐
  │              PyChatCLI                   │  ← CLI 交互层
  │  run() → input() → handle_command()     │     /help, /clear, /system ...
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │               PyChat                     │  ← 核心逻辑层
  │  send() → stream() → tool_use 循环       │     对话管理 + 工具调度
  │  _trim_history() | _execute_tools()     │
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │          Anthropic API                   │  ← API 层
  │  messages.stream()                       │     SSE 流式响应
  │  model + tools + system                 │
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │          TOOL_REGISTRY                   │  ← 工具层
  │  weather | calculate | time | run       │     name → (schema, handler)
  └─────────────────────────────────────────┘

  {Color.style('各层对应 Phase 2 的哪一课', Color.DIM)}
  {'─' * 50}
  API 层    → L11 API 基础调用
  System    → L12 Prompt 工程
  Tools     → L13 结构化输出 / Tool Use
  Stream    → L14 流式响应
  CLI + 整合 → L15 本课 — 全部融合
"""


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        print(ARCHITECTURE)
        cli = PyChatCLI()
        cli.run()

    print(f"""
  {Color.style('🎉 Phase 2 完成!', Color.GREEN + Color.BOLD)}

  你已经从一个 "只会写 Python 脚本" 的 Java 工程师,
  进化成 "能构建 AI 应用" 的开发者。

  Phase 2 技能清单:
  ✅ 调用 Claude / OpenAI API
  ✅ System Prompt 设计 (角色、约束)
  ✅ Few-shot、Chain of Thought
  ✅ Prompt 模板化
  ✅ Tool Use / Function Calling
  ✅ 流式输出 (SSE)
  ✅ 完整对话应用架构

  你的 PyChat 现在可以:
  - 查询天气
  - 执行计算
  - 运行 Python 代码
  - 管理对话历史
  - 自定义系统角色

  {Color.style('下一步: Phase 3 — RAG + 向量数据库 + Embedding', Color.CYAN)}
  你将学习: 语义检索、文档分块、向量数据库、知识库 Q&A 系统
""")


# ============================================================
# 试试看 (Try This) —— 练习实现
# ============================================================
#
# 以下练习的代码已经集成到上方的 PyChat 和 PyChatCLI 类中:
#
# 1. 翻译器工具 (translate):
#    - TOOL_REGISTRY["translate"] = (TRANSLATE_TOOL, handle_translate)
#    - 内置本地字典, 支持中英日韩法德互译
#    - 试用: /tool on translate → "把 Hello World 翻译成日语"
#
# 2. 对话保存/加载:
#    - /save [文件名]  → 保存对话为 JSON
#    - /load <文件名>   → 从 JSON 恢复对话
#    - PyChat.save(filepath) / PyChat.load(filepath)
#
# 3. 多角色切换:
#    - /role 列出预置角色
#    - /role <角色名> 切换 (代码审查专家/技术面试官/创意写作助手)
#    - 切换时自动清空历史
#    - PRESET_ROLES dict 管理所有预置角色
#
# 4. 对话历史回显:
#    - /history [N] 显示最近 N 条对话
#    - 自动跳过 system prompt, 区分 user/assistant/工具消息
#    - 长内容自动截断 (80 字符)
#
# 5. (挑战) 流式中断:
#    - 在流式输出中按 Ctrl+C 触发 KeyboardInterrupt
#    - 中断后调用 get_partial_response() 获取已输出文本
#    - _commit_partial() 将部分回复标记为 "[已中断]" 保留在历史中
#    - 原理: 在 text_stream 循环中累积 self._partial_text
#
# 6. (挑战) 上下文窗口可视化:
#    - context_bar() 方法返回进度条 [████████░░]
#    - /info 命令中展示上下文用量
#    - 绿色 (<70%), 黄色 (70-90%), 红色 (>90%)
#
# 所有功能都已集成, 运行 python phase2/15_chat_app.py 或
# python phase2/15_chat_app.py --demo 即可体验!
