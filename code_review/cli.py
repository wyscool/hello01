# ============================================================
# code_review/cli.py — 命令行审查工具
# ============================================================
# 用法:
#   python code_review/cli.py                    # 交互模式
#   python code_review/cli.py path/to/file.java  # 审查文件
#   echo "code..." | python code_review/cli.py   # 管道输入
# ============================================================

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from code_review.agent import create_review_agent


def detect_language(path_or_code: str) -> str:
    """根据文件名或代码内容推断语言。"""
    ext = Path(path_or_code).suffix if os.path.exists(path_or_code) else ""
    lang_map = {
        ".java": "java", ".py": "python", ".kt": "java",
        ".go": "go", ".rs": "rust", ".ts": "typescript",
        ".js": "javascript", ".sql": "sql",
    }
    if ext in lang_map:
        return lang_map[ext]
    # 根据代码内容推断
    if "public class" in path_or_code or "import java" in path_or_code:
        return "java"
    if "def " in path_or_code and "import " in path_or_code:
        return "python"
    return "java"  # 默认


def print_report(result: dict):
    """格式化打印审查报告。"""
    print()
    print("=" * 60)
    print("  代码审查报告")
    print("=" * 60)

    # 摘要
    summary = result.get("summary", "无")
    score = result.get("score", "N/A")
    print(f"\n  综合评分: {score}/10")
    print(f"  摘要: {summary[:200]}")

    # 发现列表
    findings = result.get("findings", [])
    if findings:
        print(f"\n  发现 {len(findings)} 个问题:")
        print(f"  {'─' * 50}")
        severities = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
        for i, f in enumerate(findings, 1):
            icon = severities.get(f.get("severity", ""), "  ")
            line = f.get("line", "?")
            title = f.get("title", "")
            desc = f.get("description", "")[:120]
            sug = f.get("suggestion", "")
            print(f"\n  {icon} [{i}] {title} (L{line})")
            print(f"     类别: {f.get('category', '')}")
            if desc:
                print(f"     说明: {desc}")
            if sug:
                print(f"     建议: {sug[:150]}")

    # 执行步骤 (Plan 模式)
    steps = result.get("steps", [])
    if steps:
        print(f"\n  {'─' * 50}")
        print(f"  审查步骤:")
        for s in steps:
            icon = "✓" if s.get("status") == "done" else "✗"
            print(f"    {icon} Step {s['step']}: {s['desc'][:60]}")

    print(f"\n{'=' * 60}\n")


def review_file(filepath: str, agent, language: str = ""):
    """审查单个文件。"""
    path = Path(filepath).expanduser()
    if not path.exists():
        print(f"错误: 文件不存在 — {filepath}")
        return

    code = path.read_text(encoding="utf-8")
    lang = language or detect_language(str(path))
    print(f"\n审查文件: {path} ({lang}, {len(code)} 字符, "
          f"{len(code.splitlines())} 行)")
    print("分析中...")

    result = agent.review(code, language=lang)
    print_report(result)


def interactive(agent):
    """交互模式 — 贴代码直接审查。"""
    print("""
╔══════════════════════════════════════════════════╗
║         AI 代码审查助手 — CLI 模式                ║
╠══════════════════════════════════════════════════╣
║  用法:                                            ║
║    1. 直接贴代码, 输入 END 结束, 开始审查          ║
║    2. /file <路径>   审查文件                      ║
║    3. /lang <java|python>  切换默认语言            ║
║    4. /help           帮助                         ║
║    5. /quit           退出                         ║
╚══════════════════════════════════════════════════╝
""")
    lang = "java"
    history: list[str] = []

    while True:
        try:
            raw = input("CR > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见!")
            break

        if not raw:
            continue

        # 命令
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            match cmd:
                case "/quit" | "/exit":
                    print("再见!")
                    break
                case "/help":
                    print("""
  命令:
    /file <路径>     审查文件
    /lang <语言>     切换默认语言 (java/python)
    /history         查看审查历史
    /quit            退出
  直接贴代码:
    粘贴代码, 输入 END 结束, 开始审查
    /lang 切换识别语言
""")
                case "/lang":
                    if arg in ("java", "python"):
                        lang = arg
                        print(f"默认语言: {lang}")
                    else:
                        print(f"当前: {lang}, 可选: java / python")
                case "/file":
                    if arg:
                        review_file(arg, agent, lang)
                    else:
                        print("用法: /file <路径>")
                case "/history":
                    if history:
                        print(f"\n审查历史 ({len(history)} 条):")
                        for i, h in enumerate(history, 1):
                            print(f"  [{i}] {h[:80]}...")
                    else:
                        print("暂无审查历史")
                case _:
                    print(f"未知命令: {cmd}, /help 查看帮助")
            continue

        # 代码输入模式
        print(f"  (贴代码, {lang} 模式。输入 END 结束)")
        lines = [raw]
        while True:
            try:
                line = input()
            except (KeyboardInterrupt, EOFError):
                break
            if line.strip() == "END":
                break
            lines.append(line)

        code = "\n".join(lines)
        if not code.strip():
            continue

        print(f"  ({len(code)} 字符, {len(code.splitlines())} 行) 分析中...")
        result = agent.review(code, language=lang)
        print_report(result)
        history.append(code[:80])


def main():
    print("=" * 60)
    print("  AI 代码审查助手")
    print("=" * 60)

    agent = create_review_agent()
    print(f"  Model: {agent.llm.model}")
    print(f"  API: {'reachable' if agent.llm.is_healthy else 'offline'}")
    print(f"  Tools: {agent.mcp.server.tool_count if agent.mcp.server else 0}")

    # 文件参数 (优先检查, 因为 python -m 时 stdin 可能不是 tty)
    if len(sys.argv) > 1:
        review_file(sys.argv[1], agent)
        return

    # 管道输入
    if not sys.stdin.isatty():
        code = sys.stdin.read()
        lang = detect_language(code)
        print(f"\n审查管道输入 ({lang}, {len(code)} 字符)...")
        result = agent.review(code, language=lang)
        print_report(result)
        return

    # 交互模式
    interactive(agent)


if __name__ == "__main__":
    main()
