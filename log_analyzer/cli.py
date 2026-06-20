# ============================================================
# log_analyzer/cli.py — 命令行接口
# ============================================================
# 用法:
#   python -m log_analyzer.cli analyze app.log
#   python -m log_analyzer.cli analyze app.log "分析错误日志"
#   python -m log_analyzer.cli analyze app.log --question "有没有OOM?"
#   python -m log_analyzer.cli stats app.log           # 仅统计
# ============================================================

import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from log_analyzer.parser import LogParser
from log_analyzer.agent import create_log_agent
from log_analyzer.config import AppConfig


def cmd_analyze(args):
    """Agent 驱动的日志分析。"""
    config = AppConfig.from_env()
    file_path = Path(args.file).expanduser().resolve()

    if not file_path.exists():
        print(f"错误: 文件不存在: {args.file}")
        sys.exit(1)

    question = args.question or "分析日志中的异常和错误"

    print(f"  文件: {file_path}")
    print(f"  大小: {file_path.stat().st_size / 1024:.1f} KB")
    print(f"  问题: {question}")
    print(f"  模型: {config.llm_model}")
    print()

    try:
        agent = create_log_agent(file_path, config)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

    print(f"  解析: {len(agent.entries)} 条日志")
    print(f"  工具: {', '.join(agent.tool_names)}")
    print()

    # 执行分析
    print("  Agent 分析中...\n")
    result = agent.analyze(question)

    print("=" * 60)
    print(result.answer)
    print("=" * 60)
    print(f"\n  迭代: {result.iterations} 轮, "
          f"耗时: {result.latency_ms:.0f}ms, "
          f"工具调用: {len(result.tool_calls)} 次")

    if args.verbose:
        print(f"\n  工具调用详情:")
        for tc in result.tool_calls:
            print(f"    第{tc['iter']}轮 | {tc['tool']} | {tc['status']}")
            for k, v in tc.get("input", {}).items():
                print(f"      {k}: {v}")


def cmd_stats(args):
    """仅统计，不调用 LLM。"""
    file_path = Path(args.file).expanduser().resolve()

    if not file_path.exists():
        print(f"错误: 文件不存在: {args.file}")
        sys.exit(1)

    config = AppConfig.from_env()
    parser = LogParser(max_file_size_mb=config.max_file_size_mb)
    entries = parser.parse_file(file_path)

    from log_analyzer.tools import tool_stats, tool_timeline, tool_top_errors
    from log_analyzer.tools import tool_count, tool_sample

    st = tool_stats(entries)
    print(f"  文件: {file_path.name}")
    print(f"  大小: {file_path.stat().st_size / 1024:.1f} KB")
    print(f"  格式: {parser.detect_format(file_path.read_text('utf-8').split(chr(10)))}")
    print()
    print(f"总行数: {st['total_lines']}")
    print(f"级别分布: {st['by_level']}")
    print(f"时间范围: {st['time_range']['start']} ~ {st['time_range']['end']}")
    print(f"错误率: {st['error_rate']}")
    print()

    tl = tool_timeline(entries)
    if tl["bucket_count"] > 0:
        print(f"时间分布 ({tl['interval']}):")
        for b in tl["buckets"]:
            bar = "█" * min(b["error"], 40)
            print(f"  {b['time']} | 总:{b['total']:4d} | 错:{b['error']:3d} | {bar}")

    print()
    te = tool_top_errors(entries, n=5)
    print(f"高频错误模式 ({te['total_errors']} 条, {te['unique_patterns']} 种):")
    for e in te["top_errors"]:
        print(f"  [{e['count']:3d}] {e['pattern'][:100]}")


def main():
    parser = argparse.ArgumentParser(
        prog="log-analyzer",
        description="LLM 驱动的日志分析 Agent",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Agent 分析日志")
    p_analyze.add_argument("file", help="日志文件路径")
    p_analyze.add_argument("question", nargs="?", default="",
                           help="分析问题 (默认: 分析日志中的异常和错误)")
    p_analyze.add_argument("--question", "-q", dest="question_kw",
                           default="", help="分析问题 (关键字参数)")
    p_analyze.add_argument("--verbose", "-v", action="store_true",
                           help="显示工具调用详情")
    p_analyze.set_defaults(func=cmd_analyze)

    # stats
    p_stats = sub.add_parser("stats", help="快速统计 (不调用 LLM)")
    p_stats.add_argument("file", help="日志文件路径")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()

    # 合并 question (位置参数或关键字参数)
    if hasattr(args, "question_kw") and args.question_kw:
        args.question = args.question_kw
    if hasattr(args, "question") and not args.question:
        args.question = ""

    args.func(args)


if __name__ == "__main__":
    main()
