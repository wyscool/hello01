# ============================================================
# log_analyzer/tools.py — 日志分析工具集
# ============================================================
# 6 个工具，每个返回 dict，通过 MCPServer 注册给 Agent 调用。
#
# 工具通过闭包捕获 entries（解析后的日志列表），Agent 无需传递数据。
# ============================================================

import re
from collections import Counter
from functools import partial

from log_analyzer.parser import LogEntry


# ============================================================
# 工具实现
# ============================================================

def tool_search(entries: list[LogEntry], keyword: str,
                context_lines: int = 3) -> dict:
    """关键词搜索，返回匹配行 + 上下文。"""
    keyword_lower = keyword.lower()
    matches: list[dict] = []

    for i, e in enumerate(entries):
        if keyword_lower not in e.raw.lower():
            continue

        start = max(0, i - context_lines)
        end = min(len(entries), i + context_lines + 1)
        context = [
            f"[{entries[j].line_number}] {entries[j].raw[:200]}"
            for j in range(start, end)
        ]
        matches.append({
            "line": e.line_number,
            "level": e.level or "?",
            "text": e.raw[:300],
            "timestamp": e.timestamp.isoformat() if e.timestamp else "",
            "context": " | ".join(context),
        })

    total = len(matches)
    return {
        "total_matches": total,
        "keyword": keyword,
        "matches": matches[:30],
        "truncated": total > 30,
    }


def tool_count(entries: list[LogEntry], level: str = "",
               keyword: str = "") -> dict:
    """统计日志条数，可按级别和关键词过滤。"""
    filtered = entries
    if level:
        lv = level.upper()
        filtered = [e for e in filtered if e.level == lv]
    if keyword:
        kw = keyword.lower()
        filtered = [e for e in filtered if kw in e.raw.lower()]

    level_counts = dict(Counter(
        e.level or "UNKNOWN" for e in filtered
    ))
    total = len(filtered)

    return {
        "total": total,
        "total_all": len(entries),
        "by_level": level_counts,
        "filter_level": level or None,
        "filter_keyword": keyword or None,
        "error_rate": f"{level_counts.get('ERROR', 0) / max(total, 1) * 100:.1f}%",
    }


def tool_sample(entries: list[LogEntry], n: int = 10,
                level: str = "") -> dict:
    """获取日志样本，默认返回最近的条目。"""
    filtered = entries
    if level:
        filtered = [e for e in filtered if e.level == level.upper()]

    samples = filtered[-n:] if filtered else []
    return {
        "samples": [
            {
                "line": s.line_number,
                "level": s.level or "?",
                "timestamp": s.timestamp.isoformat() if s.timestamp else "",
                "text": s.raw[:300],
            }
            for s in reversed(samples)
        ],
        "sample_count": len(samples),
        "filter_level": level or None,
    }


def tool_timeline(entries: list[LogEntry],
                  interval: str = "hour") -> dict:
    """日志时间分布，按 hour/day/minute 聚合。"""
    buckets: dict[str, dict] = {}

    for e in entries:
        if e.timestamp is None:
            continue

        if interval == "hour":
            key = e.timestamp.strftime("%Y-%m-%d %H:00")
        elif interval == "day":
            key = e.timestamp.strftime("%Y-%m-%d")
        elif interval == "minute":
            key = e.timestamp.strftime("%Y-%m-%d %H:%M")
        else:
            key = e.timestamp.strftime("%Y-%m-%d %H:00")

        if key not in buckets:
            buckets[key] = {"total": 0, "error": 0, "warn": 0, "info": 0}
        buckets[key]["total"] += 1
        lv = e.level or ""
        if lv in ("ERROR", "FATAL", "SEVERE"):
            buckets[key]["error"] += 1
        elif lv in ("WARN", "WARNING"):
            buckets[key]["warn"] += 1
        elif lv == "INFO":
            buckets[key]["info"] += 1

    return {
        "interval": interval,
        "buckets": [
            {"time": k, **v} for k, v in sorted(buckets.items())
        ],
        "bucket_count": len(buckets),
        "entries_with_ts": sum(b["total"] for b in buckets.values()),
    }


def tool_top_errors(entries: list[LogEntry], n: int = 10) -> dict:
    """高频错误模式 Top-N (基于消息前 80 字符聚类)。"""
    error_entries = [e for e in entries
                     if e.level in ("ERROR", "FATAL", "SEVERE")]
    patterns: Counter[str] = Counter()

    for e in error_entries:
        # 规范化: 移除数字、引号内的动态内容
        normalized = re.sub(r"\d+", "N", e.message[:80])
        normalized = re.sub(r"'[^']*'", "'...'", normalized)
        normalized = re.sub(r'"[^"]*"', '"..."', normalized)
        patterns[normalized] += 1

    return {
        "total_errors": len(error_entries),
        "unique_patterns": len(patterns),
        "top_errors": [
            {"pattern": pat, "count": cnt, "pct": f"{cnt / max(len(error_entries), 1) * 100:.1f}%"}
            for pat, cnt in patterns.most_common(n)
        ],
    }


def tool_get_section(entries: list[LogEntry], start_line: int,
                     end_line: int) -> dict:
    """获取指定行范围的原始日志。"""
    section = [e for e in entries
               if start_line <= e.line_number <= end_line]
    return {
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": len(section),
        "content": "\n".join(e.raw for e in section[:100]),
        "truncated": len(section) > 100,
    }


def tool_stats(entries: list[LogEntry]) -> dict:
    """日志整体统计: 总数、级别分布、时间范围、文件来源。"""
    levels = Counter(e.level for e in entries if e.level)
    timestamps = [e.timestamp for e in entries if e.timestamp]

    sources = Counter(e.source for e in entries)
    return {
        "total_lines": len(entries),
        "by_level": dict(levels),
        "time_range": {
            "start": min(timestamps).isoformat() if timestamps else "",
            "end": max(timestamps).isoformat() if timestamps else "",
        },
        "sources": dict(sources),
        "error_rate": f"{levels.get('ERROR', 0) / max(len(entries), 1) * 100:.1f}%",
        "entries_with_ts": len(timestamps),
    }


# ============================================================
# 工具注册 — 创建 MCPServer
# ============================================================

def create_log_server(entries: list[LogEntry],
                      server_name: str = "log-analyzer",
                      max_context: int = 10,
                      max_sample: int = 20):
    """创建并注册好所有日志分析工具的 MCPServer。

    工具函数通过闭包捕获 entries，Agent 调用时无需传日志数据。

    用法:
        from log_analyzer.tools import create_log_server
        server = create_log_server(entries)
        tools = MCPClient(server).list_tools()
    """
    from deploy.agent_core import MCPServer

    server = MCPServer(server_name)

    # 绑定 entries 参数
    search_fn = partial(tool_search, entries)
    count_fn = partial(tool_count, entries)
    sample_fn = partial(tool_sample, entries)
    timeline_fn = partial(tool_timeline, entries)
    top_errors_fn = partial(tool_top_errors, entries)
    get_section_fn = partial(tool_get_section, entries)
    stats_fn = partial(tool_stats, entries)

    server.register(
        "search",
        "搜索日志中匹配关键词的行，返回匹配行及上下文。用于定位特定错误或关键字。",
        {
            "keyword": {"type": "string", "description": "搜索关键词 (不区分大小写)"},
            "context_lines": {"type": "integer", "description": f"上下文行数，默认 3，最大 {max_context}"},
        },
        ["keyword"],
        search_fn,
    ).register(
        "count",
        "统计日志条数。可按日志级别 (ERROR/WARN/INFO/DEBUG) 和关键词过滤。",
        {
            "level": {"type": "string", "description": "级别过滤: ERROR / WARN / INFO / DEBUG。留空则统计全部"},
            "keyword": {"type": "string", "description": "关键词过滤 (不区分大小写)。留空则不过滤"},
        },
        [],
        count_fn,
    ).register(
        "sample",
        "获取日志样本，返回最近的 N 条记录。可过滤级别。用于快速了解日志内容。",
        {
            "n": {"type": "integer", "description": f"返回条数，默认 10，最大 {max_sample}"},
            "level": {"type": "string", "description": "级别过滤: ERROR / WARN / INFO。留空则不限"},
        },
        [],
        sample_fn,
    ).register(
        "timeline",
        "查看日志时间分布，按 hour/day/minute 聚合，展示每时段的总数、错误数、警告数。",
        {
            "interval": {"type": "string", "description": "聚合粒度: hour / day / minute。默认 hour"},
        },
        [],
        timeline_fn,
    ).register(
        "top_errors",
        "高频错误模式 Top-N。自动规范化消息中的数字和引用内容，聚类相同模式。",
        {
            "n": {"type": "integer", "description": "返回 Top N，默认 10"},
        },
        [],
        top_errors_fn,
    ).register(
        "get_section",
        "获取指定行范围的原始日志内容。用于查看特定时间段的详细日志。",
        {
            "start_line": {"type": "integer", "description": "起始行号"},  # ChromaDB metadata 限制: int
            "end_line": {"type": "integer", "description": "结束行号"},
        },
        ["start_line", "end_line"],
        get_section_fn,
    ).register(
        "stats",
        "查看日志整体统计: 总行数、级别分布、时间范围、错误率、文件来源。分析前先调用此工具了解全貌。",
        {},
        [],
        stats_fn,
    )

    return server
