# log_analyzer/ — 智能日志分析 Agent

LLM 驱动的日志分析工具。解析多种日志格式，通过 ReAct 循环自主分析故障根因。支持 CLI、FastAPI、Python 库三种使用方式。

## 快速开始

```bash
pip install -r log_analyzer/requirements.txt

# CLI — 统计分析 (不调用 LLM)
python log_analyzer/cli.py stats /path/to/app.log

# CLI — Agent 深度分析 (调用 LLM)
python log_analyzer/cli.py analyze /path/to/app.log --verbose

# FastAPI 服务
uvicorn log_analyzer.app:app --port 8004
```

## 架构

```
日志文件 (.log/.json/.txt, 最大 50MB)
  │
  ▼
LogParser
  ├─ 自动编码检测 (utf-8 → gbk → latin-1)
  ├─ JSON lines 格式     → {"timestamp":..., "level":..., "message":...}
  ├─ 标准日志格式         → 2024-01-01 12:00:00 [ERROR] ...
  ├─ Java 堆栈跟踪        → 多行合并为一个 LogEntry
  └─ 纯文本 (兜底)        → 每行一个 LogEntry
  │
  ▼
LogEntry[] (timestamp, level, message, raw_line, ...)
  │
  ▼
MCPServer (7 个日志分析工具)
  ├─ tool_stats        — 统计概览 (条目数、时间范围、级别分布)
  ├─ tool_top_errors   — TOP N 错误消息 (按出现次数排序)
  ├─ tool_search       — 关键词全文搜索
  ├─ tool_count        — 正则或文本匹配计数
  ├─ tool_sample       — 随机采样 N 条
  ├─ tool_timeline     — 按时间粒度 (分钟/小时/天) 聚合
  └─ tool_get_section  — 按行号范围提取原文
  │
  ▼
LogAnalysisAgent (ReAct 循环, 最多 10 次迭代)
  └─ stats → top_errors → search → get_section → 根因报告
```

## 项目结构

```
log_analyzer/
├── __init__.py          # 包文档
├── config.py            # AppConfig: 17 个字段
├── parser.py            # LogParser: 4 种格式解析 + 编码检测
├── tools.py             # 7 个日志分析工具 (tool_search/tool_stats/...)
├── agent.py             # LogAnalysisAgent: ReAct 循环
├── cli.py               # CLI 入口
├── app.py               # FastAPI 服务
├── Dockerfile
├── requirements.txt     # anthropic, fastapi, uvicorn, python-dotenv, python-multipart
└── tests/
    ├── test_parser.py
    ├── test_tools.py
    ├── test_agent.py
    └── test_config.py
```

## 各模块详解

### parser.py — 日志解析器

`LogParser` 支持的格式:

| 格式 | 示例 | 检测方式 |
|------|------|---------|
| JSON lines | `{"timestamp": "...", "level": "ERROR", "message": "..."}` | 第一行是否有效 JSON |
| 结构化文本 | `2024-06-01 10:00:00 [ERROR] [main] ...` | 正则提取 timestamp/level/thread |
| Java 堆栈 | `Exception in thread ...\n\tat ...` | 多行合并，检测 `^\s+at ` 模式 |
| 纯文本 | 任意文本 | 兜底，每行一个 LogEntry |

编码检测: 按 utf-8 → gbk → latin-1 顺序尝试，避免中文乱码。

**LogEntry 数据类**: `timestamp`, `level`, `thread`, `message`, `raw_line`, `line_number`

### tools.py — 日志分析工具

7 个工具通过 `functools.partial` 将日志数据绑定到工具处理函数:

| 工具 | 输入 | 输出 |
|------|------|------|
| `tool_stats` | — | 条目总数、时间范围、ERROR/WARN/INFO/DEBUG 计数和百分比 |
| `tool_top_errors` | `n=10` | TOP N 错误消息 (去重 + 频率排序) |
| `tool_search` | `keyword`, `level=""`, `limit=20` | 匹配的 LogEntry 列表 |
| `tool_count` | `pattern` | 正则/文本匹配的命中数 |
| `tool_sample` | `n=5`, `level=""` | 随机采样 N 条日志 |
| `tool_timeline` | `granularity="hour"` | 按时间粒度聚合的直方图数据 |
| `tool_get_section` | `start_line`, `end_line` | 原始日志行的原文 (行号范围) |

### agent.py — 分析 Agent

`LogAnalysisAgent` 执行 ReAct 循环:

```
Step 1: tool_stats()      → 了解全局 (多少条? 时间范围? 错误率?)
Step 2: tool_top_errors() → 找到最主要的错误类型
Step 3: tool_search()     → 搜索关键错误相关的上下文
Step 4: tool_get_section()→ 提取原始日志片段
Step 5: 综合分析 + 根因推断 → 结构化报告
```

配置: `agent_max_iterations=10` (可通过环境变量调整)。

## CLI 用法

```bash
# 统计分析 (纯 Python, 零 LLM 调用)
python log_analyzer/cli.py stats /var/log/app.log
# 输出: 条目数、级别分布饼图(ASCII)、时间范围、TOP 错误

# Agent 分析 (调用 LLM)
python log_analyzer/cli.py analyze /var/log/app.log --verbose

# 从管道读取
cat /var/log/app.log | python log_analyzer/cli.py analyze -
```

## API 参考

### `POST /analyze` — 分析服务器端文件

```bash
curl -X POST http://localhost:8004/analyze \
  -H "Content-Type: application/json" \
  -d '{"path": "/var/log/app.log"}'
```

### `POST /analyze/upload` — 上传日志文件分析

```bash
curl -X POST http://localhost:8004/analyze/upload \
  -F "file=@app.log" \
  -F "description=production error analysis"
```

## 依赖

| 来源 | 组件 | 用途 |
|------|------|------|
| `deploy.agent_core` | LlmClient | LLM API 调用 |
| `deploy.agent_core` | MCPServer, MCPClient | 工具注册和调用 |
| `deploy.agent_core` | TokenBudget | Token 消耗估算 |
| `deploy.observability` | JsonLogger, Trace | 结构化日志和追踪 |
| `deploy.infrastructure` | RateLimiter, HealthChecker, GracefulShutdown, ServiceStats | 生产级运维 |
| `deploy.cost_control` | ExactCache | 分析结果缓存 |

## 设计亮点

- **零 LLM 统计模式**: `stats` 子命令纯 Python 完成，不消耗 Token，秒级出结果
- **MCP 工具抽象**: 日志分析能力封装为标准 MCP 工具，可复用
- **自动编码检测**: 处理中文日志的 GBK/UTF-8 编码混乱问题
- **堆栈合并**: 多行 Java 异常堆栈合并为一个条目，避免被截断
