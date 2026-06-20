# code_review/ — AI 代码审查助手

对 Java/Python 代码执行多层 AI 审查: 静态规则检查 (零 LLM 成本) → LLM 模式检测 → 深度分析 → 综合报告。

## 设计理念

审查分四层执行，逐层深入:
1. **静态规则** — 纯正则/特征匹配，零 LLM 调用，毫秒级完成
2. **LLM 模式检测** — 让 LLM 识别代码中的反模式和常见错误
3. **深度分析** — LLM 对逻辑/安全/性能/可读性的全面审查
4. **综合报告** — 汇总所有层结果 + 可操作建议

## 快速开始

```bash
pip install -r code_review/requirements.txt

# CLI — 审查文件
python code_review/cli.py path/to/file.py

# CLI — 交互模式
python code_review/cli.py
# 粘贴代码，输入 END 结束

# CLI — 管道输入
cat file.py | python code_review/cli.py

# FastAPI 服务
uvicorn code_review.app:app --port 8001
```

## 架构

```
源代码 (Python / Java)
  │
  ▼
Step 1: 静态规则检查 (纯 Python, 零 LLM 成本)
  ├─ 命名规范: class PascalCase, function snake_case
  ├─ 行长度检查: >120 chars
  ├─ 方法长度检查: >50 lines
  ├─ 空 catch 块检测
  ├─ System.out.print() 检测 (Java)
  └─ 可变默认参数检测 (Python)
  │
  ▼
Step 2: LLM 辅助模式检测
  ├─ SQL 注入风险
  ├─ 资源泄漏 (连接未关闭)
  ├─ 空指针/NPE 风险
  └─ 不安全的 eval() / exec()
  │
  ▼
Step 3: 深度 LLM 分析
  └─ 逻辑错误 / 安全漏洞 / 性能瓶颈 / 可维护性
  │
  ▼
Step 4: 综合报告
  └─ 规则命中 + 模式检测 + 深度分析 + 修复建议
```

## 项目结构

```
code_review/
├── __init__.py          # 包文档
├── config.py            # AppConfig: 7 个字段
├── agent.py             # ReviewAgent (Plan-then-Act 4 步流程)
├── tools.py             # tool_check_style + tool_detect_patterns
├── cli.py               # CLI (交互/文件/管道 三种模式)
├── app.py               # FastAPI 服务
├── Dockerfile
├── requirements.txt     # anthropic, fastapi, uvicorn, python-dotenv
└── tests/
    ├── test_tools.py
    ├── test_agent.py
    └── test_config.py
```

## 各模块详解

### tools.py — 审查工具

**`tool_check_style(code, language)`** — 静态规则引擎:

| 规则 | Python | Java | 严重级别 |
|------|--------|------|---------|
| 类名 PascalCase | ✓ | ✓ | warning |
| 函数名 snake_case | ✓ | — | warning |
| 行长度 >120 | ✓ | ✓ | info |
| 方法长度 >50 | ✓ | ✓ | info |
| 空 catch/except | ✓ | ✓ | error |
| System.out.print | — | ✓ | warning |
| 可变对象作为默认参数 | ✓ | — | error |

**`tool_detect_patterns(code, language)`** — LLM 辅助模式检测:
- SQL 注入: 字符串拼接进 SQL
- 资源泄漏: 打开未关闭的连接/文件
- 空指针风险: 链式调用未检查 null
- 不安全执行: eval() / exec() / Runtime.exec()

### agent.py — 审查 Agent

`ReviewAgent` 使用 Plan-then-Act 模式:

```
Plan  → "我将分 4 步审查: 风格检查 → 模式检测 → 深度分析 → 报告"
Act 1 → tool_check_style(code, language)    → [规则命中列表]
Act 2 → tool_detect_patterns(code, language) → [模式检测结果]
Act 3 → LLM 深度分析 (逻辑/安全/性能/可读性) → 详细分析
Act 4 → 综合所有结果 + 可操作建议 → 最终报告
```

### cli.py — CLI 入口

```bash
# 交互模式: 粘贴代码，输入 END 结束
python code_review/cli.py
> def foo(a=[]):
>     pass
> END

# 文件模式: 审查指定文件
python code_review/cli.py /path/to/code.py

# 管道模式: 从 stdin 读取
cat code.py | python code_review/cli.py
# 或
echo 'def foo(a=[]): pass' | python code_review/cli.py
```

## API 参考

### `POST /review`

```bash
curl -X POST http://localhost:8001/review \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def foo(a=[]):\n    return a",
    "language": "python",
    "focus": ["security", "performance"]
  }'
```

`focus` 可选值: `"style"`, `"security"`, `"performance"`, `"readability"`, `"logic"`, `"all"`。

## 与其他包的关系

**code_review 是唯一不依赖 `deploy/` 的包** — 拥有独立的 `LlmClient` 和 `MCPServer`/`MCPClient` 实现。这反映了学习路径: code_review 是较早的实战项目，先写了自给自足的实现，后来才抽取 `deploy/` 作为共享基础设施。

## 测试

```bash
pytest code_review/tests/ -v
# 覆盖: 静态规则检查、模式检测、Agent 流程、配置加载
```
