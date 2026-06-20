# ============================================================
# code-review/tools.py — 代码分析专用工具
# ============================================================
# 三个工具:
#   1. check_style       — 纯规则引擎 (不调 LLM, 零成本)
#   2. detect_patterns   — 正则模式匹配 (常见 Bug 模式)
#   3. analyze_code      — LLM 深度分析 (由 Agent 在 Plan 中调用)
#
# 设计原则: 规则引擎先跑 (快+免费), LLM 做补充 (慢+花钱)。
# 类比 Java: check_style ≈ Checkstyle / SonarLint
# ============================================================

import re
import json
from pathlib import Path


# ============================================================
# 一、check_style — 规则引擎
# ============================================================

# Java 命名正则
JAVA_CLASS_RE = re.compile(r'\bclass\s+([a-zA-Z][a-zA-Z0-9]*)')
JAVA_METHOD_RE = re.compile(
    r'\b(public|private|protected)\s+(?:\w+\s+)?(\w+)\('
)
JAVA_CONSTANT_RE = re.compile(
    r'\bstatic\s+final\s+\w+\s+([A-Z_][A-Z0-9_]*)'
)

# Python 命名正则
PY_CLASS_RE = re.compile(r'class\s+([A-Z][a-zA-Z0-9]*)')
PY_FUNC_RE = re.compile(r'def\s+([a-z_][a-z0-9_]*)')
PY_CONSTANT_RE = re.compile(r'^([A-Z_][A-Z0-9_]*)\s*=', re.MULTILINE)


def tool_check_style(code: str, language: str = "java") -> dict:
    """规则引擎 — 检查代码风格和命名规范。

    纯本地计算, 不调 LLM, 零 Token 成本。
    """
    violations = []
    lines = code.split("\n")

    # --- 通用规则 ---
    for i, line in enumerate(lines, 1):
        # 行长度
        if len(line) > 120:
            violations.append({
                "line": i, "rule": "line_length",
                "severity": "info",
                "message": f"行长度 {len(line)} > 120",
            })

    # 文件长度
    if len(lines) > 500:
        violations.append({
            "line": 1, "rule": "file_length",
            "severity": "warning",
            "message": f"文件过长 ({len(lines)} 行), 建议拆分",
        })

    # --- 语言特定规则 ---
    if language == "java":
        _java_style_checks(code, lines, violations)
    elif language == "python":
        _python_style_checks(code, lines, violations)

    return {
        "violations": violations,
        "count": len(violations),
        "language": language,
        "lines_total": len(lines),
    }


def _java_style_checks(code: str, lines: list[str],
                       violations: list):
    """Java 风格检查。"""

    # 类名 PascalCase
    for m in JAVA_CLASS_RE.finditer(code):
        name = m.group(1)
        if not name[0].isupper():
            violations.append({
                "line": code[:m.start()].count("\n") + 1,
                "rule": "java_class_naming",
                "severity": "warning",
                "message": f"类名 '{name}' 应为 PascalCase (首字母大写)",
            })

    # 方法名 camelCase
    for m in JAVA_METHOD_RE.finditer(code):
        name = m.group(2)
        if name[0].isupper():
            violations.append({
                "line": code[:m.start()].count("\n") + 1,
                "rule": "java_method_naming",
                "severity": "warning",
                "message": f"方法名 '{name}' 应为 camelCase (首字母小写)",
            })

    # 空 catch 块
    for m in re.finditer(
        r'catch\s*\([^)]+\)\s*\{\s*\}', code
    ):
        line_no = code[:m.start()].count("\n") + 1
        violations.append({
            "line": line_no, "rule": "empty_catch",
            "severity": "critical",
            "message": "空的 catch 块 — 异常被静默吞掉",
        })

    # System.out.println 残留
    for m in re.finditer(r'System\.out\.println', code):
        line_no = code[:m.start()].count("\n") + 1
        violations.append({
            "line": line_no, "rule": "sysout",
            "severity": "info",
            "message": "System.out.println — 考虑用 Logger 替代",
        })

    # 方法长度
    _check_method_length(code, lines, violations, "java")


def _python_style_checks(code: str, lines: list[str],
                         violations: list):
    """Python 风格检查。"""

    # 类名 PascalCase
    for m in PY_CLASS_RE.finditer(code):
        name = m.group(1)
        if not name[0].isupper():
            violations.append({
                "line": code[:m.start()].count("\n") + 1,
                "rule": "python_class_naming",
                "severity": "warning",
                "message": f"类名 '{name}' 应为 PascalCase",
            })

    # 函数名 snake_case
    for m in PY_FUNC_RE.finditer(code):
        name = m.group(1)
        if not name.startswith("_"):
            if any(c.isupper() for c in name):
                violations.append({
                    "line": code[:m.start()].count("\n") + 1,
                    "rule": "python_func_naming",
                    "severity": "warning",
                    "message": f"函数名 '{name}' 应为 snake_case",
                })

    # 可变默认参数
    for m in re.finditer(
        r'def\s+\w+\s*\([^)]*\w+\s*=\s*(\[\]|\{\}|set\(\))', code
    ):
        line_no = code[:m.start()].count("\n") + 1
        violations.append({
            "line": line_no, "rule": "mutable_default",
            "severity": "critical",
            "message": f"可变默认参数 {m.group(1)} — "
                       f"用 None + 内部初始化替代",
        })

    # 裸 except
    for m in re.finditer(r'except\s*:', code):
        line_no = code[:m.start()].count("\n") + 1
        violations.append({
            "line": line_no, "rule": "bare_except",
            "severity": "warning",
            "message": "裸 except — 指定具体异常类型",
        })

    # 方法长度
    _check_method_length(code, lines, violations, "python")


def _check_method_length(code: str, lines: list[str],
                         violations: list, language: str):
    """检测过长的函数/方法。"""
    if language == "java":
        pattern = re.compile(
            r'(public|private|protected)\s+\w+\s+\w+\s*\([^)]*\)\s*\{'
        )
    else:
        pattern = re.compile(r'def\s+\w+\s*\([^)]*\):')

    for m in pattern.finditer(code):
        start_line = code[:m.start()].count("\n") + 1     # 1-based
        remaining = lines[start_line - 1:]                 # 包含签名行
        end_line = start_line - 1                          # 0-indexed

        if language == "java":
            brace_count = 0
            for j, line in enumerate(remaining):
                brace_count += line.count("{") - line.count("}")
                end_line = start_line - 1 + j
                if brace_count <= 0 and j > 0:
                    break
        else:
            # Python: 基于缩进检测函数体结束
            def_indent = len(lines[start_line - 1]) - len(
                lines[start_line - 1].lstrip()
            )
            for j, line in enumerate(remaining):
                stripped = line.strip()
                if j > 0 and stripped and len(line) - len(line.lstrip()) <= def_indent:
                    break
                end_line = start_line - 1 + j

        func_len = end_line - (start_line - 1) + 1
        if func_len > 50:
            violations.append({
                "line": start_line, "rule": "method_length",
                "severity": "warning",
                "message": f"函数/方法过长 ({func_len} 行), "
                           f"建议 < 50 行",
            })


# ============================================================
# 二、detect_patterns — Bug 模式检测
# ============================================================

# Java 常见 Bug 模式
JAVA_BUG_PATTERNS = [
    (
        "sql_injection",
        r'Statement\s+\w+\s*=.*\.createStatement\(\)',
        "critical",
        "使用 Statement 而非 PreparedStatement — 存在 SQL 注入风险",
    ),
    (
        "conn_not_closed",
        r'\.getConnection\([^)]*\)(?!.*\.close\(\))',
        "warning",
        "数据库连接可能未关闭 — 考虑 try-with-resources",
    ),
    (
        "null_pointer",
        r'\.get\(\w+\)\.\w+\(',
        "warning",
        "Map.get() 后直接调用方法 — 可能 NullPointerException",
    ),
    (
        "double_check_locking",
        r'if\s*\(\s*\w+\s*==\s*null\s*\)\s*\{[^}]*synchronized',
        "info",
        "双重检查锁定 — 确保 volatile 修饰",
    ),
]

# Python 常见 Bug 模式
PYTHON_BUG_PATTERNS = [
    (
        "mutable_default",
        r'def\s+\w+\s*\([^)]*\w+\s*=\s*(\[\]|\{\}|set\(\))',
        "critical",
        "可变默认参数 — 所有调用共享同一个对象",
    ),
    (
        "bare_except",
        r'except\s*:',
        "warning",
        "裸 except — 会捕获 KeyboardInterrupt 等系统异常",
    ),
    (
        "undefined_var",
        r'(?:^|\s)(\w+)\s*=\s*\1',
        "info",
        "变量赋值给自己 — 可能是拼写错误",
    ),
    (
        "unsafe_eval",
        r'\beval\s*\([^)]*\)',
        "critical",
        "使用 eval() — 代码注入风险, 考虑 ast.literal_eval",
    ),
]


def tool_detect_patterns(code: str, language: str = "java") -> dict:
    """模式匹配 — 检测常见 Bug 模式。

    纯正则匹配, 不调 LLM, 零 Token 成本。
    注意: 有误报可能, 需人工确认。
    """
    findings = []
    patterns = JAVA_BUG_PATTERNS if language == "java" else PYTHON_BUG_PATTERNS

    for rule_id, pattern_str, severity, message in patterns:
        for m in re.finditer(pattern_str, code, re.MULTILINE):
            line_no = code[:m.start()].count("\n") + 1
            snippet = code[m.start():m.end()].strip()[:80]
            findings.append({
                "line": line_no,
                "rule": rule_id,
                "severity": severity,
                "message": message,
                "snippet": snippet,
            })

    return {
        "findings": findings,
        "count": len(findings),
        "language": language,
        "note": "模式匹配有误报可能, 请人工确认",
    }


# ============================================================
# 三、read_file — 文件读取
# ============================================================

def tool_read_file(path: str, max_lines: int = 200,
                   project_root: str = ".") -> dict:
    """读取文件内容 (安全检查)。"""
    p = Path(path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()

    try:
        p.relative_to(root)
    except ValueError:
        return {"error": f"安全限制: 只能读取项目目录 ({root})"}

    if not p.exists():
        return {"error": f"文件不存在: {path}"}
    if p.is_dir():
        return {"error": f"路径是目录: {path}"}

    # 代码审查支持更多文件类型
    allowed = {".py", ".java", ".kt", ".go", ".rs", ".ts",
               ".js", ".txt", ".md", ".json", ".yml", ".yaml",
               ".xml", ".sql", ".html", ".css", ".sh", ".c", ".h",
               ".cpp", ".hpp"}
    if p.suffix not in allowed:
        return {"error": f"不支持的类型: {p.suffix}"}

    try:
        lines = p.read_text(encoding="utf-8").split("\n")
        total = len(lines)
        preview = "\n".join(lines[:max_lines])
        return {
            "path": str(p), "total_lines": total,
            "preview_lines": min(max_lines, total),
            "content": preview,
        }
    except Exception as e:
        return {"error": f"读取失败: {e}"}
