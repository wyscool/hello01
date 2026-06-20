# ============================================================
# code_review/tests/test_tools.py — 代码审查工具测试
# ============================================================

import pytest
from pathlib import Path

from code_review.tools import (
    tool_check_style, tool_detect_patterns, tool_read_file,
)


# ============================================================
# 测试用代码样本
# ============================================================

# Java: 包含多种问题的代码
JAVA_BAD_CODE = """\
package com.example;

import java.sql.Statement;
import java.sql.Connection;
import java.util.Map;

public class myapp {
    public static final int bad_constant = 42;

    public String GetUser(String id) {
        try {
            Statement stmt = getConnection().createStatement();
            String sql = "SELECT * FROM users WHERE id=" + id;
            return stmt.executeQuery(sql).toString();
        } catch (Exception e) {
        }
        return null;
    }

    private Connection getConnection() {
        Map<String, String> map = new java.util.HashMap<>();
        String val = map.get("key").toLowerCase();
        System.out.println("debug: " + val);
        return null;
    }
}
"""

# 真正包含长方法的 Java 代码 (55 行方法体)
JAVA_LONG_METHOD = (
    "public class Big {\n"
    + "    public String longOne(String x) {\n"
    + "\n".join(f"        s += \"line {i}\";" for i in range(55))
    + "\n        return s;\n    }\n}"
)

# Python: 包含多种问题的代码
PYTHON_BAD_CODE = """\
class myClass:
    pass

def MyFunc(items=[]):
    try:
        result = eval("1+1")
    except:
        pass
    x = x
    return result
"""

# Java conn_not_closed / null_pointer 专项测试数据
JAVA_CONN_LEAK = (
    "import java.sql.Connection;\n"
    "import java.sql.DriverManager;\n"
    "public class Leak {\n"
    "    public void leak() {\n"
    '        Connection c = DriverManager.getConnection("url");\n'
    "        c.createStatement();\n"
    "    }\n"
    "}"
)

JAVA_NULL_DEREF = (
    "import java.util.Map;\n"
    "import java.util.HashMap;\n"
    "public class NullDeref {\n"
    "    static final String KEY = \"k\";\n"
    "    public int danger() {\n"
    "        Map<String, String> m = new HashMap<>();\n"
    "        return m.get(KEY).length();\n"
    "    }\n"
    "}"
)


# ============================================================
# tool_check_style — Java
# ============================================================

class TestCheckStyleJava:
    def test_returns_correct_structure(self):
        result = tool_check_style(JAVA_BAD_CODE, language="java")
        assert "violations" in result
        assert "count" in result
        assert result["language"] == "java"
        assert result["lines_total"] > 0

    def test_detects_class_naming(self):
        result = tool_check_style(JAVA_BAD_CODE, language="java")
        class_violations = [
            v for v in result["violations"]
            if v["rule"] == "java_class_naming"
        ]
        assert len(class_violations) >= 1
        v = class_violations[0]
        assert "myapp" in v["message"]
        assert v["severity"] == "warning"

    def test_detects_method_naming(self):
        result = tool_check_style(JAVA_BAD_CODE, language="java")
        method_violations = [
            v for v in result["violations"]
            if v["rule"] == "java_method_naming"
        ]
        assert len(method_violations) >= 1
        assert any("GetUser" in v["message"] for v in method_violations)

    def test_detects_empty_catch(self):
        result = tool_check_style(JAVA_BAD_CODE, language="java")
        catch_violations = [
            v for v in result["violations"]
            if v["rule"] == "empty_catch"
        ]
        assert len(catch_violations) == 1
        assert catch_violations[0]["severity"] == "critical"

    def test_detects_sysout(self):
        result = tool_check_style(JAVA_BAD_CODE, language="java")
        sysout_violations = [
            v for v in result["violations"]
            if v["rule"] == "sysout"
        ]
        assert len(sysout_violations) == 1

    def test_detects_long_line(self):
        long_line_code = "public class Test { String x = \"" + "x" * 130 + "\"; }"
        result = tool_check_style(long_line_code, language="java")
        line_len = [v for v in result["violations"] if v["rule"] == "line_length"]
        assert len(line_len) >= 1

    def test_detects_long_file(self):
        many_lines = "\n".join([f"// line {i}" for i in range(501)])
        result = tool_check_style(many_lines, language="java")
        file_len = [v for v in result["violations"] if v["rule"] == "file_length"]
        assert len(file_len) == 1
        assert file_len[0]["severity"] == "warning"

    def test_detects_long_method(self):
        result = tool_check_style(JAVA_LONG_METHOD, language="java")
        method_len = [v for v in result["violations"] if v["rule"] == "method_length"]
        assert len(method_len) >= 1

    def test_clean_code_no_violations(self):
        clean = "public class Hello { public String greet(String name) { return \"Hi \" + name; } }"
        result = tool_check_style(clean, language="java")
        assert result["count"] == 0

    def test_empty_code(self):
        result = tool_check_style("", language="java")
        assert result["lines_total"] == 1
        assert result["count"] == 0


# ============================================================
# tool_check_style — Python
# ============================================================

class TestCheckStylePython:
    def test_detects_bare_except(self):
        result = tool_check_style(PYTHON_BAD_CODE, language="python")
        bare_v = [v for v in result["violations"]
                  if v["rule"] == "bare_except"]
        assert len(bare_v) == 1

    def test_detects_mutable_default(self):
        result = tool_check_style(PYTHON_BAD_CODE, language="python")
        mut_v = [v for v in result["violations"]
                 if v["rule"] == "mutable_default"]
        assert len(mut_v) == 1
        assert mut_v[0]["severity"] == "critical"

    def test_detects_long_method(self):
        code = "def long_method():\n" + "    x = 1\n" * 52
        result = tool_check_style(code, language="python")
        method_len = [v for v in result["violations"]
                      if v["rule"] == "method_length"]
        assert len(method_len) >= 1

    def test_clean_code(self):
        code = "def greet(name: str) -> str:\n    return f'Hi {name}'\n"
        result = tool_check_style(code, language="python")
        assert result["count"] == 0

    def test_py_class_regex_captures(self):
        """PY_CLASS_RE 只捕获大写开头的类名 (这是工具的自然行为)。"""
        result = tool_check_style("class MyClass:\n    pass", language="python")
        # 大写开头的类名不触发 violation
        class_v = [v for v in result["violations"]
                   if v["rule"] == "python_class_naming"]
        assert len(class_v) == 0

    def test_py_func_regex_captures(self):
        """PY_FUNC_RE 只捕获小写开头的函数名。"""
        result = tool_check_style("def my_func():\n    pass", language="python")
        func_v = [v for v in result["violations"]
                  if v["rule"] == "python_func_naming"]
        # snake_case 不触发 violation
        assert len(func_v) == 0


# ============================================================
# tool_detect_patterns — Java
# ============================================================

class TestDetectPatternsJava:
    def test_detects_sql_injection(self):
        result = tool_detect_patterns(JAVA_BAD_CODE, language="java")
        sql = [f for f in result["findings"] if f["rule"] == "sql_injection"]
        assert len(sql) == 1
        assert sql[0]["severity"] == "critical"

    def test_detects_conn_not_closed(self):
        result = tool_detect_patterns(JAVA_CONN_LEAK, language="java")
        conn = [f for f in result["findings"] if f["rule"] == "conn_not_closed"]
        assert len(conn) == 1

    def test_detects_null_pointer(self):
        result = tool_detect_patterns(JAVA_NULL_DEREF, language="java")
        null = [f for f in result["findings"] if f["rule"] == "null_pointer"]
        assert len(null) == 1

    def test_has_line_numbers(self):
        result = tool_detect_patterns(JAVA_BAD_CODE, language="java")
        for finding in result["findings"]:
            assert isinstance(finding["line"], int)
            assert finding["line"] > 0
            assert "snippet" in finding

    def test_clean_code(self):
        clean = "public class A { public void f() { int x = 1; } }"
        result = tool_detect_patterns(clean, language="java")
        assert result["count"] == 0


# ============================================================
# tool_detect_patterns — Python
# ============================================================

class TestDetectPatternsPython:
    def test_detects_mutable_default(self):
        result = tool_detect_patterns(PYTHON_BAD_CODE, language="python")
        mut = [f for f in result["findings"] if f["rule"] == "mutable_default"]
        assert len(mut) == 1
        assert mut[0]["severity"] == "critical"

    def test_detects_bare_except(self):
        result = tool_detect_patterns(PYTHON_BAD_CODE, language="python")
        bare = [f for f in result["findings"] if f["rule"] == "bare_except"]
        assert len(bare) == 1

    def test_detects_unsafe_eval(self):
        result = tool_detect_patterns(PYTHON_BAD_CODE, language="python")
        eval_f = [f for f in result["findings"] if f["rule"] == "unsafe_eval"]
        assert len(eval_f) == 1
        assert eval_f[0]["severity"] == "critical"

    def test_has_note_about_false_positives(self):
        result = tool_detect_patterns(PYTHON_BAD_CODE, language="python")
        assert "note" in result
        assert "误报" in result["note"]

    def test_empty_code(self):
        result = tool_detect_patterns("", language="python")
        assert result["count"] == 0
        assert result["findings"] == []


# ============================================================
# tool_read_file
# ============================================================

class TestReadFile:
    @pytest.fixture
    def project_root(self, tmp_path):
        return str(tmp_path)

    def test_reads_existing_file(self, project_root):
        p = Path(project_root) / "test.java"
        p.write_text("public class Test {\n    int x = 1;\n}\n", encoding="utf-8")
        result = tool_read_file(str(p), project_root=project_root)
        assert "error" not in result
        # 末尾有 \n, split 产生 4 个元素
        assert result["total_lines"] == 4
        assert "public class Test" in result["content"]

    def test_max_lines_truncation(self, project_root):
        content = "\n".join([f"line {i}" for i in range(300)])
        p = Path(project_root) / "big.java"
        p.write_text(content, encoding="utf-8")
        result = tool_read_file(str(p), max_lines=50, project_root=project_root)
        assert result["preview_lines"] == 50
        assert result["total_lines"] == 300

    def test_file_not_found(self, project_root):
        nonexistent = str(Path(project_root) / "nonexistent.java")
        result = tool_read_file(nonexistent, project_root=project_root)
        assert "error" in result
        assert "不存在" in result["error"]

    def test_directory_rejected(self, project_root):
        result = tool_read_file(project_root, project_root=project_root)
        assert "error" in result
        assert "目录" in result["error"]

    def test_path_outside_project(self, project_root):
        result = tool_read_file("/etc/passwd", project_root=project_root)
        assert "error" in result
        assert "安全限制" in result["error"]

    def test_unsupported_file_type(self, project_root):
        p = Path(project_root) / "test.exe"
        p.write_text("fake", encoding="utf-8")
        result = tool_read_file(str(p), project_root=project_root)
        assert "error" in result
        assert "不支持" in result["error"]
