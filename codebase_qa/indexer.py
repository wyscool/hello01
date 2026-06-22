# ============================================================
# codebase_qa/indexer.py — 多语言代码结构化分块
# ============================================================
# 核心技术:
#   Python: ast 模块解析，按函数/类/方法边界分块
#   Java:   正则表达式解析，按类/接口/枚举/方法/字段分块
# 每个块保留签名、docstring/Javadoc、精确行号、语言标识。
# ============================================================

import ast
import hashlib
from pathlib import Path
from dataclasses import dataclass, field


# ============================================================
# 一、CodeChunk — 一个代码块
# ============================================================

@dataclass
class CodeChunk:
    """单个代码块（一个函数、类、方法、或模块顶层代码）。"""

    name: str                           # "retry" | "MyClass" | "MyClass.method"
    type: str                           # "function" | "class" | "method" | "module_level"
    file_path: str                      # 相对路径: "phase1/04_functions.py"
    start_line: int                     # 1-based 起始行
    end_line: int                       # 1-based 结束行
    code_text: str = ""                 # 完整源码
    docstring: str = ""                 # 提取的 docstring / Javadoc
    signature: str = ""                 # 定义第一行: "def ..." | "public void ..."
    file_hash: str = ""                 # SHA-256，增量索引 + 去重
    language: str = "python"            # "python" | "java"

    @property
    def embed_text(self) -> str:
        """嵌入文本 — 语言标签 + 签名 + docstring + 代码前 500 字符。"""
        parts = [f"[{self.language}] {self.type}: {self.signature}"]
        if self.docstring:
            parts.append(self.docstring)
        parts.append(self.code_text[:500])
        return "\n".join(parts)

    @property
    def chunk_id(self) -> str:
        """ChromaDB 唯一 ID: {file_hash[:8]}:{name}:{start_line}"""
        return f"{self.file_hash[:8]}:{self.name}:{self.start_line}"


# ============================================================
# 二、CodeIndexer — AST 解析器
# ============================================================

class CodeIndexer:
    """用 AST 解析 Python 源码，按函数/类/方法边界分块。

    用法:
        indexer = CodeIndexer(exclude_dirs={"tests", "venv"})
        chunks = indexer.index_directory(Path("./my_project"))
    """

    def __init__(self, exclude_dirs: set[str] | None = None):
        self.exclude_dirs = exclude_dirs or {
            "tests", "venv", ".git", "__pycache__",
            "node_modules", "build", "dist",
        }
        self._file_hashes: dict[str, str] = {}  # path(str) → sha256

    # ----------------------------------------------------------
    # 文件发现
    # ----------------------------------------------------------

    def walk_directory(self, root: Path, extensions: set[str] | None = None) -> list[Path]:
        """递归发现代码文件，跳过 exclude_dirs。

        Args:
            root: 搜索根目录
            extensions: 文件扩展名集合，默认 {".py", ".java"}
        """
        if extensions is None:
            extensions = {".py", ".java"}
        files: list[Path] = []
        root = root.resolve()
        for ext in extensions:
            for item in root.rglob(f"*{ext}"):
                if any(excl in item.parts for excl in self.exclude_dirs):
                    continue
                files.append(item)
        return sorted(files)

    # ----------------------------------------------------------
    # 文件哈希（增量索引）
    # ----------------------------------------------------------

    @staticmethod
    def hash_file(file_path: Path) -> str:
        """SHA-256 十六进制摘要。"""
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    def is_unchanged(self, file_path: Path) -> bool:
        """文件是否未修改（哈希与缓存一致）。"""
        key = str(file_path.resolve())
        cached = self._file_hashes.get(key)
        if cached is None:
            return False
        return self.hash_file(file_path) == cached

    # ----------------------------------------------------------
    # 文件解析 — 统一入口 (按语言分发)
    # ----------------------------------------------------------

    def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """解析单个代码文件，按后缀分发到 Python / Java 解析器。

        流程:
          1. 读文件 → 计算 SHA-256 → 增量检测
          2. .py  → _parse_python_file (ast 模块)
          3. .java → _parse_java_file (regex)
          4. 统一返回 CodeChunk[]
        """
        suffix = file_path.suffix.lower()
        if suffix == ".py":
            return self._parse_python_file(file_path)
        elif suffix == ".java":
            return self._parse_java_file(file_path)
        else:
            return []

    # ----------------------------------------------------------
    # Python AST 解析
    # ----------------------------------------------------------

    def _parse_python_file(self, file_path: Path) -> list[CodeChunk]:
        """用 ast 模块解析 .py 文件。"""
        try:
            source = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError):
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []  # 忽略语法错误的文件

        file_hash_val = hashlib.sha256(source.encode()).hexdigest()

        if self.is_unchanged(file_path):
            return []

        source_lines = source.split("\n")
        root = file_path.parent.resolve()
        rel_path = str(file_path.resolve().relative_to(root)) if root in file_path.resolve().parents else file_path.name

        chunks: list[CodeChunk] = []
        covered_ranges: list[tuple[int, int]] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._build_function_chunk(
                    node, source_lines, rel_path, file_hash_val,
                    class_name=None,
                )
                chunks.append(chunk)
                covered_ranges.append((node.lineno, node.end_lineno or node.lineno))

            elif isinstance(node, ast.ClassDef):
                class_chunk = self._build_class_chunk(
                    node, source_lines, rel_path, file_hash_val,
                )
                chunks.append(class_chunk)
                covered_ranges.append((node.lineno, node.end_lineno or node.lineno))

                # 提取方法
                for body_node in node.body:
                    if isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_chunk = self._build_function_chunk(
                            body_node, source_lines, rel_path, file_hash_val,
                            class_name=node.name,
                        )
                        chunks.append(method_chunk)
                        covered_ranges.append(
                            (body_node.lineno, body_node.end_lineno or body_node.lineno)
                        )

        # 模块级代码: 未被函数/类覆盖的行
        module_text = self._extract_module_level(source_lines, covered_ranges)
        if module_text and module_text.strip():
            file_stem = file_path.stem
            chunks.append(CodeChunk(
                name=f"{file_stem}_module",
                type="module_level",
                file_path=rel_path,
                start_line=1,
                end_line=len(source_lines),
                code_text=module_text,
                docstring=ast.get_docstring(tree) or "",
                signature="",
                file_hash=file_hash_val,
            ))

        key = str(file_path.resolve())
        self._file_hashes[key] = file_hash_val
        return chunks

    # ----------------------------------------------------------
    # Java regex 解析
    # ----------------------------------------------------------
    # Java 没有内置 AST 解析器，用正则表达式提取类/接口/枚举/方法/字段。
    # 覆盖 90%+ 常见 Java 结构，零外部依赖。后续可升级为 tree-sitter。

    def _parse_java_file(self, file_path: Path) -> list[CodeChunk]:
        """用正则表达式解析 .java 文件。"""
        try:
            source = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError):
            return []

        file_hash_val = hashlib.sha256(source.encode()).hexdigest()

        if self.is_unchanged(file_path):
            return []

        source_lines = source.split("\n")
        rel_path = file_path.name

        chunks: list[CodeChunk] = []
        covered_ranges: list[tuple[int, int]] = []

        type_defs = self._find_java_types(source, source_lines)
        for td in type_defs:
            chunks.append(CodeChunk(
                name=td["name"],
                type="class",
                file_path=rel_path,
                start_line=td["start_line"],
                end_line=td["end_line"],
                code_text="\n".join(source_lines[td["start_line"] - 1:td["end_line"]]),
                docstring=td.get("javadoc", ""),
                signature=td["signature"],
                file_hash=file_hash_val,
                language="java",
            ))
            covered_ranges.append((td["start_line"], td["end_line"]))

            members = self._find_java_members(
                td["body"], td["name"], source_lines, td["body_start_line"],
            )
            for m in members:
                member_code = "\n".join(
                    source_lines[m["start_line"] - 1:m["end_line"]]
                )
                chunks.append(CodeChunk(
                    name=f"{td['name']}.{m['name']}",
                    type=m["type"],
                    file_path=rel_path,
                    start_line=m["start_line"],
                    end_line=m["end_line"],
                    code_text=member_code,
                    docstring=m.get("javadoc", ""),
                    signature=m["signature"],
                    file_hash=file_hash_val,
                    language="java",
                ))
                covered_ranges.append((m["start_line"], m["end_line"]))

        module_text = self._extract_module_level(source_lines, covered_ranges)
        if module_text and module_text.strip():
            chunks.append(CodeChunk(
                name=f"{file_path.stem}_module",
                type="module_level",
                file_path=rel_path,
                start_line=1,
                end_line=len(source_lines),
                code_text=module_text,
                docstring="",
                signature="",
                file_hash=file_hash_val,
                language="java",
            ))

        key = str(file_path.resolve())
        self._file_hashes[key] = file_hash_val
        return chunks

    @staticmethod
    def _strip_java_comments(source: str) -> str:
        """移除 Java 注释和字符串字面量，替换为空格（用于结构分析）。

        关键：保留换行符，确保行号与原文对齐。
        """
        result: list[str] = []
        i = 0
        while i < len(source):
            # 字符串字面量
            if source[i] == '"':
                result.append(' ')
                i += 1
                while i < len(source):
                    if source[i] == '\\':
                        result.append(' ')
                        i += 2
                        continue
                    if source[i] == '"':
                        break
                    if source[i] == '\n':
                        result.append('\n')
                    else:
                        result.append(' ')
                    i += 1
                if i < len(source):
                    i += 1
                continue
            # 字符字面量
            if source[i] == "'":
                result.append(' ')
                i += 1
                while i < len(source):
                    if source[i] == '\\':
                        i += 2
                        continue
                    if source[i] == "'":
                        break
                    i += 1
                if i < len(source):
                    i += 1
                continue
            # 行注释
            if source[i:i + 2] == '//':
                while i < len(source) and source[i] != '\n':
                    result.append(' ')
                    i += 1
                continue
            # 块注释 / Javadoc — 保留内部换行
            if source[i:i + 2] == '/*':
                while i < len(source) and source[i:i + 2] != '*/':
                    if source[i] == '\n':
                        result.append('\n')
                    else:
                        result.append(' ')
                    i += 1
                if i < len(source):
                    result.append('  ')
                    i += 2
                continue
            result.append(source[i])
            i += 1
        return ''.join(result)

    @staticmethod
    def _find_matching_brace(
        lines: list[str], start_line: int, start_col: int,
    ) -> int:
        """找匹配的 '}'。返回 0-based 行号，未找到返回 -1。"""
        depth = 0
        for i in range(start_line, len(lines)):
            line = lines[i]
            j = start_col if i == start_line else 0
            while j < len(line):
                c = line[j]
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        return i
                j += 1
        return -1

    def _find_java_types(
        self, source: str, source_lines: list[str],
    ) -> list[dict]:
        """查找所有 class/interface/enum/@interface 定义。"""
        import re
        clean = self._strip_java_comments(source)
        clean_lines = clean.split("\n")

        pattern = re.compile(
            r'(?:(?:public|private|protected)\s+)?'
            r'(?:(?:abstract|static|final|strictfp|sealed|non-sealed)\s+)*'
            r'(class|interface|enum|@interface)\s+(\w+)',
        )

        results: list[dict] = []
        for i, line in enumerate(clean_lines):
            m = pattern.search(line)
            if not m:
                continue
            keyword = m.group(1)
            name = m.group(2)

            # 找 class 体开括号
            brace_col = line.find('{', m.end())
            if brace_col < 0:
                for j in range(i + 1, min(i + 5, len(clean_lines))):
                    if '{' in clean_lines[j]:
                        brace_line = j
                        brace_col = clean_lines[j].find('{')
                        break
                else:
                    continue
            else:
                brace_line = i

            end_line = self._find_matching_brace(
                clean_lines, brace_line, brace_col,
            )
            if end_line < 0:
                continue

            sig = self._java_signature(source_lines, i)
            javadoc = self._extract_javadoc(source_lines, i)

            body_start = brace_line + 1
            body_end = end_line - 1
            if body_start <= body_end:
                body = "\n".join(source_lines[body_start:body_end + 1])
            else:
                body = ""

            results.append({
                "name": name,
                "keyword": keyword,
                "start_line": i + 1,
                "end_line": end_line + 1,
                "signature": sig,
                "javadoc": javadoc,
                "body": body,
                "body_start_line": body_start + 1,
            })

        return results

    def _find_java_members(
        self, body: str, class_name: str,
        source_lines: list[str], base_offset: int,
    ) -> list[dict]:
        """在类体中查找方法、构造函数和字段。"""
        if not body.strip():
            return []

        body_lines = body.split("\n")
        clean_body = self._strip_java_comments(body)
        clean_body_lines = clean_body.split("\n")

        results: list[dict] = []
        skip_until_end = -1  # 跳过已处理方法体内部的行

        for i, clean_line in enumerate(clean_body_lines):
            abs_line = base_offset + i  # 1-based

            if abs_line <= skip_until_end:
                continue

            stripped = clean_line.strip()
            if not stripped:
                continue

            # 方法 / 构造函数: 包含 '(' 和 '{'（支持单行和多行声明）"
            if '(' in stripped and '{' in stripped:
                brace_col = clean_line.rfind('{')
                end_line = self._find_matching_brace(
                    clean_body_lines, i, brace_col,
                )
                if end_line < 0:
                    continue

                abs_end = base_offset + end_line
                sig = self._java_signature(source_lines, abs_line - 1)
                javadoc = self._extract_javadoc(source_lines, abs_line - 1)
                method_name = self._extract_java_method_name(stripped, class_name)

                results.append({
                    "name": method_name,
                    "type": "method",
                    "start_line": abs_line,
                    "end_line": abs_end,
                    "signature": sig,
                    "javadoc": javadoc,
                })
                skip_until_end = abs_end
                continue

            # 字段声明: 以 ';' 结束，不含 '('
            if stripped.endswith(';') and '(' not in stripped:
                sig = self._java_signature(source_lines, abs_line - 1)
                javadoc = self._extract_javadoc(source_lines, abs_line - 1)
                field_name = self._extract_java_field_name(stripped)

                results.append({
                    "name": field_name,
                    "type": "field",
                    "start_line": abs_line,
                    "end_line": abs_line,
                    "signature": sig,
                    "javadoc": javadoc,
                })

        return results

    @staticmethod
    def _extract_java_method_name(stripped_line: str, class_name: str) -> str:
        """从方法/构造函数签名中提取方法名。

        'public void setName(String name)' → 'setName'
        'public UserService(Dependency dep)' → 'UserService'
        """
        import re
        # 移除注解（行内 @Override 等）
        clean = re.sub(r'@\w+\s*', '', stripped_line).strip()
        # 构造函数：类名后跟 '('
        m = re.match(
            r'(?:public|private|protected)?\s*'
            r'(?:static|final|abstract|synchronized|native)?\s*'
            + re.escape(class_name) + r'\s*\(',
            clean,
        )
        if m:
            return class_name  # 构造函数
        # 普通方法
        m = re.search(r'(\w+)\s*\(', clean)
        if m:
            return m.group(1)
        return "unknown"

    @staticmethod
    def _extract_java_field_name(stripped_line: str) -> str:
        """从字段声明中提取字段名。

        'private String name;' → 'name'
        'private final List<String> items;' → 'items'
        """
        import re
        # 移除结尾分号
        decl = stripped_line.rstrip(';').strip()
        # 分割并取最后一个标识符
        parts = re.split(r'[\s<>,\[\]?]+', decl)
        parts = [p for p in parts if p]
        return parts[-1] if parts else "unknown"

    @staticmethod
    def _java_signature(source_lines: list[str], start_line: int) -> str:
        """提取 Java 类型/方法定义的第一行（含注解）。"""
        sig_lines: list[str] = []
        # 收集注解（@Override, @GetMapping(...) 等）
        for j in range(start_line - 1, -1, -1):
            line = source_lines[j].strip()
            if line.startswith('@'):
                sig_lines.insert(0, line)
            else:
                break
        sig_lines.append(source_lines[start_line].strip())
        return "\n".join(sig_lines)

    @staticmethod
    def _extract_javadoc(
        source_lines: list[str], line_index: int,
    ) -> str:
        """提取声明前最近的 Javadoc 注释。

        line_index: 声明所在行 (0-based)
        会跳过空行和注解行（@Override 等），找到最近的 Javadoc。
        """
        # 向前查找 /** ... */ 或 // 注释
        comments: list[str] = []
        for j in range(line_index - 1, max(line_index - 10, -1), -1):
            line = source_lines[j].strip()
            if line.startswith('*') or line.startswith('/*') or line == '*/':
                comments.insert(0, line)
            elif line.startswith('//'):
                comments.insert(0, line)
            elif line == '' or line.startswith('@'):
                # 跳过空行和注解行（如 @Override, @Service）
                continue
            else:
                break

        if not comments:
            return ""

        text = "\n".join(comments)
        # 清理 Javadoc 标记
        import re
        text = re.sub(r'/\*\*?\s*', '', text)
        text = re.sub(r'\*/\s*', '', text)
        text = re.sub(r'^\s*\*\s?', '', text, flags=re.MULTILINE)
        return text.strip()

    # ----------------------------------------------------------
    # 构建单个 chunk
    # ----------------------------------------------------------

    def _build_function_chunk(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: list[str], rel_path: str, file_hash_val: str,
        class_name: str | None,
    ) -> CodeChunk:
        """从 FunctionDef/AsyncFunctionDef 节点构建 CodeChunk。"""
        start = node.lineno
        end = node.end_lineno or node.lineno
        code_text = "\n".join(source_lines[start - 1:end])
        name = f"{class_name}.{node.name}" if class_name else node.name
        return CodeChunk(
            name=name,
            type="method" if class_name else "function",
            file_path=rel_path,
            start_line=start,
            end_line=end,
            code_text=code_text,
            docstring=ast.get_docstring(node) or "",
            signature=self._get_signature(node, source_lines),
            file_hash=file_hash_val,
        )

    def _build_class_chunk(
        self, node: ast.ClassDef,
        source_lines: list[str], rel_path: str, file_hash_val: str,
    ) -> CodeChunk:
        """从 ClassDef 节点构建 CodeChunk。"""
        start = node.lineno
        end = node.end_lineno or node.lineno
        code_text = "\n".join(source_lines[start - 1:end])
        return CodeChunk(
            name=node.name,
            type="class",
            file_path=rel_path,
            start_line=start,
            end_line=end,
            code_text=code_text,
            docstring=ast.get_docstring(node) or "",
            signature=self._get_signature(node, source_lines),
            file_hash=file_hash_val,
        )

    # ----------------------------------------------------------
    # 签名重建 — 从 AST 节点还原定义第一行
    # ----------------------------------------------------------

    @staticmethod
    def _get_signature(
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        source_lines: list[str],
    ) -> str:
        """从 AST 节点提取定义第一行（含装饰器）。

        装饰器不包含在 .lineno 中，需要单独查找。
        """
        prefix = ""
        if node.decorator_list:
            first_decorator_lineno = node.decorator_list[0].lineno
            decorator_lines = source_lines[first_decorator_lineno - 1:node.lineno - 1]
            prefix = "\n".join(decorator_lines) + "\n"

        first_line = source_lines[node.lineno - 1].strip()

        # 处理跨行签名（参数列表分多行）
        end_lineno = node.end_lineno or node.lineno
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # 找第一个 ":" 的位置（参数列表结束后）
            body_start = node.body[0].lineno if node.body else end_lineno + 1
            sig_lines = source_lines[node.lineno - 1:body_start - 1]
            sig_text = "\n".join(sig_lines)
            # 截断到 "):" 或 ":"
            colon_idx = sig_text.rfind(":")
            if colon_idx >= 0:
                first_line = sig_text[:colon_idx + 1]
            else:
                first_line = sig_text.rstrip()
        else:
            # ClassDef
            body_start = node.body[0].lineno if node.body else end_lineno + 1
            sig_lines = source_lines[node.lineno - 1:body_start - 1]
            sig_text = "\n".join(sig_lines)
            colon_idx = sig_text.rfind(":")
            if colon_idx >= 0:
                first_line = sig_text[:colon_idx + 1]
            else:
                first_line = sig_text.rstrip()

        return prefix + first_line

    # ----------------------------------------------------------
    # 模块级代码提取
    # ----------------------------------------------------------

    @staticmethod
    def _extract_module_level(
        source_lines: list[str],
        covered_ranges: list[tuple[int, int]],
    ) -> str | None:
        """返回未被任何函数/类覆盖的行。

        covered_ranges: [(start_line, end_line), ...], 行号 1-based。
        """
        total = len(source_lines)
        covered: set[int] = set()
        for start, end in covered_ranges:
            for i in range(start, end + 1):
                covered.add(i)

        result_lines: list[str] = []
        for i, line in enumerate(source_lines, start=1):
            if i not in covered:
                result_lines.append(line)

        return "\n".join(result_lines) if result_lines else None

    # ----------------------------------------------------------
    # 批量索引
    # ----------------------------------------------------------

    def index_directory(self, root: Path, extensions: set[str] | None = None) -> list[CodeChunk]:
        """递归解析目录下所有 .py / .java 文件，返回所有代码块。"""
        files = self.walk_directory(root, extensions=extensions)
        all_chunks: list[CodeChunk] = []
        for fp in files:
            chunks = self.parse_file(fp)
            all_chunks.extend(chunks)
        return all_chunks

    def chunk_to_metadata(self, chunk: CodeChunk) -> dict:
        """CodeChunk → ChromaDB metadata dict（值类型: str/int/float/bool）。"""
        return {
            "name": chunk.name,
            "type": chunk.type,
            "file_path": chunk.file_path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "signature": chunk.signature,
            "language": chunk.language,
        }

    @property
    def indexed_files(self) -> int:
        """已索引文件数。"""
        return len(self._file_hashes)
