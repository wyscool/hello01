# ============================================================
# codebase_qa/indexer.py — AST 结构化代码分块
# ============================================================
# 核心技术: 用 Python 标准库 ast 模块解析源码，按函数/类/方法
# 边界分块，每个块保留签名、docstring、精确行号。
# 这是与 rag_kb 通用文本分块最本质的区别。
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
    docstring: str = ""                 # 提取的 docstring
    signature: str = ""                 # 定义第一行: "def retry(max_attempts: int = 3):"
    file_hash: str = ""                 # SHA-256，增量索引 + 去重

    @property
    def embed_text(self) -> str:
        """嵌入文本 — 优先签名+docstring（语义密集），再加代码前 500 字符。"""
        parts = [f"{self.type}: {self.signature}"]
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

    def walk_directory(self, root: Path) -> list[Path]:
        """递归发现 .py 文件，跳过 exclude_dirs。"""
        files: list[Path] = []
        root = root.resolve()
        for item in root.rglob("*.py"):
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
    # AST 解析 — 核心
    # ----------------------------------------------------------

    def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """解析单个 .py 文件，提取所有代码块。

        流程:
          1. 读文件 → 计算 SHA-256 → 增量检测
          2. ast.parse → 遍历顶层节点
          3. FunctionDef/AsyncFunctionDef → function chunk
          4. ClassDef → class chunk + 遍历 body 提取 methods
          5. 未被覆盖的行 → module_level chunk
        """
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

    def index_directory(self, root: Path) -> list[CodeChunk]:
        """递归解析目录下所有 .py 文件，返回所有代码块。"""
        files = self.walk_directory(root)
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
        }

    @property
    def indexed_files(self) -> int:
        """已索引文件数。"""
        return len(self._file_hashes)
