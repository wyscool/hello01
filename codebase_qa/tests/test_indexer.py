# ============================================================
# codebase_qa/tests/test_indexer.py — CodeIndexer AST 解析测试
# ============================================================

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from codebase_qa.indexer import CodeIndexer, CodeChunk


# ============================================================
# fixtures
# ============================================================

@pytest.fixture
def indexer():
    return CodeIndexer()


@pytest.fixture
def temp_py_file(tmp_path):
    """创建临时 .py 文件用于 AST 解析测试。"""
    src = tmp_path / "src"
    src.mkdir()
    file_path = src / "sample.py"
    file_path.write_text("""\"\"\"Module docstring.\"\"\"
import os
from typing import Optional

CONSTANT = 42


def simple_function():
    return 1


def func_with_docstring(a: int, b: str = "default") -> bool:
    \"\"\"This is a docstring.\"\"\"
    return a > 0


async def async_func():
    return await some_call()


class SimpleClass:
    \"\"\"Class docstring.\"\"\"

    def method_one(self):
        return 1

    async def async_method(self, x: int) -> str:
        \"\"\"Async method docstring.\"\"\"
        return str(x)


class EmptyClass:
    pass


@staticmethod
def decorated_func():
    return "decorated"


@retry(max_attempts=3, delay=0.5)
def complex_decorated():
    pass
""")
    return file_path


@pytest.fixture
def empty_py_file(tmp_path):
    """空 .py 文件。"""
    src = tmp_path / "src2"
    src.mkdir()
    fp = src / "empty.py"
    fp.write_text("")
    return fp


@pytest.fixture
def syntax_error_file(tmp_path):
    """语法错误的 .py 文件。"""
    src = tmp_path / "src3"
    src.mkdir()
    fp = src / "bad.py"
    fp.write_text("def broken(\n")
    return fp


# ============================================================
# TestCodeChunk
# ============================================================

class TestCodeChunk:
    def test_embed_text_format(self):
        chunk = CodeChunk(
            name="my_func", type="function",
            file_path="test.py", start_line=10, end_line=15,
            code_text="def my_func():\n    return 1",
            docstring="My docstring.",
            signature="def my_func():",
            file_hash="abc123def456",
        )
        text = chunk.embed_text
        assert "function: def my_func():" in text
        assert "My docstring." in text
        assert "def my_func()" in text

    def test_chunk_id_format(self):
        chunk = CodeChunk(
            name="my_func", type="function",
            file_path="test.py", start_line=10, end_line=15,
            file_hash="abc123def4567890123456789012345678901234567890123456789012345678",
        )
        assert chunk.chunk_id.startswith("abc123de")
        assert ":my_func:10" in chunk.chunk_id

    def test_chunk_ids_are_unique(self):
        """同文件不同行的同名函数应有唯一 ID。"""
        c1 = CodeChunk(name="foo", type="function", file_path="a.py",
                       start_line=10, end_line=12, file_hash="abc")
        c2 = CodeChunk(name="foo", type="function", file_path="a.py",
                       start_line=20, end_line=22, file_hash="abc")
        assert c1.chunk_id != c2.chunk_id


# ============================================================
# TestWalkDirectory
# ============================================================

class TestWalkDirectory:
    def test_finds_py_files(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("x = 1")
        (src / "b.py").write_text("y = 2")
        (src / "not_py.txt").write_text("hello")
        sub = src / "sub"
        sub.mkdir()
        (sub / "c.py").write_text("z = 3")

        indexer = CodeIndexer()
        files = indexer.walk_directory(src)
        py_files = {f.name for f in files}
        assert py_files == {"a.py", "b.py", "c.py"}

    def test_excludes_dirs(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("x = 1")
        tests_dir = src / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_main.py").write_text("def test(): pass")
        venv_dir = src / "venv"
        venv_dir.mkdir()
        (venv_dir / "lib.py").write_text("pass")

        indexer = CodeIndexer(exclude_dirs={"tests", "venv"})
        files = indexer.walk_directory(src)
        py_files = {f.name for f in files}
        assert py_files == {"main.py"}

    def test_sorted_output(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "c.py").write_text("")
        (src / "a.py").write_text("")
        (src / "b.py").write_text("")
        indexer = CodeIndexer()
        files = indexer.walk_directory(src)
        names = [f.name for f in files]
        assert names == ["a.py", "b.py", "c.py"]


# ============================================================
# TestHashFile
# ============================================================

class TestHashFile:
    def test_same_content_same_hash(self, tmp_path):
        fp1 = tmp_path / "a.py"
        fp1.write_text("x = 1")
        fp2 = tmp_path / "b.py"
        fp2.write_text("x = 1")
        assert CodeIndexer.hash_file(fp1) == CodeIndexer.hash_file(fp2)

    def test_different_content_different_hash(self, tmp_path):
        fp1 = tmp_path / "a.py"
        fp1.write_text("x = 1")
        fp2 = tmp_path / "b.py"
        fp2.write_text("x = 2")
        assert CodeIndexer.hash_file(fp1) != CodeIndexer.hash_file(fp2)

    def test_is_unchanged(self, tmp_path):
        fp = tmp_path / "test.py"
        fp.write_text("x = 1")
        indexer = CodeIndexer()
        assert not indexer.is_unchanged(fp)
        # 手动设置缓存
        indexer._file_hashes[str(fp.resolve())] = CodeIndexer.hash_file(fp)
        assert indexer.is_unchanged(fp)


# ============================================================
# TestParseFile — 核心: AST 解析
# ============================================================

class TestParseFile:
    def test_extracts_functions(self, indexer, temp_py_file):
        chunks = indexer.parse_file(temp_py_file)
        funcs = [c for c in chunks if c.type == "function"]
        names = {c.name for c in funcs}
        assert "simple_function" in names
        assert "func_with_docstring" in names
        assert "async_func" in names
        assert "decorated_func" in names
        assert "complex_decorated" in names

    def test_extracts_classes(self, indexer, temp_py_file):
        chunks = indexer.parse_file(temp_py_file)
        classes = [c for c in chunks if c.type == "class"]
        names = {c.name for c in classes}
        assert "SimpleClass" in names
        assert "EmptyClass" in names

    def test_extracts_methods(self, indexer, temp_py_file):
        chunks = indexer.parse_file(temp_py_file)
        methods = [c for c in chunks if c.type == "method"]
        names = {c.name for c in methods}
        assert "SimpleClass.method_one" in names
        assert "SimpleClass.async_method" in names

    def test_extracts_module_level(self, indexer, temp_py_file):
        chunks = indexer.parse_file(temp_py_file)
        mod = [c for c in chunks if c.type == "module_level"]
        assert len(mod) == 1
        assert "import os" in mod[0].code_text
        assert "CONSTANT" in mod[0].code_text

    def test_docstring_extraction(self, indexer, temp_py_file):
        chunks = indexer.parse_file(temp_py_file)
        func = [c for c in chunks if c.name == "func_with_docstring"][0]
        assert func.docstring == "This is a docstring."

    def test_method_docstring(self, indexer, temp_py_file):
        chunks = indexer.parse_file(temp_py_file)
        method = [c for c in chunks if c.name == "SimpleClass.async_method"][0]
        assert method.docstring == "Async method docstring."

    def test_signature_includes_decorator(self, indexer, temp_py_file):
        chunks = indexer.parse_file(temp_py_file)
        func = [c for c in chunks if c.name == "decorated_func"][0]
        assert "@staticmethod" in func.signature
        assert "def decorated_func()" in func.signature

    def test_line_numbers_accurate(self, indexer, temp_py_file):
        chunks = indexer.parse_file(temp_py_file)
        func = [c for c in chunks if c.name == "simple_function"][0]
        assert func.start_line > 0
        assert func.end_line >= func.start_line

    def test_empty_file(self, indexer, empty_py_file):
        chunks = indexer.parse_file(empty_py_file)
        # 空文件只有一个 module_level chunk
        types = {c.type for c in chunks}
        assert types == {"module_level"} or len(chunks) == 0

    def test_syntax_error_graceful(self, indexer, syntax_error_file):
        chunks = indexer.parse_file(syntax_error_file)
        assert chunks == []  # 不应崩溃

    def test_skip_unchanged_files(self, indexer, temp_py_file):
        # 第一次解析
        chunks1 = indexer.parse_file(temp_py_file)
        assert len(chunks1) > 0
        # 第二次（未修改）应返回空
        chunks2 = indexer.parse_file(temp_py_file)
        assert chunks2 == []

    def test_file_hash_stored(self, indexer, temp_py_file):
        indexer.parse_file(temp_py_file)
        key = str(temp_py_file.resolve())
        assert key in indexer._file_hashes

    def test_chunk_has_valid_metadata(self, indexer, temp_py_file):
        chunks = indexer.parse_file(temp_py_file)
        func = [c for c in chunks if c.name == "simple_function"][0]
        meta = indexer.chunk_to_metadata(func)
        assert meta["name"] == "simple_function"
        assert meta["type"] == "function"
        assert isinstance(meta["start_line"], int)
        assert isinstance(meta["end_line"], int)
        assert meta["file_path"].endswith(".py")


# ============================================================
# TestIndexDirectory
# ============================================================

class TestIndexDirectory:
    def test_indexes_multiple_files(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def foo():\n    return 1\n")
        (src / "b.py").write_text("class Bar:\n    pass\n")

        indexer = CodeIndexer()
        chunks = indexer.index_directory(src)
        func = [c for c in chunks if c.type == "function"]
        cls  = [c for c in chunks if c.type == "class"]
        assert len(func) >= 1
        assert len(cls) >= 1

    def test_incremental_reindex(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def foo():\n    return 1\n")

        indexer = CodeIndexer()
        chunks1 = indexer.index_directory(src)
        # 再次索引（未修改）
        chunks2 = indexer.index_directory(src)
        assert len(chunks2) == 0  # 无新 chunk
        assert indexer.indexed_files == 1

    def test_new_file_detected(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def foo():\n    return 1\n")

        indexer = CodeIndexer()
        indexer.index_directory(src)

        # 添加新文件
        (src / "b.py").write_text("def bar():\n    return 2\n")
        chunks2 = indexer.index_directory(src)
        assert len(chunks2) > 0  # 新文件被检测到
