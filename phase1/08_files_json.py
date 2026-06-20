# ============================================================
# Phase 1, Lesson 08: 文件与 JSON —— open()、pathlib、json
# ============================================================
#
# 本课目标:
#   1. open() —— 打开文件的正确姿势
#   2. with 语句 —— 上下文管理器, 自动关闭文件
#   3. 读取: read() / readline() / readlines()
#   4. 写入: write() / writelines()
#   5. 文件模式: 'r', 'w', 'a', 'x', 'b', '+'
#   6. encoding —— 为什么必须指定 utf-8
#   7. pathlib.Path —— 面向对象的路径操作
#   8. json 模块 —— 序列化与反序列化
#   9. json + 文件: json.dump() / json.load()
#   10. 实战: 构建一个简单的笔记存储系统
#
# 预计阅读 + 实操时间: 35-45 分钟
# ============================================================


# ------------------------------------------------------------
# 一、open() 基础 —— 打开文件的入口
# ------------------------------------------------------------
# Java:
#   BufferedReader reader = new BufferedReader(new FileReader("data.txt"));
#   // 需要 finally { reader.close(); } 或 try-with-resources
# Python:
#   f = open("data.txt", "r", encoding="utf-8")
#   返回一个文件对象 (file object), 类似 Java 的 Reader/Writer。
#
# open() 的三个关键参数:
#   file:  文件路径 (字符串或 Path 对象)
#   mode:  打开模式, 默认 'r' (只读文本)
#   encoding: 编码, 建议始终指定 utf-8

# 基本用法 —— 打开、读取、关闭
f = open("phase1/08_demo_data.txt", "r", encoding="utf-8")
content = f.read()
f.close()  # ⚠️ 必须手动关闭! 忘记 close 会导致资源泄露
print(content)


# ------------------------------------------------------------
# 二、with 语句 —— Python 的 "try-with-resources"
# ------------------------------------------------------------
# Java (since 7):
#   try (BufferedReader r = new BufferedReader(...)) {
#       // 自动 close
#   }
# Python:
#   with open(...) as f:
#       // 缩进块结束后自动 close, 即使发生异常也会关闭
#
# with 是 Python 处理文件的标准方式。永远优先使用 with!

print("=" * 50)
print("with 语句 —— 自动关闭文件")
print("=" * 50)

with open("phase1/08_demo_data.txt", "r", encoding="utf-8") as f:
    content = f.read()

# 此时文件已经关闭了
print(f"文件已关闭: {f.closed}")  # True
print(f"内容:\n{content}")


# ------------------------------------------------------------
# 三、读取文件的三种方式
# ------------------------------------------------------------
# read()       → 一次性读取整个文件到字符串 → 适合小文件
# readline()   → 读取一行 (保留换行符)       → 适合逐行处理
# readlines()  → 读取所有行到 list[str]      → 适合小文件 + 需要行列表
# for line in f → 逐行迭代 (最省内存!)      → 适合大文件

print("=" * 50)
print("读取方式对比")
print("=" * 50)

# 方式 1: read() —— 整个文件读入一个字符串
with open("phase1/08_demo_data.txt", "r", encoding="utf-8") as f:
    all_at_once = f.read()
    print(f"read() - 字符数: {len(all_at_once)}")

# 方式 2: readlines() —— 读取所有行到列表
with open("phase1/08_demo_data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    print(f"readlines() - 行数: {len(lines)}")
    for i, line in enumerate(lines[:3], 1):
        print(f"  第{i}行: {line.rstrip()}")  # rstrip() 去掉末尾换行符

# 方式 3: for line in f —— 逐行迭代 (推荐!)
# 文件对象本身就是可迭代的, 每次 yield 一行。
# 不会一次性把所有行加载到内存, 适合处理大文件。
print("\nfor line in f (逐行迭代):")
with open("phase1/08_demo_data.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if i <= 3:
            print(f"  第{i}行: {line.rstrip()}")


# ------------------------------------------------------------
# 四、写入文件 —— write() / writelines()
# ------------------------------------------------------------
# 'w' 模式: 写入 (覆盖已有内容)
# 'a' 模式: 追加 (append, 在末尾添加)
# 'x' 模式: 排他创建 (文件已存在则报错, 类似 Java 的 CREATE_NEW)

from pathlib import Path

output_dir = Path("phase1/output")
output_dir.mkdir(exist_ok=True)  # 确保目录存在

# write() —— 写入字符串
with open(output_dir / "hello.txt", "w", encoding="utf-8") as f:
    f.write("第一行\n")
    f.write("第二行\n")
    f.write("你好, 世界!\n")

print("\n" + "=" * 50)
print("写入文件")
print("=" * 50)

# 验证写入结果
with open(output_dir / "hello.txt", "r", encoding="utf-8") as f:
    print(f.read())

# writelines() —— 写入字符串列表 (不会自动加换行!)
lines_to_write = ["苹果\n", "香蕉\n", "橙子\n"]
with open(output_dir / "fruits.txt", "w", encoding="utf-8") as f:
    f.writelines(lines_to_write)

print("fruits.txt 写入完成")

# 追加模式 —— 不覆盖, 在末尾添加
with open(output_dir / "hello.txt", "a", encoding="utf-8") as f:
    f.write("这是追加的一行\n")

print("追加后:")
with open(output_dir / "hello.txt", "r", encoding="utf-8") as f:
    print(f.read())


# ------------------------------------------------------------
# 五、文件模式速查
# ------------------------------------------------------------
#   模式      读/写   文本/二进制   覆盖/追加   说明
#   'r'       读      文本          -          文件必须存在
#   'w'       写      文本          覆盖       文件不存在则创建
#   'a'       写      文本          追加       文件不存在则创建
#   'x'       写      文本          排他       文件存在则报错
#   'rb'      读      二进制        -          读取图片、PDF 等
#   'wb'      写      二进制        覆盖       写入二进制数据
#   'r+'      读写    文本          -          文件必须存在
#   'w+'      读写    文本          覆盖       -
#
# ⚠️ 默认模式是 'r', 默认编码取决于系统 (macOS/Linux 通常是 utf-8, Windows 可能是 GBK)。
#    跨平台代码必须显式指定 encoding="utf-8"!

# 演示: 读取二进制文件
import os

# 创建一个简单的二进制文件
binary_data = bytes([0x48, 0x65, 0x6C, 0x6C, 0x6F])  # "Hello" 的字节
with open(output_dir / "binary_test.bin", "wb") as f:
    f.write(binary_data)

with open(output_dir / "binary_test.bin", "rb") as f:
    raw = f.read()
    print(f"\n二进制读取: {raw} → 解码: {raw.decode('utf-8')}")


# ------------------------------------------------------------
# 六、encoding —— 为什么必须指定
# ------------------------------------------------------------
# 不指定 encoding 时, Python 使用系统默认编码。
# macOS/Linux 默认 utf-8, Windows 默认 GBK。
# 写"你好"在 macOS 上正常, 到 Windows 上就乱码。
# 所以: 始终显式指定 encoding="utf-8"。

# 演示: 读写中文
chinese_text = "Python 的文件读写非常简单\n比 Java 的 IO 流直观多了\n"

with open(output_dir / "chinese.txt", "w", encoding="utf-8") as f:
    f.write(chinese_text)

# 如果用错误的编码读取会怎样?
with open(output_dir / "chinese.txt", "r", encoding="ascii") as f:
    try:
        f.read()
    except UnicodeDecodeError as e:
        print(f"\n编码错误演示: {e}")


# ------------------------------------------------------------
# 七、pathlib.Path —— 现代路径操作
# ------------------------------------------------------------
# Java: java.nio.file.Path (since Java 7)
# Python: pathlib.Path (since 3.4, 推荐替代 os.path)
#
# Path 是面向对象的路径表示, 不是字符串。
# 好处: 跨平台 (/ vs \)、方法链式调用、直观的运算符重载。

print("=" * 50)
print("pathlib.Path 操作")
print("=" * 50)

# 创建 Path 对象
data_dir = Path("phase1/output")
config_file = data_dir / "config.json"    # / 运算符拼接路径!
print(f"拼接路径: {config_file}")

# 常用属性和方法
print(f"文件名:     {config_file.name}")       # config.json
print(f"后缀:       {config_file.suffix}")     # .json
print(f"不带后缀:   {config_file.stem}")       # config
print(f"父目录:     {config_file.parent}")     # phase1/output
print(f"绝对路径:   {config_file.resolve()}")  # /Users/.../phase1/output/config.json
print(f"是否存在:   {config_file.exists()}")    # False (还没创建)
print(f"是否是文件: {config_file.is_file()}")   # False

# 创建目录
new_dir = data_dir / "sub" / "deep"
new_dir.mkdir(parents=True, exist_ok=True)  # mkdir -p 的效果!
print(f"\n已创建目录: {new_dir}")

# 遍历目录
print(f"\n{data_dir} 目录内容:")
for item in data_dir.iterdir():
    if item.is_file():
        size = item.stat().st_size  # 文件大小 (字节)
        print(f"  📄 {item.name:<30} {size:>6} bytes")
    elif item.is_dir():
        print(f"  📁 {item.name}/")

# 通配符查找
print(f"\n所有 .txt 文件:")
for txt_file in data_dir.glob("*.txt"):
    print(f"  {txt_file.name}")

# 递归查找
print(f"\n所有 .txt 文件 (递归):")
for txt_file in data_dir.rglob("*.txt"):
    print(f"  {txt_file.relative_to(data_dir)}")

# ⚠️ 重要区别:
# Path 对象和 str 的互操作
p = Path("demo.txt")
# open(p) — ✅ Python 3.6+ 原生支持 Path 对象
# json.load(p) — ❌ json 模块不接受 Path 对象, 需要 str(p)
# os.path.join(p, "sub") — ❌ os.path 函数不接受 Path 对象


# ------------------------------------------------------------
# 八、json 模块 —— Python 与 JSON 的无缝转换
# ------------------------------------------------------------
# Java: Jackson / Gson → 需要定义类, 然后 objectMapper.readValue(...)
# Python: json 模块 → dict/list 直接转 JSON, 不需要定义类!
#
# 四个核心函数:
#   json.dumps(obj)  → Python 对象 → JSON 字符串
#   json.loads(str)  → JSON 字符串 → Python 对象
#   json.dump(obj, f) → Python 对象 → 写入 JSON 文件
#   json.load(f)     → JSON 文件 → Python 对象
#
# 记忆技巧: 带 s 的是字符串操作 (string), 不带 s 的是文件操作

import json

print("\n" + "=" * 50)
print("json 序列化与反序列化")
print("=" * 50)

# --- dumps: Python → JSON 字符串 ---
# Python 类型到 JSON 类型的自动映射:
#   dict   → JSON object
#   list   → JSON array
#   str    → JSON string
#   int/float → JSON number
#   bool   → JSON true/false  (注意: True → true, False → false)
#   None   → JSON null        (注意: None → null)

user = {
    "name": "小明",
    "age": 25,
    "skills": ["Python", "Java", "SQL"],
    "active": True,
    "address": None,
}

json_str = json.dumps(user, ensure_ascii=False, indent=2)
# ensure_ascii=False: 保留中文, 不转成 \uXXXX
# indent=2: 格式化输出, 否则是一行
print(f"dumps 结果:\n{json_str}")

# --- loads: JSON 字符串 → Python ---
parsed = json.loads(json_str)
print(f"\nloads 类型: {type(parsed)}")  # <class 'dict'>
print(f"name: {parsed['name']}, skills: {parsed['skills']}")

# --- dump: Python → JSON 文件 ---
users = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35},
]

json_path = output_dir / "users.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(users, f, ensure_ascii=False, indent=2)

print(f"\n已写入: {json_path} ({json_path.stat().st_size} bytes)")

# --- load: JSON 文件 → Python ---
with open(json_path, "r", encoding="utf-8") as f:
    loaded_users = json.load(f)

print(f"从文件读取: {len(loaded_users)} 个用户")
for u in loaded_users:
    print(f"  {u['name']}: {u['age']} 岁")


# ------------------------------------------------------------
# 九、json 序列化进阶 —— 自定义类型
# ------------------------------------------------------------
# json 默认只能序列化: dict, list, str, int, float, bool, None
# 如果是自定义类, 需要提供转换函数。

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Note:
    title: str
    content: str
    created_at: datetime
    tags: list[str]


# 自定义 JSON 编码器
class NoteEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Note):
            return {
                "title": obj.title,
                "content": obj.content,
                "created_at": obj.created_at.isoformat(),  # datetime → str
                "tags": obj.tags,
            }
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


note = Note(
    title="学习笔记",
    content="今天学了 json 模块, 非常简洁",
    created_at=datetime.now(),
    tags=["python", "json"],
)

# 方式 1: 使用自定义 Encoder
note_json = json.dumps(note, cls=NoteEncoder, ensure_ascii=False, indent=2)
print("\n" + "=" * 50)
print("自定义类型序列化")
print("=" * 50)
print(note_json)

# 方式 2: 用 default 参数 (更简洁, 不需要定义类)
def note_to_dict(obj):
    if isinstance(obj, Note):
        return {
            "title": obj.title,
            "content": obj.content,
            "created_at": obj.created_at.isoformat(),
            "tags": obj.tags,
        }
    raise TypeError(f"无法序列化类型: {type(obj)}")

note_json2 = json.dumps(note, default=note_to_dict, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# 综合实战: 构建一个简单的笔记存储系统
# ------------------------------------------------------------
# 练习文件读写 + JSON + pathlib 的综合应用。
# 这是一个命令行的笔记管理工具, 支持: 创建、列表、搜索、删除。

class NotesApp:
    """
    基于文件的笔记应用。
    每篇笔记存储为一个 .json 文件, 包含 title、content、tags、created_at。
    """

    def __init__(self, storage_dir: str = "phase1/output/notes"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create(self, title: str, content: str, tags: list[str] | None = None) -> Note:
        """创建一篇新笔记。"""
        note = Note(
            title=title,
            content=content,
            created_at=datetime.now(),
            tags=tags or [],
        )

        # 用标题生成文件名 (简单处理: 替换空格和特殊字符)
        safe_name = title.replace(" ", "_").replace("/", "-")
        note_file = self.storage_dir / f"{safe_name}.json"

        # 如果文件已存在, 加时间戳区分
        if note_file.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            note_file = self.storage_dir / f"{safe_name}_{ts}.json"

        with open(note_file, "w", encoding="utf-8") as f:
            json.dump(note, f, cls=NoteEncoder, ensure_ascii=False, indent=2)

        print(f"  笔记已保存: {note_file.name}")
        return note

    def list_all(self) -> list[dict]:
        """列出所有笔记的摘要信息。"""
        notes = []
        for note_file in sorted(self.storage_dir.glob("*.json")):
            with open(note_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            notes.append({
                "file": note_file.name,
                "title": data["title"],
                "created_at": data["created_at"],
                "tags": data.get("tags", []),
                "size": note_file.stat().st_size,
            })
        return notes

    def search(self, keyword: str) -> list[dict]:
        """在标题和内容中搜索关键词。"""
        results = []
        for note_file in self.storage_dir.glob("*.json"):
            with open(note_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if keyword.lower() in data["title"].lower() or keyword.lower() in data["content"].lower():
                results.append({
                    "file": note_file.name,
                    "title": data["title"],
                    "created_at": data["created_at"],
                })
        return results

    def delete(self, filename: str) -> bool:
        """删除指定笔记文件。"""
        note_file = self.storage_dir / filename
        if note_file.exists():
            note_file.unlink()  # 删除文件
            print(f"  已删除: {filename}")
            return True
        print(f"  文件不存在: {filename}")
        return False

    def stats(self) -> dict:
        """统计信息。"""
        all_notes = list(self.storage_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in all_notes)
        all_tags: set[str] = set()
        for note_file in all_notes:
            with open(note_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_tags.update(data.get("tags", []))
        return {
            "total_notes": len(all_notes),
            "total_size": total_size,
            "all_tags": sorted(all_tags),
        }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  综合实战: 笔记存储系统")
    print("=" * 60)

    app = NotesApp()

    # 1. 创建几篇笔记
    print("\n📝 创建笔记:")
    app.create("Python 文件操作", "open() 配合 with 语句是最佳实践", ["python", "io"])
    app.create("JSON 序列化", "json.dumps/dumps 四个函数要记牢", ["python", "json"])
    app.create("pathlib 入门", "Path 对象比 os.path 更直观, 用 / 拼接路径", ["python", "pathlib"])
    app.create("Java vs Python IO", "Python 的 IO 比 Java 简洁太多, 不需要装饰器模式", ["java", "comparison"])

    # 2. 列出所有笔记
    print("\n📋 所有笔记:")
    for note in app.list_all():
        tags_str = ", ".join(note["tags"])
        print(f"  [{note['file']}] {note['title']} ({note['size']} bytes) [{tags_str}]")

    # 3. 搜索
    print("\n🔍 搜索 'json':")
    for result in app.search("json"):
        print(f"  {result['file']}: {result['title']}")

    # 4. 统计
    stats = app.stats()
    print(f"\n📊 统计: {stats['total_notes']} 篇笔记, "
          f"共 {stats['total_size']} bytes, "
          f"标签: {stats['all_tags']}")

    # 5. 删除一篇笔记
    print("\n🗑️ 删除笔记:")
    app.delete("Java_vs_Python_IO.json")

    print(f"\n📊 删除后统计: {app.stats()['total_notes']} 篇笔记")


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 写一个函数 file_stats(path: str) -> dict:
#    统计给定文本文件的: 字符数、行数、单词数、空行数。

def file_stats(path: str) -> dict:
    """统计文本文件的字符数、行数、单词数和空行数。"""
    p = Path(path)
    if not p.exists():
        return {"error": f"文件不存在: {path}"}
    text = p.read_text(encoding="utf-8")
    lines = text.split("\n")
    empty_lines = sum(1 for line in lines if line.strip() == "")
    word_count = len(text.split())
    return {
        "file": p.name,
        "chars": len(text),
        "lines": len(lines),
        "words": word_count,
        "empty_lines": empty_lines,
    }

print("--- file_stats ---")
demo = Path("test_file_stats.txt")
demo.write_text("hello world\n\nfoo bar baz\n\n")
print(file_stats(str(demo)))
demo.unlink()
print()

#
# 2. 写一个 CSV 转 JSON 的工具:
#    def csv_to_json(csv_path: str, json_path: str) -> None:

def csv_to_json(csv_path: str, json_path: str) -> None:
    """将 CSV 文件转为 JSON 文件。第一行是列名。"""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        headers = f.readline().strip().split(",")
        for line in f:
            values = line.strip().split(",")
            record = dict(zip(headers, values))
            records.append(record)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  导出 {len(records)} 条记录: {csv_path} → {json_path}")

print("--- csv_to_json ---")
csv_path = Path("test_fruits.csv")
csv_path.write_text("name,color,price\napple,red,5\nbanana,yellow,3\n", encoding="utf-8")
csv_to_json(str(csv_path), "test_fruits.json")
print(f"  结果: {Path('test_fruits.json').read_text()}")
csv_path.unlink()
Path("test_fruits.json").unlink()
print()

#
# 3. 写一个函数 backup_files(source_dir: str, backup_dir: str, pattern: str = "*.txt"):

def backup_files(source_dir: str, backup_dir: str, pattern: str = "*.txt") -> int:
    """备份匹配 pattern 的文件到 backup_dir, 并加 .bak 后缀。"""
    src = Path(source_dir)
    dst = Path(backup_dir)
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src.glob(pattern):
        target = dst / f"{f.name}.bak"
        target.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        count += 1
        print(f"  备份: {f.name} → {target.name}")
    return count

print("--- backup_files ---")
src_dir = Path("test_backup_src")
src_dir.mkdir(exist_ok=True)
(src_dir / "a.txt").write_text("content a")
(src_dir / "b.txt").write_text("content b")
(src_dir / "c.md").write_text("markdown")
n = backup_files(str(src_dir), "test_backup_dst")
print(f"  共备份 {n} 个文件")

# 清理
import shutil
shutil.rmtree(src_dir)
shutil.rmtree("test_backup_dst")
print()

#
# 4. 扩展 NotesApp:
#    - 添加 update(filename: str, new_content: str) 方法
#    - 添加 export_all(export_path: str) 方法
#    - 添加 import_notes(import_path: str) 方法

def notesapp_update(self, filename: str, new_content: str) -> bool:
    """更新笔记内容。"""
    note_file = self.storage_dir / filename
    if not note_file.exists():
        print(f"  文件不存在: {filename}")
        return False
    data = json.loads(note_file.read_text(encoding="utf-8"))
    data["content"] = new_content
    data["updated_at"] = datetime.now().isoformat()
    note_file.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"  已更新: {filename}")
    return True

def notesapp_export_all(self, export_path: str) -> int:
    """将所有笔记导出为一个 JSON 文件。"""
    all_notes = []
    for note_file in sorted(self.storage_dir.glob("*.json")):
        all_notes.append(json.loads(note_file.read_text(encoding="utf-8")))
    Path(export_path).write_text(
        json.dumps(all_notes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  已导出 {len(all_notes)} 篇笔记 → {export_path}")
    return len(all_notes)

def notesapp_import_notes(self, import_path: str) -> int:
    """从 JSON 文件批量导入笔记。"""
    data = json.loads(Path(import_path).read_text(encoding="utf-8"))
    count = 0
    for item in data:
        title = item.get("title", "untitled")
        content = item.get("content", "")
        tags = item.get("tags", [])
        safe_name = title.replace(" ", "_").replace("/", "-")
        note_file = self.storage_dir / f"{safe_name}.json"
        if note_file.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            note_file = self.storage_dir / f"{safe_name}_{ts}.json"
        note_file.write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1
    print(f"  已导入 {count} 篇笔记 ← {import_path}")
    return count

# 绑定到 NotesApp
NotesApp.update = notesapp_update
NotesApp.export_all = notesapp_export_all
NotesApp.import_notes = notesapp_import_notes

print("--- NotesApp 扩展 ---")
app2 = NotesApp()
app2.create("test", "original content", ["test"])
app2.update("test.json", "updated content!")
app2.export_all("test_export.json")
print(f"  导出文件内容行数: {len(Path('test_export.json').read_text().splitlines())}")
app2.delete("test.json")
app2.import_notes("test_export.json")
print(f"  导入后笔记数: {app2.stats()['total_notes']}")
app2.delete("test.json")
Path("test_export.json").unlink()
print()

#
# 5. (探索) json.dumps 有个 default 参数, 可以让它序列化更多类型。

print("--- json.dumps default ---")
print(f"  datetime: {json.dumps(datetime.now(), default=str)}")
print(f"  set: {json.dumps({1, 2, 3}, default=list)}")
print(f"  complex: {json.dumps(3+4j, default=str)}")
print()

#
# 6. (挑战) 实现一个简单的日志文件轮转 (log rotation):

class RotatingLogger:
    """日志轮转器: 文件超过 max_bytes 时自动归档。"""

    def __init__(self, log_path: str, max_bytes: int = 1024):
        self.log_path = Path(log_path)
        self.max_bytes = max_bytes

    def _rotate(self):
        """将当前日志重命名为带日期的归档文件。"""
        if not self.log_path.exists():
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive = self.log_path.with_stem(
            f"{self.log_path.stem}_{ts}"
        )
        self.log_path.rename(archive)
        print(f"  🔄 轮转: {self.log_path.name} → {archive.name}")

    def write(self, message: str):
        """写入日志, 自动检查大小并轮转。"""
        if self.log_path.exists() and self.log_path.stat().st_size >= self.max_bytes:
            self._rotate()
        timestamp = datetime.now().isoformat()
        line = f"[{timestamp}] {message}\n"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

print("--- RotatingLogger ---")
logger = RotatingLogger("test_log.txt", max_bytes=100)
for i in range(20):
    logger.write(f"这是第 {i+1} 条日志, 用来测试轮转功能")
# 检查轮转文件
archives = sorted(Path(".").glob("test_log_*.txt"))
print(f"  当前日志: {Path('test_log.txt').exists()}")
print(f"  归档文件: {len(archives)} 个")
# 清理
Path("test_log.txt").unlink(missing_ok=True)
for a in archives:
    a.unlink()
print()


# 做完后告诉我:
#   - Python 的文件 IO 和 Java 的相比, 你更喜欢哪个? 为什么?
#   - pathlib 的 / 运算符拼接路径, 你觉得是好设计还是过度简化?
# 我们继续 Lesson 09: async/await 异步基础。
# ============================================================
