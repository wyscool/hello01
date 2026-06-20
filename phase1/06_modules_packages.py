# ============================================================
# Phase 1, Lesson 06: 模块与包 —— import、模块、包
# ============================================================
#
# 本课目标:
#   1. import 语句的基本用法
#   2. 模块 (module) —— 一个 .py 文件就是一个模块
#   3. 包 (package) —— 目录 + __init__.py
#   4. from ... import ... 的各种形式
#   5. import 搜索路径 (sys.path)
#   6. 相对导入 vs 绝对导入
#   7. __name__ 与 __main__ 的再次强调
#   8. __all__ —— 控制公开接口
#   9. 模块的私有约定 (_ 前缀)
#   10. 标准库初探 (os, sys, pathlib)
#
# 预计阅读 + 实操时间: 30-40 分钟
#
# 课前准备:
#   确保本文件同级目录下有 demo_pkg/ 包:
#     demo_pkg/
#       __init__.py
#       calculator.py
#       string_tools.py
# ============================================================


# ------------------------------------------------------------
# 一、什么是模块? —— 一个 .py 文件就是一个模块
# ------------------------------------------------------------
# Java: 源文件通常属于某个 package, 编译成 .class 文件。
# Python: 没有编译步骤。一个 .py 文件直接就是一个模块,
#         文件名(不含 .py)就是模块名。
#
# 例如: calculator.py → 模块名就是 "calculator"


# ------------------------------------------------------------
# 二、import 的基本用法
# ------------------------------------------------------------
# Java: import com.example.utils.Calculator;
# Python: 几种形式, 更灵活。

# 形式 1: 导入整个模块
#   使用方式: 模块名.函数名()
import calculator

print(f"3 + 5 = {calculator.add(3, 5)}")
print(f"PI = {calculator.PI}")

# 形式 2: 从模块导入特定名称
#   使用方式: 直接使用函数名, 不需要模块名前缀
from calculator import multiply, power

print(f"4 * 7 = {multiply(4, 7)}")
print(f"2^10 = {power(2, 10)}")

# 形式 3: 导入所有公开名称 (不推荐, 除非你很清楚在做什么)
#   这会导入模块里所有不以 _ 开头的名称
from calculator import *

print(f"10 - 3 = {subtract(10, 3)}")  # 因为用了 import *, subtract 可以直接用

# ⚠️ import * 的风险: 名称冲突!
# 如果两个模块都有同名函数, 后导入的会覆盖前面的。
# 生产代码中尽量避免 import *。

# 形式 4: 给模块起别名 (as)
#   当模块名太长, 或者需要避免命名冲突时使用。
import calculator as calc

print(f"别名调用: {calc.add(1, 2)}")

# 形式 5: 给导入的名称起别名
from calculator import divide as div

print(f"10 / 3 = {div(10, 3):.2f}")


# ------------------------------------------------------------
# 三、包 (Package) —— 组织多个模块的目录
# ------------------------------------------------------------
# Java: package com.example.utils;
#       包结构对应目录结构: com/example/utils/
# Python: 包就是一个包含 __init__.py 的目录。
#         __init__.py 可以为空, 也可以包含初始化代码。

# 导入包中的模块:
import demo_pkg.calculator

print(f"\n包导入: {demo_pkg.calculator.add(10, 20)}")

# 从包导入模块:
from demo_pkg import string_tools

print(f"元音数量: {string_tools.count_vowels('Hello World')}")

# 从包的模块导入特定函数:
from demo_pkg.string_tools import capitalize_words, is_palindrome

print(f"首字母大写: {capitalize_words('hello python')}")
print(f"是否是回文: {is_palindrome('A man a plan a canal Panama')}")


# ------------------------------------------------------------
# 四、import 的执行机制 —— 重要!
# ------------------------------------------------------------
# 当你 import 一个模块时, Python 会:
#   1. 找到模块文件
#   2. 执行模块中的**顶层代码** (函数定义、类定义、变量赋值等)
#   3. 创建一个模块对象, 把执行结果(变量、函数、类)存进去
#   4. 在 sys.modules 中缓存, 供后续导入复用
#
# 这意味着: 一个模块在一个程序运行期间只会被执行一次!
# 即使多次 import, 也是复用缓存的结果。

print(f"\ncalculator 模块已加载: {'calculator' in dir()}")

# 验证: 再次 import 不会重新执行模块代码
import calculator as calc2  # 和之前的 calc 指向同一个模块对象
print(f"calc is calc2: {calc is calc2}")  # True


# ------------------------------------------------------------
# 五、import 搜索路径 —— Python 怎么找到模块?
# ------------------------------------------------------------
# Python 按以下顺序查找模块:
#   1. 内置模块 (builtins)
#   2. sys.path 列表中的目录
#
# sys.path 包括:
#   - 当前运行脚本所在的目录 (或当前工作目录)
#   - PYTHONPATH 环境变量中的目录
#   - Python 安装时的默认目录 (标准库位置)

import sys

print("\nimport 搜索路径 (sys.path 前 5 项):")
for i, path in enumerate(sys.path[:5], 1):
    print(f"  {i}. {path}")

# 你可以临时添加搜索路径:
# sys.path.append("/path/to/your/modules")
# 但通常更好的做法是: 用 PYTHONPATH 环境变量, 或者安装为 pip 包。


# ------------------------------------------------------------
# 六、相对导入 vs 绝对导入
# ------------------------------------------------------------
# 绝对导入: 从项目的"根"开始指定完整路径
#   import demo_pkg.calculator
#   from demo_pkg import string_tools
#
# 相对导入: 以当前模块为基准, 用 . 和 .. 表示位置
#   from . import calculator          ← 同一包内的 calculator 模块
#   from .. import some_module        ← 上一级目录的模块
#   from .string_tools import count_vowels  ← 同一包内的 string_tools 模块
#
# ⚠️ 相对导入只能在"包内部"使用!
#   如果直接运行包含相对导入的脚本, 会报错:
#   "ImportError: attempted relative import with no known parent package"
#
#   解决方案: 用 python -m 运行, 或者使用绝对导入。

# 本文件演示的是绝对导入, 因为我们在包外部运行它。


# ------------------------------------------------------------
# 七、__name__ 与 __main__ —— 再次强调
# ------------------------------------------------------------
# 这是 Lesson 01 讲过的内容, 现在结合模块理解更深入。
#
# 当一个 .py 文件被直接运行时: __name__ == "__main__"
# 当一个 .py 文件被 import 时:    __name__ == "模块名" (如 "calculator")
#
# 实际效果:
#   运行 python calculator.py → calculator.__name__ == "__main__"
#   在另一个文件里 import calculator → calculator.__name__ == "calculator"

# 验证:
print(f"\ncalculator 模块的 __name__: {calculator.__name__}")
print(f"本文件的 __name__: {__name__}")


# ------------------------------------------------------------
# 八、__all__ —— 控制 from module import * 导出的内容
# ------------------------------------------------------------
# 在模块或 __init__.py 中定义 __all__ 列表,
# 可以精确控制 from module import * 时导出哪些名称。

# 查看 calculator 模块的 __all__ (如果没有定义, 默认导出所有非 _ 开头的名称)
print(f"\ncalculator.__all__: {getattr(calculator, '__all__', '未定义 (默认导出所有非 _ 名称)')}")

# 查看 demo_pkg 的 __all__
print(f"demo_pkg.__all__: {demo_pkg.__all__}")
# 所以 from demo_pkg import * 只会导出 __version__


# ------------------------------------------------------------
# 九、模块的私有约定 —— _ 前缀
# ------------------------------------------------------------
# Python 没有 private 关键字, 用命名约定:
#   _name    —— "内部使用, 外部不要碰" (单下划线)
#   __name   —— 名称改写 (Lesson 05 讲过)

# calculator 模块里定义了 _validate_number, 但建议不要外部使用:
# calculator._validate_number(42)  # 能运行, 但不推荐

# from module import * 不会导入 _ 开头的名称:
# from calculator import *  →  _validate_number 不会被导入


# ------------------------------------------------------------
# 十、标准库初探 —— os, sys, pathlib
# ------------------------------------------------------------
# Python 的标准库非常丰富。这里快速了解几个常用模块,
# 后续课程会深入使用。

import os

# os.path —— 路径操作 (旧式, 现在推荐用 pathlib)
print(f"\n当前工作目录: {os.getcwd()}")
print(f"当前文件绝对路径: {os.path.abspath(__file__)}")
print(f"当前文件所在目录: {os.path.dirname(os.path.abspath(__file__))}")

# pathlib —— 面向对象的路径操作 (现代推荐方式)
from pathlib import Path

current_file = Path(__file__).resolve()  # 当前文件的 Path 对象
print(f"Path 方式 - 文件名: {current_file.name}")
print(f"Path 方式 - 后缀: {current_file.suffix}")
print(f"Path 方式 - 父目录: {current_file.parent}")

# 用 Path 拼接路径 (跨平台, 自动处理 / 和 \)
data_dir = current_file.parent / "data" / "config.json"
print(f"拼接路径: {data_dir}")

# sys —— 系统相关
print(f"\nPython 版本: {sys.version}")
print(f"命令行参数: {sys.argv}")  # 运行脚本时传入的参数


# ------------------------------------------------------------
# 综合实战: 模块化的配置加载器
# ------------------------------------------------------------
# 体会"把功能拆成多个模块"的好处。

import json
from pathlib import Path


def load_config(name: str = "default") -> dict:
    """
    从配置文件加载配置。
    配置文件约定放在当前目录的 config/ 子目录下。
    """
    config_dir = Path(__file__).parent / "config"
    config_file = config_dir / f"{name}.json"

    if not config_file.exists():
        return {"error": f"配置文件不存在: {config_file}"}

    return json.loads(config_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print()
    print("=" * 50)
    print("综合实战: 配置加载器")
    print("=" * 50)

    # 尝试加载配置 (config 目录可能不存在, 用 try 处理)
    try:
        cfg = load_config("app")
        print(f"配置内容: {cfg}")
    except FileNotFoundError:
        print("config/ 目录不存在, 这是预期的 (我们没有创建配置文件)。")
        print("你可以尝试:")
        print("  1. 创建 phase1/config/ 目录")
        print("  2. 在里面放一个 app.json 文件")
        print("  3. 再次运行本脚本")


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 新建一个 phase1/my_utils.py 文件, 里面写一个函数:
#    def greet(name: str) -> str:
#        return f"你好, {name}!"
#    然后在本文件的 __main__ 块里 import 并调用它。

import sys
sys.path.insert(0, str(Path(__file__).parent))
from my_utils import greet
print("--- my_utils.greet ---")
print(greet("小明"))
print()

#
# 2. 在 demo_pkg/ 下新建 math_extra.py, 提供 factorial(n) 函数。
#    然后在本文件里用 from demo_pkg.math_extra import factorial 导入并使用。

from demo_pkg.math_extra import factorial as fact
print("--- factorial ---")
for i in range(6):
    print(f"factorial({i}) = {fact(i)}")
print()

#
# 3. 打印 sys.path, 找到标准库的安装位置。
#    去那个目录逛逛, 看看 os.py、json.py 等标准库模块长什么样。

print("--- sys.path ---")
for p in sys.path[:5]:
    print(f"  {p}")
print("  ... (省略剩余)")
# 通常标准库在 sys.path 中类似: /opt/homebrew/.../python3.12/
# 可以去那个目录的 Lib/ 子目录看 os.py、json.py
print()

#
# 4. 用 pathlib 列出当前目录下的所有 .py 文件:

print("--- 当前目录 .py 文件 ---")
for f in Path(".").glob("*.py"):
    print(f"  {f}")
print()

#
# 5. (挑战) 写一个函数 find_module(name: str) -> Path:
#    给定模块名, 在 sys.path 中查找对应的 .py 文件并返回其 Path。

def find_module(name: str) -> Path | None:
    """在 sys.path 中查找模块对应的 .py 文件。"""
    for entry in sys.path:
        candidate = Path(entry) / f"{name}.py"
        if candidate.exists():
            return candidate
    return None

print("--- find_module ---")
for mod in ["os", "json", "calculator", "my_utils"]:
    found = find_module(mod)
    if found:
        print(f"  {mod:12} → {found}")
    else:
        print(f"  {mod:12} → 未找到")
print()


# 做完后告诉我:
#   - 相对导入和绝对导入, 你觉得哪个更不容易出错?
#   - 逛标准库目录时, 有什么意外发现?
# 我们继续 Lesson 07: 错误与异常。
# ============================================================
