"""
demo_pkg —— 演示用 Python 包。

这是一个最小化的包结构示例:
    demo_pkg/
        __init__.py       ← 包初始化文件 (就是这个文件)
        calculator.py     ← 计算器模块
        string_tools.py   ← 字符串工具模块

__init__.py 的作用:
    1. 告诉 Python"这个目录是一个包"
    2. 初始化包级别的变量、导入子模块
    3. 控制 from demo_pkg import * 时导出什么
"""

__version__ = "1.0.0"
__author__ = "Learner"

# 当有人写 "from demo_pkg import *" 时, 默认导出的名称
__all__ = ["__version__"]
