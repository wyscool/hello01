# phase1/ — Python 基础 + 工程化 (10 课)

Phase 1 是 Python 入门阶段，面向有 Java 经验的开发者。10 节课从零开始覆盖 Python 核心语法，通过 Java 类比降低学习曲线。

## 学习目标

完成 Phase 1 后应能:
- 理解 Python 与 Java 的核心语法差异 (动态类型、缩进块、鸭子类型)
- 使用 Python 标准数据结构 (list/dict/set/tuple) 和推导式
- 定义函数 (含装饰器、闭包)、类 (含魔术方法、dataclass) 和模块
- 处理异常、读写文件、序列化 JSON
- 编写异步代码 (asyncio) 和单元测试 (pytest)

## 课程列表

| # | 文件 | 主题 | Java 类比 |
|---|------|------|-----------|
| 01 | `01_basics.py` | 变量、类型、字符串、f-string | `var` vs 动态类型, `String.format()` vs f-string |
| 02 | `02_control_flow.py` | 条件判断、循环、推导式 | `for(;;)` vs `for in`, Stream API vs 推导式 |
| 03 | `03_collections.py` | list/dict/set/tuple、切片、生成器 | `ArrayList` vs list, `HashMap` vs dict, `HashSet` vs set |
| 04 | `04_functions.py` | 函数定义、参数类型、装饰器、闭包 | `@Override` vs 装饰器, 方法引用 vs 闭包 |
| 05 | `05_classes.py` | 类、继承、魔术方法、property、dataclass | `class` vs `__init__`, `@Data` vs `@dataclass` |
| 06 | `06_modules_packages.py` | import、包结构、`__init__.py` | `package` vs 文件系统包, `import` vs `from...import` |
| 07 | `07_errors.py` | try/except、自定义异常、上下文管理器 | `try-catch-finally` vs `try-except-finally`, `Closeable` vs context manager |
| 08 | `08_files_json.py` | 文件读写、JSON、序列化 | `FileReader` vs `open()`, `Jackson` vs `json` |
| 09 | `09_async_basics.py` | asyncio、协程、异步上下文管理器 | `CompletableFuture` vs `asyncio`, `@Async` vs `await` |
| 10 | `10_pytest_basics.py` | pytest、fixture、mock、参数化 | JUnit vs pytest, Mockito vs unittest.mock |

## 运行方式

```bash
# 每节课是独立可运行的 .py 文件
python phase1/01_basics.py

# 在 PyCharm 中: 右键文件 → Run
```

## 辅助文件

| 文件 | 说明 |
|------|------|
| `conftest.py` | pytest 共享 fixtures (shared_calculator 等) |
| `calculator.py` | 教学用计算器模块 (用于 10_pytest_basics.py) |
| `test_calculator.py` | calculator 的测试文件 |
| `my_utils.py` | 演示用工具函数模块 |
| `03_exercise_06_my_zip.py` | 练习解答: 手动实现 zip() |
| `04_exercise_answers.py` | 练习解答: 函数相关内容 |
| `demo_pkg/` | 教学用 Python 包示例 |
| `08_demo_data.txt` | 文件读写教学的示例数据 |

## 教学风格

每节课遵循统一结构:
1. **文件头注释** — 课号、学习目标、预计时间
2. **分段讲解** — 每个知识点带 Java 类比的注释块
3. **可运行代码** — 每个代码片段都能直接执行
4. **主程序块** — `if __name__ == "__main__":` 展示运行结果
5. **试试看** — 4-6 个课后练习 (注释形式)

## 前置要求

- Java 后端开发经验 (理解 OOP、异常、集合框架、泛型)
- Python 3.12+ 已安装
- IDE (PyCharm 推荐)
