# ============================================================
# Phase 1, Lesson 04: 函数 —— def, *args, **kwargs, lambda
# ============================================================
#
# 本课目标:
#   1. def 定义函数
#   2. 位置参数、关键字参数、默认参数
#   3. *args —— 接收任意多个位置参数
#   4. **kwargs —— 接收任意多个关键字参数
#   5. 仅限关键字参数
#   6. 可变默认参数的陷阱 (Python 经典 bug)
#   7. Lambda 表达式
#   8. 函数是第一等公民 (闭包、高阶函数)
#   9. Docstring 和类型提示
#
# 预计阅读 + 实操时间: 35-45 分钟
# ============================================================
from tkinter.font import names


# ------------------------------------------------------------
# 一、基础定义 —— 比 Java 简洁, 但没有重载
# ------------------------------------------------------------
# Java:
#   public static int add(int a, int b) { return a + b; }
# Python:

def add(a: int, b: int) -> int:
    """返回两个整数的和。"""
    return a + b

# ⚠️ Python **没有方法重载**!
# 你不能定义两个同名但参数不同的函数:
#   def add(a, b): ...
#   def add(a, b, c): ...   # 这会覆盖上面那个!
#
# 替代方案: 用默认参数、*args、或者类型检查。


# ------------------------------------------------------------
# 二、调用方式 —— 位置参数 vs 关键字参数
# ------------------------------------------------------------
# Java 调用函数: 只能按位置传参。
# Python 调用函数: 可以按位置, 也可以按名字!

def describe_person(name: str, age: int, city: str = "Unknown") -> str:
    """
    name 和 age 是必填参数 (没有默认值)。
    city 是可选参数 (有默认值 "Unknown")。
    """
    return f"{name}, {age} 岁, 来自 {city}"

# 方式 1: 位置参数 (和 Java 一样)
print(describe_person("Alice", 30))

# 方式 2: 关键字参数 (可以打乱顺序!)
print(describe_person(age=25, name="Bob", city="上海"))

# 方式 3: 混用 (位置参数必须在关键字参数前面)
print(describe_person("Charlie", 28, city="北京"))

# ⚠️ 不能这样: describe_person(name="Dave", 30)  # 位置参数在关键字参数后面会报错!


# ------------------------------------------------------------
# 三、*args —— 接收任意多个位置参数
# ------------------------------------------------------------
# Java 用可变参数: void printAll(String... items)
# Python 用 *args:

def sum_all(*numbers: int) -> int:
    """
    *numbers 会把所有多余的位置参数打包成一个 tuple。
    例如: sum_all(1, 2, 3) → numbers = (1, 2, 3)
    """
    total = 0
    for n in numbers:
        total += n
    return total

# 调用:
print(sum_all())           # 0
print(sum_all(1, 2))       # 3
print(sum_all(1, 2, 3, 4)) # 10

# *args 这个名字是约定, 可以改, 但强烈建议保持。
# args 的类型永远是 tuple, 即使只传了一个参数。

# 更 Pythonic 的写法 (用内置 sum):
def sum_all_v2(*numbers: int) -> int:
    return sum(numbers)


# ------------------------------------------------------------
# 四、**kwargs —— 接收任意多个关键字参数
# ------------------------------------------------------------
# Java 没有直接等价物。kwargs 让你接收"任意数量的命名参数"。

def build_user(name: str, **kwargs: str) -> dict:
    """
    **kwargs 会把所有多余的关键字参数打包成一个 dict。
    例如: build_user("Alice", age="30", city="北京")
          → kwargs = {"age": "30", "city": "北京"}
    """
    result = {"name": name}
    result.update(kwargs)  # 把 kwargs 字典合并进来
    return result

# 调用:
user = build_user("Alice", age="30", city="北京", job="工程师")
print(user)
# {'name': 'Alice', 'age': '30', 'city': '北京', 'job': '工程师'}

# kwargs 也是约定名, 类型永远是 dict。


# ------------------------------------------------------------
# 五、组合使用: 固定参数 + *args + **kwargs
# ------------------------------------------------------------
# 这是 Python 函数签名的"终极形态", 也是很多框架 API 的样子。

def flexible(a: int, b: int = 10, *args: int, **kwargs: str) -> None:
    """
    a:     必填位置参数
    b:     可选位置参数 (默认 10)
    *args: 多余的位置参数 → tuple
    **kwargs: 多余的关键字参数 → dict
    """
    print(f"a={a}, b={b}")
    print(f"args={args}")
    print(f"kwargs={kwargs}")

flexible(1)                           # a=1, b=10, args=(), kwargs={}
flexible(1, 20)                       # a=1, b=20, args=(), kwargs={}
flexible(1, 20, 30, 40)               # a=1, b=20, args=(30, 40), kwargs={}
flexible(1, 20, 30, x="hello", y="world")  # kwargs={"x": "hello", "y": "world"}


# ------------------------------------------------------------
# 六、仅限关键字参数 —— 用 * 隔开
# ------------------------------------------------------------
# 如果你希望某些参数"必须"用关键字传参, 可以在签名里放一个裸 *:

def create_user(name: str, *, age: int, city: str = "Unknown") -> dict:
    """
    * 后面的参数只能用关键字传参!
    create_user("Alice", age=30)        # OK
    create_user("Alice", 30)            # 报错! age 不能按位置传
    """
    return {"name": name, "age": age, "city": city}

print(create_user("Alice", age=30, city="上海"))

# 这个技巧在生产代码中很常见 —— 强制调用者"显式"传参, 避免误传。


# ------------------------------------------------------------
# 七、⚠️ 可变默认参数的陷阱 —— Python 最著名的 bug 之一
# ------------------------------------------------------------
# 这是每个 Python 程序员都会踩、都会被坑、都会记住的陷阱。

def append_item_bad(item: str, items: list = []) -> list:
    """
    ❌ 错误的写法!
    默认参数 `items=[]` 在函数**定义时**就被创建了,
    不是每次调用时创建。所以多次调用会共享同一个 list!
    """
    items.append(item)
    return items

# 试试:
print(append_item_bad("a"))   # ['a']     ← 看起来对
print(append_item_bad("b"))   # ['a', 'b'] ← 什么?! 上次的 'a' 还在!

# 正确的写法:
def append_item_good(item: str, items: list[str] | None = None) -> list[str]:
    """
    ✅ 正确的写法!
    用 None 做默认值, 在函数内部判断并创建新 list。
    """
    if items is None:
        items = []
    items.append(item)
    return items

print(append_item_good("a"))   # ['a']
print(append_item_good("b"))   # ['b']  ← 每次独立

# 核心规则: **默认参数在定义时求值, 且只求一次。**
# 可变对象 (list, dict, set) 绝不能做默认参数!


# ------------------------------------------------------------
# 八、Lambda 表达式 —— 单行匿名函数
# ------------------------------------------------------------
# Java:  (a, b) -> a + b
# Python: lambda a, b: a + b
#
# ⚠️ Python 的 lambda **只能写一行表达式**, 不能写语句 (如 if/else 块、赋值等)。

# 简单运算:
square = lambda x: x * x
print(square(5))  # 25

# 传给高阶函数 (最常见用法):
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_by_abs = sorted(numbers, key=lambda x: -x)  # 按负数排序 = 降序
print(sorted_by_abs)

# lambda 适合"用过即弃"的简单逻辑。复杂的逻辑请用 def。


# ------------------------------------------------------------
# 九、函数是第一等公民 —— 闭包
# ------------------------------------------------------------
# Python 函数可以: 作为参数、作为返回值、存进变量、放进列表。
# Java 8+ 也有函数式编程, 但 Python 从诞生就支持。

# 1. 函数作为参数 (高阶函数)
def apply_operation(a: int, b: int, operation) -> int:
    return operation(a, b)

print(apply_operation(3, 4, lambda x, y: x + y))  # 7
print(apply_operation(3, 4, lambda x, y: x * y))  # 12

# 2. 函数返回函数 (闭包)
def make_multiplier(factor: int):
    """返回一个"乘以 factor"的函数。"""
    def multiplier(x: int) -> int:
        return x * factor  # multiplier "记住"了 factor 的值
    return multiplier

double = make_multiplier(2)   # double(x) = x * 2
triple = make_multiplier(3)   # triple(x) = x * 3

print(double(5))   # 10
print(triple(5))   # 15

# 这就是"闭包": 内部函数记住了外部函数的变量。
# 在 AI 开发中, 闭包常用于: 创建配置化的工具函数、装饰器等。


# ------------------------------------------------------------
# 十、装饰器初探 (了解即可, 后面深入)
# ------------------------------------------------------------
# 装饰器是"接收函数、返回函数"的高阶函数的语法糖。
# 这是 Python 的一个强大特性, 框架代码中很常见。

def my_logger(func):
    """一个最简单的装饰器: 在函数执行前后打印日志。"""
    def wrapper(*args, **kwargs):
        print(f"[LOG] 开始调用 {func.__name__}")
        result = func(*args, **kwargs)
        print(f"[LOG] {func.__name__} 返回 {result}")
        return result
    return wrapper

@my_logger  # ← 等价于 say_hello = my_logger(say_hello)
def say_hello(name: str) -> str:
    return f"Hello, {name}!"

print(say_hello("Alice"))

# 输出顺序:
# [LOG] 开始调用 say_hello
# [LOG] say_hello 返回 Hello, Alice!
# Hello, Alice!


# ------------------------------------------------------------
# 综合实战: 灵活的数据处理器
# ------------------------------------------------------------

def process_records(records: list[dict], *fields: str, **options) -> list[dict]:
    """
    处理记录列表, 提取指定字段, 支持可选配置。

    参数:
        records: 记录列表, 每个记录是 dict
        *fields: 要提取的字段名
        **options: 可选配置
            - skip_empty: bool, 是否跳过空值 (默认 False)
            - default_value: 缺失字段的默认值 (默认 None)
    """
    skip_empty = options.get("skip_empty", False)
    default_value = options.get("default_value", None)

    result = []
    for record in records:
        processed = {}
        for field in fields:
            value = record.get(field, default_value)
            if skip_empty and value in (None, "", []):
                continue
            processed[field] = value
        if processed:  # 不为空才添加
            result.append(processed)
    return result


if __name__ == "__main__":
    print()
    print("=" * 50)
    print("综合实战: 数据处理器")
    print("=" * 50)

    data = [
        {"name": "Alice", "age": 30, "city": "北京"},
        {"name": "Bob", "age": 25},
        {"name": "Charlie", "city": "上海"},
    ]

    # 只提取 name 和 age
    print(process_records(data, "name", "age"))

    # 提取所有字段, 跳过空值
    print(process_records(data, "name", "age", "city", skip_empty=True))

    # 指定缺失字段的默认值
    print(process_records(data, "name", "age", "job", default_value="未知"))


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 写一个函数 greet(*names: str, greeting: str = "你好") -> None:
#    接收任意多个名字, 用 greeting 前缀逐个打招呼。
#    例如: greet("Alice", "Bob", greeting="Hi")
#          输出: Hi, Alice! 和 Hi, Bob!
#    注意: greeting 必须是关键字参数 (用 * 隔开)。

def greet(*names: str, greeting: str = "你好") -> None:
    for s in names:
        print(f"{greeting} {s}!")

print("--- greet ---")
greet("Alice", "Bob", greeting="Hi")
print()

#
# 2. 写一个函数 merge_dicts(*dicts: dict) -> dict:
#    接收任意多个字典, 合并成一个。
#    后面的字典覆盖前面的同 key 值。
#    例如: merge_dicts({"a": 1}, {"b": 2}, {"a": 3}) → {"a": 3, "b": 2}
#    提示: 先创建空 dict, 然后逐个 update。

def merge_dicts(*dicts: dict) -> dict:
    result = {}
    for d in dicts:
        result.update(d)
    return result

print(merge_dicts({"a": 1}, {"b": 2}, {"a": 3}))

#
# 3. 用闭包实现一个计数器:
#    def make_counter() -> callable:
#        # 返回一个函数, 每次调用返回递增的整数 (0, 1, 2, ...)
#    counter = make_counter()
#    print(counter())  # 0
#    print(counter())  # 1
#    print(counter())  # 2
#    # 再创建一个新的 counter2, 它应该从 0 开始独立计数

def make_counter():
    count = 0
    def inner_counter():
        nonlocal count
        count += 1
        return count
    return inner_counter

print("--- make_counter ---")
counter1 = make_counter()
print(counter1())  # 1
print(counter1())  # 2
print(counter1())  # 3

counter2 = make_counter()  # 独立从 1 开始
print(counter2())  # 1
print()

#
# 4. (回顾陷阱) 下面的代码有什么问题? 运行看看, 然后修复:
#    def add_timestamp(data: dict = {}) -> dict:
#        from datetime import datetime
#        data["timestamp"] = datetime.now().isoformat()
#        return data

from datetime import datetime

# ❌ 有问题的版本: 可变默认参数在函数定义时只创建一次
def add_timestamp_bug(data: dict = {}) -> dict:
    data["timestamp"] = datetime.now().isoformat()
    return data

# 演示问题: 多次调用不传参, 它们共享同一个 dict
print("--- 陷阱演示 ---")
d1 = add_timestamp_bug()
d2 = add_timestamp_bug()
print(f"d1={d1}\nd2={d2}")
print(f"d1 is d2? {d1 is d2}")  # True! 同一个对象

# ✅ 修复版本: 用 None 做哨兵值
def add_timestamp(data: dict | None = None) -> dict:
    if data is None:
        data = {}
    data["timestamp"] = datetime.now().isoformat()
    return data

print("\n--- 修复后 ---")
d3 = add_timestamp()
d4 = add_timestamp()
print(f"d3={d3}\nd4={d4}")
print(f"d3 is d4? {d3 is d4}")  # False! 独立对象
print()

#
# 5. (挑战) 写一个装饰器 @retry(max_attempts=3):
#    被装饰的函数如果抛出异常, 自动重试最多 3 次。
#    每次重试前打印 "第 N 次重试...".
#    如果 3 次都失败, 抛出最后一次的异常。
#    提示: 用 while 循环 + try/except。

import functools
import time

def retry(max_attempts: int = 3, delay: float = 0.5):
    """装饰器: 自动重试失败的函数。

    Args:
        max_attempts: 最大尝试次数 (含首次)
        delay: 重试间隔秒数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        print(f"第 {attempt} 次重试... ({e})")
                        time.sleep(delay)
            raise last_exception  # type: ignore
        return wrapper
    return decorator

# 演示: 模拟不稳定函数
@retry(max_attempts=3, delay=0.1)
def unstable_network_call():
    import random
    if random.random() < 0.7:
        raise ConnectionError("网络超时")
    return "数据获取成功!"

print("--- @retry 演示 ---")
try:
    result = unstable_network_call()
    print(f"结果: {result}")
except ConnectionError as e:
    print(f"3 次重试后仍然失败: {e}")
print()
#    被装饰的函数如果抛出异常, 自动重试最多 3 次。
#    每次重试前打印 "第 N 次重试...".
#    如果 3 次都失败, 抛出最后一次的异常。
#    提示: 用 while 循环 + try/except。
#
# 做完后告诉我:
#   - 陷阱题 (第 4 题) 你预判对了吗?
#   - 闭包计数器 (第 3 题) 和 Java 的匿名类实现相比, 哪个更简洁?
# 我们继续 Lesson 05: 类与面向对象。
# ============================================================
