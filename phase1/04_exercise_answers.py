# ============================================================
# Exercise 3, 4, 5 详细解答
# ============================================================


# ============================================================
# 第 3 题: 用闭包实现计数器
# ============================================================
# 核心难点: 内部函数要"修改"外部函数的变量。
# Python 里如果内部函数想修改外部函数的变量, 必须用 `nonlocal` 声明。

def make_counter():
    """
    返回一个计数器函数, 每次调用返回递增的整数。

    闭包的关键: counter() 函数"捕获"了 make_counter 里的 count 变量,
    即使 make_counter 已经执行完毕, count 仍然活在内存中,
    被 counter() 引用着。
    """
    count = 0  # 这是外部函数(make_counter)的局部变量

    def counter() -> int:
        nonlocal count  # ← 关键! 声明"我要修改外部函数的 count"
        # 如果没有 nonlocal, Python 会认为你在创建一个新的局部变量 count,
        # 然后报错: "local variable 'count' referenced before assignment"
        current = count
        count += 1
        return current

    return counter


# 测试:
print("=" * 50)
print("Exercise 3: 闭包计数器")
print("=" * 50)

c1 = make_counter()
print(f"c1 第 1 次: {c1()}")  # 0
print(f"c1 第 2 次: {c1()}")  # 1
print(f"c1 第 3 次: {c1()}")  # 2

c2 = make_counter()
print(f"c2 第 1 次: {c2()}")  # 0 ← 独立的计数器!
print(f"c1 第 4 次: {c1()}")  # 3 ← c1 继续计数, 不受 c2 影响

# Java 类比:
#   这类似于 Java 中返回一个实现了 Callable/Supplier 的匿名内部类,
#   匿名类捕获了外部方法的局部变量。
#   但 Python 的 nonlocal 显式声明了"我要修改捕获的变量",
#   Java 的匿名类只能捕获 final/effectively final 变量, 不能直接修改。


# ============================================================
# 第 4 题: 修复可变默认参数的陷阱
# ============================================================
# 问题代码:
#   def add_timestamp(data: dict = {}) -> dict:
#       ...
#
# 问题所在:
#   `data: dict = {}` 中的 `{}` 在函数**定义时**就被创建了,
#   不是每次调用时创建。所有调用共享同一个 dict 对象!

# ❌ 错误版本 (运行看效果):
def add_timestamp_bad(data: dict = {}) -> dict:
    from datetime import datetime
    data["timestamp"] = datetime.now().isoformat()
    return data


# ✅ 正确版本:
def add_timestamp_good(data: dict | None = None) -> dict:
    """
    用 None 做默认值, 在函数内部判断并创建新 dict。

    为什么不能用空 dict `{}` 做默认值?
    因为 `{}` 在函数定义时求值一次, 之后所有调用共享它。
    None 是不可变的, 安全。我们在函数体内根据 None 创建新的 dict。
    """
    if data is None:
        data = {}  # ← 每次调用时创建全新的 dict

    from datetime import datetime
    data["timestamp"] = datetime.now().isoformat()
    return data


print()
print("=" * 50)
print("Exercise 4: 可变默认参数陷阱")
print("=" * 50)

# 看看错误版本的诡异行为:
print("错误版本:")
r1 = add_timestamp_bad()
print(f"第 1 次调用: {r1}")
r2 = add_timestamp_bad()
print(f"第 2 次调用: {r2}")
# 你会发现 r1 和 r2 是同一个对象, 而且 r1 里也有了第 2 次的时间戳!
print(f"r1 is r2? {r1 is r2}")  # True! 同一个对象!

# 正确版本:
print("\n正确版本:")
r3 = add_timestamp_good()
print(f"第 1 次调用: {r3}")
r4 = add_timestamp_good()
print(f"第 2 次调用: {r4}")
print(f"r3 is r4? {r3 is r4}")  # False, 各自独立


# ============================================================
# 第 5 题: 带参数的重试装饰器 @retry(max_attempts=3)
# ============================================================
# 这是三道题中最难的, 涉及"装饰器工厂"的概念。
#
# 回顾基础装饰器:
#   @my_logger
#   def func(): ...
#   等价于: func = my_logger(func)
#
# 带参数的装饰器:
#   @retry(max_attempts=3)
#   def func(): ...
#   等价于: func = retry(max_attempts=3)(func)
#
# 也就是说 retry(max_attempts=3) 返回一个"真正的装饰器",
# 这个装饰器再接收 func 作为参数。
#
# 所以我们需要三层嵌套:
#   retry(参数) → 返回 decorator
#   decorator(func) → 返回 wrapper
#   wrapper(*args, **kwargs) → 执行 func 并处理重试逻辑

def retry(max_attempts: int = 3):
    """
    装饰器工厂: 接收配置参数, 返回真正的装饰器。
    """

    def decorator(func):
        """
        真正的装饰器: 接收被装饰的函数, 返回包装函数。
        """

        def wrapper(*args, **kwargs):
            """
            包装函数: 实际执行的地方, 包含重试逻辑。
            """
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    # 尝试执行原函数
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts:
                        # 最后一次也失败了, 抛出异常
                        print(f"第 {attempt} 次尝试失败, 放弃。")
                        raise last_exception
                    else:
                        # 还没用完次数, 打印重试信息
                        print(f"第 {attempt} 次失败, 准备重试...")

            # 理论上不会执行到这里, 但为了类型检查器不报错:
            return None

        return wrapper

    return decorator


# 测试: 创建一个会随机失败的函数
import random


@retry(max_attempts=3)
def unreliable_operation() -> str:
    """
    模拟一个不可靠的操作: 30% 概率成功, 70% 概率抛异常。
    """
    if random.random() < 0.3:  # 30% 概率成功
        return "操作成功!"
    raise RuntimeError("连接超时")


print()
print("=" * 50)
print("Exercise 5: 重试装饰器")
print("=" * 50)

# 运行多次, 观察重试行为
for trial in range(1, 6):
    print(f"\n--- 第 {trial} 轮测试 ---")
    try:
        result = unreliable_operation()
        print(f"最终结果: {result}")
    except RuntimeError as e:
        print(f"最终失败: {e}")


# ============================================================
# 补充: 装饰器三层结构的记忆口诀
# ============================================================
#
# 不带参数的装饰器 (2 层):
#   def decorator(func):      # 第 1 层: 接收函数
#       def wrapper(*a, **k): # 第 2 层: 包装逻辑
#           ...
#       return wrapper
#       return decorator
#
# 带参数的装饰器 (3 层):
#   def factory(config):      # 第 1 层: 接收配置
#       def decorator(func):  # 第 2 层: 接收函数
#           def wrapper(*a, **k):  # 第 3 层: 包装逻辑
#               ...
#           return wrapper
#       return decorator
#   return factory
#
# 记忆口诀: "参数一层、函数一层、逻辑一层"
# ============================================================
