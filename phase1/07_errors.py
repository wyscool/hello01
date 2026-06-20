# ============================================================
# Phase 1, Lesson 07: 错误与异常 —— try/except/finally、raise
# ============================================================
#
# 本课目标:
#   1. try / except / finally 基本结构
#   2. 捕获特定异常类型
#   3. except ... as e —— 获取异常对象
#   4. try 的 else 子句 —— Python 独有
#   5. raise —— 抛出异常
#   6. 自定义异常 —— 继承 Exception
#   7. 异常层次结构
#   8. 最佳实践: 不要裸 except!
#   9. 异常链: raise ... from ...
#   10. 断言 assert
#
# 预计阅读 + 实操时间: 30-40 分钟
# ============================================================


# ------------------------------------------------------------
# 一、基本结构: try / except / finally
# ------------------------------------------------------------
# Java:
#   try { ... } catch (SomeException e) { ... } finally { ... }
# Python:
#   try: ... except SomeException as e: ... finally: ...

# 基本示例:
def safe_divide(a: float, b: float) -> float | None:
    """安全除法, 捕获除零错误。"""
    try:
        result = a / b
        print(f"计算成功: {a} / {b} = {result}")
        return result
    except ZeroDivisionError:
        print("❌ 错误: 除数不能为 0")
        return None


print("=" * 50)
print("基本 try/except")
print("=" * 50)

safe_divide(10, 2)
safe_divide(10, 0)


# ------------------------------------------------------------
# 二、获取异常对象 —— except ... as e
# ------------------------------------------------------------
# 你可以在 except 后面用 as 获取异常实例, 查看详细信息。

def parse_int(text: str) -> int | None:
    """将字符串转为整数, 捕获转换错误。"""
    try:
        return int(text)
    except ValueError as e:
        # e 是 ValueError 的实例, 可以打印它的消息
        print(f"转换失败: '{text}' → {e}")
        return None


print()
print(parse_int("42"))       # 42
print(parse_int("abc"))      # None, 打印错误信息


# ------------------------------------------------------------
# 三、捕获多种异常 —— 多个 except 或元组
# ------------------------------------------------------------

def read_index(items: list, index: int) -> any:
    """安全读取列表元素, 处理两种错误。"""
    try:
        return items[index]
    except IndexError:
        print(f"索引越界: {index}, 列表长度: {len(items)}")
        return None
    except TypeError:
        print(f"索引类型错误: 期望整数, 得到 {type(index).__name__}")
        return None


print()
print(read_index([1, 2, 3], 5))       # IndexError
print(read_index([1, 2, 3], "a"))     # TypeError


# 另一种写法: 用一个 except 捕获多种异常
def safe_operation():
    try:
        # 可能抛出 FileNotFoundError 或 PermissionError 的代码
        pass
    except (FileNotFoundError, PermissionError) as e:
        print(f"文件操作失败: {e}")


# ------------------------------------------------------------
# 四、try 的 else 子句 —— Python 独有!
# ------------------------------------------------------------
# else 块在 try 块**没有发生异常时**执行。
# 用来放"只有成功时才需要执行的代码", 让逻辑更清晰。

def process_number(text: str) -> None:
    try:
        num = int(text)
    except ValueError:
        print(f"'{text}' 不是有效数字")
    else:
        # 只有转换成功才执行
        print(f"转换成功, 结果是 {num}")
        print(f"平方: {num ** 2}")
    finally:
        # 无论是否异常, 都执行
        print("处理结束")


print()
print("=" * 50)
print("try/else/finally")
print("=" * 50)

process_number("7")
process_number("xyz")

# 为什么用 else?
#   如果不写 else, 成功后的代码要放在 try 块里。
#   但那些代码本身可能抛出异常, 会被 except 捕获, 造成混淆。
#   放在 else 里, 明确表达"这是成功后的逻辑"。


# ------------------------------------------------------------
# 五、raise —— 抛出异常
# ------------------------------------------------------------
# Java: throw new IllegalArgumentException("...");
# Python: raise ValueError("...")

def validate_age(age: int) -> None:
    """验证年龄, 不合法时抛出异常。"""
    if not isinstance(age, int):
        raise TypeError(f"年龄必须是整数, 得到 {type(age).__name__}")
    if age < 0:
        raise ValueError(f"年龄不能为负数: {age}")
    if age > 150:
        raise ValueError(f"年龄不合理: {age}")


print()
print("=" * 50)
print("raise 抛出异常")
print("=" * 50)

for test_age in [25, -5, 200, "abc"]:
    try:
        validate_age(test_age)
        print(f"✓ {test_age} 岁: 合法")
    except (ValueError, TypeError) as e:
        print(f"✗ {test_age}: {e}")


# ------------------------------------------------------------
# 六、自定义异常 —— 继承 Exception
# ------------------------------------------------------------
# Python 没有 checked/unchecked 异常之分, 所有异常都是 unchecked。
# 自定义异常只需继承 Exception (或它的子类)。

class BusinessError(Exception):
    """业务逻辑错误基类。"""
    pass


class InsufficientBalanceError(BusinessError):
    """余额不足。"""
    def __init__(self, balance: float, required: float) -> None:
        self.balance = balance
        self.required = required
        super().__init__(f"余额不足: 当前 {balance}, 需要 {required}")


class AccountFrozenError(BusinessError):
    """账户已冻结。"""
    pass


class BankAccount:
    def __init__(self, balance: float = 0, frozen: bool = False) -> None:
        self.balance = balance
        self.frozen = frozen

    def withdraw(self, amount: float) -> float:
        if self.frozen:
            raise AccountFrozenError("账户已冻结, 无法操作")
        if amount > self.balance:
            raise InsufficientBalanceError(self.balance, amount)
        self.balance -= amount
        return self.balance


print()
print("=" * 50)
print("自定义异常")
print("=" * 50)

acc = BankAccount(balance=100)

try:
    acc.withdraw(50)
    print(f"取款 50 成功, 余额: {acc.balance}")
    acc.withdraw(200)
except InsufficientBalanceError as e:
    print(f"业务错误: {e}")
    print(f"  当前余额: {e.balance}, 需要: {e.required}")


# ------------------------------------------------------------
# 七、异常层次结构
# ------------------------------------------------------------
# Python 的所有异常都继承自 BaseException。
# 通常我们处理的是 Exception 及其子类 (不包括 SystemExit、KeyboardInterrupt)。

# 异常层次 (简化):
# BaseException
#   ├── SystemExit          # sys.exit() 触发
#   ├── KeyboardInterrupt   # Ctrl+C 触发
#   └── Exception           # 所有普通异常的基类
#         ├── ValueError
#         ├── TypeError
#         ├── KeyError
#         ├── IndexError
#         ├── FileNotFoundError
#         └── ...

print()
print("=" * 50)
print("异常层次")
print("=" * 50)

def show_hierarchy(exc_class: type, indent: int = 0) -> None:
    """打印异常的继承链。"""
    prefix = "  " * indent
    print(f"{prefix}{exc_class.__name__}")
    for base in exc_class.__bases__:
        if base is not object:
            show_hierarchy(base, indent + 1)


print("InsufficientBalanceError 的继承链:")
show_hierarchy(InsufficientBalanceError)

# 捕获父类 = 捕获所有子类
# 例如: except BusinessError 会捕获 InsufficientBalanceError 和 AccountFrozenError


# ------------------------------------------------------------
# 八、⚠️ 不要裸 except! —— Python 中最常见的错误
# ------------------------------------------------------------
# 裸 except (不写异常类型) 会捕获所有异常, 包括 SystemExit 和 KeyboardInterrupt!
# 这意味着用户按 Ctrl+C 都停不下来。

def bad_example():
    try:
        1 / 0
    except:  # ❌ 不要这样写!
        print("出错了 (但我不知道是什么错)")


def good_example():
    try:
        1 / 0
    except ZeroDivisionError as e:  # ✅ 明确指定异常类型
        print(f"除零错误: {e}")


# 如果确实需要捕获所有异常, 用 except Exception:
def acceptable_example():
    try:
        risky_operation()
    except Exception as e:  # ✅ 捕获所有普通异常, 但不捕获 SystemExit
        print(f"意外错误: {e}")


print()
print("=" * 50)
print("最佳实践")
print("=" * 50)
print("❌ 不要: except:")
print("✅ 要:   except SpecificError:")
print("✅ 要:   except Exception as e:  (如果确实需要捕获所有)")


# ------------------------------------------------------------
# 九、异常链 —— raise ... from ...
# ------------------------------------------------------------
# 当你捕获一个异常, 想抛出一个更有意义的异常时,
# 用 from 保留原始异常的信息, 方便调试。

def load_config_v2(path: str) -> dict:
    """加载配置文件, 将底层错误包装为业务错误。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            import json
            return json.load(f)
    except FileNotFoundError as e:
        # 用 from 保留原始异常链
        raise BusinessError(f"配置文件不存在: {path}") from e
    except json.JSONDecodeError as e:
        raise BusinessError(f"配置文件格式错误: {path}") from e


print()
print("=" * 50)
print("异常链")
print("=" * 50)

try:
    load_config_v2("不存在的文件.json")
except BusinessError as e:
    print(f"业务层错误: {e}")
    print(f"原始原因: {e.__cause__}")  # 查看原始异常


# ------------------------------------------------------------
# 十、断言 assert —— 调试辅助
# ------------------------------------------------------------
# assert 用于检查"不应该发生"的条件。如果条件为 False, 抛出 AssertionError。
# 生产环境运行时可以关闭断言 (python -O)。

def calculate_discount(price: float, discount: float) -> float:
    assert price >= 0, f"价格不能为负数: {price}"
    assert 0 <= discount <= 1, f"折扣必须在 0-1 之间: {discount}"
    return price * (1 - discount)


print()
print("=" * 50)
print("断言 assert")
print("=" * 50)

print(f"折扣后价格: {calculate_discount(100, 0.2)}")

# 下面的调用会触发 AssertionError:
# print(calculate_discount(-10, 0.2))


# ------------------------------------------------------------
# 综合实战: 健壮的文件处理
# ------------------------------------------------------------

from pathlib import Path


def process_data_file(file_path: str) -> list[dict] | None:
    """
    处理数据文件, 返回记录列表。
    处理所有可能的错误, 给出清晰的错误信息。
    """
    path = Path(file_path)

    try:
        # 检查文件存在
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path.absolute()}")

        # 检查是否是文件 (不是目录)
        if not path.is_file():
            raise ValueError(f"路径不是文件: {path}")

        # 读取内容
        content = path.read_text(encoding="utf-8")

        # 解析 JSON
        import json
        data = json.loads(content)

        # 验证数据格式
        if not isinstance(data, list):
            raise ValueError("数据必须是列表格式")

        for i, record in enumerate(data):
            if not isinstance(record, dict):
                raise ValueError(f"第 {i} 条记录不是字典: {type(record)}")

        return data

    except FileNotFoundError as e:
        print(f"[错误] {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"[错误] JSON 解析失败 (第 {e.lineno} 行): {e.msg}")
        return None
    except ValueError as e:
        print(f"[错误] 数据验证失败: {e}")
        return None
    except Exception as e:
        print(f"[错误] 意外错误: {type(e).__name__}: {e}")
        return None


if __name__ == "__main__":
    print()
    print("=" * 50)
    print("综合实战: 文件处理")
    print("=" * 50)

    # 测试不存在的文件
    process_data_file("不存在的数据.json")

    # 测试有效文件 (我们临时创建一个)
    test_file = Path("test_data.json")
    test_file.write_text('[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]')
    result = process_data_file(str(test_file))
    print(f"处理结果: {result}")
    test_file.unlink()  # 清理


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 写一个函数 get_element(data: list, index: int, default=None):
#    安全获取列表元素。如果索引越界, 返回 default 而不是抛异常。

def get_element(data: list, index: int, default=None):
    """安全获取列表元素, 索引越界返回 default。"""
    try:
        return data[index]
    except IndexError:
        return default

print("--- get_element ---")
nums = [10, 20, 30]
print(get_element(nums, 1))        # 20
print(get_element(nums, 5))        # None
print(get_element(nums, 5, "缺"))  # "缺"
print()

#
# 2. 创建自定义异常 NetworkError, 并写一个函数 fetch_data(url: str):
#    如果 url 不包含 "http", 抛出 NetworkError。
#    用 try/except 调用它, 捕获并打印错误。

class NetworkError(Exception):
    """网络相关异常。"""
    pass

def fetch_data(url: str) -> str:
    if "http" not in url:
        raise NetworkError(f"无效的 URL (缺少 http): {url}")
    return f"[模拟数据] 从 {url} 获取的数据"

print("--- NetworkError ---")
for url in ["https://api.example.com", "ftp://bad.url", "invalid"]:
    try:
        result = fetch_data(url)
        print(f"  ✅ {url}: {result}")
    except NetworkError as e:
        print(f"  ❌ {e}")
print()

#
# 3. 写一个计算器函数 calculate(a, b, op):
#    op 可以是 "+", "-", "*", "/"。
#    对以下错误分别抛出不同的异常:
#      - 除以零 → ZeroDivisionError
#      - 不支持的运算符 → ValueError
#      - a 或 b 不是数字 → TypeError
#    然后写一个调用者, 用多个 except 分别处理。

def calculate(a, b, op: str):
    """简单计算器, 根据 op 执行对应运算。"""
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError(f"操作数必须是数字, 得到 {type(a).__name__}, {type(b).__name__}")
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        if b == 0:
            raise ZeroDivisionError("除数不能为零")
        return a / b
    else:
        raise ValueError(f"不支持的运算符: {op!r}, 支持: + - * /")

print("--- calculate ---")
cases = [
    (10, 3, "+"), (10, 3, "-"), (10, 3, "*"), (10, 3, "/"),
    (10, 0, "/"),  (10, 3, "^"),  ("x", 3, "+"),
]
for a, b, op in cases:
    try:
        result = calculate(a, b, op)
        print(f"  {a} {op} {b} = {result}")
    except ZeroDivisionError as e:
        print(f"  ❌ 除以零: {e}")
    except ValueError as e:
        print(f"  ❌ 运算符错误: {e}")
    except TypeError as e:
        print(f"  ❌ 类型错误: {e}")
print()

#
# 4. (回顾) 用 try/except/else/finally 写一个文件拷贝函数:

def copy_file(src: str, dst: str) -> bool:
    """拷贝文件, 演示 try/except/else/finally 完整用法。"""
    try:
        content = Path(src).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"  源文件不存在: {src}")
        return False
    else:
        Path(dst).write_text(content, encoding="utf-8")
        print(f"  拷贝成功: {src} → {dst}")
        return True
    finally:
        print("  操作完成")

print("--- copy_file ---")
# 创建测试文件
Path("test_src.txt").write_text("Hello, try/except/else/finally!")
copy_file("test_src.txt", "test_dst.txt")
copy_file("不存在.txt", "test_dst.txt")
Path("test_src.txt").unlink(missing_ok=True)
Path("test_dst.txt").unlink(missing_ok=True)
print()

#
# 5. (挑战) 实现一个重试装饰器 @retry(max_attempts=3, exceptions=(Exception,)):
#    只捕获指定的异常类型, 其他异常直接抛出。

import functools
import time as _time

def retry(max_attempts: int = 3, delay: float = 0.3,
          exceptions: tuple = (Exception,)):
    """装饰器: 只重试指定的异常类型, 其他异常直接抛出。"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        print(f"  第 {attempt} 次重试... ({type(e).__name__}: {e})")
                        _time.sleep(delay)
                # 不在此元组中的异常直接向上抛出 (不会被此处的 except 捕获)
            raise last_exception
        return wrapper
    return decorator

@retry(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
def unstable_api():
    import random
    r = random.random()
    if r < 0.6:
        raise ConnectionError("网络波动")
    # TypeError 不在重试列表, 会直接抛出
    if r < 0.7:
        raise TypeError("不应该重试的错误")
    return "OK"

print("--- @retry (指定异常) ---")
try:
    result = unstable_api()
    print(f"  结果: {result}")
except TypeError:
    print("  TypeError 直接抛出, 未重试 (预期行为)")
except ConnectionError as e:
    print(f"  重试 3 次后仍失败: {e}")
print()


# 做完后告诉我:
#   - 自定义异常和 Java 的相比, 哪个更简洁?
#   - "不要裸 except" 这条规则, 你在 Java 里有没有对应的习惯?
# 我们继续 Lesson 08: 文件与 JSON。
# ============================================================
