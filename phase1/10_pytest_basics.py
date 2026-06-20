# ============================================================
# Phase 1, Lesson 10: pytest 测试基础
# ============================================================
#
# 本课目标:
#   1. 安装 pytest、运行第一个测试
#   2. 测试函数: test_ 前缀、assert 断言
#   3. 测试异常: pytest.raises()
#   4. fixture —— pytest 的依赖注入神器
#   5. parametrize —— 一份测试跑多组数据
#   6. conftest.py —— 共享 fixture
#   7. 测试类组织 (TestClass)
#   8. 常用命令行参数 (-v, -k, -s, --lf)
#   9. 标记: @pytest.mark.skip / skipif / xfail
#   10. pytest vs JUnit 对比 —— 给 Java 开发者的速查表
#
# 预计阅读 + 实操时间: 35-45 分钟
#
# 课前准备:
#   确保已安装 pytest:
#     conda activate myhello
#     pip install pytest
#   验证: pytest --version
#
# ⚠️ 本课的结构和之前不太一样:
#   1. 先在本文件里读概念和示例
#   2. 实际的测试写在 test_calculator.py 里
#   3. 用命令行 pytest 运行测试
#   因为 pytest 是命令行工具, 不是通过 __main__ 运行的。
# ============================================================

from pathlib import Path
import pytest


# ------------------------------------------------------------
# 一、pytest 是什么? —— Python 测试框架的"事实标准"
# ------------------------------------------------------------
# Java 生态: JUnit 5 + Mockito + AssertJ
# Python 生态: pytest + unittest.mock
#
# 为什么 pytest 赢了?
#   - 不需要继承 TestCase (对比 unittest)
#   - 不需要 @Test 注解, 函数名 test_ 开头即可
#   - assert 直接断言, 不需要 assertEquals(a, b) 等一堆方法
#   - fixture 比 @BeforeEach/@BeforeAll 灵活得多
#   - 插件生态强大 (pytest-cov, pytest-asyncio, pytest-mock...)
#
# 安装:
#   pip install pytest
#   验证: pytest --version
#
# 最简测试:
#   def test_one_plus_one():
#       assert 1 + 1 == 2
#
# 运行:
#   pytest test_calculator.py -v


# ------------------------------------------------------------
# 二、基本测试 —— 函数名以 test_ 开头
# ------------------------------------------------------------
# 惯例: 测试文件以 test_ 开头或 _test 结尾
#       测试函数以 test_ 开头
#       测试类以 Test 开头 (不含 __init__)

# 被测试的代码 (通常这些在单独的文件里)
def add(a: int, b: int) -> int:
    return a + b


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


def is_palindrome(text: str) -> bool:
    """检查字符串是否为回文。"""
    cleaned = text.replace(" ", "").lower()
    return cleaned == cleaned[::-1]


class Calculator:
    """简单的计算器, 作为测试目标。"""
    def __init__(self) -> None:
        self.history: list[str] = []

    def add(self, a: float, b: float) -> float:
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a: float, b: float) -> float:
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def clear_history(self) -> None:
        self.history.clear()


# --- 对应的测试函数 ---
# 语法极致简洁: 就 assert + 布尔表达式
# Python 的 assert 在 pytest 中会自动产生友好的错误信息

def test_add_basic():
    """最基本的测试。"""
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(100, 200) == 300


def test_divide_normal():
    """正常除法。"""
    assert divide(10, 2) == 5.0
    assert divide(7, 2) == 3.5


def test_is_palindrome():
    """回文判断。"""
    assert is_palindrome("radar") == True
    assert is_palindrome("A man a plan a canal Panama") == True
    assert is_palindrome("hello") == False

# 对比 JUnit:
#   @Test                                    ← 不需要!
#   void testAddBasic() {                    ← def test_add_basic():
#       assertEquals(3, add(1, 2));          ← assert add(1, 2) == 3
#       assertEquals(0, add(-1, 1));         ← assert add(-1, 1) == 0
#       assertEquals(300, add(100, 200));    ← assert add(100, 200) == 300
#   }


# ------------------------------------------------------------
# 三、测试异常 —— pytest.raises()
# ------------------------------------------------------------
# JUnit: assertThrows(ValueError.class, () -> divide(10, 0));
# pytest: with pytest.raises(ValueError):

def test_divide_by_zero():
    """验证除以 0 抛出 ValueError。"""
    with pytest.raises(ValueError) as exc_info:
        divide(10, 0)

    # 可选: 检查异常消息
    assert "除数不能为 0" in str(exc_info.value)


def test_divide_by_zero_match():
    """用 match 参数直接匹配异常消息。"""
    with pytest.raises(ValueError, match="除数不能为 0"):
        divide(10, 0)

    # match 是正则, 所以 match="除数.*0" 也行


# ------------------------------------------------------------
# 四、fixture —— pytest 的杀手级功能
# ------------------------------------------------------------
# fixture = 可复用的测试前置条件 (setup / teardown)
# JUnit: @BeforeEach / @BeforeAll / @AfterEach / @AfterAll
# pytest: @pytest.fixture —— 更灵活, 通过参数注入使用
#
# 核心思想:
#   - 定义 fixture 函数, 返回测试需要的"资源"
#   - 测试函数在参数中声明需要哪个 fixture
#   - pytest 自动调用 fixture 并注入返回值
#   - 这就是依赖注入 (DI) 在测试中的体现!

@pytest.fixture
def calculator() -> Calculator:
    """创建一个全新的计算器实例, 每个测试用独立的实例。"""
    print("\n[fixture] 创建 Calculator")  # -s 参数可见
    return Calculator()


@pytest.fixture
def sample_numbers() -> list[int]:
    """提供一组测试用的数字。"""
    return [1, 2, 3, 5, 8, 13]


def test_calculator_add(calculator: Calculator):
    """fixture 通过参数注入! calculator 是上面 fixture 的返回值。"""
    result = calculator.add(3, 4)
    assert result == 7
    assert len(calculator.history) == 1


def test_calculator_subtract(calculator: Calculator):
    """每个测试获得独立的 calculator 实例。"""
    result = calculator.subtract(10, 3)
    assert result == 7
    assert len(calculator.history) == 1  # 不是 2! 因为这是新实例


def test_fixture_uses_fixture(calculator: Calculator, sample_numbers: list[int]):
    """一个测试可以使用多个 fixture。"""
    total = sum(sample_numbers)
    result = calculator.add(total, 0)
    assert result == sum(sample_numbers)


# fixture scope: 控制 fixture 的生命周期
#   function (默认): 每个测试函数调用一次
#   class:   每个测试类调用一次
#   module:  每个测试模块调用一次
#   session: 整个测试会话调用一次

@pytest.fixture(scope="module")
def expensive_resource():
    """模块级别: 整个测试文件共享一个实例。"""
    print("\n[fixture] 创建昂贵的资源 (module scope, 只执行一次)")
    return {"db": "connected", "cache": "warmed"}


def test_expensive_1(expensive_resource: dict):
    assert expensive_resource["db"] == "connected"


def test_expensive_2(expensive_resource: dict):
    assert expensive_resource["cache"] == "warmed"
    # expensive_resource 和 test_expensive_1 是同一个实例


# setup / teardown 模式 (使用 yield):
@pytest.fixture
def temp_file():
    """测试前后自动创建和清理文件。"""
    file_path = Path("test_temp.txt")
    file_path.write_text("test data", encoding="utf-8")
    print(f"\n[fixture] 创建临时文件: {file_path}")

    yield file_path  # ← yield 之后是 teardown!

    # 清理代码
    if file_path.exists():
        file_path.unlink()
    print(f"[fixture] 删除临时文件: {file_path}")


def test_temp_file(temp_file: Path):
    assert temp_file.exists()
    assert temp_file.read_text(encoding="utf-8") == "test data"
    # 测试结束后, fixture 自动删除文件


# ------------------------------------------------------------
# 五、parametrize —— 一份测试, 多组数据
# ------------------------------------------------------------
# JUnit: @ParameterizedTest + @ValueSource / @CsvSource / @MethodSource
# pytest: @pytest.mark.parametrize —— 比 JUnit 更直观

@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (100, -50, 50),
    (999, 1, 1000),
])
def test_add_parametrized(a: int, b: int, expected: int):
    """pytest 会为每组参数生成一个独立的测试用例。"""
    assert add(a, b) == expected


@pytest.mark.parametrize("text, expected", [
    ("radar", True),
    ("hello", False),
    ("", True),                    # 空字符串是回文吗? 技术上是的
    ("a", True),                   # 单字符
    ("Madam", True),               # 大小写
    ("A Santa at NASA", True),     # 忽略空格
])
def test_palindrome_parametrized(text: str, expected: bool):
    assert is_palindrome(text) == expected


# parametrize 叠加: 笛卡尔积!
@pytest.mark.parametrize("a", [1, 2, 3])
@pytest.mark.parametrize("b", [10, 20])
def test_multiply_combinations(a: int, b: int):
    """生成 3 × 2 = 6 个测试用例!"""
    assert a * b >= 0


# ------------------------------------------------------------
# 六、conftest.py —— 跨文件共享 fixture
# ------------------------------------------------------------
# 把 fixture 定义放在 conftest.py 中,
# 同级及子目录下的所有测试文件都能自动使用, 无需 import!
#
# 这是 pytest 最精妙的设计之一。
# JUnit 需要继承 BaseTest 或 @Import, pytest 直接按目录层级自动发现。
#
# 本目录的 conftest.py 已经提供了 shared_calculator fixture。
# 下面这个测试可以直接使用它:

def test_shared_fixture(shared_calculator: Calculator):
    """shared_calculator 定义在 conftest.py, 这里直接使用。"""
    result = shared_calculator.add(10, 20)
    assert result == 30


# ------------------------------------------------------------
# 七、测试类 —— 组织相关测试
# ------------------------------------------------------------
# 测试类以 Test 开头, 不含 __init__。
# 类内的测试方法共享类级别的 fixture。

class TestCalculator:
    """关于 Calculator 的一组测试。"""

    @pytest.fixture
    def calc(self) -> Calculator:
        """这个 fixture 只在 TestCalculator 内可用。"""
        return Calculator()

    def test_add(self, calc: Calculator):
        assert calc.add(1, 2) == 3

    def test_subtract(self, calc: Calculator):
        assert calc.subtract(10, 3) == 7

    def test_history_tracks_operations(self, calc: Calculator):
        """测试历史记录是否正确追踪。"""
        calc.add(1, 2)
        calc.subtract(5, 3)
        assert len(calc.history) == 2
        assert "1 + 2 = 3" in calc.history[0]
        assert "5 - 3 = 2" in calc.history[1]


class TestDivide:
    """关于除法的测试。"""

    def test_normal_division(self):
        assert divide(10, 2) == 5.0

    def test_negative_division(self):
        assert divide(-10, 2) == -5.0

    def test_division_by_zero(self):
        with pytest.raises(ValueError):
            divide(10, 0)


# ------------------------------------------------------------
# 八、常用命令行参数
# ------------------------------------------------------------
# 运行本文件的所有测试:
#   pytest phase1/10_pytest_basics.py -v
#
#   常用参数:
#   -v           详细输出 (显示每个测试名)
#   -vv          更详细 (显示断言细节)
#   -k "关键词"  只运行匹配关键词的测试 (如 -k "divide")
#   -s           显示 print 输出 (默认不显示)
#   -x           第一个失败后立即停止
#   --lf         只运行上次失败的测试 (--last-failed)
#   --ff         先跑上次失败的, 再跑其他的 (--failed-first)
#   --tb=short   简化回溯信息
#   --tb=long    完整回溯信息
#   --maxfail=3  失败 3 个后停止
#   -q           安静模式
#   --durations=5 显示最慢的 5 个测试


# ------------------------------------------------------------
# 九、标记 (markers) —— skip / skipif / xfail
# ------------------------------------------------------------

@pytest.mark.skip(reason="这个测试还没写完")
def test_not_ready():
    """skip: 跳过这个测试。"""
    assert 1 == 2  # 永远不会执行


@pytest.mark.skipif(
    "sys.version_info < (3, 10)",
    reason="需要 Python 3.10+ 的 match/case 语法"
)
def test_requires_modern_python():
    """skipif: 条件性跳过。Python 3.10 之前跳过。"""
    status = 200
    match status:
        case 200:
            assert True
        case _:
            assert False


@pytest.mark.xfail(reason="已知 bug, 下个版本修复")
def test_known_bug():
    """xfail: 预期会失败。通过了反而报 XPASS (意外通过)。"""
    result = add(1, 1)
    assert result == 3  # 故意写错的预期


# 自定义 marker (在 conftest.py 中注册后可用):
@pytest.mark.slow
def test_slow_operation():
    """标记为慢速测试, 可以用 pytest -m "not slow" 跳过。"""
    import time
    time.sleep(0.1)
    assert True


# ------------------------------------------------------------
# 十、pytest vs JUnit 速查表
# ------------------------------------------------------------
# 给 Java 开发者的对照表。

CHEATSHEET = """
pytest vs JUnit 5 对照表
═══════════════════════════════════════════════════════════════
功能                JUnit 5                     pytest
───────────────────────────────────────────────────────────────
测试标识            @Test                       def test_xxx():
测试类              class XxxTest               class TestXxx:
断言                assertEquals(a, b)           assert a == b
异常断言            assertThrows(X.class, ...)   pytest.raises(X):
前置条件            @BeforeEach                 @pytest.fixture
所有测试前/后       @BeforeAll / @AfterAll      scope="module" fixture
参数化测试          @ParameterizedTest          @pytest.mark.parametrize
跳过测试            @Disabled                   @pytest.mark.skip
条件跳过            @DisabledOnOs(...)           @pytest.mark.skipif(...)
预期失败            - (没有内置)                 @pytest.mark.xfail
分组运行            @Tag("slow")                @pytest.mark.slow
Mock                Mockito                     unittest.mock / pytest-mock
覆盖率              JaCoCo                      pytest-cov
并行测试            junit-platform              pytest-xdist
运行单个测试类      mvn -Dtest=XxxTest          pytest test_xxx.py::TestXxx
运行单个测试方法    mvn -Dtest=XxxTest#method   pytest test_xxx.py::TestXxx::test_method
═══════════════════════════════════════════════════════════════
"""


if __name__ == "__main__":
    print("=" * 60)
    print("  pytest 测试基础 —— 概念篇")
    print("=" * 60)
    print()
    print("本文件包含的是测试概念和示例代码。")
    print("实际测试运行请使用命令行:")
    print()
    print("  # 运行本文件的所有测试")
    print("  pytest phase1/10_pytest_basics.py -v")
    print()
    print("  # 运行带参数化的测试")
    print("  pytest phase1/10_pytest_basics.py -v -k parametrized")
    print()
    print("  # 运行 Calculator 类的测试")
    print("  pytest phase1/10_pytest_basics.py -v -k Calculator")
    print()
    print("  # 跳过慢速测试")
    print("  pytest phase1/10_pytest_basics.py -v -m \"not slow\"")
    print()
    print("💡 你也可以在 PyCharm 中右键本文件 → Run pytest")
    print()
    print(CHEATSHEET)
    print("=" * 60)
    print("🎉 Phase 1 完成! 接下来进入 Phase 2: LLM API + Prompt 工程")
    print("=" * 60)
