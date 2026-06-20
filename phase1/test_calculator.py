"""
calculator 模块的测试文件。

运行方式:
    pytest phase1/test_calculator.py -v
    或
    pytest phase1/ -v  (发现本目录下所有测试)

与 Lesson 06 的 calculator.py 配合使用。
"""
import pytest
import sys
from pathlib import Path

# 确保能 import calculator 模块
sys.path.insert(0, str(Path(__file__).parent))

import calculator


# ============================================================
# 基本功能测试
# ============================================================

def test_add_positive():
    assert calculator.add(3, 5) == 8

def test_add_negative():
    assert calculator.add(-3, -5) == -8

def test_add_mixed():
    assert calculator.add(10, -3) == 7


def test_subtract():
    assert calculator.subtract(10, 3) == 7

def test_multiply():
    assert calculator.multiply(4, 7) == 28

def test_divide():
    assert calculator.divide(10, 2) == 5.0

def test_power():
    assert calculator.power(2, 10) == 1024


# ============================================================
# 异常测试
# ============================================================

def test_divide_by_zero():
    with pytest.raises(ValueError, match="除数不能为 0"):
        calculator.divide(10, 0)


# ============================================================
# parametrize 参数化
# ============================================================

@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),      # 负数
    (100, -50, 50),   # 混合正负
    (999, 1, 1000),   # 大数
])
def test_add_parametrized(a, b, expected):
    assert calculator.add(a, b) == expected


@pytest.mark.parametrize("dividend, divisor, expected", [
    (10, 2, 5.0),
    (7, 2, 3.5),      # 浮点结果
    (-10, 2, -5.0),   # 负数
    (0, 5, 0.0),      # 0 除以任何数
])
def test_divide_parametrized(dividend, divisor, expected):
    assert calculator.divide(dividend, divisor) == expected


# ============================================================
# fixture 测试
# ============================================================

def test_pi_is_constant():
    assert calculator.PI == 3.141592653589793  # math.pi 的值


def test_calculator_module_has_all_functions():
    """确认模块导出了所有关键函数。"""
    functions = ['add', 'subtract', 'multiply', 'divide', 'power']
    for func in functions:
        assert hasattr(calculator, func), f"缺少函数: {func}"
