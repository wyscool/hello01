"""计算器模块 —— Lesson 06 的示例模块。

这个模块直接放在 phase1/ 目录下, 演示"模块"的概念:
    import calculator  → 导入这个文件

对比 demo_pkg/calculator.py:
    import demo_pkg.calculator  → 导入包中的模块
"""

PI = 3.141592653589793


def add(a: float, b: float) -> float:
    """返回 a + b。"""
    return a + b


def subtract(a: float, b: float) -> float:
    """返回 a - b。"""
    return a - b


def multiply(a: float, b: float) -> float:
    """返回 a * b。"""
    return a * b


def divide(a: float, b: float) -> float:
    """返回 a / b。"""
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


def power(base: float, exp: float) -> float:
    """返回 base 的 exp 次方。"""
    return base ** exp


# _ 前缀表示"模块内部使用, 不建议外部调用"
def _validate_number(x) -> None:
    """验证输入是否为数字。"""
    if not isinstance(x, (int, float)):
        raise TypeError(f"期望数字, 得到 {type(x)}")
