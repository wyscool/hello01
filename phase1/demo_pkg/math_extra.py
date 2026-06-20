"""math_extra — 额外数学工具。"""


def factorial(n: int) -> int:
    """计算 n! (阶乘)。"""
    if n < 0:
        raise ValueError("阶乘只支持非负整数")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def fibonacci(n: int) -> int:
    """返回第 n 个斐波那契数 (1-indexed)。"""
    if n <= 0:
        raise ValueError("n 必须 > 0")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
