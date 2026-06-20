# ============================================================
# Exercise 06 解答: 自己实现 my_zip
# ============================================================
# 这道题的核心不是"语法", 而是"思路":
#   怎么用同一个索引 i, 同时访问两个列表?
#   循环该在什么时候停下来?
#
# 我会给你三个版本, 从"最像 Java 的思维"到"最 Pythonic"。
# ============================================================


def my_zip_v1(list1: list, list2: list) -> list[tuple]:
    """
    版本 1: while 循环 —— 最直观, 最像 Java 的思维。

    核心思路:
        1. 用一个索引 i, 同时访问 list1[i] 和 list2[i]
        2. 循环条件: i 必须同时小于两个列表的长度
           这样就能自动"以最短的为准"截断
        3. 每次把 (list1[i], list2[i]) 组成 tuple, 放进结果
        4. i += 1 前进

    类比 Java:
        就像你同时遍历两个 ArrayList, 用同一个下标 i。
    """
    result: list[tuple] = []

    i = 0
    while i < len(list1) and i < len(list2):
        pair = (list1[i], list2[i])  # 从两个列表各取一个, 组成 tuple
        result.append(pair)
        i += 1

    return result


def my_zip_v2(list1: list, list2: list) -> list[tuple]:
    """
    版本 2: for + range —— 更 Pythonic, 更简洁。

    关键洞察:
        "以最短的为准"的数学表达就是 min(len(list1), len(list2))。
        不需要在循环条件里写 and, 直接算出来循环次数就行。
    """
    result: list[tuple] = []

    # 例如: list1 有 3 个元素, list2 有 5 个 → min=3 → 循环 3 次
    shortest = min(len(list1), len(list2))

    for i in range(shortest):
        result.append((list1[i], list2[i]))

    return result


def my_zip_v3(list1: list, list2: list) -> list[tuple]:
    """
    版本 3: 列表推导式 —— 一行搞定, Python 风格的极致。

    但这有一个重要区别:
        v1/v2 是一步步 append, 内存是逐步分配的。
        v3 是一次性创建整个列表, 内存一次性分配。
        对于大数据量, v3 可能更快(但可读性稍差)。
    """
    return [(list1[i], list2[i]) for i in range(min(len(list1), len(list2)))]


# ------------------------------------------------------------
# 测试
# ------------------------------------------------------------
if __name__ == "__main__":
    names = ["Alice", "Bob", "Charlie"]
    grades = [90, 85, 78]

    print("=" * 50)
    print("my_zip 实现测试")
    print("=" * 50)

    print(f"v1: {my_zip_v1(names, grades)}")
    print(f"v2: {my_zip_v2(names, grades)}")
    print(f"v3: {my_zip_v3(names, grades)}")
    print(f"内置 zip: {list(zip(names, grades))}")

    # 长度不同的情况
    short = ["a", "b"]
    long = [1, 2, 3, 4]
    print(f"\n长度不同: {my_zip_v1(short, long)}")
    # 输出: [('a', 1), ('b', 2)] —— 只取前两个, 长的部分被丢弃

    # 性能对比 (感受一下差距)
    import time

    big1 = list(range(100_000))
    big2 = list(range(100_000))

    t0 = time.time()
    my_zip_v2(big1, big2)
    t1 = time.time()
    print(f"\nmy_zip_v2 耗时: {(t1 - t0) * 1000:.2f} ms")

    t0 = time.time()
    list(zip(big1, big2))  # 内置 zip 返回的是迭代器, 转 list 才公平对比
    t1 = time.time()
    print(f"内置 zip 耗时: {(t1 - t0) * 1000:.2f} ms")


# ============================================================
# 思考题 (做完上面的运行后, 想想这些)
# ============================================================
#
# 1. 如果 list1 有 100 个元素, list2 只有 3 个, my_zip 返回几个 tuple?
#    答案由谁决定? (提示: 看循环条件或 min())
#
# 2. 为什么内置 zip() 比你的实现快几十倍?
#    提示: Python 是解释型语言, 你写的每一行 Python 代码
#          都要被解释器逐行翻译执行。而内置 zip() 是用 C 写的,
#          直接编译成机器码运行。这叫做"C 扩展加速"。
#
# 3. (进阶挑战) 如果要实现"以最长的为准",
#    短的那个自动用 None 填充, 怎么改?
#    例如: my_zip_long(["a", "b"], [1, 2, 3]) → [("a", 1), ("b", 2), (None, 3)]
#
# 4. (回顾) 为什么我们在 v3 里写了 list(tuple) 而不是直接返回 zip()?
#    提示: zip() 返回的不是 list, 而是"迭代器"(iterator),
#          这是一个"惰性计算"的对象, 不是立刻算出来的。
#          这在 Python 中非常常见, 我们后面会专门讲。
# ============================================================
