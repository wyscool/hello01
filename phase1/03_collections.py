# ============================================================
# Phase 1, Lesson 03: 集合类型 —— list / dict / set / tuple + 推导式
# ============================================================
#
# 本课目标:
#   1. list —— Python 最常用的数据结构, 类似 Java 的 ArrayList
#   2. tuple —— 不可变的 list, 支持"解包"操作
#   3. dict —— 键值对映射, 类似 Java 的 HashMap
#   4. set —— 无序不重复集合, 类似 Java 的 HashSet
#   5. 列表/字典/集合推导式 —— Python 独有的优雅语法
#   6. 常用内置函数: len, sum, max, min, sorted, enumerate, zip
#
# 本课是 Python 最重要的一课之一。Java 的集合 API 很完整但很冗长,
# Python 的集合操作极为简洁。
#
# 预计阅读 + 实操时间: 40-50 分钟
# ============================================================


# ------------------------------------------------------------
# 一、list —— Python 的"万能动态数组"
# ------------------------------------------------------------
# Java: ArrayList<String> fruits = new ArrayList<>();
#       fruits.add("apple");
# Python: fruits = ["apple", "banana", "cherry"]  # 直接字面量定义

fruits = ["apple", "banana", "cherry"]

# 常用操作:
fruits.append("date")  # 尾部添加 (Java: add())
fruits.insert(1, "avocado")  # 在索引1插入 (Java: add(index, element))
fruits.remove("banana")  # 移除第一个匹配值 (Java: remove(Object))
last = fruits.pop()  # 移除并返回最后一个 (Java: remove(size()-1))
first = fruits.pop(0)  # 移除并返回指定索引 (Java: remove(index))

# 查:
print(f"长度: {len(fruits)}")  # 长度 (Java: size())
print(f"是否含 apple: {'apple' in fruits}")  # 是否包含 (Java: contains())
print(f"cherry 的索引: {fruits.index('cherry')}")  # 查找 (Java: indexOf())

# 遍历:
for fruit in fruits:
    print(fruit)

# ⚠️ Python list 可以放不同类型的数据 (Java 也可以, 用 Object)
# mixed = [1, "hello", True, None, [1, 2, 3]]  # 合法但通常不推荐


# ------------------------------------------------------------
# 二、切片 (Slicing) —— Python 最强大的特性之一
# ------------------------------------------------------------
# Java 没有切片。你要取子数组, 必须用循环或 subList()。
# Python: list[start:end:step]   三个参数都是可选的

numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 基本切片:
print(f"[2:5]  = {numbers[2:5]}")  # [2, 3, 4] —— 从索引2开始, 到索引5(不包含)
print(f"[:3]   = {numbers[:3]}")  # [0, 1, 2] —— 从头开始, 到3(不含)
print(f"[7:]   = {numbers[7:]}")  # [7, 8, 9] —— 从7开始到末尾
print(f"[:]    = {numbers[:]}")  # 复制整个列表 (浅拷贝)

# 步长:
print(f"[::2]  = {numbers[::2]}")  # [0, 2, 4, 6, 8] —— 每隔一个取一个
print(f"[1::2] = {numbers[1::2]}")  # [1, 3, 5, 7, 9] —— 从1开始每隔一个
print(f"[::-1] = {numbers[::-1]}")  # [9, 8, ..., 0] —— 反转! 常用技巧

# 负索引: 从末尾开始计数
print(f"[-1]   = {numbers[-1]}")  # 9 —— 最后一个
print(f"[-3:]  = {numbers[-3:]}")  # [7, 8, 9] —— 最后三个
print(f"[:-2]  = {numbers[:-2]}")  # [0..7] —— 除最后两个

# 切片的安全特性: 超出范围不报错, 返回空列表
print(f"[100:200] = {numbers[100:200]}")  # [] (不是 IndexError!)

# ------------------------------------------------------------
# 三、tuple —— 不可变的 list
# ------------------------------------------------------------
# tuple 和 list 很像, 但不能增删改元素。
# 用 () 定义, 但实际上通常用逗号分隔 (括号经常省略)。

point = (3, 4)  # 坐标
rgb = (255, 128, 0)  # 颜色
empty = ()  # 空 tuple
single = (1,)  # ⚠️ 单个元素的 tuple 注意逗号! (1,) != (1)

# 为什么用 tuple?
#   1. 不可变 = 安全 (可作为 dict 的 key 或 set 的元素)
#   2. 比 list 稍快
#   3. 语义: 表示"固定的一组数据" (如坐标、配置项、函数返回值)

# --- 元组解包 —— Python 最优雅的语法之一 ---
x, y = (3, 4)  # x=3, y=4
# 等价于 Java: int x = point.get(0); int y = point.get(1); (太啰嗦了)

# 交换变量 (不用临时变量):
a, b = 10, 20
a, b = b, a  # 交换! 背后就是 tuple 解包


# Java 必须写: int temp = a; a = b; b = temp;

# 函数多返回值 (本质是返回 tuple):
def divide(a: int, b: int) -> tuple[int, int]:
    """返回 (商, 余数)"""
    return a // b, a % b


quotient, remainder = divide(17, 5)  # 自动解包
print(f"商={quotient}, 余={remainder}")

# 用 _ 忽略不需要的值:
first, _, third = (1, 2, 3)  # first=1, third=3, 2 被丢弃

# ------------------------------------------------------------
# 四、dict —— 键值对映射
# ------------------------------------------------------------
# Java: Map<String, Integer> scores = new HashMap<>();
# Python: 花括号, key:value 对

scores = {
    "Alice": 90,
    "Bob": 85,
    "Charlie": 78,
}

# 增删改查:
scores["David"] = 92  # 添加/更新 (Java: put())
alice_score = scores["Alice"]  # 访问 (Java: get())
# ⚠️ 访问不存在的 key 会抛 KeyError! 更安全的方式:
alice_score = scores.get("Alice", 0)  # 默认值0 (Java: getOrDefault())

del scores["Charlie"]  # 删除 (Java: remove())

# 遍历:
for name in scores:  # 遍历 keys
    print(f"key: {name}")

for score in scores.values():  # 遍历 values
    print(f"value: {score}")

for name, score in scores.items():  # 遍历 (key, value) 对 —— 最常用
    print(f"{name}: {score}")

# 判断 key 存在:
if "Alice" in scores:  # 等价于 Java: containsKey()
    print("Alice 在成绩单里")

# dict 的 key 必须是"不可变"的 (hashable): str, int, float, tuple, bool
# list 和 dict 不能作为 dict 的 key (因为它们是可变的)


# ------------------------------------------------------------
# 五、set —— 无序不重复集合
# ------------------------------------------------------------
# Java: Set<String> tags = new HashSet<>();
# Python: 花括号但不含 key, 或用 set() 构造函数

tags = {"python", "ai", "backend", "python"}  # 重复的 "python" 被自动去重
print(f"tags = {tags}")  # 顺序不保证

# 集合运算 (非常常用):
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(f"a | b = {a | b}")  # 并集 (Java: union())
print(f"a & b = {a & b}")  # 交集 (Java: intersection())
print(f"a - b = {a - b}")  # 差集 (Java: difference())
print(f"a ^ b = {a ^ b}")  # 对称差集 (Java: symmetricDifference())

# 成员判断 (比 list 快得多, O(1) vs O(n)):
print(f"3 在 a 里? {3 in a}")  # True

# ------------------------------------------------------------
# 六、推导式 —— Python 的灵魂
# ------------------------------------------------------------
# 这是 Java 程序员第一次看到会觉得"这是魔法"的语法。
# 本质上: 用一行代码替代 for 循环 + append。

# --- 1. 列表推导式: [表达式 for 变量 in 可迭代对象 if 条件] ---

# 传统写法 (Java 风格):
squares_old = []
for i in range(10):
    squares_old.append(i * i)

# 推导式写法:
squares = [i * i for i in range(10)]
# 读法: "i*i, 对每一个 i 在 range(10) 中"
print(f"squares = {squares}")

# 带条件:
even_squares = [i * i for i in range(10) if i % 2 == 0]
# 读法: "i*i, 对每一个 i 在 range(10) 中, 如果 i 是偶数"
print(f"even_squares = {even_squares}")

# --- 2. 字典推导式: {key: value for ...} ---
names = ["Alice", "Bob", "Charlie"]
name_lengths = {name: len(name) for name in names}
# {'Alice': 5, 'Bob': 3, 'Charlie': 7}
print(f"name_lengths = {name_lengths}")

# 过滤 + 转换:
passed_students = {name: score for name, score in scores.items() if score >= 80}
print(f"及格: {passed_students}")

# --- 3. 集合推导式: {表达式 for ...} ---
unique_lengths = {len(name) for name in names}
print(f"unique_lengths = {unique_lengths}")  # {3, 5} (集合去重)

# ⚠️ 没有 tuple 推导式! (i*i for i in range(10)) 是生成器表达式, 后面讲

# --- 嵌套推导式 (不要太深, 可读性第一) ---
matrix = [[i * j for j in range(1, 4)] for i in range(1, 4)]
print(f"乘法表: {matrix}")
# [[1, 2, 3], [2, 4, 6], [3, 6, 9]]


# ------------------------------------------------------------
# 七、常用内置函数
# ------------------------------------------------------------
nums = [3, 1, 4, 1, 5, 9, 2, 6]

print(f"长度: {len(nums)}")
print(f"求和: {sum(nums)}")
print(f"最大: {max(nums)}")
print(f"最小: {min(nums)}")
print(f"排序(新列表): {sorted(nums)}")  # 返回新列表, 不修改原列表
print(f"降序: {sorted(nums, reverse=True)}")

# enumerate (索引+值, 同时拿到):
for i, val in enumerate(["a", "b", "c"], start=1):
    print(f"{i}: {val}")

# zip (并行迭代):
names = ["Alice", "Bob", "Charlie"]
grades = [90, 85, 78]
for name, grade in zip(names, grades):
    print(f"{name}: {grade}")
# ⚠️ zip 以最短的为准, 超出部分被截断 (不会报错)

# 解压 (unzip) —— * 是解包操作符:
pairs = [("a", 1), ("b", 2), ("c", 3)]
letters, nums = zip(*pairs)  # *pairs 把列表展开成三个参数
print(f"letters = {letters}, nums = {nums}")


# ------------------------------------------------------------
# 综合实战: 处理学生成绩单
# ------------------------------------------------------------

def analyze_scores(data: dict[str, int]) -> dict:
    """
    分析成绩, 返回统计信息。
    """
    if not data:
        return {"error": "没有数据"}

    total = sum(data.values())
    avg = total / len(data)

    return {
        "count": len(data),
        "total": total,
        "average": round(avg, 2),
        "最高分": max(data, key=data.get),  # key=data.get 指定比较依据
        "最低分": min(data, key=data.get),
        "及格名单": [name for name, s in data.items() if s >= 60],
        "优秀名单": {name: s for name, s in data.items() if s >= 90},
    }


if __name__ == "__main__":
    print()
    print("=" * 50)
    print("成绩单分析")
    print("=" * 50)

    exam_scores = {
        "Alice": 95,
        "Bob": 58,
        "Charlie": 82,
        "David": 91,
        "Eve": 67,
    }

    result = analyze_scores(exam_scores)
    for key, value in result.items():
        print(f"{key}: {value}")

# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 用切片取列表的奇数位元素 (索引 1, 3, 5...):
#    nums = [0,1,2,3,4,5,6,7,8,9]
#    # 你的代码 —— 只用切片, 不写循环

nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
n = nums[1::2]
print(f"n = {n}")


#
# 2. 写一个函数 reverse_string(s: str) -> str, 不用循环, 只用切片。
#    输入 "hello", 返回 "olleh"。

def reverse_string(s: str) -> str:
    return s[::-1]


r = reverse_string("hello")
print(f"r = {r}")

#
# 3. 用字典推导式把列表变成 {值: 索引} 映射:
#    fruits = ["apple", "banana", "cherry"]
#    # 结果: {"apple": 0, "banana": 1, "cherry": 2}

fruits = ["apple", "banana", "cherry"]
r_fruits = {v: i for (i, v) in enumerate(fruits)}
print(f"r_fruits = {r_fruits}")

#
# 4. 用 set 运算找出两个列表的共同元素:
#    list1 = [1, 2, 3, 4, 5]
#    list2 = [4, 5, 6, 7, 8]
#    # 提示: 先转 set, 再 &

list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]
set1 = set(list1)
set2 = set(list2)
set3 = set1 & set2

print(f"set3 = {set3}")


#
# 5. (挑战) 实现一个函数 find_common_keys(dict1, dict2):
#    返回两个字典中共同的 key。
#    要求用 set 运算, 不用循环。

def find_common_keys(dict1, dict2):
    set4 = set(dict1.keys())
    set5 = set(dict2.keys())
    return set4 & set5


set6 = find_common_keys({"Alice": 20, "Bob": 40, "Charlie": 30}, {"Alice": 70, "Bob": 80})

print(set6)


#
# 6. (挑战) 实现你自己的 zip: my_zip(list1, list2) -> list[tuple]:
#    不用 zip() 函数, 自己写循环实现。
#    然后比较: 你的实现和内置 zip() 谁短? 谁快? 为什么?

def my_zip(list1, list2) -> list[tuple]:
    return [(list1[i], list2[i]) for i in range(min(len(list1), len(list2)))]

names = ["Alice", "Bob", "Charlie"]
grades = [90, 85, 78]
for name, grade in my_zip(names, grades):
    print(f"{name}: {grade}")

#
# 做完后告诉我:
#   - 哪一题让你"啊哈"了一下?
#   - 切片和推导式, 哪个让你更想多看几遍?
# 我们继续 Lesson 04: 函数 (def, *args, **kwargs, lambda)。
# ============================================================
