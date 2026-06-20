# ============================================================
# Phase 1, Lesson 02: 控制流 —— if / for / while
# ============================================================
#
# 本课目标:
#   1. if / elif / else  与 Java 的区别
#   2. for 循环 —— Python 的 for 是"迭代器风格", 不是 C 风格
#   3. range() —— 当你确实需要数字序列时
#   4. while 循环
#   5. break / continue
#   6. 真值与假值 (truthy / falsy) —— Python 独有的概念
#   7. 比较与布尔运算符 (and / or / not, 不是 &&/||/!)
#   8. 三元表达式 (x if cond else y)
#   9. for / while 的 else 子句 —— Python 独有的怪异语法
#   10. match / case 一瞥 (Python 3.10+, 比 Java switch 强大很多)
#
# 如何运行: 同 Lesson 01
# 预计阅读 + 实操时间: 30-40 分钟
# ============================================================
from types import NoneType

# ------------------------------------------------------------
# 一、if / elif / else —— 注意三点小区别
# ------------------------------------------------------------
# Java:
#   if (x > 0) { ... } else if (x < 0) { ... } else { ... }
# Python:
#   - 条件外**不需要** ()  (虽然加了也不报错)
#   - "else if" 写成 elif (一个单词)
#   - 用 : + 缩进, 不用 {}

score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "D"

print(f"score={score}, grade={grade}")


# ------------------------------------------------------------
# 二、比较与布尔运算符 —— Java 用符号, Python 用单词
# ------------------------------------------------------------
# Java:   &&     ||     !       ==      !=
# Python: and    or     not     ==      !=     (符号不能用!)

x = 10
y = 20

# 多条件组合: 用 and / or / not, 不能用 && / ||
if x > 0 and y > 0:
    print("两者都为正")

if x > 100 or y > 10:
    print("至少一个满足")

if not (x == y):
    print("x 不等于 y")

# 一个 Python 特有的优雅写法: 链式比较
age = 25
if 18 <= age < 60:   # 相当于  age >= 18 and age < 60
    print("成年且非老年")
# Java 必须写: if (age >= 18 && age < 60)


# ------------------------------------------------------------
# 三、真值与假值 (truthy / falsy) —— 重要!
# ------------------------------------------------------------
# Java 的 if 条件**必须**是 boolean。
# Python 的 if 条件**可以是任何值**, Python 会自动判断它"算 True 还是算 False"。
#
# 以下值算 False (falsy):
#   False, None, 0, 0.0, "", [], {}, (), set()
# 其他几乎所有值都算 True (truthy)。
#
# 实战意义: 简化"判断空"的写法。

items = []
if items:           # 等价于  if len(items) > 0
    print("有数据")
else:
    print("列表为空")  # 会执行这个

name = ""
if not name:        # 等价于  if name == "" or name is None
    print("名字为空")

# 这种写法很常见, 你会在 Python 代码里见到很多。
# 但有个陷阱: 当你真的想区分 None 和 0 / 空字符串时, 必须显式比较:
value = 0
if value is None:   # 显式判断 None, 用 `is`
    print("value is None")
else:
    print(f"value is {value}")  # 走这里, 因为 0 不是 None
# 这里如果错写成  if not value: 会把 0 也当作"空", 是一个常见的 bug 源。


# ------------------------------------------------------------
# 四、for 循环 —— Python 的 for 是迭代器风格
# ------------------------------------------------------------
# Java 经典 for:  for (int i = 0; i < 10; i++) { ... }
# Java for-each:  for (String s : list) { ... }
#
# Python **没有** C 风格的 for(init; cond; step), 只有迭代器风格:
#   for 变量 in 可迭代对象:
#       ...

# 1. 遍历列表
for fruit in ["apple", "banana", "cherry"]:
    print(f"水果: {fruit}")

# 2. 遍历字符串 (字符串就是字符序列)
for char in "hello":
    print(char, end=" ")  # end=" " 表示用空格代替默认的换行
print()  # 最后补一个换行

# 3. 需要索引时, 用 enumerate()
for index, fruit in enumerate(["apple", "banana", "cherry"]):
    print(f"#{index}: {fruit}")
# Java 类比: 像 for (int i=0; i<list.size(); i++) 但更优雅


# ------------------------------------------------------------
# 五、range() —— 生成数字序列
# ------------------------------------------------------------
# 如果你确实需要"i 从 0 到 9", 用 range():

for i in range(5):       # 0, 1, 2, 3, 4 (默认从 0 开始, 不包含上界)
    print(i, end=" ")
print()

for i in range(2, 8):    # 2, 3, 4, 5, 6, 7 (指定起点)
    print(i, end=" ")
print()

for i in range(0, 10, 2):  # 0, 2, 4, 6, 8 (指定步长)
    print(i, end=" ")
print()

# 重要观念: range() 不是立即生成所有数字, 而是"按需生成"的迭代器。
# range(10**9) 不会占用 10 亿个数字的内存。
# 这是 Python "懒求值"思维的开始, 后面 Phase 3 处理大数据时会更深入。


# ------------------------------------------------------------
# 六、while 循环
# ------------------------------------------------------------
# Java 和 Python 都有 while, 用法几乎一样。Python 没有 do-while。

count = 0
while count < 3:
    print(f"count = {count}")
    count += 1   # Python 没有 ++, 用 += 1

# 经典 while True + break (相当于 do-while):
attempts = 0
while True:
    attempts += 1
    if attempts >= 3:
        print(f"尝试了 {attempts} 次, 退出")
        break


# ------------------------------------------------------------
# 七、break / continue
# ------------------------------------------------------------
# 用法和 Java 完全一样:
#   break    立即退出循环
#   continue 跳过本次迭代, 进入下一次

# 例子: 找第一个偶数
for n in [1, 3, 5, 4, 7, 8]:
    if n % 2 == 0:
        print(f"第一个偶数是 {n}")
        break

# 例子: 跳过 0, 累加正数
total = 0
for n in [3, -1, 0, 5, -2, 7]:
    if n <= 0:
        continue   # 跳过非正数
    total += n
print(f"正数累加结果: {total}")


# ------------------------------------------------------------
# 八、三元表达式 (Python 的 ?: )
# ------------------------------------------------------------
# Java:   String s = (age >= 18) ? "adult" : "minor";
# Python: s = "adult" if age >= 18 else "minor"   # 顺序: 值-条件-值
#
# 读法: "adult, 如果 age>=18, 否则 minor"
# 刚开始你会觉得别扭, 习惯后会觉得读起来很自然 (像英语)。

age = 17
status = "成年" if age >= 18 else "未成年"
print(f"age={age}, status={status}")


# ------------------------------------------------------------
# 九、for/while 的 else 子句 —— Python 独有的怪异语法
# ------------------------------------------------------------
# 这个语法 90% 的人第一次见会觉得"什么鬼"。
#
# 规则: 循环正常结束 (没有被 break 打断) 时, 执行 else 块。
#       如果 break 退出了, else 块不执行。
#
# 应用场景: "找一遍, 没找到就做点什么"

target = 99
for n in [1, 2, 3, 4, 5]:
    if n == target:
        print(f"找到了 {target}")
        break
else:
    # 注意: 这个 else 是和 for 配对的, 不是和 if 配对
    print(f"没找到 {target}")

# 用 Java 实现同样逻辑通常要用一个 boolean found 标志位。
# Python 这个 else 让代码更紧凑, 但它的语义是反直觉的, 你看到时记住即可。


# ------------------------------------------------------------
# 十、match / case 一瞥 (Python 3.10+)
# ------------------------------------------------------------
# Java 14+ 有 switch 表达式。Python 3.10 引入了更强大的 match/case, 支持"模式匹配"。
# 这里只看个简单例子, 后面用到时再深入。

def describe(value):
    match value:
        case 0:
            return "零"
        case 1 | 2 | 3:        # 多个值匹配同一分支
            return "小数字"
        case int() if value > 100:   # 带条件的模式
            return "大整数"
        case str():            # 类型模式: 匹配任何字符串
            return f"字符串 '{value}'"
        case _:                # _ 表示"匹配任何", 相当于 default
            return "其他"

print(describe(0))
print(describe(2))
print(describe(200))
print(describe("hello"))
print(describe(3.14))


# ------------------------------------------------------------
# 综合实战: 猜数字游戏 (简化版)
# ------------------------------------------------------------
# 体会控制流的组合。生产代码会用 input(), 这里我们用预设答案演示。

def play_guess_game(secret: int, guesses: list[int]) -> int:
    """
    模拟猜数字游戏。返回猜中所用的步数, 若没猜中返回 -1。
    """
    for i, guess in enumerate(guesses, start=1):
        if guess == secret:
            print(f"第 {i} 次: {guess} —— 猜中了!")
            return i
        elif guess < secret:
            print(f"第 {i} 次: {guess} —— 小了")
        else:
            print(f"第 {i} 次: {guess} —— 大了")
    else:
        print("没猜中, 答案是", secret)
        print("你今天不在状态")
        return -1


if __name__ == "__main__":
    print()
    print("=" * 50)
    print("猜数字游戏演示")
    print("=" * 50)
    play_guess_game(secret=42, guesses=[10, 80, 50, 45, 100])


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. FizzBuzz 经典题 (1..20):
#    - 能被 3 整除: 打印 "Fizz"
#    - 能被 5 整除: 打印 "Buzz"
#    - 同时被 3 和 5 整除: 打印 "FizzBuzz"
#    - 其余打印数字本身
#    用 for + range + if/elif/else 完成。

for i in range(20):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)

#
# 2. 修改上面的 play_guess_game, 把 for...else 用上:
#    如果循环正常跑完没猜中, 在 else 块里打印 "你今天不在状态"。

def play_guess_game_v2(secret: int, guesses: list[int]) -> int:
    """猜数字游戏 — 用 for...else 重构。"""
    for i, guess in enumerate(guesses, 1):
        if guess == secret:
            print(f"第 {i} 次: {guess} —— 猜对了!")
            return i
        elif guess < secret:
            print(f"第 {i} 次: {guess} —— 小了")
        else:
            print(f"第 {i} 次: {guess} —— 大了")
    else:
        print("没猜中, 答案是", secret)
        print("你今天不在状态")
        return -1

print("--- for...else 版本 ---")
play_guess_game_v2(secret=42, guesses=[10, 80, 50])
print()

#
# 3. 写一个函数 is_truthy(value) -> str:
#    用 if 判断, 返回 "truthy" 或 "falsy"。
#    然后用 for 循环测试以下值, 输出每个值和它的结果:
#       [0, 1, -1, "", "hi", None, [], [0], {}, {"a":1}, True, False]
#    猜测每一个的结果, 再运行验证。

def is_truthy(value) -> str:
    # Python 的 bool() 内置了 truthy/falsy 规则, 一行搞定
    return "truthy" if value else "falsy"

for x in [0, 1, -1, "", "hi", None, [], [0], {}, {"a":1}, True, False]:
    print(is_truthy(x))

#
# 4. (挑战) 不用 max(), 写一个函数 find_max(numbers: list[int]) -> int,
#    返回列表里最大的数。考虑: 空列表时该返回什么? 怎么处理这种情况?

def find_max(numbers: list[int]) -> int | None:
    """不用 max() 找到列表最大值。空列表返回 None。"""
    if not numbers:
        return None
    current_max = numbers[0]
    for n in numbers[1:]:
        if n > current_max:
            current_max = n
    return current_max

print("--- find_max ---")
print(find_max([3, 7, 2, 9, 1]))  # 9
print(find_max([-5, -1, -10]))     # -1
print(find_max([]))                  # None
print()

#
# 4. (挑战) 不用 max(), 写一个函数 find_max(numbers: list[int]) -> int,
#    返回列表里最大的数。考虑: 空列表时该返回什么? 怎么处理这种情况?
#
# 做完后告诉我:
#   - 哪一题让你卡住了 (或者没卡住的话, 哪一题让你最有收获)
#   - 对 truthy/falsy 和 for-else 的感受
# 我们再进入 Lesson 03: 集合 (list / dict / set / tuple) + 列表推导式。
# ============================================================
