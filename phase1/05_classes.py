# ============================================================
# Phase 1, Lesson 05: 类与面向对象
# ============================================================
#
# 本课目标:
#   1. class 定义与 __init__
#   2. self —— 必须显式写出
#   3. 实例变量 vs 类变量
#   4. 方法类型: 实例方法、类方法(@classmethod)、静态方法(@staticmethod)
#   5. @property —— Python 的 getter/setter
#   6. 特殊方法 (__str__, __repr__, __eq__, __lt__ 等)
#   7. 继承与 super()
#   8. 多重继承与 MRO (方法解析顺序)
#   9. @dataclass —— 自动生成样板代码
#   10. "私有"变量的约定
#
# 预计阅读 + 实操时间: 40-50 分钟
# ============================================================


# ------------------------------------------------------------
# 一、基础定义 —— 没有 public/private, self 要显式写
# ------------------------------------------------------------
# Java:
#   public class User {
#       private String name;
#       private int age;
#       public User(String name, int age) { ... }
#   }
# Python:

class User:
    """用户类。"""

    def __init__(self, name: str, age: int) -> None:
        """
        初始化方法。创建对象时自动调用。
        注意: 这不是"构造函数", 真正的构造是 __new__(很少重写)。
        """
        self.name = name  # 实例变量 —— 用 self. 前缀
        self.age = age

    def introduce(self) -> str:
        """实例方法。第一个参数必须是 self (相当于 Java 的 this)。"""
        return f"我是 {self.name}, {self.age} 岁"


# 使用:
alice = User("Alice", 30)
print(alice.introduce())

# ⚠️ 重要区别:
#   - Python 没有 new 关键字: User("Alice", 30) 直接创建
#   - self 必须显式写出: 定义时写 def method(self), 调用时不用传 self
#   - 没有 public/private: 所有属性都是公开的, 用命名约定表示"私有"


# ------------------------------------------------------------
# 二、实例变量 vs 类变量
# ------------------------------------------------------------
# 类变量: 定义在类体里、方法外, 被所有实例共享 (类似 Java 的 static 字段)。
# 实例变量: 定义在 __init__ 里, 每个实例独立。

class Dog:
    species = "Canis familiaris"  # ← 类变量, 所有 Dog 共享

    def __init__(self, name: str) -> None:
        self.name = name  # ← 实例变量, 每个 Dog 独立


d1 = Dog("Buddy")
d2 = Dog("Max")

print(d1.species, d2.species)  # 都是 "Canis familiaris"
print(d1.name, d2.name)        # "Buddy" "Max"

# 通过类修改类变量 → 影响所有实例:
Dog.species = "Canis lupus"
print(d1.species)  # "Canis lupus"

# 通过实例修改 → 只会给该实例创建一个"遮蔽"的实例变量:
d1.species = "Mutant"
print(d1.species)  # "Mutant" (实例变量遮蔽了类变量)
print(d2.species)  # "Canis lupus" (不受影响)
print(Dog.species)  # "Canis lupus" (类变量本身没变)


# ------------------------------------------------------------
# 三、方法类型 —— 实例方法、类方法、静态方法
# ------------------------------------------------------------
# Java 区分实例方法和 static 方法。
# Python 有三种:

class MyClass:
    count: int = 0  # 类变量

    def __init__(self, value: int) -> None:
        self.value = value

    # 1. 实例方法 —— 最常用的, 第一个参数是 self
    def instance_method(self) -> str:
        return f"实例方法, value={self.value}"

    # 2. 类方法 —— 第一个参数是 cls (类本身), 可以访问/修改类变量
    @classmethod
    def class_method(cls) -> str:
        return f"类方法, count={cls.count}"

    # 3. 静态方法 —— 不接收 self 也不接收 cls, 和普通函数一样
    @staticmethod
    def static_method(x: int, y: int) -> int:
        return x + y


obj = MyClass(10)
print(obj.instance_method())   # 实例方法
print(MyClass.class_method())  # 类方法 —— 可以用类名或实例调用
print(MyClass.static_method(3, 4))  # 静态方法

# 什么时候用哪种?
#   实例方法: 需要访问 self 的属性/方法
#   类方法:   需要访问类变量, 或实现"替代构造函数" (见下方)
#   静态方法: 逻辑上属于类但不需要访问类/实例的状态


# ------------------------------------------------------------
# 四、替代构造函数 —— 类方法的经典用法
# ------------------------------------------------------------
# Java 用重载构造函数。Python 没有重载, 用类方法实现。

class Date:
    def __init__(self, year: int, month: int, day: int) -> None:
        self.year = year
        self.month = month
        self.day = day

    @classmethod
    def from_string(cls, date_str: str) -> "Date":
        """从字符串创建 Date, 例如 '2026-05-19'。"""
        year, month, day = map(int, date_str.split("-"))
        return cls(year, month, day)  # ← cls(...) 等价于 Date(...)

    @classmethod
    def today(cls) -> "Date":
        """创建今天的日期。"""
        from datetime import date as _date
        d = _date.today()
        return cls(d.year, d.month, d.day)

    def __str__(self) -> str:
        return f"{self.year}-{self.month:02d}-{self.day:02d}"


d1 = Date(2026, 5, 19)
d2 = Date.from_string("2026-12-25")
d3 = Date.today()
print(d1, d2, d3)


# ------------------------------------------------------------
# 五、@property —— Python 风格的 getter/setter
# ------------------------------------------------------------
# Java:  private field + getter + setter
# Python: 用 @property 把方法变成"属性"

class Temperature:
    def __init__(self, celsius: float) -> None:
        self._celsius = celsius  # _ 前缀表示"内部使用"

    @property
    def celsius(self) -> float:
        """getter —— 访问 temp.celsius 时自动调用。"""
        return self._celsius

    @celsius.setter
    def celsius(self, value: float) -> None:
        """setter —— temp.celsius = 30 时自动调用。"""
        if value < -273.15:
            raise ValueError("温度不能低于绝对零度")
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        """派生属性 —— 自动计算, 不需要存储。"""
        return self._celsius * 9 / 5 + 32


temp = Temperature(25)
print(f"摄氏: {temp.celsius}")      # 像访问属性一样, 不用括号!
print(f"华氏: {temp.fahrenheit}")  # 自动计算

temp.celsius = 30                  # 像设置属性一样, 自动调用 setter
print(f"更新后: {temp.celsius}")

# temp.celsius = -300  # 会抛出 ValueError


# ------------------------------------------------------------
# 六、特殊方法 (Dunder Methods) —— 让类"Pythonic"
# ------------------------------------------------------------
# 双下划线方法让自定义类可以像内置类型一样工作。

class Vector:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        """str(obj) 和 print(obj) 时调用。面向用户的友好表示。"""
        return f"Vector({self.x}, {self.y})"

    def __repr__(self) -> str:
        """repr(obj) 和交互式环境里直接输入 obj 时调用。面向开发者的精确表示。"""
        return f"Vector(x={self.x}, y={self.y})"

    def __eq__(self, other: object) -> bool:
        """== 运算符。"""
        if not isinstance(other, Vector):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __add__(self, other: "Vector") -> "Vector":
        """+ 运算符。"""
        return Vector(self.x + other.x, self.y + other.y)

    def __len__(self) -> int:
        """len(obj) —— 虽然语义不太对, 但演示一下。"""
        return 2


v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1)           # Vector(1, 2)         ← __str__
print(repr(v1))     # Vector(x=1, y=2)     ← __repr__
print(v1 == v2)     # False                ← __eq__
print(v1 + v2)      # Vector(4, 6)         ← __add__
print(len(v1))      # 2                    ← __len__


# ------------------------------------------------------------
# 七、继承与 super()
# ------------------------------------------------------------
# Python 3 的 super() 不需要参数 (自动推断)。

class Animal:
    def __init__(self, name: str) -> None:
        self.name = name

    def speak(self) -> str:
        return "..."

    def __str__(self) -> str:
        return f"Animal({self.name})"


class Cat(Animal):
    def __init__(self, name: str, breed: str) -> None:
        super().__init__(name)  # ← 调用父类的 __init__, 不需要写 Cat, self
        self.breed = breed

    def speak(self) -> str:
        return f"{self.name} 说: 喵~"

    def __str__(self) -> str:
        return f"Cat({self.name}, {self.breed})"


cat = Cat("咪咪", "英短")
print(cat)
print(cat.speak())

# isinstance 检查:
print(isinstance(cat, Cat))     # True
print(isinstance(cat, Animal))  # True
print(issubclass(Cat, Animal))  # True


# ------------------------------------------------------------
# 八、多重继承与 MRO (方法解析顺序)
# ------------------------------------------------------------
# Python 支持多重继承! 这是 Java 不支持的 (Java 用接口)。
# MRO (Method Resolution Order) 决定了调用顺序。

class Flyer:
    def move(self) -> str:
        return "飞"

class Swimmer:
    def move(self) -> str:
        return "游"

class Duck(Flyer, Swimmer):  # ← 多重继承
    pass


duck = Duck()
print(duck.move())  # "飞" —— 因为 Flyer 在 MRO 中排在 Swimmer 前面

# 查看 MRO:
print(Duck.__mro__)  # (<class 'Duck'>, <class 'Flyer'>, <class 'Swimmer'>, <class 'object'>)

# ⚠️ 多重继承很强大但容易复杂。生产代码中建议:
#   - 优先考虑组合 (has-a) 而非继承 (is-a)
#   - 如果必须用, 用 mixin 模式 (提供特定功能的小类)


# ------------------------------------------------------------
# 九、@dataclass —— 自动生成样板代码
# ------------------------------------------------------------
# Python 3.7+ 引入。自动为你生成 __init__, __repr__, __eq__ 等。
# 这是 Python 版的 "Java Record" (但功能更强)。

from dataclasses import dataclass, field


@dataclass
class Product:
    """有了 @dataclass, 不需要手写 __init__, __repr__, __eq__。"""
    name: str
    price: float
    quantity: int = 0  # 默认值

    @property
    def total_value(self) -> float:
        return self.price * self.quantity


p1 = Product("iPhone", 9999.0, 2)
p2 = Product("iPhone", 9999.0, 2)
print(p1)            # Product(name='iPhone', price=9999.0, quantity=2)
print(p1 == p2)      # True —— 自动实现了 __eq__
print(p1.total_value)  # 19998.0

# 更高级: 用 field() 做更精细控制
@dataclass
class Config:
    name: str
    tags: list[str] = field(default_factory=list)  # 默认空列表, 每个实例独立!
    debug: bool = field(default=False, repr=False)  # repr 时不显示


c = Config("app")
c.tags.append("python")
print(c)  # Config(name='app', tags=['python'])


# ------------------------------------------------------------
# 十、"私有"变量的约定
# ------------------------------------------------------------
# Python 没有真正的私有。用命名约定:
#   _name    —— "内部使用, 请不要直接访问" (受保护)
#   __name   —— "私有, 会触发名称改写 (name mangling)"

class BankAccount:
    def __init__(self, balance: float) -> None:
        self._balance = balance       # _ 前缀: "受保护"
        self.__pin = "1234"           # __ 前缀: "私有" (名称改写)

    def deposit(self, amount: float) -> None:
        if amount > 0:
            self._balance += amount

    @property
    def balance(self) -> float:
        return self._balance


account = BankAccount(1000)
print(account.balance)      # 1000 —— 通过 property 访问

# print(account._balance)   # 可以访问, 但"不应该"
# print(account.__pin)      # AttributeError! 名称被改写了
# 实际上它被改名成了 _BankAccount__pin:
print(account._BankAccount__pin)  # "1234" —— 技术上还是可以访问

# 结论: Python 的"私有"是"约定", 不是"强制"。
# 哲学: "我们都是成年人" (We are all consenting adults)。


# ------------------------------------------------------------
# 综合实战: 一个简单的配置管理器
# ------------------------------------------------------------

from dataclasses import dataclass
from typing import Any


@dataclass
class Setting:
    key: str
    value: Any
    description: str = ""


class ConfigManager:
    """简单的配置管理器, 演示类、继承、property 的综合使用。"""

    def __init__(self) -> None:
        self._settings: dict[str, Setting] = {}

    def set(self, key: str, value: Any, description: str = "") -> None:
        self._settings[key] = Setting(key, value, description)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._settings:
            return self._settings[key].value
        return default

    @property
    def all_keys(self) -> list[str]:
        return list(self._settings.keys())

    @property
    def count(self) -> int:
        return len(self._settings)

    def __str__(self) -> str:
        lines = [f"ConfigManager ({self.count} settings):"]
        for s in self._settings.values():
            desc = f" # {s.description}" if s.description else ""
            lines.append(f"  {s.key} = {s.value}{desc}")
        return "\n".join(lines)


if __name__ == "__main__":
    print()
    print("=" * 50)
    print("综合实战: 配置管理器")
    print("=" * 50)

    config = ConfigManager()
    config.set("host", "localhost", "服务器地址")
    config.set("port", 8080, "端口号")
    config.set("debug", True)

    print(config)
    print(f"host = {config.get('host')}")
    print(f"所有 key: {config.all_keys}")


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 给 Vector 类添加 __sub__ 方法, 实现 - 运算符 (向量减法)。
#    测试: Vector(5, 5) - Vector(2, 3) → Vector(3, 2)

# 扩展 Vector 类 (monkey patch, 演示用; 正式代码应在类定义内添加)
def vector_sub(self, other: "Vector") -> "Vector":
    if not isinstance(other, Vector):
        return NotImplemented
    return Vector(self.x - other.x, self.y - other.y)

Vector.__sub__ = vector_sub

print("--- __sub__ ---")
v3 = Vector(5, 5) - Vector(2, 3)
print(v3)  # Vector(3, 2)
print()


#
# 2. 创建一个 @dataclass 表示 Rectangle(宽, 高):
#    - 添加 @property 计算面积和周长
#    - 添加类方法 from_square(cls, side) 创建正方形

@dataclass
class Rectangle:
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    @classmethod
    def from_square(cls, side: float) -> "Rectangle":
        return cls(width=side, height=side)

print("--- Rectangle ---")
r1 = Rectangle(10, 5)
print(f"面积={r1.area}, 周长={r1.perimeter}")
sq = Rectangle.from_square(4)
print(f"正方形: {sq}, 面积={sq.area}")
print()


#
# 3. 实现一个继承 ConfigManager 的 EnvConfigManager:
#    - 添加 load_from_env() 方法, 从 os.environ 读取特定前缀的变量
#    - 例如: 环境变量 APP_HOST=localhost → config.set("host", "localhost")

import os

class EnvConfigManager(ConfigManager):
    """从环境变量加载配置的 ConfigManager 子类。"""

    def load_from_env(self, prefix: str = "APP_") -> None:
        """读取以 prefix 开头的环境变量, 去掉前缀后设为配置项。

        例如: APP_HOST=localhost → config.set("host", "localhost")
             APP_PORT=8080      → config.set("port", "8080")
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                self.set(config_key, value, f"from env:{key}")

print("--- EnvConfigManager ---")
# 演示: 手动设置两个环境变量然后读取
os.environ["APP_HOST"] = "localhost"
os.environ["APP_PORT"] = "8080"
env_config = EnvConfigManager()
env_config.load_from_env(prefix="APP_")
print(env_config)
print(f"host={env_config.get('host')}, port={env_config.get('port')}")
# 清理
os.environ.pop("APP_HOST", None)
os.environ.pop("APP_PORT", None)
print()


#
# 4. (回顾陷阱) 下面的代码有什么问题?
#    @dataclass
#    class Bad:
#        items: list = []
#    # 创建两个 Bad 实例, 分别往 items append 不同元素, 观察结果

print("--- 陷阱: 可变默认参数 ---")

# Python 3.11+ 的 @dataclass 已经自动拦截 list/dict 等可变默认值并抛出 ValueError。
# 但普通 class 的默认参数陷阱仍然存在:

class Bad:
    # ❌ 这个 [] 在函数定义时只创建一次, 所有实例共享!
    def __init__(self, items: list = []):
        self.items = items

b1 = Bad()
b2 = Bad()
b1.items.append("a")
b2.items.append("b")
print(f"b1.items = {b1.items}")  # ['a', 'b'] ← 共享了!
print(f"b2.items = {b2.items}")  # ['a', 'b']
print(f"b1.items is b2.items? {b1.items is b2.items}")  # True!

# ✅ 正确做法: None 哨兵
class Good:
    def __init__(self, items: list | None = None):
        self.items = items if items is not None else []

g1 = Good()
g2 = Good()
g1.items.append("a")
g2.items.append("b")
print(f"g1.items = {g1.items}")  # ['a']
print(f"g2.items = {g2.items}")  # ['b']
print(f"g1.items is g2.items? {g1.items is g2.items}")  # False!

# 如果确实需要 @dataclass + 可变默认值, 用 field(default_factory=list):
#   from dataclasses import field
#   items: list = field(default_factory=list)
print()


#
# 5. (挑战) 实现一个简单的上下文管理器类 Timer:
#    with Timer() as t:
#        time.sleep(1)
#    print(f"耗时: {t.elapsed} 秒")
#    提示: 需要实现 __enter__ 和 __exit__ 方法。

import time

class Timer:
    """上下文管理器 —— 测量 with 块的执行时间。"""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.elapsed = self.end - self.start
        return False  # 不抑制异常

print("--- Timer ---")
with Timer() as t:
    time.sleep(0.5)
    total = sum(range(1000000))
print(f"耗时: {t.elapsed:.3f} 秒")
print()


# 做完后告诉我:
#   - dataclass 和 Java Record 相比, 你更喜欢哪个?
#   - Python "约定式私有"和 Java "强制式私有", 你怎么看?
# 我们继续 Lesson 06: 模块与包。
# ============================================================
