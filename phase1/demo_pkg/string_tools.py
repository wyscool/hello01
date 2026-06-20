"""字符串工具模块 —— 提供文本处理功能。"""

VOWELS = "aeiouAEIOU"


def count_vowels(text: str) -> int:
    """统计字符串中元音字母的数量。"""
    return sum(1 for char in text if char in VOWELS)


def capitalize_words(text: str) -> str:
    """将每个单词首字母大写。"""
    return " ".join(word.capitalize() for word in text.split())


def reverse_words(text: str) -> str:
    """反转单词顺序。"""
    return " ".join(reversed(text.split()))


def is_palindrome(text: str) -> bool:
    """判断是否为回文 (忽略大小写和非字母字符)。"""
    cleaned = "".join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]


# 模块级别的私有变量
_INTERNAL_COUNTER = 0


def _internal_helper(text: str) -> str:
    """内部辅助函数 —— 不建议外部直接调用。"""
    return f"[内部处理] {text}"
