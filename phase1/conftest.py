"""
conftest.py — pytest 共享 fixture 配置。

本文件中的 fixture 对整个 phase1/ 目录下的测试自动可见,
无需手动 import!

这是 pytest 最优雅的设计之一:
  把 fixture 放进 conftest.py → 同级及子目录的测试都能直接使用
  类比: Spring 的 @Configuration + @Bean, 但按目录层级自动扫描
"""
import pytest
import sys
from pathlib import Path


@pytest.fixture
def shared_calculator():
    """模块级别的共享 Calculator。

    这个 fixture 在 10_pytest_basics.py 的 test_shared_fixture 中使用,
    演示了 conftest.py 自动发现机制: 测试不需要 import 它!
    """
    sys.path.insert(0, str(Path(__file__).parent))

    # 导入 10_pytest_basics.py 中定义的 Calculator
    from importlib import import_module
    module = import_module('10_pytest_basics')
    Calculator = getattr(module, 'Calculator')
    return Calculator()


# ============================================================
# 自定义 marker 注册
# ============================================================

def pytest_configure(config):
    """注册自定义 mark, 避免 pytest 警告。"""
    config.addinivalue_line("markers", "slow: 标记慢速测试")
