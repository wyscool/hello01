# ============================================================
# deploy/tests/test_agent.py — DevAssistant 集成测试
# ============================================================
# 用 EvalSuite 对部署的 Agent 做回归测试。
#
# 运行:
#   python deploy/tests/test_agent.py
#
# 这可以集成到 CI/CD:
#   每次改 Prompt 或换模型后, 跑一次确保没有退化。
# ============================================================

import sys
import time
from pathlib import Path

# 确保项目根在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deploy.config import AppConfig
from deploy.agent_core import create_agent

# 简化: 手工定义测试用例做回归验证
# 进阶: 可以把 phase5/41_evaluation.py 的 EvalSuite 复制到 deploy/
#       实现更完整的评估流水线

# ============================================================
# 构建 Agent
# ============================================================

config = AppConfig.from_env()
config.llm_model = "claude-sonnet-4-6"
agent = create_agent(config)

print("=" * 60)
print("  DevAssistant 集成测试")
print("=" * 60)
print(f"  Model: {config.llm_model}")
print(f"  API: {'reachable' if agent.llm.is_healthy else 'offline'}")
print()

# ============================================================
# 测试用例
# ============================================================

TESTS = [
    {
        "name": "数学计算",
        "task": "计算 (100 + 50) / 3 的结果, 只返回数字",
        "mode": "quick",
        "checks": ["50", "50.0"],  # 答案应包含 50
    },
    {
        "name": "时间查询",
        "task": "现在是哪一年? 只返回年份数字",
        "mode": "quick",
        "checks": ["2026"],
    },
    {
        "name": "工具调用",
        "task": "帮我计算 2 的 10 次方",
        "mode": "quick",
        "checks": ["1024"],
    },
]

passed = 0
failed = 0

for i, tc in enumerate(TESTS):
    print(f"  [{i + 1}/{len(TESTS)}] {tc['name']}...", end=" ")
    start = time.time()

    try:
        result = agent.ask(tc["task"], mode=tc["mode"])
        answer = result.get("answer", "")
        elapsed = (time.time() - start) * 1000

        # 检查
        ok = any(check in answer for check in tc["checks"])
        if ok:
            print(f"✓ ({elapsed:.0f}ms)")
            passed += 1
        else:
            print(f"✗ 期望包含 {tc['checks']}")
            print(f"    实际: {answer[:100]}...")
            failed += 1
    except Exception as e:
        print(f"✗ error: {e}")
        failed += 1

# ============================================================
# 报告
# ============================================================

print(f"\n{'=' * 60}")
print(f"  结果: {passed}/{len(TESTS)} 通过")
if failed > 0:
    print(f"  {failed} 个测试失败")
    sys.exit(1)
else:
    print(f"  全部通过!")
print(f"{'=' * 60}")
