"""
AI 应用开发 —— Python 学习课程追踪器
=====================================

这个文件是你的学习进度看板。随时可以运行它查看当前进度：
    python curriculum.py

课程文件都在 phase1/ 目录下，每节课一个可运行的 .py 文件。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Lesson:
    file_name: str
    topic: str
    completed: bool = False
    notes: str = ""


@dataclass
class Phase:
    number: int
    title: str
    duration: str
    description: str
    lessons: list[Lesson]


# ============================================================
# 完整课程路线
# ============================================================

CURRICULUM: list[Phase] = [
    Phase(
        number=1,
        title="Python 基础 + 工程化",
        duration="4-6 周",
        description="Python 语法、数据类型、控制流、函数、类、模块。目标: 能独立写 Python 脚本。",
        lessons=[
            Lesson("01_basics.py", "变量、类型、打印、type hints、__main__", completed=True),
            Lesson("02_control_flow.py", "if/elif/else、for、while、truthy/falsy、match/case", completed=True),
            Lesson("03_collections.py", "list/切片、tuple/解包、dict、set、推导式、内置函数", completed=True),
            Lesson("04_functions.py", "def、*args、**kwargs、lambda、闭包、装饰器", completed=True),
            Lesson("05_classes.py", "class、__init__、方法、dataclass、继承、MRO", completed=True),
            Lesson("06_modules_packages.py", "import、模块、包、__init__.py", completed=True),
            Lesson("07_errors.py", "try/except/finally、raise、自定义异常", completed=True),
            Lesson("08_files_json.py", "open()、pathlib、json 模块", completed=True),
            Lesson("09_async_basics.py", "async/await、asyncio 基础", completed=True),
            Lesson("10_pytest_basics.py", "pytest 测试基础", completed=True),
        ]
    ),
    Phase(
        number=2,
        title="LLM API 调用 + Prompt 工程",
        duration="4-6 周",
        description="调用大模型 API、Prompt 设计、流式响应、工具调用。目标: 做出第一个 AI 应用。",
        lessons=[
            Lesson("11_llm_api_intro.py", "Claude / OpenAI API 基础调用", completed=True),
            Lesson("12_prompt_engineering.py", "Prompt 设计模式、角色、少样本", completed=True),
            Lesson("13_structured_output.py", "JSON 输出、函数调用", completed=True),
            Lesson("14_streaming.py", "流式响应、实时输出", completed=True),
            Lesson("15_chat_app.py", "构建一个对话应用", completed=True),
        ]
    ),
    Phase(
        number=3,
        title="RAG + 向量数据库 + Embedding",
        duration="4-6 周",
        description="语义检索、向量数据库、文档分块、检索流水线。目标: 做出知识库问答系统。",
        lessons=[
            Lesson("21_embeddings.py", "Embedding 概念与 API 调用", completed=True),
            Lesson("22_vector_db.py", "向量数据库 (pgvector / Pinecone)", completed=True),
            Lesson("23_document_processing.py", "文档加载、分块、清洗", completed=True),
            Lesson("24_retrieval_pipeline.py", "检索 + 重排序 + 生成", completed=True),
            Lesson("25_knowledge_base.py", "构建完整知识库 Q&A 系统", completed=True),
        ]
    ),
    Phase(
        number=4,
        title="Agent + 工具调用 + MCP",
        duration="6-8 周",
        description="Agent 循环、工具调用、多步推理、MCP 协议。目标: 做出有用的智能体。",
        lessons=[
            Lesson("31_agent_basics.py", "Agent 概念、ReAct 模式", completed=True),
            Lesson("32_tool_use.py", "工具定义、调用、结果处理", completed=True),
            Lesson("33_planning.py", "多步规划、反思、自我修正", completed=True),
            Lesson("34_mcp_protocol.py", "Model Context Protocol 协议", completed=True),
            Lesson("35_agent_project.py", "端到端 Agent 项目", completed=True),
        ]
    ),
    Phase(
        number=5,
        title="AI 工程化",
        duration="持续",
        description="评估、可观测性、成本控制、生产部署。目标: 做出生产级 AI 系统。",
        lessons=[
            Lesson("41_evaluation.py", "评估框架、指标设计", completed=True),
            Lesson("42_observability.py", "日志、追踪、监控", completed=True),
            Lesson("43_cost_control.py", "缓存、批处理、成本控制", completed=True),
            Lesson("44_production.py", "部署、扩展、高可用", completed=True),
        ]
    ),
]


def print_progress() -> None:
    """打印当前学习进度。"""
    print("=" * 60)
    print("  AI 应用开发学习进度")
    print("=" * 60)

    total_lessons = 0
    completed_lessons = 0

    for phase in CURRICULUM:
        phase_total = len(phase.lessons)
        phase_done = sum(1 for l in phase.lessons if l.completed)
        total_lessons += phase_total
        completed_lessons += phase_done

        # Phase header
        status = "✅" if phase_done == phase_total and phase_total > 0 else "📚"
        if phase_done == 0:
            status = "⬜"

        print(f"\n{status} Phase {phase.number}: {phase.title}")
        print(f"   预计: {phase.duration} | {phase_done}/{phase_total} 课完成")
        print(f"   {phase.description}")

        # Lessons
        for lesson in phase.lessons:
            mark = "  ✅" if lesson.completed else "  ⬜"
            print(f"{mark} {lesson.file_name:<30} {lesson.topic}")

    # Summary
    pct = completed_lessons / total_lessons * 100 if total_lessons > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"  总进度: {completed_lessons}/{total_lessons} 课完成 ({pct:.1f}%)")
    print(f"{'=' * 60}")

    # What's next
    next_phase: Optional[Phase] = None
    next_lesson: Optional[Lesson] = None
    for phase in CURRICULUM:
        for lesson in phase.lessons:
            if not lesson.completed:
                next_phase = phase
                next_lesson = lesson
                break
        if next_lesson:
            break

    if next_lesson and next_phase:
        print(f"\n🎯 下一课: phase{next_phase.number}/{next_lesson.file_name}")
        print(f"   主题: {next_lesson.topic}")
        print(f"   运行: python phase{next_phase.number}/{next_lesson.file_name}")
        print(f"\n💡 提示: 修改本文件中对应 lesson 的 completed=True 来标记进度")
    else:
        print("\n🎉 恭喜! 所有课程已完成!")


def mark_completed(file_name: str) -> None:
    """标记某节课为已完成。"""
    for phase in CURRICULUM:
        for lesson in phase.lessons:
            if lesson.file_name == file_name:
                lesson.completed = True
                print(f"✅ 已标记完成: {file_name}")
                return
    print(f"❌ 未找到课程: {file_name}")


if __name__ == "__main__":
    print_progress()

    # 示例: 标记课程完成 (取消注释下面这行并修改文件名)
    # mark_completed("06_modules_packages.py")
