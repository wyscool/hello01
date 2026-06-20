# ============================================================
# log_analyzer/ — 智能日志分析 Agent
# ============================================================
# LLM 驱动的日志分析工具，自动发现异常、追踪根因、生成报告。
#
# 组件:
#   LogParser          — 多格式日志解析 (JSON lines / 文本 / Java 异常栈)
#   LogAnalysisAgent   — ReAct Agent, LLM 自主决定分析策略
#   AppConfig          — 24 字段配置
#
# 用法:
#   CLI:  python -m log_analyzer.cli analyze app.log
#   API:  uvicorn log_analyzer.app:app --port 8003
#   Code: from log_analyzer.agent import create_log_agent
# ============================================================
