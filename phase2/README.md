# phase2/ — LLM API + Prompt 工程 (5 课)

Phase 2 进入 AI 应用开发核心领域: 调用大语言模型 API、设计提示词、处理结构化输出和流式响应。

## 学习目标

完成 Phase 2 后应能:
- 使用 Anthropic Python SDK 调用 LLM (Message API)
- 设计 system prompt、few-shot 示例和角色扮演提示
- 通过 tool use 实现结构化 JSON 输出
- 处理 SSE 流式响应
- 构建带上下文管理的多轮对话应用

## 课程列表

| # | 文件 | 主题 | 核心内容 |
|---|------|------|---------|
| 11 | `11_llm_api_intro.py` | LLM API 入门 | Anthropic SDK 安装配置、`messages.create()`、Message/ContentBlock 类型、`ANTHROPIC_API_KEY` 和 `ANTHROPIC_BASE_URL` 环境变量 |
| 12 | `12_prompt_engineering.py` | Prompt 工程 | System prompt 设计、role-playing、few-shot、思维链 (Chain-of-Thought)、输出格式约束 |
| 13 | `13_structured_output.py` | 结构化输出 | Tool use 实现 JSON 结构化输出、Pydantic 验证、schema 定义 |
| 14 | `14_streaming.py` | 流式响应 | SSE 事件流、`text_delta` 增量、实时输出展示 |
| 15 | `15_chat_app.py` | 对话应用 | 多轮对话、消息历史管理、token 限制处理、简单 chatbot |

## 运行方式

```bash
# 设置 API 密钥 (项目根目录 .env 已配置)
export ANTHROPIC_API_KEY=sk-xxx
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic

# 运行任意课程
python phase2/11_llm_api_intro.py
```

## 关键技术点

### Anthropic SDK 调用模式

```python
import anthropic
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.content[0].text)
```

### DeepSeek 兼容代理

本项目的 `.env` 指向 DeepSeek API (`https://api.deepseek.com/anthropic`)，该 API 兼容 Anthropic SDK 的 `/v1/messages` 端点格式。

## 前置要求

- 完成 Phase 1 (Python 基础)
- 理解 HTTP API 基本概念
- DeepSeek API Key (或 Anthropic 官方 API Key)
