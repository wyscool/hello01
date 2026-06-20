# ============================================================
# codebase_qa/generator.py — AnswerGenerator
# ============================================================
# 包装 LlmClient，构建代码搜索专用 system prompt，
# 将检索到的代码片段格式化后发送给 LLM 生成答案。
# ============================================================


class AnswerGenerator:
    """构建 codebase Q&A prompt，调用 LLM 生成带引用的答案。"""

    CODE_QA_SYSTEM = (
        "你是一个代码库问答助手。根据提供的代码片段回答用户问题。\n"
        "规则:\n"
        "  1. 每个回答必须引用具体的文件路径和行号"
        " (格式: [path/to/file.py:起始行-结束行])\n"
        "  2. 如果代码片段中没有相关信息，明确说"
        " '代码库中未找到相关信息'\n"
        "  3. 区分 '代码中实际存在的' 和 '基于代码推断的'\n"
        "  4. 对于 '在哪里' 类问题，先列出位置再展示代码\n"
        "  5. 对于 '怎么实现' 类问题，展示完整实现并解释关键逻辑\n"
        "  6. 对于 '哪些文件使用' 类问题，列出所有相关文件路径\n"
        "  7. 用中文回答，保持技术准确性"
    )

    def __init__(self, llm_client):
        self._llm = llm_client

    def build_user_prompt(self, query: str, results: list) -> str:
        """将检索结果格式化为 LLM user prompt。

        格式:
          [1] path/to/file.py:10-25 (score: 0.92, type: function)
          ```python
          def some_func():
              ...
          ```
        """
        parts = [f"用户问题: {query}\n\n代码库中检索到的相关代码片段:\n"]

        for i, r in enumerate(results, start=1):
            file_path = r.metadata.get("file_path", "unknown")
            start_line = r.metadata.get("start_line", "?")
            end_line = r.metadata.get("end_line", "?")
            code_type = r.metadata.get("type", "?")
            name = r.metadata.get("name", "?")
            parts.append(
                f"[{i}] {file_path}:{start_line}-{end_line}"
                f" ({code_type}: {name}, score: {r.score})\n"
                f"```python\n{r.text}\n```\n"
            )

        return "\n".join(parts)

    def generate(self, query: str, results: list,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        if not results:
            return "代码库中未找到相关信息。请确认相关代码已索引（POST /index）。"

        user_prompt = self.build_user_prompt(query, results)
        response = self._llm.create(
            messages=[{"role": "user", "content": user_prompt}],
            system=self.CODE_QA_SYSTEM,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self._llm.get_text(response)
