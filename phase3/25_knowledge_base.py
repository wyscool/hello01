# ============================================================
# Phase 3, Lesson 25: 知识库 Q&A 系统 —— Phase 3 收官项目
# ============================================================
#
# 本课目标:
#   融合 Phase 3 全部技能，构建一个完整的终端知识库问答应用。
#   和 L15 的 PyChat 不同, 这次 LLM 的回答基于你的私有文档。
#
#   融合的技能:
#     Lesson 21: Embedding (sentence-transformers 本地向量化)
#     Lesson 22: ChromaDB (文档存储 + ANN 搜索)
#     Lesson 23: 文档处理 (加载、清洗、分块)
#     Lesson 24: 检索流水线 (Retrieval → Rerank → Generate)
#
#   新增知识:
#     1. 知识库管理 — add/list/remove 命令
#     2. 多格式文档导入 — .txt / .md / .json
#     3. 流式 RAG 回答 — 边检索边生成
#     4. 会话历史管理 — 多轮追问
#     5. 终端 UI — 颜色、进度、格式化
#     6. 应用架构 — KnowledgeBase + QASystem + CLI 三层
#
# 预计阅读 + 实操时间: 60-70 分钟
#
# 前置: 已完成 Lesson 21-24
# ============================================================

import os
import sys
import re
import json
import time
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


# ============================================================
# 〇、环境准备
# ============================================================

# --- Anthropic API (流式 + 非流式) ---
from anthropic import Anthropic
from anthropic.types import Message

api_key = os.getenv("ANTHROPIC_API_KEY")
base_url = os.getenv("ANTHROPIC_BASE_URL")

client_kwargs = {"api_key": api_key} if api_key else {}
if base_url:
    client_kwargs["base_url"] = base_url
llm_client = Anthropic(**client_kwargs)


def _get_text(response: Message) -> str:
    parts = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts)


try:
    llm_client.messages.create(
        model="claude-sonnet-4-6", max_tokens=10,
        messages=[{"role": "user", "content": "ping"}],
    )
    api_ok = True
except Exception:
    api_ok = False

if api_ok:
    print("✅ API 连接正常")
else:
    print("⚠️  API 不可用, 将以模拟模式运行")

# --- Embedding 模型 ---
from sentence_transformers import SentenceTransformer

print("  加载 embedding 模型...")
st_model = SentenceTransformer("all-MiniLM-L6-v2")


class EmbedFn:
    """sentence-transformers → ChromaDB 1.5+ embedding function."""

    def name(self) -> str:
        return "all-MiniLM-L6-v2"

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return st_model.encode(input, convert_to_numpy=True).tolist()

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return st_model.encode(input, convert_to_numpy=True).tolist()

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_query(input)


# --- ChromaDB ---
import chromadb

db_client = chromadb.PersistentClient(path="./phase3/chroma_db")

# --- numpy ---
import numpy as np

print()


# ============================================================
# 一、终端颜色工具
# ============================================================

class Color:
    """ANSI 颜色码。类比 Java AnsiConstants。"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    @staticmethod
    def style(text: str, *codes: str) -> str:
        return "".join(codes) + text + Color.RESET


# ============================================================
# 二、系统架构概览
# ============================================================
# 三层设计:
#
#   CLI 层    → 处理用户输入、命令解析、格式化输出
#   QASystem  → 编排检索 + 生成、管理对话历史
#   KnowledgeBase → 文档管理 (入库、搜索、删除)
#
#   ┌─────────────────────────────────────────┐
#   │  CLI: /add /list /remove /stats /help   │
#   ├─────────────────────────────────────────┤
#   │  QASystem                                │
#   │    ask(query)                            │
#   │      ├── KnowledgeBase.search(query)     │
#   │      ├── ContextBuilder.build(...)       │
#   │      └── LLM.generate(prompt)            │
#   ├─────────────────────────────────────────┤
#   │  KnowledgeBase                           │
#   │    ├── ChromaDB (向量存储)               │
#   │    └── DocumentProcessor (文档处理)      │
#   └─────────────────────────────────────────┘
#
# 类比 Java:
#   CLI          ≈ Controller 层 (Spring @Controller)
#   QASystem     ≈ Service 层 (Spring @Service)
#   KnowledgeBase ≈ Repository 层 (Spring @Repository)
#   ChromaDB     ≈ 数据库 (MySQL/PostgreSQL)

print("=" * 60)
print("  PyKB — 本地知识库问答系统")
print("=" * 60)
print("""
  三层架构:
    CLI (输入/输出) → QASystem (编排) → KnowledgeBase (存储)
""")


# ============================================================
# 三、KnowledgeBase —— 知识库管理层
# ============================================================
# 类比 Java:
#   KnowledgeBase ≈ @Repository — 封装 ChromaDB 的所有 CRUD 操作

class KnowledgeBase:
    """知识库: 管理文档的增删查。

    底层是 ChromaDB collection, 上层是面向用户的文件操作。
    """

    def __init__(self, collection_name: str = "pykb_main"):
        self.collection = db_client.get_or_create_collection(
            name=collection_name,
            embedding_function=EmbedFn(),
        )

    # --- 文档加载工具 (内联, 避免 import L23 导致副作用) ---

    @staticmethod
    def _load_file(file_path: Path) -> str | None:
        """加载文件, 自动处理编码。"""
        for enc in ["utf-8", "gbk", "latin-1"]:
            try:
                return file_path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return None

    @staticmethod
    def _clean_text(text: str) -> str:
        """文本清洗。"""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """简单固定大小分块 (优先按段落边界切分)。"""
        if len(text) <= chunk_size:
            return [text]

        # 先按双换行切 (段落), 再合并短段落
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= chunk_size:
                current = (current + "\n\n" + para).strip() if current else para
            else:
                if current:
                    chunks.append(current)
                # 如果单个段落就超过 chunk_size, 硬切
                if len(para) > chunk_size:
                    for i in range(0, len(para), chunk_size - overlap):
                        chunks.append(para[i:i + chunk_size])
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks

    # --- 公开 API ---

    def add_file(self, file_path: Path) -> int:
        """添加单个文件到知识库。返回 chunk 数量。"""
        content = self._load_file(file_path)
        if not content:
            print(f"  {Color.style('✗', Color.RED)} 无法读取: {file_path.name}")
            return 0

        content = self._clean_text(content)
        chunks = self._chunk_text(content, chunk_size=500, overlap=50)

        if not chunks:
            return 0

        # 生成唯一 ID 前缀
        prefix = file_path.stem[:20].replace(" ", "_")
        # 删除旧版本 (如果重新导入)
        existing = self.collection.get(include=[])["ids"]
        to_delete = [eid for eid in existing if eid.startswith(prefix)]
        if to_delete:
            self.collection.delete(ids=to_delete)

        # 批量添加
        ids = [f"{prefix}_{i}" for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=[{
                "source": file_path.name,
                "chunk": i,
                "chunks_total": len(chunks),
                "suffix": file_path.suffix,
                "added_at": datetime.now().isoformat(),
            } for i in range(len(chunks))],
        )
        return len(chunks)

    def add_directory(self, dir_path: Path, patterns: tuple = ("*.txt", "*.md")) -> int:
        """批量导入目录。返回总 chunk 数。"""
        total = 0
        for pattern in patterns:
            for fp in sorted(dir_path.glob(pattern)):
                n = self.add_file(fp)
                if n > 0:
                    print(f"  {Color.style('✓', Color.GREEN)} {fp.name} → {n} chunks")
                total += n
        return total

    def list_docs(self) -> list[dict]:
        """列出知识库中所有文档 (去重后的 source 列表)。"""
        if self.collection.count() == 0:
            return []

        all_meta = self.collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in all_meta["metadatas"]:
            source = meta.get("source", "unknown")
            if source not in seen:
                seen[source] = {
                    "source": source,
                    "chunks": 1,
                    "added_at": meta.get("added_at", "")[:19],
                }
            else:
                seen[source]["chunks"] += 1
        return list(seen.values())

    def remove_doc(self, source: str) -> int:
        """删除指定来源的所有 chunk。返回删除数量。"""
        all_data = self.collection.get(include=["metadatas"])
        to_delete = [
            eid for eid, meta in zip(all_data["ids"], all_data["metadatas"])
            if meta.get("source") == source
        ]
        if to_delete:
            self.collection.delete(ids=to_delete)
        return len(to_delete)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """语义搜索。返回 [{id, text, score, source, chunk}, ...]"""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        items = []
        for doc_id, text, meta, dist in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0],
        ):
            score = 1.0 / (1.0 + dist)
            items.append({
                "id": doc_id,
                "text": text,
                "score": round(score, 4),
                "source": meta.get("source", "unknown"),
                "chunk": meta.get("chunk", 0),
            })

        return items

    def stats(self) -> dict:
        """知识库统计信息。"""
        docs = self.list_docs()
        return {
            "total_docs": len(docs),
            "total_chunks": self.collection.count(),
            "sources": [d["source"] for d in docs],
        }

    @property
    def count(self) -> int:
        return self.collection.count()


# ============================================================
# 四、QASystem —— 问答编排层
# ============================================================
# 类比 Java:
#   QASystem ≈ @Service — 编排 KnowledgeBase + LLM 完成问答

class QASystem:
    """知识库问答系统。

    维护对话历史, 支持多轮追问和流式输出。
    """

    DEFAULT_SYSTEM = """你是一个基于文档的知识库助手。回答规则:
1. 只根据提供的文档内容回答, 不要使用文档外的知识
2. 如果文档中没有相关信息, 明确说 "文档中未找到相关信息"
3. 回答时引用具体的文档来源 (如 "根据《Redis 入门指南》...")
4. 如果多个文档有相关信息, 综合回答
5. 保持简洁、准确"""

    def __init__(self, kb: KnowledgeBase, model: str = "claude-sonnet-4-6"):
        self.kb = kb
        self.model = model
        self.history: list[dict] = []     # 对话历史
        self.max_history = 6              # 保留最近 3 轮 (6 条消息)

    def ask(self, query: str, stream: bool = False) -> dict:
        """执行一次 RAG 问答。

        Returns:
          {answer, sources, latency_ms}
        """
        start = time.time()

        # ① 检索
        raw_results = self.kb.search(query, top_k=5)

        # ② 去噪
        results = [r for r in raw_results if r["score"] >= 0.3]
        if not results:
            results = raw_results[:3] if raw_results else []

        # ③ 构建上下文
        if results:
            context_parts = []
            for i, r in enumerate(results, 1):
                context_parts.append(
                    f"[文档{i}] 来源: {r['source']}\n{r['text']}"
                )
            context = "\n\n---\n\n".join(context_parts)

            user_prompt = f"""以下是相关的参考文档:

{context}

---
基于以上文档, 请回答用户的问题。

用户问题: {query}"""
        else:
            user_prompt = f"""知识库中没有找到与以下问题相关的文档。

用户问题: {query}

请诚实告知用户知识库中没有相关信息, 并建议用户尝试不同的问法或添加相关文档。"""

        # ④ 构建消息列表 (含历史)
        system_msg = self.DEFAULT_SYSTEM
        messages = self._build_messages(user_prompt)

        # ⑤ 生成
        if not api_ok:
            answer = self._mock_answer(query, results)
        elif stream:
            answer = self._stream_generate(messages, system_msg)
        else:
            answer = self._generate(messages, system_msg)

        latency = (time.time() - start) * 1000

        # ⑥ 更新历史
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})
        self._trim_history()

        return {
            "query": query,
            "answer": answer,
            "sources": results,
            "latency_ms": latency,
        }

    def _build_messages(self, current_prompt: str) -> list[dict]:
        """构建消息列表, 拼接历史对话。"""
        messages = []
        for msg in self.history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
        messages.append({"role": "user", "content": current_prompt})
        return messages

    def _generate(self, messages: list[dict], system: str) -> str:
        """非流式生成。"""
        try:
            response = llm_client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                system=system,
                messages=messages,
            )
            return _get_text(response)
        except Exception as e:
            return f"[调用失败: {e}]"

    def _stream_generate(self, messages: list[dict], system: str) -> str:
        """流式生成 (实时打印)。"""
        full_text = []
        try:
            with llm_client.messages.stream(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                system=system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full_text.append(text)
            print()  # 流结束换行
            return "".join(full_text)
        except Exception as e:
            return f"[流式调用失败: {e}]"

    def _mock_answer(self, query: str, results: list[dict]) -> str:
        """模拟生成 (API 不可用时的后备)。"""
        if not results:
            return f"[模拟] 知识库中没有找到与 '{query}' 相关的文档。"
        lines = [f"[模拟] 基于 {len(results)} 条检索结果:"]
        for i, r in enumerate(results, 1):
            preview = r["text"][:80].replace("\n", " ")
            lines.append(f"  [{i}] {r['source']}: {preview}...")
        return "\n".join(lines)

    def _trim_history(self):
        """裁剪历史, 防止 token 爆炸。"""
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def clear_history(self):
        self.history = []


# ============================================================
# 五、CLI 交互界面
# ============================================================
# 类比 Java:
#   CLI ≈ @Controller — 处理用户输入, 调用 Service, 格式化输出

class CLI:
    """知识库问答系统命令行界面。"""

    def __init__(self):
        self.kb = KnowledgeBase()
        self.qa = QASystem(self.kb)
        self.stream_mode = False
        self.running = True

    def run(self):
        """主循环。"""
        self._print_welcome()

        while self.running:
            try:
                user_input = input(
                    Color.style("\n📚 你: ", Color.BOLD + Color.CYAN)
                ).strip()
            except (KeyboardInterrupt, EOFError):
                print("\n")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                self._handle_command(user_input)
            else:
                self._handle_question(user_input)

        self._print_goodbye()

    def _handle_command(self, cmd: str):
        """命令调度。"""
        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        match action:
            case "/help":
                self._cmd_help()
            case "/add":
                self._cmd_add(arg)
            case "/list":
                self._cmd_list()
            case "/remove":
                self._cmd_remove(arg)
            case "/stats":
                self._cmd_stats()
            case "/stream":
                self._cmd_stream(arg)
            case "/clear":
                self._cmd_clear()
            case "/quit" | "/exit":
                self.running = False
            case _:
                print(f"  {Color.style('未知命令', Color.RED)}: {cmd}")
                print(f"  输入 {Color.style('/help', Color.YELLOW)} 查看可用命令")

    def _handle_question(self, question: str):
        """处理用户提问。"""
        if self.kb.count == 0:
            print(f"\n  {Color.style('⚠️  知识库为空!', Color.YELLOW)}")
            print(f"  请先用 {Color.style('/add <文件或目录>', Color.CYAN)} 导入文档")
            print(f"  示例: {Color.style('/add phase3/docs/', Color.DIM)}")
            return

        print()  # 空行

        if self.stream_mode:
            print(Color.style("🤖 AI: ", Color.BOLD + Color.GREEN), end="")
            sys.stdout.flush()
            result = self.qa.ask(question, stream=True)
        else:
            result = self.qa.ask(question, stream=False)
            print(Color.style("🤖 AI: ", Color.BOLD + Color.GREEN) + result["answer"])

        # 打印来源
        if result["sources"]:
            sources = result["sources"]
            print(f"\n  {Color.style('📖 来源', Color.DIM)} "
                  f"({len(sources)} 条, {result['latency_ms']:.0f}ms):")
            for i, src in enumerate(sources, 1):
                score_str = f"{src['score']:.3f}"
                print(f"  [{i}] {src['source']} "
                      f"{Color.style(f'(相关度: {score_str})', Color.DIM)}")
        else:
            print(f"\n  {Color.style('(知识库中无相关文档)', Color.DIM)}")

    # --- 命令实现 ---

    def _cmd_help(self):
        print(f"""
  {Color.style('PyKB 命令列表', Color.BOLD)}
  {'─' * 40}
  {Color.style('/add <路径>', Color.CYAN)}    导入文档或目录 (支持 .txt .md .json)
  {Color.style('/list', Color.CYAN)}         列出知识库中的所有文档
  {Color.style('/remove <文件名>', Color.CYAN)}  删除指定文档
  {Color.style('/stats', Color.CYAN)}        查看知识库统计
  {Color.style('/stream on|off', Color.CYAN)}  切换流式输出
  {Color.style('/clear', Color.CYAN)}        清除对话历史
  {Color.style('/help', Color.CYAN)}         显示此帮助
  {Color.style('/quit', Color.CYAN)}         退出程序

  {Color.style('直接输入问题即可开始问答', Color.DIM)}
""")

    def _cmd_add(self, path_str: str):
        if not path_str:
            print(f"  {Color.style('用法: /add <文件或目录路径>', Color.YELLOW)}")
            return

        p = Path(path_str).expanduser().resolve()
        if not p.exists():
            print(f"  {Color.style('✗', Color.RED)} 路径不存在: {p}")
            return

        print(f"  导入中...")
        if p.is_file():
            n = self.kb.add_file(p)
            if n > 0:
                print(f"  {Color.style('✓', Color.GREEN)} {p.name} → {n} chunks")
            # 重建 QASystem 以感知新数据 (可选)
        elif p.is_dir():
            total = self.kb.add_directory(p)
            print(f"  {Color.style('✓', Color.GREEN)} 共导入 {total} 个 chunks")
        else:
            print(f"  {Color.style('✗', Color.RED)} 不支持的路径类型")

    def _cmd_list(self):
        docs = self.kb.list_docs()
        if not docs:
            print(f"  {Color.style('知识库为空', Color.DIM)}")
            return

        print(f"\n  {Color.style('知识库文档列表', Color.BOLD)} ({len(docs)} 个文档)")
        print(f"  {'─' * 50}")
        for i, doc in enumerate(docs, 1):
            print(f"  [{i}] {doc['source']:<25s} "
                  f"{doc['chunks']} chunks  "
                  f"{Color.style(doc['added_at'], Color.DIM)}")

    def _cmd_remove(self, source: str):
        if not source:
            print(f"  {Color.style('用法: /remove <文件名>  (用 /list 查看)', Color.YELLOW)}")
            return

        n = self.kb.remove_doc(source)
        if n > 0:
            print(f"  {Color.style('✓', Color.GREEN)} 已删除 {source} ({n} chunks)")
        else:
            print(f"  {Color.style('✗', Color.RED)} 未找到: {source}")

    def _cmd_stats(self):
        stats = self.kb.stats()
        print(f"""
  {Color.style('知识库统计', Color.BOLD)}
  {'─' * 30}
  文档数:    {stats['total_docs']}
  Chunk 总数: {stats['total_chunks']}
  流式输出:  {'开启' if self.stream_mode else '关闭'}
  对话轮次:  {len(self.qa.history) // 2}
""")
        if stats["sources"]:
            print(f"  文档列表:")
            for s in stats["sources"]:
                print(f"    - {s}")

    def _cmd_stream(self, arg: str):
        if arg.lower() in ("on", "true", "1", "开启"):
            self.stream_mode = True
            print(f"  {Color.style('✓', Color.GREEN)} 流式输出已开启")
        elif arg.lower() in ("off", "false", "0", "关闭"):
            self.stream_mode = False
            print(f"  {Color.style('✓', Color.GREEN)} 流式输出已关闭")
        else:
            status = "开启" if self.stream_mode else "关闭"
            print(f"  流式输出: {status}")
            print(f"  用法: {Color.style('/stream on', Color.CYAN)} "
                  f"或 {Color.style('/stream off', Color.CYAN)}")

    def _cmd_clear(self):
        self.qa.clear_history()
        print(f"  {Color.style('✓', Color.GREEN)} 对话历史已清除")

    def _print_welcome(self):
        print(f"""
  {Color.style('欢迎使用 PyKB — 本地知识库问答系统', Color.BOLD + Color.GREEN)}
  {'─' * 50}
  基于你的私有文档回答问题, 数据完全本地处理。

  快速开始:
    {Color.style('/add phase3/docs/', Color.CYAN)}        导入示例文档
    {Color.style('直接输入问题', Color.CYAN)}             开始检索问答
    {Color.style('/help', Color.CYAN)}                   查看所有命令

  按 Ctrl+C 退出
""")

        # 自动加载 docs/ 目录 (如果知识库为空且有 docs 目录)
        docs_dir = Path(__file__).parent / "docs"
        if self.kb.count == 0 and docs_dir.exists():
            print(f"  {Color.style('⚡ 首次启动, 自动导入示例文档...', Color.YELLOW)}")
            total = self.kb.add_directory(docs_dir)
            print(f"  {Color.style('✓', Color.GREEN)} 已导入 {total} 个 chunks, 可以直接开始提问\n")

    def _print_goodbye(self):
        print(f"\n  {Color.style('再见! 知识库数据已持久化到 ChromaDB。', Color.DIM)}")
        print(f"  下次启动会自动加载已有文档。\n")


# ============================================================
# 六、非交互模式 —— 适合脚本调用
# ============================================================
# 除了交互式 CLI, 也支持命令行一次性问答:

def one_shot(query: str, kb: KnowledgeBase | None = None):
    """一次性问答 (非交互模式)。"""
    if kb is None:
        kb = KnowledgeBase()
    qa = QASystem(kb)

    if kb.count == 0:
        docs_dir = Path(__file__).parent / "docs"
        if docs_dir.exists():
            kb.add_directory(docs_dir)

    result = qa.ask(query, stream=False)
    print(f"\n{Color.style('Q:', Color.BOLD)} {result['query']}")
    print(f"{Color.style('A:', Color.BOLD)} {result['answer']}")
    if result["sources"]:
        print(f"\n{Color.style('来源:', Color.DIM)}")
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] {src['source']} ({src['score']:.3f})")
    latency_str = f"{result['latency_ms']:.0f}ms"
    print(f"\n{Color.style(f'耗时: {latency_str}', Color.DIM)}")


# ============================================================
# 七、模块演示 —— 非交互式验证全链路
# ============================================================

print("\n" + "=" * 60)
print("模块演示: 全链路验证 (非交互)")
print("=" * 60)

# 如果 pykb_main collection 为空, 自动导入
kb_demo = KnowledgeBase()
if kb_demo.count == 0:
    docs_dir = Path(__file__).parent / "docs"
    if docs_dir.exists():
        print(f"  自动导入 docs/ ...")
        kb_demo.add_directory(docs_dir)

# 非交互问答测试
if kb_demo.count > 0:
    qa_demo = QASystem(kb_demo)
    test_queries = [
        "Redis 有哪些数据结构?",
        "怎样备份 MySQL 数据库?",
        "Python 有什么核心特性?",
    ]

    print(f"  知识库: {kb_demo.count} chunks, {len(kb_demo.list_docs())} 个文档\n")

    for i, q in enumerate(test_queries, 1):
        print(f"  {Color.style(f'[{i}]', Color.DIM)} {q}")
        result = qa_demo.ask(q, stream=False)
        answer_preview = textwrap.shorten(result["answer"], width=120, placeholder="...")
        print(f"  → {answer_preview}")
        if result["sources"]:
            top_source = result["sources"][0]["source"]
            latency_str = f"{result['latency_ms']:.0f}ms"
            print(f"  {Color.style(f'   来源: {top_source} ({latency_str})', Color.DIM)}")
        print()
else:
    print(f"  {Color.style('知识库为空, 跳过演示', Color.YELLOW)}")
    print(f"  提示: 运行 CLI 模式会自动导入示例文档, 或手动 /add")

print("=" * 60)
print("模块演示完成 — 进入 CLI 交互模式\n")


# ============================================================
# 八、入口
# ============================================================

if __name__ == "__main__":
    # 支持命令行参数: python phase3/25_knowledge_base.py "Redis 有哪些数据结构?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"  非交互模式 (查询: {query})\n")
        kb = KnowledgeBase()
        if kb.count == 0:
            docs_dir = Path(__file__).parent / "docs"
            if docs_dir.exists():
                kb.add_directory(docs_dir)
        one_shot(query, kb)
    else:
        # 交互模式
        cli = CLI()
        cli.run()

    print("=" * 60)
    print("  Lesson 25 完成! Phase 3 收官!")
    print("=" * 60)
    print(f"""
  回顾: 你学会了什么?

  Phase 3 技能树:
    L21 Embedding       — 文本 → 向量, 语义相似度
    L22 ChromaDB        — 向量存储, ANN 搜索, Metadata 过滤
    L23 文档处理        — 加载、清洗、三种分块策略
    L24 检索流水线      — Retrieve → Rerank → Context → Generate
    L25 知识库 Q&A ★    — 以上四课的完整整合

  PyKB 架构:
    ┌──────────────────────────────────────┐
    │  CLI    /add /list /remove /stream   │  ← 你交互的界面
    │  QASystem  ask() 检索+生成+历史      │  ← 业务编排
    │  KnowledgeBase  search/add/remove    │  ← 数据管理
    │  ChromaDB + sentence-transformers    │  ← 存储 + 向量
    │  Anthropic API (DeepSeek 兼容)       │  ← 生成
    └──────────────────────────────────────┘

  🎯 下一阶段: Phase 4 — Agent + MCP
     Lesson 31: Agent 基础 & ReAct 模式
     让 LLM 不只是回答问题, 而是能自主规划、执行多步任务!

  运行方式:
    python phase3/25_knowledge_base.py              # 交互模式
    python phase3/25_knowledge_base.py "你的问题"    # 一次性问答
""")


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 导入你自己的文档:
#    找一个你常用的笔记、项目 README 或技术博客,
#    用 /add 导入, 然后针对内容提问:
#    - 检索到的 Top-3 是否和你的问题相关?
#    - LLM 的回答有没有"幻觉" (编造文档中没有的内容)?
#
# 2. 对比流式 vs 非流式:
#    用 /stream on 开启流式, 提问同一个问题。
#    - 用户体感上, 流式和非流式哪个更好?
#    - 流式输出时, 你还能看清来源引用吗?
#
# 3. 测试分块参数对答案的影响:
#    修改 KnowledgeBase._chunk_text 的 chunk_size:
#    - chunk_size=200 → 小块, 精细匹配, 但上下文少
#    - chunk_size=1000 → 大块, 更多上下文, 但检索精度下降
#    用 /remove 删除文档, /add 重新导入, 对比效果。
#
# 4. 实现 JSON 文档导入:
#    扩展 add_file(), 识别 .json 后缀:
#    - 读取 JSON, 假设格式为 [{"title": ..., "body": ...}]
#    - title 存入 metadata, body 作为文本分块
#    提示: 用 json.loads() 解析, 然后走和 .txt 一样的流程。
#
# 5. (挑战) 实现多 Collection 切换:
#    给 CLI 加一个 /use <collection_name> 命令:
#    - 切换当前使用的 collection
#    - 不同项目用不同 collection (如 "work_docs", "study_notes")
#    提示: 在 CLI 中维护当前 collection_name, 重建 KnowledgeBase。
#
# 6. (思考) 知识库的质量评估:
#    写 10 个你预期能从文档中找到答案的问题,
#    逐一测试, 记录:
#    - 有多少问题的 Top-1 结果就是正确答案?
#    - 有多少问题的 LLM 回答完全准确?
#    - 不准确的原因是检索没找对, 还是生成出了问题?
#    这就是 RAG 评估的雏形 (Phase 5 会深入)。
#
# 做完后告诉我:
#   - 你导入自己的文档后, 问答效果如何?
#   - 你觉得 RAG 方式相比直接问 LLM, 最大的优势是什么?
# 我们进入 Phase 4: Agent + MCP!
# ============================================================


# ============================================================
# 试试看 — 练习实现
# ============================================================

print("\n" + "=" * 60)
print("试试看: 练习实现")
print("=" * 60)

# ----------------------------------------------------------
# 练习 1: 导入自己的文档并测试
# ----------------------------------------------------------
print("\n--- 练习 1: 导入自己的文档 ---")

print("""
  操作步骤:
    1. 准备你自己的文档 (笔记、README、博客等), 放到 phase3/docs/ 目录
    2. 运行 CLI 交互模式: python phase3/25_knowledge_base.py
    3. 输入 /add phase3/docs/ 导入所有文档
    4. 针对内容提问, 观察 Top-3 是否相关

  CLI 交互示例:
    📚 你: /add phase3/docs/
      导入中...
      ✓ redis_guide.md → 5 chunks
      ✓ mysql_backup.md → 3 chunks
      ...

    📚 你: Redis 支持哪些数据结构?
    🤖 AI: 根据《Redis 入门指南》, Redis 支持以下数据结构:
          String、Hash、List、Set、Sorted Set...
    📖 来源 (3 条, 245ms):
    [1] redis_guide.md (相关度: 0.852)
    ...

  评估要点:
    1. Top-3 检索结果是否和问题相关?
    2. LLM 的回答有没有"幻觉" (编造文档中没有的内容)?
    3. 如果答错了, 是检索没找对 → 还是 LLM 没理解对?
""")

# 非交互式测试 — 用已导入的文档
test_kb = KnowledgeBase()
qa_test = QASystem(test_kb)

# 如果 pykb_main 已有数据, 做一次问答测试
if test_kb.count > 0:
    print("  当前知识库状态:")
    for d in test_kb.list_docs():
        print(f"    {d['source']}: {d['chunks']} chunks")

    test_q = "Redis 有哪些数据结构?"
    result = qa_test.ask(test_q, stream=False)
    print(f"\n  测试问答:")
    print(f"  Q: {test_q}")
    print(f"  A: {result['answer'][:200]}...")
    if result["sources"]:
        print(f"  来源: {[s['source'] for s in result['sources']]}")
else:
    print("  (知识库为空, 请先用 /add 导入文档再测试)")

# ----------------------------------------------------------
# 练习 2: 流式 vs 非流式对比
# ----------------------------------------------------------
print("\n--- 练习 2: 流式 vs 非流式对比 ---")

if test_kb.count > 0:
    import time as time_mod

    compare_q = "怎样备份数据库?"

    # 非流式
    print(f"\n  [非流式] Q: {compare_q}")
    start = time_mod.time()
    result_ns = qa_test.ask(compare_q, stream=False)
    ns_time = (time_mod.time() - start) * 1000
    print(f"  回答: {result_ns['answer'][:150]}...")
    print(f"  延迟: {ns_time:.0f}ms (全部内容一起返回)")
    print(f"  来源: {[s['source'] for s in result_ns['sources']]}")

    # 流式
    if api_ok:
        print(f"\n  [流式] Q: {compare_q}")
        print(f"  流式输出 (实时): ", end="", flush=True)
        start = time_mod.time()
        # 注意: 流式已经在 qa_test.ask 内部打印了
        result_s = qa_test.ask(compare_q, stream=True)
        # 上面已经实时打印了, 这里不重复打印
        s_time = (time_mod.time() - start) * 1000
        print(f"  延迟: {s_time:.0f}ms (首位时间更短)")
    else:
        print(f"\n  [流式] API 不可用, 跳过")

print(f"""
  用户体验对比:
  ┌──────────┬──────────────────┬─────────────────────────┐
  │ 模式      │ 体感              │ 适用场景                  │
  ├──────────┼──────────────────┼─────────────────────────┤
  │ 非流式    │ 等 2-3 秒, 全部出  │ 短回答、批量查询、脚本     │
  │ 流式      │ 立刻开始出字       │ 长回答、聊天、交互体验优先  │
  └──────────┴──────────────────┴─────────────────────────┘

  流式的工程价值:
    1. 首位延迟 (TTFB) 比非流式短 10-50 倍
    2. 用户感觉"系统在思考"而不是"卡住了"
    3. 但流式输出后, 来源引用需要等流结束才能打印
""")

# ----------------------------------------------------------
# 练习 3: 分块参数对答案的影响
# ----------------------------------------------------------
print("\n--- 练习 3: 分块参数对答案的影响 ---")

print("""
  实验方法:
    1. 修改 KnowledgeBase._chunk_text 的 chunk_size
    2. 用 /remove 删除文档, /add 重新导入
    3. 用同一个问题对比效果

  对比实验设计:
""")

# 用内存中的文档模拟对比 (不修改实际 KnowledgeBase)
sample_long_text = """
Python 是一门解释型、动态类型的编程语言。
它由 Guido van Rossum 于 1991 年首次发布。
Python 的设计哲学强调代码可读性和简洁的语法。
Python 的核心特性包括:
1. 列表推导式 — 用一行代码生成新列表
2. 装饰器 — 修改函数行为的语法糖
3. 生成器 — 惰性计算,节省内存
4. asyncio — 异步编程框架
5. 类型提示 — 可选的静态类型标注
Python 广泛应用于 Web 开发、数据科学、AI、自动化脚本等领域。
Django 和 Flask 是两个流行的 Python Web 框架。
NumPy、Pandas、Scikit-learn 是数据科学的核心库。
PyTorch 和 TensorFlow 是主流的深度学习框架。
"""

if test_kb.count == 0:
    print("  (知识库为空, 用模拟数据演示分块对比)")

for cs in [200, 500, 1000]:
    chunks = KnowledgeBase._chunk_text(sample_long_text, chunk_size=cs)
    print(f"\n  chunk_size={cs}: {len(chunks)} chunks")
    for i, c in enumerate(chunks[:4]):
        print(f"    [{i}] ({len(c)}字): {c[:60]}...")
    if len(chunks) > 4:
        print(f"    ... 共 {len(chunks)} chunks")

print(f"""
  预期效果:
    chunk_size=200: 检索精确 (每个 chunk 聚焦一个小主题)
                    但上下文少, LLM 可能无法完整回答综合问题
    chunk_size=500: 平衡 (推荐默认)
                    一个段落约 3-5 个句子, 信息完整 + 检索精准
    chunk_size=1000: 上下文多, 但检索精度下降
                     可能搜到不相关的 chunk, LLM 被干扰

  关键洞察:
    分块大小决定了 RAG 的"分辨率"
    太小: 分辨率高但视野窄
    太大: 视野宽但分辨率低
    最优值取决于你的文档类型和查询模式
""")

# ----------------------------------------------------------
# 练习 4: JSON 文档导入
# ----------------------------------------------------------
print("\n--- 练习 4: JSON 文档导入 ---")

# 扩展 KnowledgeBase 的 add_file 方法以支持 JSON
class KnowledgeBaseWithJson(KnowledgeBase):
    """扩展 KnowledgeBase, 支持 JSON 格式文档导入。"""

    def add_file(self, file_path: Path) -> int:
        """
        添加单个文件, 支持 .txt / .md / .json。

        JSON 格式要求:
          [{"title": "...", "body": "..."}, ...]
          或 {"title": "...", "body": "..."}
          或 [{"content": "..."}, ...]
        """
        suffix = file_path.suffix.lower()

        if suffix == ".json":
            return self._add_json_file(file_path)
        else:
            # 调用父类的文本文件处理
            return super().add_file(file_path)

    def _add_json_file(self, file_path: Path) -> int:
        """导入 JSON 格式的文档。"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  {Color.style('✗', Color.RED)} JSON 解析失败: {e}")
            return 0

        # 统一成列表
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            print(f"  {Color.style('✗', Color.RED)} JSON 格式不支持 (需要数组或对象)")
            return 0

        total_chunks = 0
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            # 提取文本和标题
            body = item.get("body") or item.get("content") or ""
            title = item.get("title") or item.get("name") or f"item_{i}"

            if not body.strip():
                continue

            # 清洗 + 分块
            body = self._clean_text(body)
            chunks = self._chunk_text(body, chunk_size=500, overlap=50)

            # 生成唯一 ID
            safe_title = re.sub(r"[^\w]", "_", title)[:30]
            prefix = f"{file_path.stem}_{safe_title}"

            # 删除旧版本
            existing = self.collection.get(include=[])["ids"]
            to_delete = [eid for eid in existing if eid.startswith(prefix)]
            if to_delete:
                self.collection.delete(ids=to_delete)

            # 添加
            ids = [f"{prefix}_{j}" for j in range(len(chunks))]
            self.collection.add(
                ids=ids,
                documents=chunks,
                metadatas=[{
                    "source": file_path.name,
                    "title": title,
                    "chunk": j,
                    "chunks_total": len(chunks),
                    "item_index": i,
                    "suffix": ".json",
                    "added_at": datetime.now().isoformat(),
                } for j in range(len(chunks))],
            )
            total_chunks += len(chunks)

        return total_chunks


# 创建示例 JSON 并测试导入
SAMPLE_KB_JSON = [
    {"title": "Python 列表推导", "body": "列表推导式是 Python 的特色语法, 用 [expr for x in iterable if cond] 一行生成新列表。比传统的 for 循环更简洁高效。"},
    {"title": "Docker Compose", "body": "docker-compose.yml 文件定义多容器应用。通过 services 指定各个容器, 使用 networks 和 volumes 管理网络和存储。"},
    {"title": "Git Rebase", "body": "git rebase 将当前分支的提交移到目标分支的最新提交之后, 产生线性历史。和 merge 不同, rebase 不产生额外的合并提交。"},
]

sample_json_path = Path(__file__).parent / "docs" / "_demo_kb.json"
sample_json_path.write_text(
    json.dumps(SAMPLE_KB_JSON, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"  创建示例 JSON: {sample_json_path}")
print(f"  包含 {len(SAMPLE_KB_JSON)} 条文档")

# 用扩展的 KnowledgeBase 导入
kb_json = KnowledgeBaseWithJson("pykb_json_test")
n = kb_json.add_file(sample_json_path)
print(f"  导入结果: {n} chunks")

# 验证: 查看 metadata 中的 title
if n > 0:
    sample_data = kb_json.collection.get(limit=3, include=["documents", "metadatas"])
    print(f"\n  导入验证 (前 3 条):")
    for doc_id, text, meta in zip(sample_data["ids"], sample_data["documents"], sample_data["metadatas"]):
        title = meta.get("title", "N/A")
        print(f"  [{doc_id}] title={title}: {text[:50]}...")

    # 搜索测试
    results = kb_json.search("容器编排", top_k=3)
    if results:
        print(f"\n  搜索 '容器编排' Top-3:")
        for r in results:
            print(f"    [{r['score']:.3f}] {r['source']} (title={r.get('id', '')}): "
                  f"{r['text'][:50]}...")

# 清理
try:
    db_client.delete_collection("pykb_json_test")
except Exception:
    pass

print(f"""
  JSON 导入要点:
    1. title → metadata (不参与 embedding, 但可以展示和过滤)
    2. body  → 分块 + embedding (搜索的主要对象)
    3. 每条 JSON 记录独立分块, 互不干扰
    4. 可选: 在 metadata 中增加 tags、date、author 等扩展字段

  这和 rag_kb/ 项目中的 JSON 导入逻辑一致。
""")

# ----------------------------------------------------------
# 练习 5: 多 Collection 切换 (挑战)
# ----------------------------------------------------------
print("\n--- 练习 5: 多 Collection 切换 (挑战) ---")


class MultiCollectionCLI:
    """
    支持多 Collection 切换的 CLI。

    在标准 CLI 基础上添加 /use 命令:
      /use <collection_name>  切换到指定的 collection
      /use                    列出可用的 collection

    不同项目用不同 collection:
      work_docs    — 工作文档
      study_notes  — 学习笔记
      project_x    — 某个项目的文档
    """

    def __init__(self):
        self.current_collection = "pykb_main"
        self.collections: dict[str, KnowledgeBase] = {}

    def get_kb(self) -> KnowledgeBase:
        """获取当前 collection 的 KnowledgeBase (懒加载)。"""
        if self.current_collection not in self.collections:
            self.collections[self.current_collection] = KnowledgeBase(self.current_collection)
        return self.collections[self.current_collection]

    def list_collections(self) -> list[str]:
        """列出 ChromaDB 中所有 collection。"""
        try:
            cols = db_client.list_collections()
            return [c.name for c in cols]
        except Exception:
            return list(self.collections.keys())

    def switch_collection(self, name: str) -> str:
        """切换到指定的 collection。"""
        existing = self.list_collections()
        if name in existing:
            self.current_collection = name
            return f"已切换到 collection: {name}"
        else:
            # 新 collection, 会在第一次使用时自动创建
            self.current_collection = name
            return f"已创建并切换到新 collection: {name}"

    def import_to_current(self, path: str) -> int:
        """导入文档到当前 collection。"""
        kb = self.get_kb()
        p = Path(path).expanduser().resolve()
        if not p.exists():
            print(f"  {Color.style('✗', Color.RED)} 路径不存在: {p}")
            return 0

        if p.is_file():
            return kb.add_file(p)
        elif p.is_dir():
            return kb.add_directory(p)
        return 0

    def show_status(self):
        """显示当前状态。"""
        kb = self.get_kb()
        docs = kb.list_docs()
        all_cols = self.list_collections()

        print(f"""
  {Color.style('多 Collection 管理', Color.BOLD)}
  {'─' * 40}
  当前: {Color.style(self.current_collection, Color.CYAN)}
  文档数: {len(docs)}
  Chunks: {kb.count}
  可用 collections: {', '.join(all_cols) if all_cols else '(空)'}
""")


# 演示多 collection 管理
multi = MultiCollectionCLI()

# 切换到不同 collection 并导入
print("  多 Collection 管理演示:")

# 收集已存在的 collection, 避免重复创建造成混乱
existing_cols = multi.list_collections()

multi.switch_collection("study_notes")
if "study_notes" not in existing_cols:
    # 导入一些文档到 study_notes
    docs_dir = Path(__file__).parent / "docs"
    if docs_dir.exists():
        n = multi.import_to_current(str(docs_dir))
        if n > 0:
            print(f"  study_notes: 已导入 {n} chunks")

multi.switch_collection("work_docs")
if "work_docs" not in existing_cols:
    # 导入少量文件到 work_docs
    if docs_dir.exists():
        n = multi.import_to_current(str(docs_dir))
        if n > 0:
            print(f"  work_docs: 已导入 {n} chunks")

# 显示状态
multi.show_status()

# 在 study_notes 中搜索
multi.switch_collection("study_notes")
kb_study = multi.get_kb()
if kb_study.count > 0:
    results = kb_study.search("Redis", top_k=2)
    print(f"  [study_notes] 搜索 'Redis':")
    for r in results:
        print(f"    [{r['score']:.3f}] {r['source']}: {r['text'][:50]}...")

print(f"""
  Collection 切换的使用场景:
    1. 按项目分: work_docs / side_project / study_notes
    2. 按语言分: python_docs / java_docs / devops_docs
    3. 按版本分: docs_v1 / docs_v2 (A/B 测试检索质量)

  CLI 命令设计:
    /use work_docs      切换到工作文档
    /use                列出所有 collection
    /stats              显示当前 collection 统计

  类比 Java:
    MultiCollectionCLI ≈ 多租户路由
    每个 collection ≈ 一个独立的 schema/database
    /use 命令 ≈ SET search_path = 'work_docs'
""")

# 清理演示创建的额外 collection
try:
    from chromadb.api.client import SharedSystemClient
    # 只保留主要 collection
    keep = {"tech_docs", "processed_docs", "rag_demo", "pykb_main", "dev_notes"}
    for col in db_client.list_collections():
        if col.name not in keep:
            try:
                db_client.delete_collection(col.name)
            except Exception:
                pass
except Exception:
    pass

# ----------------------------------------------------------
# 练习 6: 知识库质量评估 (思考)
# ----------------------------------------------------------
print("\n--- 练习 6: 知识库质量评估 (思考) ---")

print("""
  思考: 如何系统评估你的知识库问答质量?

  ┌─────────────────────────────────────────────────────────┐
  │                RAG 评估框架                              │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │  1. 检索评估 (Retrieval Evaluation)                     │
  │     ┌──────────────────────────────────────────────┐    │
  │     │ 指标: Recall@K, MRR, NDCG                    │    │
  │     │ 方法: 准备 10+ 个问答对, 标注"理想文档"       │    │
  │     │ 检查: 理想文档是否出现在 Top-K 结果中?        │    │
  │     └──────────────────────────────────────────────┘    │
  │                                                         │
  │  2. 生成评估 (Generation Evaluation)                    │
  │     ┌──────────────────────────────────────────────┐    │
  │     │ 指标: 准确性、完整性、幻觉率                  │    │
  │     │ 方法: 人工审查 LLM 回答 vs 源文档             │    │
  │     │ 检查: LLM 有没有编造文档中不存在的内容?       │    │
  │     └──────────────────────────────────────────────┘    │
  │                                                         │
  │  3. 端到端评估 (End-to-End)                             │
  │     ┌──────────────────────────────────────────────┐    │
  │     │ 方法: RAGAS 框架 (Lesson 41 会详细讲)        │    │
  │     │ 指标: Faithfulness, Answer Relevance,        │    │
  │     │       Context Precision, Context Recall       │    │
  │     └──────────────────────────────────────────────┘    │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

  动手设计你自己的评估:

  Step 1: 准备 10 个问题, 你能从已导入的文档中找到答案
  Step 2: 对每个问题标注 1-3 个"理想文档" (你预期会检索到的)
  Step 3: 逐一测试, 记录:

  评估表:
""")

# 实际的评估执行 (如果有数据)
if test_kb.count > 0:
    eval_questions = [
        ("Redis 有哪些数据结构?", ["redis_guide.md"]),
        ("怎样备份 MySQL?", ["mysql_backup.md"]),
        ("Python 是什么语言?", ["python_intro.txt"]),
        ("Docker 是什么?", ["docker_intro.md"]),
        ("PostgreSQL 怎么备份?", ["pg_backup.md"]),
    ]

    print(f"  {'问题':<35} {'Top-1命中':<10} {'Recall@3':<10} {'LLM准确?':<10}")
    print(f"  {'─' * 65}")

    eval_kb = test_kb  # 使用已导入文档的知识库
    eval_qa = QASystem(eval_kb)

    total_top1 = 0
    total_accurate = 0

    for query, ideal_sources in eval_questions:
        result = eval_qa.ask(query, stream=False)
        sources = result["sources"]

        # Top-1 命中检查
        top1_source = sources[0]["source"] if sources else ""
        top1_hit = any(ideal in top1_source for ideal in ideal_sources)
        if top1_hit:
            total_top1 += 1

        # Recall@3 检查
        retrieved_sources = set(s["source"] for s in sources[:3])
        ideal_set = set(ideal_sources)
        recall = len(retrieved_sources & ideal_set) / max(len(ideal_set), 1)

        # LLM 准确性 (简化: 检查是否引用了正确的文档)
        answer = result["answer"]
        accurate = any(ideal in answer for ideal in ideal_sources)
        if accurate:
            total_accurate += 1

        print(f"  {query:<35} {'是' if top1_hit else '否':<10} "
              f"{recall:.2f}       {'是' if accurate else '否':<10}")

    n_q = len(eval_questions)
    print(f"\n  汇总: Top-1 命中 {total_top1}/{n_q} ({total_top1/n_q:.0%})  "
          f"| LLM 准确 {total_accurate}/{n_q} ({total_accurate/n_q:.0%})")

else:
    print("  (知识库为空, 请先导入文档再运行评估)")

print(f"""
  评估结果的诊断:

  If Top-1 命中率低 (检索问题):
    → 检查 chunk 策略是否合适
    → 检查 embedding 模型对中文的支持
    → 考虑添加查询改写

  If Top-1 命中率高但 LLM 准确率低 (生成问题):
    → 检查系统 prompt 是否强调"只基于文档回答"
    → Chunk 可能太大, 包含太多噪声
    → 考虑增加相似度阈值过滤

  If 两者都低:
    → 文档本身可能不够全面
    → 或者你的问题超出了文档覆盖范围

  这是 RAG 系统化的第一步, Phase 5 会深入讲:
    - RAGAS 框架自动化评估
    - 评估驱动的迭代优化
    - 线上监控 + 反馈闭环
""")

print("\n" + "=" * 60)
print("  试试看练习完成!")
print("  Phase 3 所有练习已实现。")
print("=" * 60)
