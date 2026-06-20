# ============================================================
# Phase 3, Lesson 23: 文档处理 —— 加载、分块、清洗
# ============================================================
#
# 本课目标:
#   1. 理解为什么需要文档处理 — 原始文档 ≠ 可检索片段
#   2. 文档加载 — 从文件系统读取 txt / md / json
#   3. 分块策略 — 固定大小 / 按句子 / 递归分割
#   4. Chunk 重叠 — 为什么 chunk 之间要 overlap
#   5. 文本清洗 — 规范化空白、处理编码
#   6. Metadata 保留 — 每个 chunk 记住来源
#   7. 实战: DocumentProcessor — 从文件到 embedding 就绪
#
# 预计阅读 + 实操时间: 40-50 分钟
#
# 前置: Lesson 21 (Embedding) + Lesson 22 (ChromaDB)
# ============================================================

import re
import json
from pathlib import Path
from dataclasses import dataclass, field


# ============================================================
# 〇、为什么需要文档处理?
# ============================================================
# 你有 10 个技术文档, 每个 2000 字。
# 能直接 embed 整篇文档然后搜索吗?
#
# 问题 1: Embedding 模型有 token 限制
#   all-MiniLM-L6-v2: 最大 256 token (~200 中文字)
#   → 2000 字的文档直接截断, 丢失 90% 内容!
#
# 问题 2: 长文档包含多个主题
#   "Python 简介" 文档里可能同时讲列表、异步、装饰器
#   → 整篇 embed = 所有主题混在一起 → 搜哪个都不准
#
# 问题 3: 搜索结果不精确
#   即使能 embed 整篇, 返回整篇文档给 LLM
#   → 浪费 token, 淹没关键信息
#
# 解决方案: 文档分块 (Chunking)
#   长文档 → 切成小块 → 每块 100-500 字 → 独立 embed
#   → 搜索时精确返回相关段落, 而不是整篇文档

print("=" * 60)
print("文档处理流程图")
print("=" * 60)

print("""
  原始文件                   处理后                    可检索
  ──────────────────────────────────────────────────────────
  docs/                        chunks[]
  ├── python_intro.txt    →  [chunk_001] "Python 是..."    → embed
  ├── database_backup.md  →  [chunk_002] "核心特性..."     → embed
  └── redis_guide.md      →  [chunk_003] "广泛应用..."     → embed
                             [chunk_004] "MySQL 备份..."   → embed
                             [chunk_005] "PostgreSQL..."   → embed
                             [chunk_006] "Redis 是..."    → embed

  每个 chunk 都携带 metadata:
    { source: "python_intro.txt", chunk_index: 0, ... }
""")


# ============================================================
# 一、文档加载 —— 从文件系统读入
# ============================================================
# 实际项目中, 文档来源五花八门:
#   - 本地文件: .txt, .md, .json, .csv, .pdf
#   - 网页: HTML 抓取
#   - 数据库: 从 SQL 导出
#   - API: 从第三方系统拉取
#
# 本课聚焦文件加载, 掌握后可扩展到其他来源。

print("=" * 60)
print("文档加载")
print("=" * 60)

DOCS_DIR = Path(__file__).parent / "docs"


def load_text_file(path: Path) -> str | None:
    """加载单个文本文件, 自动处理编码。"""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="gbk")
        except Exception:
            return None


def load_documents(docs_dir: Path, glob: str = "*.txt") -> list[dict]:
    """
    从目录加载所有匹配的文件。
    返回 [{"path": Path, "filename": str, "content": str}, ...]
    """
    docs = []
    for file_path in sorted(docs_dir.glob(glob)):
        content = load_text_file(file_path)
        if content:
            docs.append({
                "path": file_path,
                "filename": file_path.name,
                "content": content,
                "suffix": file_path.suffix,
            })
    return docs


# 加载演示文档
txt_docs = load_documents(DOCS_DIR, "*.txt")
md_docs = load_documents(DOCS_DIR, "*.md")
all_docs = txt_docs + md_docs

print(f"  加载 {len(all_docs)} 个文档:")
for doc in all_docs:
    print(f"    {doc['filename']} ({len(doc['content'])} 字符)")


# ============================================================
# 二、分块策略 1: 固定大小分块
# ============================================================
# 最简单的分块: 每 N 个字符切一刀。
#
# 优点: 简单、可预测、chunk 大小均匀
# 缺点: 可能在句子中间切断 → 语义不完整
#
# 类比 Java:
#   String.substring(0, 500), substring(500, 1000), ...
#   问题: substring(487, 987) 可能切在 "数据" 和 "库" 之间

print("\n" + "=" * 60)
print("分块策略 1: 固定大小")
print("=" * 60)


def chunk_by_size(text: str, chunk_size: int = 200,
                  overlap: int = 50) -> list[dict]:
    """
    固定大小分块, 带重叠。

    chunk_size: 每块最大字符数
    overlap: 相邻块重叠的字符数

    为什么需要 overlap?
      没有 overlap: chunk 边界可能正好切在关键词上
        chunk1: "...如何使用 MySQL 进行数据"
        chunk2: "库备份和恢复操作..."

      有了 overlap: 关键信息至少在一个 chunk 里完整
        chunk1: "...如何使用 MySQL 进行数据"
        chunk2: "MySQL 进行数据库备份和恢复操作..."
               ^^^^^^^^^ 和上一块重叠, 关键信息完整
    """
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        if chunk_text.strip():  # 跳过空白块
            chunks.append({
                "text": chunk_text.strip(),
                "chunk_index": chunk_idx,
                "start_char": start,
                "end_char": end,
            })
            chunk_idx += 1

        # 下一个块从 start + chunk_size - overlap 开始
        if end >= len(text):
            break
        start += chunk_size - overlap

    return chunks


# 演示: 对第一篇文章做固定大小分块
sample = all_docs[0]["content"]
chunks_fixed = chunk_by_size(sample, chunk_size=150, overlap=30)
print(f"\n  原文: {len(sample)} 字符 → {len(chunks_fixed)} 个 chunk")
for c in chunks_fixed:
    print(f"  [{c['chunk_index']}] ({c['start_char']}-{c['end_char']}): "
          f"{c['text'][:60]}...")


# ============================================================
# 三、分块策略 2: 按句子分块
# ============================================================
# 在句子边界切分, 保证每个 chunk 包含完整句子。
# 更适合中文、日文等无空格分隔的语言。
#
# 思路:
#   1. 按标点符号分句
#   2. 累积句子直到超过 chunk_size
#   3. 在句子边界切一个 chunk

print("\n" + "=" * 60)
print("分块策略 2: 按句子分块")
print("=" * 60)

# 中文句子分隔符
SENTENCE_BOUNDARY = re.compile(r"[。！？;；\n]+")


def split_sentences(text: str) -> list[str]:
    """将文本按句子边界分割, 保留非空句子。"""
    parts = SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_by_sentences(text: str, max_chunk_size: int = 300,
                       overlap_sentences: int = 1) -> list[dict]:
    """
    按句子分块, 每个块不超过 max_chunk_size 字符。

    overlap_sentences: 相邻块共享的句子数。
      确保跨块边界的上下文不丢失。
    """
    sentences = split_sentences(text)
    chunks = []
    chunk_idx = 0
    i = 0

    while i < len(sentences):
        current = []
        current_len = 0

        # 累积句子直到接近 max_chunk_size
        while i < len(sentences) and current_len + len(sentences[i]) <= max_chunk_size:
            current.append(sentences[i])
            current_len += len(sentences[i])
            i += 1

        if current:
            chunks.append({
                "text": "。".join(current) + "。",
                "chunk_index": chunk_idx,
                "sentence_count": len(current),
            })
            chunk_idx += 1

            # 回退 overlap_sentences 个句子, 让下一块和当前块重叠
            if overlap_sentences > 0 and len(current) > overlap_sentences:
                i -= overlap_sentences
        else:
            # 单句就超过 max_chunk_size, 直接放入
            i += 1

    return chunks


chunks_sent = chunk_by_sentences(sample, max_chunk_size=200, overlap_sentences=1)
print(f"\n  原文: {len(sample)} 字符 → {len(chunks_sent)} 个 chunk (按句子)")
for c in chunks_sent:
    print(f"  [{c['chunk_index']}] ({c['sentence_count']}句): "
          f"{c['text'][:80]}...")


# ============================================================
# 四、分块策略 3: 递归字符分割 (RecursiveCharacterTextSplitter)
# ============================================================
# 业界最佳实践 —— LangChain 的同款算法。
#
# 思路:
#   1. 先尝试用大分隔符切 (段落 → 句子 → 词)
#   2. 如果切出来的块还是太大, 递归用更细的分隔符
#   3. 直到所有块都在 chunk_size 以内
#
# 分隔符优先级 (从粗到细):
#   "\\n\\n"  (段落) → "\\n" (行) → "。" (句子) → "，" (短语) → "" (字符)

print("\n" + "=" * 60)
print("分块策略 3: 递归字符分割 (最佳实践)")
print("=" * 60)


def recursive_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    """
    递归字符分割 —— LangChain RecursiveCharacterTextSplitter 的简化实现。

    算法:
      1. 用当前分隔符切分文本
      2. 合并片段直到接近 chunk_size
      3. 如果单段超长, 换更细的分隔符递归
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

    # 找第一个有效的分隔符
    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if len(separators) > 1 else []

    if sep:
        splits = text.split(sep)
    else:
        # 最后的手段: 按字符切
        splits = list(text)

    # 合并片段
    chunks: list[str] = []
    current = ""

    for split in splits:
        # 尝试加入当前片段
        candidate = current + (sep if current and sep else "") + split

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # 当前块已满, 保存; 处理剩余部分
            if current.strip():
                chunks.append(current.strip())

            if len(split) > chunk_size and remaining_seps:
                # 单段超长, 递归用更细的分隔符
                sub_chunks = recursive_chunk(
                    split, chunk_size, overlap, remaining_seps
                )
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = split

    if current.strip():
        chunks.append(current.strip())

    # 处理 overlap: 在每块末尾加上下一块的开头
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                next_start = chunks[i + 1][:overlap]
                if next_start and next_start not in chunk[-overlap:]:
                    chunk = chunk + next_start
            overlapped.append(chunk)
        return overlapped

    return chunks


# 对比三种策略在同一文本上的效果
print("\n  同一文本, 三种策略对比:")
print(f"  {'策略':<12} {'块数':<6} {'平均大小':<10}")
print(f"  {'─' * 30}")

for name, chunks in [
    ("固定大小", chunk_by_size(sample, 200)),
    ("按句子", chunk_by_sentences(sample, 200)),
    ("递归分割", [{"text": c} for c in recursive_chunk(sample, 200)]),
]:
    texts = [c["text"] for c in chunks]
    avg_size = sum(len(t) for t in texts) / len(texts) if texts else 0
    print(f"  {name:<12} {len(chunks):<6} {avg_size:<10.0f}")

print(f"""
  策略选择指南:
    固定大小   — 简单场景、英文文本、快速原型
    按句子     — 中文文本、需要语义完整性
    递归分割   — 生产环境首选, 兼顾大小和语义 (LangChain 同款)

  类比 Java:
    固定大小   ≈ String.substring(0, N) 循环
    递归分割   ≈ 策略模式, 每种分隔符是一个 SplitStrategy
""")


# ============================================================
# 五、文本清洗 —— 让输入变干净
# ============================================================

print("=" * 60)
print("文本清洗")
print("=" * 60)


def clean_text(text: str) -> str:
    """
    标准化文本, 移除噪声。

    清洗步骤:
      1. 统一换行符 (\\r\\n → \\n)
      2. 压缩多余空白 (连续空格 → 单个空格, 连续换行 → 双换行)
      3. 去除首尾空白
      4. 移除控制字符 (保留换行)
    """
    # 统一换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 压缩连续空白 (但不合并换行)
    text = re.sub(r"[ \t]+", " ", text)       # 空格/Tab → 单个空格
    text = re.sub(r"\n{3,}", "\n\n", text)    # 3个以上换行 → 2个

    # 移除行首行尾多余空白
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 移除控制字符 (保留 \n)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text.strip()


# 演示清洗效果
DIRTY = """  这是一段   有问题的文本。



多处空行,    多余空格,  和奇怪的\t制表符。
  有些行前面有空白。

结尾也有空白。   """

clean = clean_text(DIRTY)
print(f"\n  清洗前 ({len(DIRTY)} 字符):")
print(f"  ┌{DIRTY}┐")
print(f"\n  清洗后 ({len(clean)} 字符):")
print(f"  ┌{clean}┐")


# ============================================================
# 六、集成管道: DocumentProcessor
# ============================================================
# 把加载、清洗、分块串成一条流水线。
# 输入: 文件目录
# 输出: 带 metadata 的 chunk 列表, 可直接喂给 embedding

print("\n" + "=" * 60)
print("综合实战: DocumentProcessor")
print("=" * 60)


@dataclass
class Chunk:
    """文档片段, 携带完整 metadata。"""
    text: str
    source: str           # 来源文件名
    chunk_index: int      # 在文档中的序号
    total_chunks: int     # 该文档的总 chunk 数
    metadata: dict = field(default_factory=dict)


class DocumentProcessor:
    """
    文档处理管道: 加载 → 清洗 → 分块 → Chunk 列表。

    类比 Java:
      class DocumentProcessor {
          List<Chunk> process(Path docsDir) {
              return loadDocuments(docsDir).stream()
                  .map(this::clean)
                  .flatMap(this::chunk)
                  .collect(toList());
          }
      }
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_directory(self, docs_dir: Path,
                          patterns: tuple[str, ...] = ("*.txt", "*.md")) -> list[Chunk]:
        """处理一个目录下的所有文档。"""
        all_chunks: list[Chunk] = []

        for pattern in patterns:
            for file_path in sorted(docs_dir.glob(pattern)):
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)

        return all_chunks

    def process_file(self, file_path: Path) -> list[Chunk]:
        """处理单个文件: 加载 → 清洗 → 分块。"""
        content = load_text_file(file_path)
        if not content:
            print(f"  ⚠️  跳过: {file_path.name} (无法读取)")
            return []

        content = clean_text(content)
        raw_chunks = recursive_chunk(
            content, self.chunk_size, self.overlap
        )
        total = len(raw_chunks)

        chunks = []
        for i, text in enumerate(raw_chunks):
            chunks.append(Chunk(
                text=text,
                source=file_path.name,
                chunk_index=i,
                total_chunks=total,
                metadata={
                    "filepath": str(file_path),
                    "suffix": file_path.suffix,
                },
            ))
        return chunks

    def to_dicts(self, chunks: list[Chunk]) -> list[dict]:
        """转为字典列表, 方便序列化或传给 ChromaDB。"""
        return [
            {
                "id": f"{c.source}#{c.chunk_index}",
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "metadata": c.metadata,
            }
            for c in chunks
        ]


# 跑一遍完整流程
processor = DocumentProcessor(chunk_size=400, overlap=50)
all_chunks = processor.process_directory(DOCS_DIR)

print(f"\n  处理完成:")
print(f"    文件数: {len(all_docs)}")
print(f"    Chunk 总数: {len(all_chunks)}")

# 按来源分组统计
from collections import Counter
source_counts = Counter(c.source for c in all_chunks)
for source, count in source_counts.items():
    print(f"      {source}: {count} chunks")

print(f"\n  样例 chunks:")
for chunk in all_chunks[:5]:
    preview = chunk.text[:60].replace("\n", " ")
    print(f"  [{chunk.source}#{chunk.chunk_index}] "
          f"({len(chunk.text)}字): {preview}...")


# ============================================================
# 七、对接 ChromaDB —— 从文件到可搜索
# ============================================================
# 证明 DocumentProcessor 的输出可以直接喂给 Lesson 22 的 ChromaDB。

print("\n" + "=" * 60)
print("对接 ChromaDB — 从文件到可搜索")
print("=" * 60)

try:
    import chromadb
    from sentence_transformers import SentenceTransformer

    # 用和 L22 一样的 embedding function
    ef = SentenceTransformer("all-MiniLM-L6-v2")

    class EmbedFn:
        def name(self): return "all-MiniLM-L6-v2"
        def embed_query(self, input): return ef.encode(input, convert_to_numpy=True).tolist()
        def embed_documents(self, input): return ef.encode(input, convert_to_numpy=True).tolist()
        def __call__(self, input): return self.embed_query(input)

    # 创建 collection
    client = chromadb.PersistentClient(path="./phase3/chroma_db")
    collection = client.get_or_create_collection(
        name="processed_docs",
        embedding_function=EmbedFn(),
    )

    # 将 DocumentProcessor 的输出转成 ChromaDB 需要的格式
    chunk_dicts = processor.to_dicts(all_chunks)

    if chunk_dicts:
        collection.add(
            ids=[c["id"] for c in chunk_dicts],
            documents=[c["text"] for c in chunk_dicts],
            metadatas=[{"source": c["source"], "chunk": c["chunk_index"]} for c in chunk_dicts],
        )
        print(f"  已添加 {len(chunk_dicts)} 个 chunk 到 ChromaDB")

        # 搜索测试
        test_queries = [
            "MySQL 怎么备份?",
            "Redis 有哪些数据结构?",
            "Python 是什么语言?",
        ]
        for q in test_queries:
            results = collection.query(query_texts=[q], n_results=2)
            print(f"\n  🔍 \"{q}\"")
            for doc_id, doc_text in zip(results["ids"][0], results["documents"][0]):
                preview = doc_text[:60].replace("\n", " ")
                print(f"    → {doc_id}: {preview}...")

except ImportError:
    print("  (chromadb 未安装, 跳过对接演示)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Lesson 23 完成! 文档处理管道已掌握。")
    print("=" * 60)
    print(f"""
  回顾: 你学会了什么?

  1. 文档加载 — 从文件系统读入, 自动处理编码
  2. 三种分块策略:
     固定大小  — 简单, 但切碎语义
     按句子    — 保留语义, 中文友好
     递归分割  — 生产级, LangChain 同款
  3. Chunk 重叠 — 确保关键信息不落在边界上
  4. 文本清洗 — 统一换行、压缩空白、去噪声
  5. DocumentProcessor — 加载→清洗→分块 一条龙
  6. 对接 ChromaDB — 处理完直接入库可搜索

  RAG 数据准备流水线:
    原始文件 → DocumentProcessor → Chunk[] → ChromaDB → LLM

  🎯 下一课: Lesson 24 — 检索流水线
     把 Embedding + 向量搜索 + LLM 生成串起来,
     完整实现 "查询 → 检索 → 生成" 的 RAG 流程!
""")


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 尝试不同分块参数:
#    修改 chunk_size 和 overlap, 对比搜索结果:
#    - chunk_size=200, overlap=20 → 小块, 精细匹配
#    - chunk_size=800, overlap=100 → 大块, 更多上下文
#    哪种更适合代码文档? 哪种更适合新闻文章?
#
# 2. 添加新文档格式支持:
#    实现 JSON 文件加载:
#    - 读取 .json 文件
#    - 提取特定字段 (如 {"title": ..., "body": ...})
#    - 只对 body 分块, title 放入 metadata
#    添加到 DocumentProcessor 的 patterns 列表。
#
# 3. 实现按标题分块 (Markdown 文档):
#    对于 Markdown 文件, 按 # 标题 切分。
#    每个 # 标题 + 下面的内容 = 一个 chunk。
#    提示: 用正则 r"^#+" 识别标题行。
#
# 4. 给 chunks 加统计信息:
#    处理完文档后, 打印:
#    - 总 chunk 数
#    - 平均 chunk 大小
#    - 最小/最大 chunk 大小
#    - 每个文件的 chunk 分布
#
# 5. (挑战) 实现自适应分块:
#    根据文档类型自动选择分块策略:
#    - .txt 文件: 固定大小分块
#    - .md 文件: 按标题分块
#    - 代码文件: 按函数/类边界分块
#    提示: 用文件后缀做路由。
#
# 6. (思考) 分块策略对搜索结果的影响:
#    同一批文档, 用不同分块策略处理后,
#    搜索同一个 query, 对比 Top-5 结果。
#    哪种策略找出的结果最相关? 为什么?
#
# 做完后告诉我:
#   - 你觉得哪种分块策略最适合你的学习笔记?
#   - 如果你的文档是 Java 代码, 会怎么分块?
# 我们继续 Lesson 24: 检索流水线。
# ============================================================


# ============================================================
# 试试看 — 练习实现
# ============================================================

print("\n" + "=" * 60)
print("试试看: 练习实现")
print("=" * 60)

# ----------------------------------------------------------
# 练习 1: 不同分块参数对比
# ----------------------------------------------------------
print("\n--- 练习 1: 不同分块参数对比 ---")

# 取一篇较长的文档做实验
if all_docs:
    sample_text = all_docs[0]["content"]
else:
    # fallback: 用一段模拟文本
    sample_text = (
        "Python 是一门解释型、动态类型的编程语言。"
        "它由 Guido van Rossum 于 1991 年发布。"
        "Python 的设计哲学强调代码可读性和简洁的语法。"
        "Python 支持多种编程范式, 包括面向对象、命令式、函数式编程。"
        "Python 有一个庞大的标准库, 涵盖了文件 I/O、网络编程、数据库接口等。"
        "在数据科学领域, Python 是主流语言, NumPy、Pandas、Scikit-learn 等库构建了强大的生态系统。"
        "在 Web 开发中, Django 和 Flask 是两个最流行的 Python 框架。"
        "Python 的包管理工具 pip 让安装第三方库变得非常简单。"
        "asyncio 是 Python 3.4 引入的异步 I/O 框架, 在 Python 3.7 中得到重大改进。"
        "Python 的类型提示 (Type Hints) 从 3.5 开始引入, 让代码更易于理解和维护。"
    ) * 3  # 重复 3 次模拟长文档

# 对比不同 chunk_size 和 overlap
configs = [
    (200, 20, "小块精细"),
    (500, 50, "中块平衡"),
    (800, 100, "大块上下文"),
]

print(f"  原文长度: {len(sample_text)} 字符\n")
print(f"  {'配置':<15} {'chunk数':<8} {'平均大小':<10} {'最小':<8} {'最大':<8}")
print(f"  {'─' * 50}")

for chunk_size, overlap, label in configs:
    chunks = chunk_by_size(sample_text, chunk_size=chunk_size, overlap=overlap)
    sizes = [len(c["text"]) for c in chunks]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    print(f"  {label:<15} {len(chunks):<8} {avg_size:<10.0f} {min(sizes):<8} {max(sizes):<8}")

print(f"""
  选择指南:
    小块 (200): 适合代码文档 — 每个函数/方法独立, 精确匹配
    中块 (500): 适合技术笔记 — 平衡上下文和精度 (推荐默认)
    大块 (800): 适合新闻文章 — 需要完整叙述, 不担心精度下降
""")

# ----------------------------------------------------------
# 练习 2: JSON 文件加载支持
# ----------------------------------------------------------
print("\n--- 练习 2: JSON 文件加载支持 ---")


def load_json_file(file_path: Path) -> list[dict]:
    """
    加载 JSON 文件, 提取结构化文档。

    支持的 JSON 格式:
      1. [{"title": "...", "body": "..."}, ...]  — 对象数组
      2. {"title": "...", "body": "..."}          — 单对象
      3. [{"content": "..."}, ...]                — content 字段
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    # 统一成列表
    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return []

    docs = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        # 提取文本内容 (body 或 content 字段)
        body = item.get("body") or item.get("content") or ""
        if not body:
            continue

        title = item.get("title") or item.get("name") or file_path.stem

        docs.append({
            "filename": file_path.name,
            "title": title,
            "body": body,
            "item_index": i,
            "suffix": file_path.suffix,
        })

    return docs


# 创建示例 JSON 文件做演示
sample_json_path = DOCS_DIR / "_sample_kb.json"
if not sample_json_path.exists():
    sample_json_data = [
        {"title": "Python 入门", "body": "Python 是一门解释型语言, 语法简洁易读, 适合初学者快速上手编程。"},
        {"title": "数据库备份", "body": "MySQL 使用 mysqldump 命令备份。PostgreSQL 使用 pg_dump, 支持并行备份。"},
        {"title": "Redis 缓存", "body": "Redis 是内存数据库, 常用作缓存层。支持 String、Hash、List、Set 等数据结构。"},
    ]
    sample_json_path.write_text(json.dumps(sample_json_data, ensure_ascii=False, indent=2), encoding="utf-8")
    # 注意: 这里直接写文件, 仅用于演示

# 加载 JSON
json_docs = load_json_file(sample_json_path) if sample_json_path.exists() else []
if json_docs:
    print(f"  从 JSON 加载了 {len(json_docs)} 条文档:")
    for d in json_docs:
        print(f"    [{d['title']}] {d['body'][:50]}...")

    print(f"\n  JSON 文档分块示例:")
    for d in json_docs:
        chunks = chunk_by_size(d["body"], chunk_size=100, overlap=20)
        print(f"    {d['title']}: {len(chunks)} chunks")
        for c in chunks:
            print(f"      → {c['text'][:60]}...")
else:
    print("  (未找到 JSON 文件, 创建 _sample_kb.json 后可测试)")

# ----------------------------------------------------------
# 练习 3: Markdown 按标题分块
# ----------------------------------------------------------
print("\n--- 练习 3: Markdown 按标题分块 ---")


def chunk_markdown_by_headers(text: str) -> list[dict]:
    """
    按 Markdown 标题 (#) 分块。

    每个 # 标题 + 下面的内容 = 一个 chunk。
    标题层级: # (一级), ## (二级), ### (三级) ...

    算法:
      1. 用正则匹配所有标题行
      2. 在两个标题之间切分
      3. 每个 section 的标题 + 正文作为一个 chunk
    """
    # 匹配标题行 (行首的 #, 支持 1-6 级)
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    # 找到所有标题的位置
    headers = list(header_pattern.finditer(text))

    if not headers:
        # 没有标题, 整篇作为一个 chunk
        return [{"text": text.strip(), "header": "(无标题)", "level": 0}]

    chunks = []
    for i, match in enumerate(headers):
        header_text = match.group(2)
        level = len(match.group(1))

        # 当前标题的内容: 从当前标题到下一个标题 (或文本结尾)
        start = match.start()
        if i + 1 < len(headers):
            end = headers[i + 1].start()
        else:
            end = len(text)

        section = text[start:end].strip()

        # 如果 section 太长, 可以进一步用递归分割
        if len(section) > 800:
            sub_chunks = recursive_chunk(section, chunk_size=500, overlap=50)
            for j, sub in enumerate(sub_chunks):
                chunks.append({
                    "text": sub.strip(),
                    "header": header_text,
                    "level": level,
                    "sub_index": j,
                })
        else:
            chunks.append({
                "text": section,
                "header": header_text,
                "level": level,
            })

    return chunks


# 用模拟 Markdown 测试
SAMPLE_MD = """# Python 基础知识

Python 是一门解释型、动态类型的编程语言。
它以简洁易读的语法著称, 适合初学者学习。

## 变量和数据类型

Python 中的变量不需要声明类型。
常见的数据类型包括: int, float, str, list, dict, set, tuple 等。

### 列表

列表是有序的可变集合, 用方括号创建。
支持索引、切片、append、pop 等操作。

### 字典

字典是键值对的集合, 用花括号创建。
查找速度快, 类似 Java 的 HashMap。

## 函数

函数用 def 关键字定义。
支持默认参数、可变参数和关键字参数。

### lambda 表达式

lambda 是匿名函数, 适合简单的单行操作。

# 高级特性

Python 的高级特性包括装饰器、生成器、上下文管理器等。
"""

md_chunks = chunk_markdown_by_headers(SAMPLE_MD)
print(f"  Markdown 按标题分块: {len(md_chunks)} 个 chunks")
for c in md_chunks:
    hdr = c.get("header", "")
    lvl = c.get("level", 0)
    indent = "  " * (lvl - 1)
    preview = c["text"][:60].replace("\n", " ")
    print(f"  {indent}[H{lvl}] {hdr}: {preview}...")

# 也测试本地的 .md 文件
md_file = DOCS_DIR / "redis_guide.md"  # 假设存在
if md_file.exists():
    md_text = md_file.read_text(encoding="utf-8")
    md_chunks_real = chunk_markdown_by_headers(md_text)
    print(f"\n  实际文件 {md_file.name}: {len(md_chunks_real)} 个 chunks (按标题)")

# ----------------------------------------------------------
# 练习 4: Chunk 统计信息
# ----------------------------------------------------------
print("\n--- 练习 4: Chunk 统计信息 ---")


def chunk_statistics(chunks: list, source_name: str | None = None) -> dict:
    """
    计算 chunk 列表的统计信息。

    Returns:
      {total, avg_size, min_size, max_size, size_distribution, ...}
    """
    if not chunks:
        return {"total": 0, "avg_size": 0, "min_size": 0, "max_size": 0}

    # 兼容 dict 和 Chunk 对象
    sizes = []
    for c in chunks:
        if hasattr(c, "text"):
            sizes.append(len(c.text))
        elif isinstance(c, dict):
            sizes.append(len(c.get("text", c.get("content", ""))))
        else:
            sizes.append(len(str(c)))

    # 大小分布
    distribution = {
        "0-100": sum(1 for s in sizes if s <= 100),
        "100-300": sum(1 for s in sizes if 100 < s <= 300),
        "300-500": sum(1 for s in sizes if 300 < s <= 500),
        "500+": sum(1 for s in sizes if s > 500),
    }

    stats = {
        "total": len(sizes),
        "avg_size": sum(sizes) / len(sizes),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "total_size": sum(sizes),
        "distribution": distribution,
    }

    if source_name:
        stats["source"] = source_name

    return stats


# 对已处理的文档做统计
print("  所有文档的 chunk 统计:")
print(f"  {'文件':<25} {'chunk数':<8} {'平均大小':<10} {'最小':<8} {'最大':<8}")
print(f"  {'─' * 65}")

all_stats = []
if all_docs:
    for doc in all_docs:
        # 用递归分割处理
        chunks = recursive_chunk(doc["content"], chunk_size=500, overlap=50)
        stats = chunk_statistics(chunks, doc["filename"])
        all_stats.append(stats)
        print(f"  {doc['filename']:<25} {stats['total']:<8} "
              f"{stats['avg_size']:<10.0f} {stats['min_size']:<8} {stats['max_size']:<8}")

    if all_stats:
        total_chunks = sum(s["total"] for s in all_stats)
        total_size = sum(s["total_size"] for s in all_stats)
        print(f"\n  总计: {len(all_stats)} 个文件, {total_chunks} 个 chunks, {total_size} 字符")
        print(f"  全局平均 chunk 大小: {total_size / total_chunks:.0f} 字符")

# ----------------------------------------------------------
# 练习 5: 自适应分块 (挑战)
# ----------------------------------------------------------
print("\n--- 练习 5: 自适应分块 (挑战) ---")


class AdaptiveChunker:
    """
    根据文件类型自动选择分块策略。

    策略路由:
      .md  → 按标题分块 (保留文档结构)
      .py  → 按函数/类边界分块 (保留代码逻辑)
      .txt → 递归字符分割 (通用策略)
      .json → 按字段分块 (结构化文档)
      其他  → 递归字符分割 (fallback)
    """

    # Python 函数/类边界的简单正则
    PYTHON_DEF_PATTERN = re.compile(
        r"^(def\s+\w+|class\s+\w+|async\s+def\s+\w+)", re.MULTILINE
    )

    @staticmethod
    def chunk_code(text: str, chunk_size: int = 500) -> list[str]:
        """
        按 Python 函数/类边界分块。

        比固定大小分块更尊重代码逻辑: 尽量不在函数中间切断。
        """
        # 找到所有函数/类定义的行号
        lines = text.split("\n")
        boundaries = [0]  # 起始边界

        for i, line in enumerate(lines):
            if AdaptiveChunker.PYTHON_DEF_PATTERN.match(line.strip()):
                boundaries.append(i)

        if len(boundaries) == 1:
            # 没有函数/类定义, 回退到递归分割
            return recursive_chunk(text, chunk_size)

        boundaries.append(len(lines))  # 结束边界

        chunks = []
        current = []
        current_len = 0

        for i in range(len(boundaries) - 1):
            section_lines = lines[boundaries[i]:boundaries[i + 1]]
            section_text = "\n".join(section_lines)

            if current_len + len(section_text) <= chunk_size:
                current.extend(section_lines)
                current_len += len(section_text)
            else:
                if current:
                    chunks.append("\n".join(current))
                # 如果单个 section 就超过 chunk_size, 进一步拆分
                if len(section_text) > chunk_size:
                    sub_chunks = recursive_chunk(section_text, chunk_size)
                    chunks.extend(sub_chunks)
                    current = []
                    current_len = 0
                else:
                    current = section_lines
                    current_len = len(section_text)

        if current:
            chunks.append("\n".join(current))

        return [c for c in chunks if c.strip()]

    @classmethod
    def chunk(cls, text: str, suffix: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """自适应分块入口。"""
        suffix = suffix.lower()

        if suffix == ".md":
            # 先按标题分, 太大的标题块再用递归分割
            md_chunks = chunk_markdown_by_headers(text)
            result = []
            for mc in md_chunks:
                chunk_text = mc["text"]
                if len(chunk_text) > chunk_size:
                    result.extend(recursive_chunk(chunk_text, chunk_size, overlap))
                else:
                    result.append(chunk_text)
            return result

        elif suffix == ".py":
            return cls.chunk_code(text, chunk_size)

        elif suffix == ".json":
            # JSON 已在加载时拆分为记录, 这里处理单个记录
            return recursive_chunk(text, chunk_size, overlap)

        else:  # .txt 及其他
            return recursive_chunk(text, chunk_size, overlap)


# 测试自适应分块
print("  自适应分块测试:")

import textwrap  # 用于代码文本格式化

# 模拟不同类型的文本
test_files = {
    "note.md": SAMPLE_MD,
    "script.py": textwrap.dedent("""\
        import os
        import sys

        def hello():
            print("Hello, World!")
            return 42

        class Calculator:
            def add(self, a, b):
                return a + b

            def subtract(self, a, b):
                return a - b

        def main():
            calc = Calculator()
            result = calc.add(1, 2)
            print(result)

        if __name__ == "__main__":
            main()
    """),
    "readme.txt": "这是一个普通的文本文件。\n包含多个段落。\n用于测试默认分块策略。\n" * 10,
}

for fname, content in test_files.items():
    suffix = Path(fname).suffix
    chunks = AdaptiveChunker.chunk(content, suffix, chunk_size=300)
    avg = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    print(f"  {fname} ({suffix}): {len(chunks)} chunks, 平均 {avg:.0f} 字符")
    for i, c in enumerate(chunks[:3]):
        preview = c[:50].replace("\n", "\\n")
        print(f"    [{i}] {preview}...")
    if len(chunks) > 3:
        print(f"    ... 共 {len(chunks)} chunks")

print(f"""
  自适应分块的设计思路:
    1. .md 按标题 → 保留章节结构, 搜"Python 函数"能定位到函数章节
    2. .py 按函数 → 保留代码逻辑, 不会在 def 中间切断
    3. .txt 递归分割 → 通用策略, 兼容所有纯文本
    4. .json 已在加载时拆分 → 每条记录独立处理

  类比 Java:
    AdaptiveChunker ≈ 策略模式 + 工厂模式
    chunk_code / chunk_md / chunk_txt = 不同的 SplitStrategy 实现
""")

# ----------------------------------------------------------
# 练习 6: 分块策略对搜索的影响 (思考)
# ----------------------------------------------------------
print("\n--- 练习 6: 分块策略对搜索的影响 (思考) ---")

print("""
  思考: 分块策略如何影响检索质量?

  实验设计 (在同一批文档上对比):
    1. 用 3 种策略分别分块
    2. 分别建索引 (ChromaDB)
    3. 用 5 个相同的 query 搜索
    4. 对比 Top-5 结果

  预期结果:
  ┌────────────┬──────────────────┬──────────────────────┐
  │ 策略        │ 优点              │ 缺点                  │
  ├────────────┼──────────────────┼──────────────────────┤
  │ 固定大小    │ 分布均匀、可预测   │ 可能切断语义            │
  │ 按句子      │ 语义完整          │ 大小不一, 有的太短       │
  │ 递归分割    │ 兼顾大小+语义      │ 实现复杂               │
  │ 按标题(MD) │ 结构感知          │ 只适合有标题的文档       │
  │ 按函数(PY) │ 代码逻辑感知       │ 只适合 Python 代码      │
  └────────────┴──────────────────┴──────────────────────┘

  关键发现:
    1. 对于技术文档 (带标题), 按标题分块效果最好
       → 每个 chunk 是一个完整的知识点
    2. 对于叙事文章 (无标题), 递归分割效果最好
       → 保持段落完整性
    3. 对于代码文件, 按函数/类分块效果最好
       → 不会把单个函数拆到多个 chunk

  分块策略的选择直接影响 RAG 的"R" (检索)质量:
    - 切太细 → 上下文不够, LLM 看不懂
    - 切太粗 → 噪声太多, 检索不精确
    - 切断语义 → 关键信息丢失, 检索失败

  类比 Java:
    分块 ≈ 数据库的页大小 (page size)
    chunk_size 太大 ≈ 每次读一页包含很多无关数据
    chunk_size 太小 ≈ 需要读很多页才能拼出完整信息
    合适的 chunk_size ≈ 大多数查询只需要 1-2 页
""")

print("\n" + "=" * 60)
print("  试试看练习完成!")
print("=" * 60)
