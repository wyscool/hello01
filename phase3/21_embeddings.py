# ============================================================
# Phase 3, Lesson 21: Embedding 概念与 API 调用
# ============================================================
#
# 本课目标:
#   1. 理解 Embedding 是什么 — 把文本变成数字向量
#   2. 为什么向量能表达语义 — "猫"和"狗"的向量比"猫"和"汽车"更接近
#   3. 余弦相似度 — 衡量两个向量有多"近"
#   4. 用 sentence-transformers 本地生成 Embedding (免费, 无需 API)
#   5. 批量 Embedding — 效率与成本
#   6. 实战: 语义搜索 — 用自然语言搜索文档
#   7. 实战: 语义去重 — 找出意思相同但表述不同的文本
#
# 预计阅读 + 实操时间: 40-50 分钟
#
# 前置: Phase 2 完成 (有 API 调用经验)
# 依赖: numpy, sentence-transformers (已安装)
# ============================================================

import time
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import numpy as np


# ============================================================
# 〇、Embedding 是什么? —— 把文字变成数字
# ============================================================
# 计算机不理解文字, 只理解数字。
# Embedding = 把一段文本映射到一个高维向量 (如 1536 个 float 的数组)。
#
# 关键性质: 语义相近的文本 → 向量也相近
#   "猫是一种宠物"         → [0.12, -0.34, 0.56, ...]
#   "狗是人类的朋友"       → [0.11, -0.32, 0.54, ...]  ← 和上面很像!
#   "今天股市大涨"         → [0.89, 0.72, -0.15, ...]  ← 完全不同!
#
# 类比 Java:
#   Embedding ≈ 语义化的 hashCode()
#   hashCode:   "hello" → 99162322     (只要内容相同, hash 就相同)
#   Embedding:  "你好"  → [0.1, 0.3, ...] (语义相近, 向量就相近)
#
#   普通 hash:  "猫" 和 "猫咪" 的 hash 完全不同 (差一个字就不一样)
#   Embedding:  "猫" 和 "猫咪" 的向量非常接近 (语义相似!)
#
# 应用场景:
#   - 语义搜索:  "怎么备份数据库?" → 找到 "PostgreSQL 备份指南"
#   - 文本聚类:  自动把相似文档归为一组
#   - 推荐系统:  "看过 A 的人还看了 B"
#   - RAG:      把用户问题和知识库都向量化, 找到最相关的文档

print("=" * 60)
print("Embedding 直觉理解")
print("=" * 60)

# 用 2D 向量演示概念 (实际 embedding 是 1536 维或更高)
# 2D 的好处: 可以画在纸上, 直观理解
print("""
  2D 空间中的语义关系 (简化示意):

      ↑ 维度 2
      │
  0.8 │              🐕 狗(0.7, 0.75)
      │          🐈 猫(0.6, 0.7)
  0.6 │
      │
  0.4 │
      │                                  📈 股票(0.8, 0.15)
  0.2 │                              📊 K线(0.75, 0.1)
      │
      └──────────────────────────────────────────→ 维度 1
               0.2    0.4    0.6    0.8

  "猫"和"狗"的向量很接近  → 都是宠物
  "猫"和"股票"的向量很远   → 完全不相关
  "股票"和"K线"的向量很近  → 都是金融话题
""")


# ============================================================
# 一、余弦相似度 —— 衡量向量的"距离"
# ============================================================
# 两个向量的"接近程度"通常用余弦相似度衡量:
#
#   cos(A, B) = (A · B) / (|A| × |B|)
#
#   分子: 点积 (dot product) — 两个向量对应位置相乘再求和
#   分母: 各自的模长乘积 — 做归一化
#
#   结果范围: [-1, 1]
#     1.0  = 方向完全一致 (语义完全相同)
#     0.0  = 正交, 不相关
#    -1.0  = 方向完全相反
#
# 类比 Java:
#   cos 相似度 ≈ String.equals() 的语义版
#   "备份数据库" != "PostgreSQL 备份指南" (字面不同)
#   cos(embed("备份数据库"), embed("PostgreSQL 备份指南")) ≈ 0.85 (语义接近!)

print("\n" + "=" * 60)
print("余弦相似度 — 手写实现")
print("=" * 60)


def cosine_similarity(a: list[float] | np.ndarray,
                      b: list[float] | np.ndarray) -> float:
    """
    计算两个向量的余弦相似度。

    纯 Python 实现, 不依赖任何库。
    和 numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b)) 等价。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# 用假想的 3D embedding 演示
cat_vec    = [0.8, 0.6, 0.1]   # 猫
dog_vec    = [0.7, 0.7, 0.1]   # 狗 — 和猫很像
stock_vec  = [0.1, 0.0, 1.0]   # 股票 — 和猫狗都不像

print(f"\n  cos(猫, 狗)   = {cosine_similarity(cat_vec, dog_vec):.4f}   ← 高, 它们都是宠物")
print(f"  cos(猫, 股票) = {cosine_similarity(cat_vec, stock_vec):.4f}   ← 低, 不相关")
print(f"  cos(狗, 股票) = {cosine_similarity(dog_vec, stock_vec):.4f}   ← 低, 不相关")
print(f"  cos(猫, 猫)   = {cosine_similarity(cat_vec, cat_vec):.4f}   ← 1.0, 和自身完全一致")


# ============================================================
# 二、加载 Embedding 模型 —— 本地运行, 免费
# ============================================================
# 我们使用 sentence-transformers 库, 模型 all-MiniLM-L6-v2:
#   - 384 维向量 (比 OpenAI 的 1536 维更小, 但质量够用)
#   - 本地 CPU 运行, 免费, 不需要 API Key
#   - 模型大小 ~80MB, 首次运行会自动下载
#   - 单条文本 ~10ms, 批量更高效
#
# 类比 Java:
#   本地 embedding ≈ 本地编译 vs 调远程编译服务
#   好处: 离线可用、无费用、无延迟
#   代价: 占用本地内存/CPU
#
# 备选方案:
#   - OpenAI text-embedding-3-small (1536 维, API 调用, $0.02/百万 token)
#   - Voyage AI voyage-3 (Anthropic 推荐, API 调用)
#   - 国内: 智谱 Embedding、百度千帆 Embedding API

print("\n" + "=" * 60)
print("加载本地 Embedding 模型")
print("=" * 60)

embedding_available = False

try:
    from sentence_transformers import SentenceTransformer

    print("  正在加载模型 all-MiniLM-L6-v2 ...")
    print("  (首次运行会下载 ~80MB, 之后会缓存)")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    EMBEDDING_DIM = model.get_embedding_dimension()
    print(f"\n  ✅ 模型加载成功!")
    print(f"  模型: all-MiniLM-L6-v2")
    print(f"  维度: {EMBEDDING_DIM}")
    print(f"  运行位置: CPU (本地)")

    # 测试一条
    test_vec = model.encode("Python 是一门优雅的编程语言")
    print(f"  前 8 个值: {[round(float(v), 6) for v in test_vec[:8]]}")
    print(f"  ... ")
    print(f"  后 8 个值: {[round(float(v), 6) for v in test_vec[-8:]]}")
    embedding_available = True

except ImportError:
    print(f"\n  ⚠️  sentence-transformers 未安装")
    print(f"  安装: pip install sentence-transformers")
    print(f"  将以模拟向量演示概念")
except Exception as e:
    print(f"\n  ⚠️  模型加载失败: {e}")
    print(f"  将以模拟向量演示概念")


def get_embedding(text: str | list[str]) -> np.ndarray:
    """
    统一的 embedding 函数。
    单条文本 → (dim,) 向量; 多条文本 → (N, dim) 矩阵。
    """
    if embedding_available:
        return model.encode(text, convert_to_numpy=True)
    else:
        # 模拟: 用 hash 生成一致的随机向量
        if isinstance(text, str):
            rng = np.random.RandomState(hash(text) % (2**31))
            return rng.randn(EMBEDDING_DIM if 'EMBEDDING_DIM' in dir() else 384).astype(float)
        else:
            rng = np.random.RandomState(42)
            dim = EMBEDDING_DIM if 'EMBEDDING_DIM' in dir() else 384
            return rng.randn(len(text), dim).astype(float)


# ============================================================
# 三、用真实 Embedding 测试语义相似度
# ============================================================

print("\n" + "=" * 60)
print("语义相似度实测")
print("=" * 60)

TEST_PAIRS = [
    ("猫喜欢吃鱼", "狗喜欢吃骨头", "动物饮食 — 应该比较高"),
    ("猫喜欢吃鱼", "今天股市大跌", "宠物 vs 金融 — 应该很低"),
    ("Python 列表推导式", "[x for x in range(10)]", "概念 vs 代码 — 应该相关"),
    ("如何备份 MySQL 数据库", "PostgreSQL 数据备份与恢复指南", "同任务不同数据库 — 应该较高"),
]

for text_a, text_b, description in TEST_PAIRS:
    emb_a = get_embedding(text_a)
    emb_b = get_embedding(text_b)
    sim = cosine_similarity(emb_a, emb_b)
    model_tag = "" if embedding_available else " (模拟)"
    print(f"\n  A: {text_a}")
    print(f"  B: {text_b}")
    print(f"  相似度: {sim:.4f}  ({description}{model_tag})")


# ============================================================
# 四、语义搜索 —— 用自然语言找文档
# ============================================================
# RAG 的核心第一步: 把用户问题和文档库都向量化, 找到最相关的文档。
#
# 流程:
#   1. 预先: 把所有文档 embed, 存起来
#   2. 查询时: 把用户问题 embed
#   3. 计算: 用户问题向量 vs 每个文档向量 → 相似度排序
#   4. 返回: Top-K 最相似的文档
#
# 类比 Java:
#   传统搜索:  keyword LIKE '%备份%'           → 字面匹配
#   语义搜索:  cos(embed(问题), embed(文档))    → 语义匹配

print("\n" + "=" * 60)
print("语义搜索 — 手写实现")
print("=" * 60)

# 文档库: 模拟一个技术文档集合
DOCUMENTS = [
    "Python 是一门解释型、动态类型的编程语言, 以简洁易读著称。",
    "Java 是静态类型的面向对象语言, 运行在 JVM 上, 强调类型安全。",
    "PostgreSQL 备份可以使用 pg_dump 命令, 支持全量和增量备份。",
    "MySQL 数据备份常用 mysqldump, 也可以使用 XtraBackup 做热备份。",
    "Redis 是内存数据库, 常用作缓存, 支持持久化到磁盘。",
    "Docker 是一种容器化技术, 可以把应用及其依赖打包成镜像。",
    "RESTful API 设计应该使用正确的 HTTP 方法和状态码。",
    "pytest 是 Python 的测试框架, 支持参数化测试和 fixture。",
]


class SemanticSearcher:
    """
    最简语义搜索引擎。

    类比 Java:
      class SemanticSearcher {
          Map<String, float[]> docEmbeddings;  // 文档 → 向量
          EmbeddingService embeddingService;   // 嵌入服务
      }
    """

    def __init__(self):
        self.docs: list[str] = []
        self.embeddings: np.ndarray | None = None

    def index(self, documents: list[str]):
        """对文档建索引: embed 所有文档。"""
        self.docs = documents
        print(f"  Embedding {len(documents)} 篇文档...", end=" ", flush=True)
        start = time.time()
        self.embeddings = get_embedding(documents)  # 批量 encode
        elapsed = time.time() - start
        label = "" if embedding_available else " (模拟)"
        print(f"完成, 耗时 {elapsed:.1f}s{label}")

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """搜索与 query 最相关的 top_k 篇文档。"""
        if self.embeddings is None:
            raise ValueError("请先调用 index() 建索引")

        q_vec = get_embedding(query)

        # 2. 计算查询与所有文档的余弦相似度
        # 向量化计算: q_vec @ embeddings.T / (norm(q) * norms) → 一次算出所有相似度
        q_norm = np.linalg.norm(q_vec)
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        similarities = np.dot(self.embeddings, q_vec) / (doc_norms * q_norm)

        # 3. 排序, 返回 top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.docs[i], float(similarities[i])) for i in top_indices]
        return results


searcher = SemanticSearcher()
searcher.index(DOCUMENTS)

# 测试不同查询
QUERIES = [
    "怎么备份数据库?",
    "Python 是什么?",
    "如何部署应用?",
]

for query in QUERIES:
    print(f"\n  🔍 查询: \"{query}\"")
    results = searcher.search(query, top_k=3)
    for i, (doc, score) in enumerate(results):
        marker = "→" if i == 0 else "  "
        print(f"  {marker} [{score:.3f}] {doc[:60]}...")


# ============================================================
# 五、语义去重 —— 找出意思相同的文本
# ============================================================
# 另一个实用场景: 在大量文本中找出语义重复的内容。
# 比如用户反馈中 "太慢了" 和 "响应速度不行" 其实是同一个问题。

print("\n" + "=" * 60)
print("语义去重")
print("=" * 60)

FEEDBACK = [
    "应用启动太慢了, 要等 10 秒",
    "启动速度很慢, 大约需要 10 秒钟",
    "界面很漂亮, 我喜欢这个设计",
    "UI 设计得不错, 颜色搭配很好",
    "支付功能有 bug, 无法完成付款",
    "点击支付按钮后报错, 不能付款",
]


def find_duplicates(texts: list[str], threshold: float = 0.85) -> list[tuple[int, int, float]]:
    """
    找出语义重复的文本对。
    返回 [(i, j, similarity)] 列表。
    """
    vectors = get_embedding(texts)  # 批量 encode

    duplicates = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(vectors[i], vectors[j])
            if sim >= threshold:
                duplicates.append((i, j, sim))
    return duplicates


dups = find_duplicates(FEEDBACK)
if dups:
    print(f"\n  发现 {len(dups)} 对语义重复 (阈值 0.85):")
    for i, j, sim in dups:
        print(f"\n  [{sim:.3f}]")
        print(f"    A: {FEEDBACK[i]}")
        print(f"    B: {FEEDBACK[j]}")
else:
    print(f"\n  未发现语义重复 (阈值 0.85)")
    if not embedding_available:
        print(f"  ⚠️ 模拟数据可能不够准确, 建议配置 OPENAI_API_KEY 后验证")


# ============================================================
# 六、Embedding 的工程注意事项
# ============================================================

print("\n" + "=" * 60)
print("工程要点 & 成本")
print("=" * 60)

print("""
  📊 all-MiniLM-L6-v2 (本课使用的本地模型):
  ┌────────────────────┬──────────────────┐
  │ 维度                │ 384              │
  │ 模型大小             │ ~80MB            │
  │ 速度 (CPU)          │ ~100 条/秒        │
  │ 费用                │ 免费              │
  │ 是否需要联网          │ 仅首次下载        │
  └────────────────────┴──────────────────┘

  📊 OpenAI text-embedding-3-small (备选):
  ┌────────────────────┬──────────────────┐
  │ 维度                │ 1536             │
  │ 价格 (input)        │ $0.02 / 百万 token│
  │ 速度                │ ~1000 条/秒       │
  │ 适合                │ 生产环境, 高并发   │
  └────────────────────┴──────────────────┘

  ⚠️ 关键注意:

  1. 文本长度限制:
     all-MiniLM-L6-v2 最大 256 token (约 200 中文字)。
     超长文档需要分块 (chunk) — Lesson 23 会详细讲。

  2. 批量调用:
     model.encode(["text1", "text2", ...]) 比逐条 encode 快 5-10 倍。
     batch 底层用了矩阵运算优化。
     类比 Java: batch insert vs 逐条 insert。

  3. 缓存:
     文档不变的情况下, embedding 不需要重新计算。
     生产环境通常把向量存在数据库中 (Lesson 22 会讲)。

  4. 模型选择:
     all-MiniLM-L6-v2:     384 维, 本地免费, 学习/原型够用
     text-embedding-3-small: 1536 维, API 调用, 生产质量
     all-mpnet-base-v2:    768 维, 本地, 质量更高但更慢
""")


# ============================================================
# 七、综合实战: 本地文档搜索器
# ============================================================
# 模拟一个场景: 你有一个 Python 学习笔记目录, 想用自然语言搜索。

print("=" * 60)
print("综合实战: 笔记搜索引擎")
print("=" * 60)

# 模拟 "笔记文件"
NOTES = {
    "python_list.md": "Python 列表是有序的可变集合, 支持切片、推导式、append/pop 等操作。列表可以包含任意类型的元素, 包括其他列表。",
    "python_dict.md": "Python 字典是键值对的集合, 类似 Java HashMap。查找速度快, 键必须是可哈希类型。常用方法: get, items, keys, values。",
    "python_async.md": "Python asyncio 提供异步编程支持。使用 async/await 关键字定义协程。事件循环管理和调度协程的执行。适合 IO 密集型任务。",
    "python_test.md": "pytest 是 Python 最流行的测试框架。支持参数化测试、fixture、mock。用 assert 语句写断言, 不需要记住各种 assertXxx 方法。",
    "java_vs_python.md": "Java 需要编译, Python 是解释型。Java 静态类型, Python 动态类型。Java 适合大型企业应用, Python 适合快速开发和数据科学。",
}

note_searcher = SemanticSearcher()
notes_text = list(NOTES.values())
note_names = list(NOTES.keys())
note_searcher.index(notes_text)

print()
search_queries = [
    "怎样创建键值对数据结构?",
    "如何测试 Python 代码?",
    "列表有哪些操作方法?",
    "异步编程怎么写?",
]

for query in search_queries:
    print(f"  🔍 \"{query}\"")
    results = note_searcher.search(query, top_k=2)
    for i, (doc, score) in enumerate(results):
        idx = notes_text.index(doc)
        fname = note_names[idx]
        print(f"    {i+1}. {fname} (相似度: {score:.3f})")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Lesson 21 完成! Embedding 概念已掌握。")
    print("=" * 60)
    print(f"""
  回顾: 你学会了什么?

  1. Embedding = 把文本变成向量
     "猫"和"狗"的向量接近; "猫"和"股票"的向量远离

  2. 余弦相似度 cos(A,B) = A·B / (|A|×|B|)
     范围 [-1, 1], 越接近 1 越相似

  3. 本地 Embedding (sentence-transformers):
     - all-MiniLM-L6-v2: 384 维, 免费, CPU 运行
     - model.encode() 批量调用, 比逐条快 5-10 倍

  4. 语义搜索 = embed(查询) + 找最近向量 + 返回对应文档
     这就是 RAG 的 "R" (Retrieval) 部分!

  5. 语义去重 = 比较所有文档对的相似度, 找出重复

  类比 Java:
    传统搜索:  String.contains("备份")     → 字面匹配
    语义搜索:  cos(embed(q), embed(doc))   → 语义匹配

    搜索引擎进化:
      LIKE '%keyword%'  →  Elasticsearch (倒排索引)  →  向量搜索 (Embedding)
      字面匹配             词频+相关性                     语义理解

  🎯 下一课: Lesson 22 — 向量数据库 (pgvector / Pinecone)
     把向量存起来, 做大规模语义搜索!
""")


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 扩大文档库:
#    往 DOCUMENTS 列表中添加 10 条你自己的文本 (工作文档、笔记、博客),
#    测试语义搜索能否找到正确的文档。
#    记录至少 3 个"搜索到了但字面完全不匹配"的例子。
#
# 2. 对比实验 — 字面搜索 vs 语义搜索:
#    实现一个简单的字面搜索函数 (用 Python 的 in 关键字),
#    和 SemanticSearcher 对比。找一个"语义相关但字面不同"的查询,
#    比较两者的结果。
#
# 3. 调整去重阈值:
#    修改 find_duplicates 的 threshold 参数:
#    - threshold=0.95: 只找出几乎完全相同的文本
#    - threshold=0.70: 找出表述不同但意思相似的文本
#    哪种更适合"用户反馈去重"场景?
#
# 4. 探索 Embedding 的"算术":
#    计算 "国王 - 男人 + 女人" 的向量, 看看它和 "女王" 的相似度。
#    用 OpenAI embedding 试一下:
#      king  = embed("国王")
#      man   = embed("男人")
#      woman = embed("女人")
#      queen = embed("女王")
#      result = king - man + woman
#      print(cos(result, queen))  # 应该接近 1!
#    提示: 用 numpy 数组做加减法。
#
# 5. (探索) 尝试不同的 Embedding 模型:
#    如果你有其他 API Key, 试试:
#    - Voyage AI: voyage-3 (Anthropic 推荐)
#    - 智谱: embedding-2
#    对比同一批文本在不同模型下的相似度排序是否一致。
#
# 6. (思考) 维度 vs 效果:
#    text-embedding-3-small 支持缩短维度 (如 512 维)。
#    尝试用 dimensions=512 参数获取低维向量,
#    对比 1536 维和 512 维的搜索效果差异。
#    提示: text-embedding-3-small 的 dimensions 参数。
#
# 做完后告诉我:
#   - 你觉得语义搜索比字面搜索强在哪? 有什么局限?
#   - 你工作中哪些场景可以用 Embedding 改进?
# 我们继续 Lesson 22: 向量数据库。
# ============================================================


# ============================================================
# 试试看 — 练习实现
# ============================================================

print("\n" + "=" * 60)
print("试试看: 练习实现")
print("=" * 60)

# ----------------------------------------------------------
# 练习 1: 扩大文档库
# ----------------------------------------------------------
# 往 DOCUMENTS 列表中添加 10 条自定义文本，测试语义搜索
print("\n--- 练习 1: 扩大文档库 ---")

MY_DOCUMENTS = DOCUMENTS + [
    "Java 的 Spring Boot 框架通过自动配置简化了微服务开发，内嵌 Tomcat 容器。",
    "Kubernetes 是容器编排平台，支持自动扩缩容、服务发现和负载均衡。",
    "Git 分支管理常用 git flow 工作流，feature 分支从 develop 分出，完成后合并回去。",
    "Linux 常用命令: ls 列出文件、grep 文本搜索、ps 查看进程、top 监控资源。",
    "HTTP 状态码: 200 成功、301 永久重定向、404 未找到、500 服务器内部错误。",
    "SQL 优化常用的手段包括: 建索引、避免 SELECT *、用 EXPLAIN 分析执行计划。",
    "消息队列 Kafka 用于异步解耦，Producer 发送消息，Consumer 消费消息，Broker 存储。",
    "微服务架构中，服务间通信可以用 REST API 或 gRPC，后者基于 HTTP/2 性能更好。",
    "Python 虚拟环境 venv 用于隔离项目依赖，每个项目有独立的 site-packages。",
    "CI/CD 流水线通常包含: 代码检查 → 单元测试 → 构建镜像 → 部署到测试/生产环境。",
]

my_searcher = SemanticSearcher()
my_searcher.index(MY_DOCUMENTS)

# 搜索"字面不匹配但语义相关"的例子
print("  语义搜索测试 (扩大后的文档库):")
expanded_queries = [
    "怎么把应用装到容器里?",     # 期望命中 Kubernetes/Docker
    "代码提交的流程是什么?",     # 期望命中 Git
    "怎么让项目依赖不冲突?",     # 期望命中 Python venv
    "服务器报错 500 是什么意思?", # 期望命中 HTTP 状态码
    "怎么把服务拆成小的?",       # 期望命中微服务
]
for q in expanded_queries:
    results = my_searcher.search(q, top_k=2)
    print(f"\n  🔍 \"{q}\"")
    for doc, score in results:
        print(f"    [{score:.3f}] {doc[:60]}...")

# ----------------------------------------------------------
# 练习 2: 字面搜索 vs 语义搜索对比
# ----------------------------------------------------------
print("\n--- 练习 2: 字面搜索 vs 语义搜索 ---")


def literal_search(query: str, documents: list[str]) -> list[tuple[str, bool]]:
    """
    简单的字面搜索: 用 Python 的 in 关键字匹配。
    返回 [(文档, 是否匹配), ...]
    """
    results = []
    for doc in documents:
        # 简单策略: query 中的关键词出现在 doc 中就算匹配
        keywords = query.replace("?", "").replace("?", "").split()
        matched = any(kw in doc for kw in keywords if len(kw) >= 2)
        results.append((doc, matched))
    return results


print("  对比实验 — 语义相关但字面不同:")
contrast_query = "怎么把代码放到服务器上运行?"
print(f"  查询: \"{contrast_query}\"")

print("\n  [字面搜索] 关键词匹配结果:")
lit_results = literal_search(contrast_query, MY_DOCUMENTS)
found_literal = False
for doc, matched in lit_results:
    if matched:
        print(f"    ✓ 匹配: {doc[:60]}...")
        found_literal = True
if not found_literal:
    print(f"    ✗ 没有文档包含查询中的关键词!")
    print(f"    → 字面搜索完全失败, 因为\"部署\"的表达方式不同")

print("\n  [语义搜索] 向量相似度结果:")
sem_results = my_searcher.search(contrast_query, top_k=3)
for doc, score in sem_results:
    print(f"    [{score:.3f}] {doc[:60]}...")
print("    → 语义搜索能找到 Docker/K8s/CI/CD 相关内容, 尽管字面不匹配")

# ----------------------------------------------------------
# 练习 3: 调整去重阈值
# ----------------------------------------------------------
print("\n--- 练习 3: 调整去重阈值 ---")

# 用更多样的反馈数据测试
MORE_FEEDBACK = FEEDBACK + [
    "付款的时候总是失败，试了好几次都不行",  # 和"支付功能有 bug"语义接近
    "界面颜色搭配得很好看",                  # 和"UI 设计得不错"接近
    "启动加载时间太长",                      # 和"启动太慢"接近
    "这个按钮的颜色可以再调一下",            # 比较独特
]

print("  同一批反馈, 不同阈值下的去重结果:")

for threshold in [0.95, 0.85, 0.70]:
    dups = find_duplicates(MORE_FEEDBACK, threshold=threshold)
    print(f"\n  threshold={threshold}: 发现 {len(dups)} 对重复")
    for i, j, sim in dups:
        print(f"    [{sim:.3f}] {MORE_FEEDBACK[i][:40]}... <-> {MORE_FEEDBACK[j][:40]}...")

print("""
  结论:
    threshold=0.95: 只找出几乎完全相同的表述 → 适合"抄袭检测"
    threshold=0.85: 找出相同含义的不同表述 → 适合"用户反馈去重"(推荐)
    threshold=0.70: 范围太宽, 可能误判 → 适合"话题聚类"
""")

# ----------------------------------------------------------
# 练习 4: Embedding 算术 (国王 - 男人 + 女人 ≈ 女王)
# ----------------------------------------------------------
print("\n--- 练习 4: Embedding 算术 ---")

if embedding_available:
    king = get_embedding("国王")
    man = get_embedding("男人")
    woman = get_embedding("女人")
    queen = get_embedding("女王")
    king_man_woman = king - man + woman

    # 余弦相似度
    sim_result = cosine_similarity(king_man_woman, queen)
    sim_king_queen = cosine_similarity(king, queen)
    sim_man_woman = cosine_similarity(man, woman)

    print(f"  cos(国王 - 男人 + 女人, 女王) = {sim_result:.4f}")
    print(f"    → 越接近 1.0 说明向量算术成立!")
    print(f"  对比: cos(国王, 女王) = {sim_king_queen:.4f}")
    print(f"  对比: cos(男人, 女人) = {sim_man_woman:.4f}")

    # 多试几组类比
    print("\n  更多类比实验:")
    pairs = [
        ("中国", "北京", "日本", "东京"),     # 首都类比
        ("中国", "北京", "法国", "巴黎"),
        ("男人", "国王", "女人", "女王"),     # 性别类比
        ("好", "更好", "快", "更快"),        # 比较级类比
    ]
    for a, b, c, expected in pairs:
        vec = get_embedding(a) - get_embedding(b) + get_embedding(c)
        expected_vec = get_embedding(expected)
        sim = cosine_similarity(vec, expected_vec)
        # 也计算直接相似度作对比
        direct_sim = cosine_similarity(get_embedding(c), get_embedding(expected))
        print(f"  {a} - {b} + {c} ≈ {expected}?  cos={sim:.3f}  "
              f"(直接 cos({c}, {expected})={direct_sim:.3f})")
else:
    print("  (需要 sentence-transformers, 当前为模拟模式, 跳过)")

# ----------------------------------------------------------
# 练习 5: 尝试不同的 Embedding 模型 (探索)
# ----------------------------------------------------------
print("\n--- 练习 5: 尝试不同 Embedding 模型 ---")

print("""
  探索: 多模型对比

  以下代码演示如何对比不同 embedding 模型的相似度排序。
  由于 all-MiniLM-L6-v2 是本地唯一可用的模型, 这里用模拟不同维度的
  模型来展示对比框架。

  如果要实际对比其他模型, 可以:
""")

if embedding_available:
    # 对比 all-MiniLM-L6-v2 (384维) 和 all-mpnet-base-v2 (768维, 如果可用)
    print("  当前模型: all-MiniLM-L6-v2 (384维)")

    try:
        model2 = SentenceTransformer("all-mpnet-base-v2")
        print("  ✅ 也加载了 all-mpnet-base-v2 (768维) 用于对比")

        test_texts = [
            "Python 是一门编程语言",
            "Java 是面向对象语言",
            "今天天气很好",
            "数据库备份方法",
            "异步编程模型",
        ]

        # 用两个模型分别 encode
        emb1 = model.encode(test_texts, convert_to_numpy=True)
        emb2 = model2.encode(test_texts, convert_to_numpy=True)

        # 计算每种模型下的相似度矩阵
        print(f"\n  all-MiniLM-L6-v2 (384维) 相似度矩阵:")
        for i in range(len(test_texts)):
            for j in range(i + 1, len(test_texts)):
                sim = cosine_similarity(emb1[i], emb1[j])
                print(f"    '{test_texts[i][:20]}' vs '{test_texts[j][:20]}': {sim:.3f}")

        print(f"\n  all-mpnet-base-v2 (768维) 相似度矩阵:")
        for i in range(len(test_texts)):
            for j in range(i + 1, len(test_texts)):
                sim = cosine_similarity(emb2[i], emb2[j])
                print(f"    '{test_texts[i][:20]}' vs '{test_texts[j][:20]}': {sim:.3f}")

        print(f"\n  观察: 两种模型的相对排序是否一致? 绝对分值是否有差异?")

    except Exception as e:
        print(f"  all-mpnet-base-v2 不可用: {e}")
        print("  提示: pip install sentence-transformers 后会包含多个预训练模型")

print("""
  补充: 如果用 API 模型 (OpenAI / Voyage / 智谱), 对比要点:
    1. 同一批文本的相似度排序是否一致 (Spearman 相关系数)
    2. 不同领域的表现差异 (代码 vs 自然语言 vs 多语言)
    3. 成本/延迟的权衡
""")

# ----------------------------------------------------------
# 练习 6: 维度 vs 效果 (思考)
# ----------------------------------------------------------
print("\n--- 练习 6: 维度 vs 效果 (思考) ---")

print("""
  思考: 维度越高 = 效果越好?

  Embedding 的维度是对语义信息"压缩"的程度:

  ┌─────────────────┬──────────┬──────────┬─────────────────────┐
  │ 维度             │ 信息量    │ 速度      │ 适用场景             │
  ├─────────────────┼──────────┼──────────┼─────────────────────┤
  │ 1536 (OpenAI)   │ 最高      │ 最慢      │ 生产环境, 高精度要求   │
  │ 768  (mpnet)    │ 高        │ 中等      │ 平衡选择             │
  │ 384  (MiniLM)   │ 中等      │ 快        │ 原型/本地/资源受限    │
  │ 256             │ 较低      │ 很快      │ 粗粒度分类           │
  │ 128             │ 低        │ 最快      │ 对精度要求不高的场景   │
  └─────────────────┴──────────┴──────────┴─────────────────────┘

  OpenAI text-embedding-3-small 支持 dimensions 参数:
    - dimensions=1536: 全维度, 最高精度
    - dimensions=512:  保留约 98% 的效果, 存储减少 67%

  关键发现 (来自 OpenAI 官方):
    缩短到 256 维时, MIRACL 基准仍保留 ~96% 的性能。
    这意味着大多数应用不需要全维度。

  取舍原则:
    1. 如果向量存储是主要成本 → 用更低维度
    2. 如果检索质量是核心指标 → 用全维度
    3. 如果做聚类/分类 → 512 维通常够用
    4. 如果做语义搜索 → 768 维是比较好的平衡点

  类比 Java:
    向量维度 ≈ 图片分辨率
    1536 维 ≈ 4K 高清   — 细节丰富, 但占用大
    384 维  ≈ 720p      — 大多数场景够用, 加载快
    128 维  ≈ 缩略图    — 只能看个大概
""")

print("\n" + "=" * 60)
print("  试试看练习完成!")
print("=" * 60)
