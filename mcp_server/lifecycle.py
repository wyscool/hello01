# ============================================================
# mcp_server/lifecycle.py — 生命周期管理 + AppContext
# ============================================================
# 对应 codebase_qa/app.py 的 lifespan() async context manager。
# 负责:
#   1. 启动时加载 Embedding 模型（最耗时，最先加载）
#   2. 初始化 ChromaDB 持久化连接
#   3. 组装 QAPipeline (Retriever → Reranker → Generator)
#   4. 初始化查询缓存和组件健康检查
#   5. 关闭时清理资源
#
# 类比 Java:
#   AppContext  = Spring ApplicationContext (Bean 容器)
#   lifespan    = @PostConstruct / @PreDestroy
# ============================================================

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppContext:
    """MCP Server 全局依赖容器。

    lifespan 初始化的所有组件放在这里，
    tool 函数通过 ctx.request_context.lifespan_context 访问。

    类比 Java: 相当于 Spring 的 ApplicationContext，
    getBean(QAPipeline.class) 变成 ctx.qa_pipeline。
    """
    qa_pipeline: object | None = None       # QAPipeline
    code_indexer: object | None = None      # CodeIndexer
    chroma_collection: object | None = None  # ChromaDB Collection
    embed_fn: object | None = None           # EmbeddingFunction
    llm_client: object | None = None         # LlmClient
    query_cache: object | None = None        # ExactCache
    config: object | None = None             # McpServerConfig


@asynccontextmanager
async def create_lifespan(server) -> AsyncIterator[AppContext]:
    """MCP Server 完整生命周期。

    startup: 加载模型 → 连接数据库 → 组装 pipeline → 预热缓存
    shutdown: 清理资源 → 打印统计
    """
    from mcp_server.config import McpServerConfig
    from codebase_qa.pipeline import EmbeddingFunction
    from codebase_qa.indexer import CodeIndexer
    from codebase_qa.retriever import Retriever, Reranker
    from codebase_qa.generator import AnswerGenerator
    from codebase_qa.pipeline import QAPipeline
    from deploy.agent_core import LlmClient
    from deploy.cost_control import ExactCache

    config = McpServerConfig.from_env()
    ctx = AppContext(config=config)
    qa_cfg = config.qa

    print(file=sys.stderr)
    print("  Codebase QA MCP Server", file=sys.stderr)
    print(f"  LLM: {qa_cfg.llm_model}", file=sys.stderr)
    print(f"  Embedding: {qa_cfg.embedding_model}", file=sys.stderr)
    print(f"  Transport: {config.transport}", file=sys.stderr)

    # 1. Embedding 模型（最耗时，先加载，触发下载 + 预热）
    print("  Loading embedding model...", file=sys.stderr)
    embed_fn = EmbeddingFunction(model_name=qa_cfg.embedding_model)
    embed_fn.embed_query(["ping"])
    ctx.embed_fn = embed_fn
    print(f"  Embedding: dim={embed_fn.dimension} ✓", file=sys.stderr)

    # 2. ChromaDB
    import chromadb
    from chromadb.config import Settings
    persist_dir = Path(qa_cfg.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(
        name=qa_cfg.collection_name,
        embedding_function=embed_fn,
    )
    ctx.chroma_collection = collection
    print(f"  ChromaDB: {qa_cfg.collection_name}, "
          f"chunks={collection.count()} ✓", file=sys.stderr)

    # 3. CodeIndexer
    code_indexer = CodeIndexer(exclude_dirs=qa_cfg.exclude_set)
    ctx.code_indexer = code_indexer
    print(f"  Indexer: exclude={qa_cfg.exclude_set} ✓", file=sys.stderr)

    # 4. LlmClient
    llm_client = LlmClient(
        api_key=qa_cfg.llm_api_key,
        base_url=qa_cfg.llm_base_url,
        model=qa_cfg.llm_model,
        max_retries=qa_cfg.llm_max_retries,
        timeout=qa_cfg.llm_timeout_seconds,
    )
    ctx.llm_client = llm_client
    print(f"  LLM: {'reachable' if llm_client.is_healthy else 'offline'} ✓",
          file=sys.stderr)

    # 5. QAPipeline 组装
    retriever = Retriever(collection)
    reranker = Reranker(embed_fn)
    generator = AnswerGenerator(llm_client)
    qa_pipeline = QAPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        top_k=qa_cfg.top_k,
        min_score=qa_cfg.min_score,
        use_mmr=qa_cfg.use_mmr,
        mmr_lambda=qa_cfg.mmr_lambda,
    )
    ctx.qa_pipeline = qa_pipeline
    print(f"  Pipeline: top_k={qa_cfg.top_k}, "
          f"mmr={'on' if qa_cfg.use_mmr else 'off'} ✓",
          file=sys.stderr)

    # 6. Query Cache
    if qa_cfg.cache_enabled:
        ctx.query_cache = ExactCache(
            max_size=qa_cfg.cache_max_size,
            ttl_seconds=qa_cfg.cache_ttl_seconds,
        )
        print(f"  Cache: max={qa_cfg.cache_max_size}, "
              f"ttl={qa_cfg.cache_ttl_seconds}s ✓",
              file=sys.stderr)

    print("  Ready ✓ (waiting for MCP requests via stdin...)",
          file=sys.stderr)
    print(file=sys.stderr)

    try:
        yield ctx
    finally:
        print(file=sys.stderr)
        print("  MCP Server shutting down.", file=sys.stderr)
        if ctx.query_cache:
            stats = ctx.query_cache.stats()
            print(f"  Cache hits: {stats.get('hits', 0)}", file=sys.stderr)
