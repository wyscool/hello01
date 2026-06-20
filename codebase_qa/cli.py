# ============================================================
# codebase_qa/cli.py — CLI 入口 (Click)
# ============================================================
# 通过 pyproject.toml 的 [project.scripts] 注册为 cqa 命令:
#   cqa ask "问题"       — 搜索代码库
#   cqa index <dirs...>  — 索引代码目录
#   cqa status           — 查看索引状态
#   cqa serve            — 启动 FastAPI 服务
#   cqa ui               — 启动 Streamlit Web UI
#
# 类比 Java:
#   pyproject.toml [project.scripts] = Maven <mainClass>
#   click 装饰器 = Spring Shell / Picocli 注解
#   @click.group() = @Command(name="cqa", subcommands={...})
# ============================================================

import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# 加载 .env (项目根目录)
_project_root = Path(__file__).parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


# ============================================================
# 延迟加载 — 避免未执行命令时加载大模型
# ============================================================

def _get_config():
    """延迟加载 AppConfig（仅在需要时读取环境变量）。"""
    from codebase_qa.config import AppConfig
    return AppConfig.from_env()


def _get_pipeline():
    """延迟加载完整的 QAPipeline（仅在 ask/server/ui 命令时初始化）。

    这个函数类似 Spring 的 @Lazy + @Autowired——
    只在真正调用时才初始化 embedding 模型和 ChromaDB。
    """
    from codebase_qa.pipeline import EmbeddingFunction, QAPipeline
    from codebase_qa.retriever import Retriever, Reranker
    from codebase_qa.generator import AnswerGenerator
    from deploy.agent_core import LlmClient

    cfg = _get_config()

    click.echo(f"  Loading embedding model ({cfg.embedding_model})...", err=True)
    embed_fn = EmbeddingFunction(model_name=cfg.embedding_model)
    embed_fn.embed_query(["ping"])

    import chromadb
    from chromadb.config import Settings
    persist_dir = Path(cfg.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=cfg.collection_name,
        embedding_function=embed_fn,
    )

    llm = LlmClient(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
        max_retries=cfg.llm_max_retries,
        timeout=cfg.llm_timeout_seconds,
    )

    retriever = Retriever(collection)
    reranker = Reranker(embed_fn)
    generator = AnswerGenerator(llm)
    pipeline = QAPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        top_k=cfg.top_k,
        min_score=cfg.min_score,
        use_mmr=cfg.use_mmr,
        mmr_lambda=cfg.mmr_lambda,
    )
    return pipeline, collection, cfg, embed_fn


# ============================================================
# CLI 主入口
# ============================================================

@click.group()
@click.version_option(version="0.1.0", prog_name="cqa")
def main():
    """Codebase Q&A — 用自然语言搜索你的 Python 代码库。

    类比: 相当于给 IDE 装了 AI 搜索引擎，直接问"retry 装饰器在哪"就能定位。

    \b
    快速开始:
      cqa index ./myproject       # 1. 索引代码目录
      cqa ask "retry在哪里"       # 2. 搜索
      cqa status                  # 3. 查看状态
      cqa serve                   # 4. 启动 API 服务 (可选)
      cqa ui                      # 5. 启动 Web UI (可选)
    """
    pass


# ============================================================
# cqa ask — 搜索代码库
# ============================================================

@main.command()
@click.argument("question")
@click.option("--top-k", "-k", default=5, show_default=True,
              help="返回结果数")
@click.option("--filter", "-f", "filter_type", default=None,
              type=click.Choice(["function", "class", "method", "module_level"]),
              help="按代码类型过滤")
@click.option("--no-mmr", is_flag=True, help="禁用 MMR 多样性重排")
@click.option("--verbose", "-v", is_flag=True, help="显示完整源码片段")
def ask(question, top_k, filter_type, no_mmr, verbose):
    """用自然语言搜索代码库。

    \b
    示例:
      cqa ask "异步重试装饰器retry在哪里"
      cqa ask "LlmClient" --filter class
      cqa ask "ChromaDB" -k 10 --verbose
    """
    pipeline, _, _, _ = _get_pipeline()

    if no_mmr:
        pipeline.use_mmr = False

    click.echo(f"  搜索: {question}", err=True)
    if filter_type:
        click.echo(f"  过滤: {filter_type}", err=True)

    with click.progressbar(length=1, label="  检索中", show_eta=False) as bar:
        response = pipeline.ask(question, filter_type=filter_type)
        bar.update(1)

    click.echo()
    click.secho(response.answer, fg="white")

    if not response.sources:
        return

    click.echo()
    click.secho(f"  {len(response.sources)} 个来源  "
                f"({response.latency_ms:.0f}ms)", fg="green")

    for i, src in enumerate(response.sources):
        meta = src.metadata
        file_line = f"{meta.get('file_path', '?')}:{meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
        click.echo()
        click.secho(f"  [{i + 1}] {file_line}", fg="cyan", bold=True)
        click.echo(f"  {meta.get('name', '?')} ({meta.get('type', '?')})  "
                   f"score: {src.score:.2%}")

        if verbose:
            click.echo()
            # 语法高亮代码
            try:
                from pygments import highlight
                from pygments.lexers import PythonLexer
                from pygments.formatters import TerminalFormatter
                code = src.text[:800]
                click.echo(highlight(code, PythonLexer(), TerminalFormatter()))
            except ImportError:
                click.echo(src.text[:800])


# ============================================================
# cqa index — 索引代码目录
# ============================================================

@main.command()
@click.argument("dirs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--exclude", "-e", default="tests,venv,.git,__pycache__,node_modules,build,dist",
              show_default=True, help="排除目录 (逗号分隔)")
def index(dirs, exclude):
    """索引一个或多个 Python 代码目录。

    将 .py 文件的函数/类/方法按 AST 结构分块并存入 ChromaDB。已修改的文件自动增量更新。

    \b
    示例:
      cqa index ./phase1
      cqa index ./phase1 ./deploy ./codebase_qa
      cqa index . --exclude "tests,venv"
    """
    from codebase_qa.pipeline import EmbeddingFunction
    from codebase_qa.indexer import CodeIndexer

    cfg = _get_config()
    exclude_set = {d.strip() for d in exclude.split(",") if d.strip()}

    click.echo(f"  Embedding: {cfg.embedding_model}", err=True)

    embed_fn = EmbeddingFunction(model_name=cfg.embedding_model)
    embed_fn.embed_query(["ping"])

    import chromadb
    from chromadb.config import Settings
    persist_dir = Path(cfg.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=cfg.collection_name,
        embedding_function=embed_fn,
    )

    indexer = CodeIndexer(exclude_dirs=exclude_set)
    total_chunks = 0

    for d in dirs:
        dir_path = Path(d).resolve()
        click.echo(f"  索引: {dir_path}", err=True)

        chunks = indexer.index_directory(dir_path)

        if not chunks:
            click.echo(f"    (无新文件)", err=True)
            continue

        ids = [c.chunk_id for c in chunks]
        docs = [c.embed_text for c in chunks]
        metas = [indexer.chunk_to_metadata(c) for c in chunks]

        # 删除旧数据 (upsert)
        for c in chunks:
            try:
                existing = collection.get(where={"file_path": c.file_path})
                if existing and existing.get("ids"):
                    collection.delete(ids=existing["ids"])
            except Exception:
                pass

        collection.add(ids=ids, documents=docs, metadatas=metas)
        total_chunks += len(chunks)
        click.echo(f"    {len(chunks)} 个新代码块", err=True)

    click.echo()
    click.secho(f"  索引完成: {total_chunks} 个代码块, "
                f"collection 总量 = {collection.count()}", fg="green")


# ============================================================
# cqa status — 查看状态
# ============================================================

@main.command()
def status():
    """查看索引统计和当前配置。

    \b
    示例:
      cqa status
    """
    cfg = _get_config()

    import chromadb
    from chromadb.config import Settings
    try:
        client = chromadb.PersistentClient(
            path=cfg.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(
            name=cfg.collection_name,
            embedding_function=None,
        )
        chunk_count = collection.count()
    except Exception:
        chunk_count = "(ChromaDB 未就绪)"

    click.secho("Codebase Q&A 状态", bold=True)
    click.echo()

    click.secho("[索引]", bold=True, fg="cyan")
    click.echo(f"  ChromaDB 目录: {cfg.chroma_persist_dir}")
    click.echo(f"  Collection:    {cfg.collection_name}")
    click.echo(f"  代码块总数:    {chunk_count}")

    click.echo()
    click.secho("[模型]", bold=True, fg="cyan")
    click.echo(f"  Embedding: {cfg.embedding_model}")
    click.echo(f"  LLM:       {cfg.llm_model}")

    click.echo()
    click.secho("[检索参数]", bold=True, fg="cyan")
    click.echo(f"  top_k:       {cfg.top_k}")
    click.echo(f"  min_score:   {cfg.min_score}")
    click.echo(f"  MMR:         {'on' if cfg.use_mmr else 'off'} "
               f"(λ={cfg.mmr_lambda})")

    click.echo()
    click.secho("[缓存]", bold=True, fg="cyan")
    click.echo(f"  enabled:     {cfg.cache_enabled}")
    click.echo(f"  TTL:         {cfg.cache_ttl_seconds}s")
    click.echo(f"  max_size:    {cfg.cache_max_size}")


# ============================================================
# cqa serve — FastAPI 服务
# ============================================================

@main.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8003, show_default=True)
@click.option("--reload", is_flag=True, help="开发模式 (自动重载)")
def serve(host, port, reload):
    """启动 FastAPI 服务 (4 个端点 + Streamlit UI 链接)。

    等同: uvicorn codebase_qa.app:app --host 0.0.0.0 --port 8003

    \b
    端点:
      GET  /health  — 健康检查
      GET  /status  — 详细状态
      POST /index   — 索引目录
      POST /query   — 搜索代码
    """
    click.echo(f"  启动 FastAPI 服务: http://{host}:{port}", err=True)
    click.echo(f"  文档: http://{host}:{port}/docs", err=True)
    click.echo()

    import uvicorn
    uvicorn.run(
        "codebase_qa.app:app",
        host=host,
        port=port,
        reload=reload,
    )


# ============================================================
# cqa ui — Streamlit Web UI
# ============================================================

@main.command()
@click.option("--port", default=8501, show_default=True)
def ui(port):
    """启动 Streamlit Web UI。

    在浏览器中提供搜索界面，含配置侧边栏和源码展示。
    等同: streamlit run codebase_qa/ui/app.py

    \b
    打开 http://localhost:8501 即可使用。
    """
    import subprocess
    ui_path = Path(__file__).parent / "ui" / "app.py"
    click.echo(f"  启动 Streamlit UI: http://localhost:{port}", err=True)
    click.echo()
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(ui_path),
        "--server.port", str(port),
    ])
