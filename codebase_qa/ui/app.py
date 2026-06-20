# ============================================================
# codebase_qa/ui/app.py — Codebase Q&A Web UI
# ============================================================
# 基于 Streamlit 的代码库问答界面。
# 封装 codebase_qa.QAPipeline，提供搜索框 + 结果展示 + 侧边栏配置。
#
# 用法: streamlit run codebase_qa/ui/app.py
#
# 类比 Java: 相当于给 Spring Boot 服务加一个 Thymeleaf/Vaadin 前端，
# 但 Streamlit 是纯 Python，不需要写 HTML/CSS/JS。
# ============================================================

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from codebase_qa.pipeline import EmbeddingFunction, QAPipeline
from codebase_qa.indexer import CodeIndexer
from codebase_qa.retriever import Retriever, Reranker
from codebase_qa.generator import AnswerGenerator
from codebase_qa.config import AppConfig
from deploy.agent_core import LlmClient

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="Codebase Q&A",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Codebase Q&A")
st.caption("用自然语言搜索你的 Python 代码库")

# ============================================================
# 初始化 — 缓存资源 (只加载一次)
# ============================================================

@st.cache_resource(show_spinner="正在加载 embedding 模型...")
def load_embedding(model_name: str) -> EmbeddingFunction:
    fn = EmbeddingFunction(model_name=model_name)
    fn.embed_query(["ping"])
    return fn


@st.cache_resource(show_spinner="正在连接 ChromaDB...")
def load_chroma(persist_dir: str, collection_name: str, _embed_fn: EmbeddingFunction):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=_embed_fn,
    )


@st.cache_resource(show_spinner="正在初始化 LLM 客户端...")
def load_pipeline(_cfg: AppConfig, _embed_fn: EmbeddingFunction, _collection) -> QAPipeline:
    llm = LlmClient(
        api_key=_cfg.llm_api_key,
        base_url=_cfg.llm_base_url,
        model=_cfg.llm_model,
        max_retries=_cfg.llm_max_retries,
        timeout=_cfg.llm_timeout_seconds,
    )
    retriever = Retriever(_collection)
    reranker = Reranker(_embed_fn)
    generator = AnswerGenerator(llm)
    return QAPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        top_k=_cfg.top_k,
        min_score=_cfg.min_score,
        use_mmr=_cfg.use_mmr,
        mmr_lambda=_cfg.mmr_lambda,
    )


# ============================================================
# 侧边栏 — 配置和状态
# ============================================================

with st.sidebar:
    st.header("配置")

    cfg = AppConfig.from_env()

    # 搜索参数
    top_k = st.slider("返回结果数 (top_k)", 1, 20, cfg.top_k)
    min_score = st.slider("最低相关度 (min_score)", 0.0, 1.0, cfg.min_score, 0.05)
    use_mmr = st.checkbox("MMR 多样性重排", cfg.use_mmr,
                          help="平衡相关性和结果多样性，避免返回重复内容")
    mmr_lambda = st.slider("MMR λ (越大越偏向相关性)", 0.0, 1.0, cfg.mmr_lambda, 0.05,
                           disabled=not use_mmr)

    filter_type = st.selectbox(
        "代码类型过滤",
        ["全部", "function", "method", "class", "module_level"],
        index=0,
    )

    st.divider()

    # 状态信息
    st.header("状态")
    try:
        col = load_chroma(cfg.chroma_persist_dir, cfg.collection_name,
                          load_embedding(cfg.embedding_model))
        st.metric("已索引代码块", col.count())
    except Exception:
        st.warning("ChromaDB 未就绪")

    st.caption(f"Embedding: {cfg.embedding_model}")
    st.caption(f"LLM: {cfg.llm_model}")

# ============================================================
# 加载资源
# ============================================================

try:
    embed_fn = load_embedding(cfg.embedding_model)
    collection = load_chroma(cfg.chroma_persist_dir, cfg.collection_name, embed_fn)
    pipeline = load_pipeline(cfg, embed_fn, collection)
    pipeline.top_k = top_k
    pipeline.min_score = min_score
    pipeline.use_mmr = use_mmr
    pipeline.mmr_lambda = mmr_lambda
    service_ready = True
except Exception as e:
    st.error(f"服务初始化失败: {e}")
    service_ready = False

# ============================================================
# 搜索框
# ============================================================

col1, col2 = st.columns([6, 1])
with col1:
    query = st.text_input(
        "输入你的问题",
        placeholder="例如：异步重试装饰器在哪个文件？LlmClient 怎么实现的？哪些文件用到了 ChromaDB？",
        label_visibility="collapsed",
    )
with col2:
    search_clicked = st.button("搜索", use_container_width=True, type="primary",
                               disabled=not service_ready)

# 示例问题
if not query:
    examples = [
        "异步重试装饰器 retry 在哪里？",
        "LlmClient 是怎么实现的？",
        "哪些文件用到了 ChromaDB？",
        "QAPipeline.ask() 的完整流程是什么？",
    ]
    st.caption("试试这些问题：")
    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with cols[i]:
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                query = example
                search_clicked = True

# ============================================================
# 执行搜索
# ============================================================

if search_clicked and query.strip() and service_ready:
    effective_filter = None if filter_type == "全部" else filter_type

    with st.spinner("正在搜索..."):
        response = pipeline.ask(query.strip(), filter_type=effective_filter)

    # ---- 结果展示 ----
    st.divider()

    # 答案
    st.subheader("答案")
    st.markdown(response.answer)

    # 元信息
    meta_cols = st.columns(4)
    meta_cols[0].metric("延迟", f"{response.latency_ms:.0f}ms")
    meta_cols[1].metric("来源数", len(response.sources))
    if response.sources:
        meta_cols[2].metric("最高相关度", f"{max(s.score for s in response.sources):.2%}")
    meta_cols[3].metric("filter", filter_type)

    # 来源详情
    if response.sources:
        st.divider()
        st.subheader("来源")

        for i, source in enumerate(response.sources):
            meta = source.metadata
            file_path = meta.get("file_path", "?")
            name = meta.get("name", "?")
            src_type = meta.get("type", "?")
            start_line = meta.get("start_line", "?")
            end_line = meta.get("end_line", "?")

            with st.expander(
                f"[{i + 1}] {file_path}:{start_line}-{end_line}  "
                f"**{name}** ({src_type})  —  {source.score:.2%}",
                expanded=(i == 0),
            ):
                # 代码块
                code_text = source.text
                # 截断过长代码
                if len(code_text) > 2000:
                    code_text = code_text[:2000] + "\n# ... (代码过长，已截断)"
                st.code(code_text, language="python",
                        line_numbers=True if start_line != "?" else False)

    # 无结果
    elif "未找到" in response.answer:
        st.info("代码库中未找到相关信息，换个关键词试试？")

# ============================================================
# 底部
# ============================================================

st.divider()
st.caption(
    f"Codebase Q&A Web UI  |  "
    f"Embedding: {cfg.embedding_model} ({embed_fn.dimension}d)  |  "
    f"ChromaDB: {cfg.collection_name}  |  "
    f"LLM: {cfg.llm_model}"
)
