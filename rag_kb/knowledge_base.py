# ============================================================
# rag_kb/knowledge_base.py — KnowledgeBase CRUD
# ============================================================
# ChromaDB 封装: add / list / get / remove / search / stats
#
# 类比 Java: KnowledgeBase ≈ @Repository
# ============================================================

import time
import shutil
from pathlib import Path
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from rag_kb.pipeline import (
    EmbeddingFunction, DocumentProcessor, Chunk,
)


@dataclass
class DocInfo:
    source: str
    chunks: int
    added_at: str


class KnowledgeBase:
    """知识库管理层 — 封装 ChromaDB CRUD。

    所有文档增删改查通过此类, 不直接操作 ChromaDB collection。
    """

    def __init__(self, collection_name: str = "rag_kb_main",
                 persist_dir: str = "./rag_kb/data/chroma_db",
                 embedding_function: EmbeddingFunction | None = None,
                 processor: DocumentProcessor | None = None,
                 chunk_size: int = 500, overlap: int = 50):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._embed_fn = embedding_function or EmbeddingFunction()

        # 确保 ChromaDB 持久化目录存在
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
        )

        self._processor = processor or DocumentProcessor(
            chunk_size=chunk_size, overlap=overlap
        )

    # ============================================================
    # Ingest
    # ============================================================

    def add_file(self, file_path: Path) -> int:
        """导入单个文件, 自动处理重复导入 (先删旧再导入)。"""
        source = file_path.name
        self.remove_doc(source)

        chunks = self._processor.process_file(file_path)
        if not chunks:
            return 0

        ids = [f"{source}_{c.chunk_index}" for c in chunks]
        docs = [c.text for c in chunks]
        metadatas = [{
            "source": source,
            "chunk_index": c.chunk_index,
            "total_chunks": c.total_chunks,
            "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        } for c in chunks]

        self._collection.add(ids=ids, documents=docs, metadatas=metadatas)
        return len(chunks)

    def add_directory(self, dir_path: Path,
                      patterns: tuple[str, ...] = (
                          "*.txt", "*.md", "*.json", "*.py", "*.java",
                      )) -> int:
        """批量导入目录, 返回总 chunk 数。"""
        total = 0
        for pattern in patterns:
            for fp in dir_path.glob(pattern):
                if fp.is_file():
                    total += self.add_file(fp)
        return total

    # ============================================================
    # Query
    # ============================================================

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """语义搜索, 返回 {id, text, score, source, ...} 列表。"""
        result = self._collection.query(
            query_texts=[query], n_results=top_k,
        )
        results: list[dict] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        for i in range(len(ids)):
            dist = distances[i] if i < len(distances) else 1.0
            results.append({
                "id": ids[i],
                "text": docs[i] if i < len(docs) else "",
                "score": round(1.0 / (1.0 + dist), 4),
                "source": (metadatas[i] or {}).get("source", "") if i < len(metadatas) else "",
                "chunk": (metadatas[i] or {}).get("chunk_index", 0) if i < len(metadatas) else 0,
            })

        return results

    # ============================================================
    # List / Get
    # ============================================================

    def list_docs(self) -> list[dict]:
        """文档列表 (按 source 去重)。"""
        all_data = self._collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])
        sources: dict[str, dict] = {}
        for m in metadatas:
            src = m.get("source", "unknown")
            if src not in sources:
                sources[src] = {
                    "source": src,
                    "chunks": 0,
                    "added_at": m.get("added_at", ""),
                }
            sources[src]["chunks"] += 1
        return list(sources.values())

    def get_doc(self, source: str) -> dict | None:
        """获取某文档的全部 chunk。"""
        all_data = self._collection.get(
            include=["documents", "metadatas"],
        )
        metadatas = all_data.get("metadatas", [])
        docs = all_data.get("documents", [])
        ids = all_data.get("ids", [])

        chunks_info: list[dict] = []
        for i, m in enumerate(metadatas):
            if m.get("source") == source:
                chunks_info.append({
                    "chunk_index": m.get("chunk_index", i),
                    "text": docs[i] if i < len(docs) else "",
                })

        if not chunks_info:
            return None

        chunks_info.sort(key=lambda x: x["chunk_index"])
        return {
            "source": source,
            "total_chunks": len(chunks_info),
            "chunks": chunks_info,
        }

    # ============================================================
    # Remove
    # ============================================================

    def remove_doc(self, source: str) -> int:
        """删除某文档的全部 chunk, 返回删除数量。"""
        all_data = self._collection.get(include=["metadatas"])
        ids_to_delete: list[str] = []
        for i, m in enumerate(all_data.get("metadatas", [])):
            if m.get("source") == source:
                ids_to_delete.append(all_data["ids"][i])

        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    # ============================================================
    # Stats / Health
    # ============================================================

    def stats(self) -> dict:
        """知识库统计信息。"""
        docs = self.list_docs()
        return {
            "collection_name": self.collection_name,
            "total_docs": len(docs),
            "total_chunks": self.count,
            "sources": [d["source"] for d in docs],
            "persist_dir": self.persist_dir,
        }

    @property
    def count(self) -> int:
        return self._collection.count()

    @property
    def is_connected(self) -> bool:
        """ChromaDB 心跳检测。"""
        try:
            self._collection.count()
            return True
        except Exception:
            return False

    @property
    def collection(self):
        """暴露给 Retriever 使用。"""
        return self._collection


# ============================================================
# 工厂函数
# ============================================================

def create_knowledge_base(config=None) -> KnowledgeBase:
    """从 AppConfig 创建完整配置的 KnowledgeBase。"""
    if config is None:
        from rag_kb.config import AppConfig
        config = AppConfig.from_env()

    embed_fn = EmbeddingFunction(model_name=config.embedding_model)
    processor = DocumentProcessor(
        chunk_size=config.chunk_size, overlap=config.chunk_overlap
    )

    return KnowledgeBase(
        collection_name=config.chroma_collection_name,
        persist_dir=config.chroma_persist_dir,
        embedding_function=embed_fn,
        processor=processor,
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )
