# ============================================================
# RAG Knowledge Base API — Docker 镜像
# ============================================================
# 构建:  docker build -t rag-kb:latest -f rag_kb/Dockerfile .
# 运行:  docker run -p 8002:8002 --env-file .env rag-kb:latest
# ============================================================

FROM python:3.12-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY rag_kb/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY rag_kb/ rag_kb/
COPY deploy/ deploy/

# ChromaDB 数据目录
RUN mkdir -p /app/rag_kb/data/chroma_db

# 非 root 用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8002

# start-period 30s: embedding 模型 (~80MB) 加载需要时间
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["python", "-m", "uvicorn", "rag_kb.app:app", "--host", "0.0.0.0", "--port", "8002"]
