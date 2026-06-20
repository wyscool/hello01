FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY codebase_qa/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预下载 bge-m3 模型（~2.2GB），避免首次启动时下载超时
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY codebase_qa/ codebase_qa/
COPY deploy/ deploy/

ENV HOST=0.0.0.0
ENV PORT=8003
ENV EMBEDDING_MODEL=BAAI/bge-m3

EXPOSE 8003

CMD ["uvicorn", "codebase_qa.app:app", "--host", "0.0.0.0", "--port", "8003"]
