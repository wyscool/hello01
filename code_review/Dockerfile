# ============================================================
# Code Review API — Docker 镜像
# ============================================================
# 构建:  docker build -t code-review:latest -f code_review/Dockerfile .
# 运行:  docker run -p 8001:8001 --env-file .env code-review:latest
# ============================================================

FROM python:3.12-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY code_review/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY code_review/ code_review/

# 非 root 用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "-m", "uvicorn", "code_review.app:app", "--host", "0.0.0.0", "--port", "8001"]
