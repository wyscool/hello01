# DevAssistant API — Docker 镜像
#
# 构建:
#   docker build -t dev-assistant:latest -f deploy/Dockerfile .
#
# 运行:
#   docker run -p 8000:8000 --env-file .env dev-assistant:latest

FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖 (sentence-transformers 需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY deploy/ deploy/
COPY .env .env

# 创建非 root 用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动
CMD ["python", "-m", "uvicorn", "deploy.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
