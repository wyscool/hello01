# ============================================================
# Log Analysis Agent API — Docker 镜像
# ============================================================
# 构建:  docker build -t log-analyzer:latest -f log_analyzer/Dockerfile .
# 运行:  docker run -p 8003:8003 --env-file .env log-analyzer:latest
# 挂载:  docker run -p 8003:8003 --env-file .env -v /var/log:/logs:ro log-analyzer:latest
# ============================================================

FROM python:3.12-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY log_analyzer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY log_analyzer/ log_analyzer/
COPY deploy/ deploy/

# 日志分析工作目录
RUN mkdir -p /app/logs

# 非 root 用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

CMD ["python", "-m", "uvicorn", "log_analyzer.app:app", "--host", "0.0.0.0", "--port", "8003"]
