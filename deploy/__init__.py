# deploy/ — DevAssistant 生产部署
#
# 将 Phase 4 的 DevAssistant Agent 包装为 FastAPI 服务，
# 集成 Phase 5 的可观测性、成本控制和基础设施组件。
#
# 启动:
#   uvicorn deploy.app:app --host 0.0.0.0 --port 8000
#
# 验证:
#   curl http://localhost:8000/health
#   curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
#        -d '{"task": "计算 sqrt(256)", "mode": "quick"}'
