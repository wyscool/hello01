# rag_kb/ — RAG Knowledge Base Q&A Service
#
# 启动:
#   uvicorn rag_kb.app:app --port 8002
#
# 用法:
#   1. 导入文档:
#      curl -X POST http://localhost:8002/ingest \
#        -H "Content-Type: application/json" \
#        -d '{"path": "./docs/", "type": "directory"}'
#
#   2. 提问:
#      curl -X POST http://localhost:8002/query \
#        -H "Content-Type: application/json" \
#        -d '{"question": "MySQL 怎么备份?"}'
#
#   3. 管理:
#      curl http://localhost:8002/docs
#      curl -X DELETE http://localhost:8002/docs/redis_guide.md
