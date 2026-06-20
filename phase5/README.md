# phase5/ — AI 工程化 (4 课)

Phase 5 是最终阶段，将前面的 Agent/RAG 项目提升为生产级服务: 评估系统、可观测性、成本控制和容器化部署。

## 学习目标

完成 Phase 5 后应能:
- 构建 RAG 质量评估框架 (准确率、召回率、F1)
- 实现结构化日志和分布式追踪
- 设计 Token 预算和成本控制系统
- 用 FastAPI + Docker 部署 AI 服务 (中间件、健康检查、优雅关闭)

## 课程列表

| # | 文件 | 主题 | 核心内容 |
|---|------|------|---------|
| 41 | `41_evaluation.py` | 评估框架 | 测试集生成、检索命中率 (hit@k)、答案准确性评估、对比实验设计 |
| 42 | `42_observability.py` | 可观测性 | `JsonLogger` (结构化 JSON 日志)、`Trace`/`Span` (分布式追踪)、`TokenMonitor` (消耗追踪) |
| 43 | `43_cost_control.py` | 成本控制 | `ExactCache` (LRU+TTL)、`SemanticCache` (embedding 相似度)、`ModelRouter` (任务 → cheap/normal/premium 模型)、`SmartClient` (两层缓存 + LLM 编排) |
| 44 | `44_production.py` | 生产部署 | FastAPI lifespan、4 个中间件链 (RequestId/Logging/ShutdownGate/RateLimit)、`HealthChecker`、Docker 构建 |

## 课程 → 生产代码映射

Phase 5 的每个模块直接对应 `deploy/` 包中的生产实现:

| 课程 | deploy/ 文件 | 说明 |
|------|-------------|------|
| 42 | `deploy/observability.py` | JsonLogger、Trace/Span、TokenMonitor 的开箱即用版 |
| 43 | `deploy/cost_control.py` | ExactCache、SemanticCache、ModelRouter、SmartClient |
| 44 | `deploy/app.py` + `deploy/infrastructure.py` | FastAPI 组装、4 个中间件、HealthChecker、GracefulShutdown |

## 运行方式

```bash
python phase5/44_production.py    # FastAPI 部署示例
```

## 生产就绪检查清单

完成 Phase 5 后的项目应具备:
- [ ] 结构化日志 (每行 JSON，含 timestamp/level/request_id)
- [ ] 请求级追踪 (Trace/Span，可串联调用链)
- [ ] Token 消耗监控 (按模型、按任务追踪)
- [ ] 查询缓存 (精确匹配 + 语义相似)
- [ ] 模型路由 (简单任务用 cheap 模型降低成本)
- [ ] 速率限制 (滑动窗口，防滥用)
- [ ] 健康检查 (liveness + readiness，K8s 兼容)
- [ ] 优雅关闭 (不丢请求，宽限期等待)
- [ ] Docker 镜像 (多阶段构建，非 root 用户)

## 前置要求

- 完成 Phase 1-4
- 理解 FastAPI、Docker 基本概念
- 理解什么是生产级服务 (Java 对比: Spring Boot Actuator、Micrometer、Resilience4j)
