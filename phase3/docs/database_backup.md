# 数据库备份指南

## MySQL 备份

MySQL 提供了多种备份方式：

1. **mysqldump** —— 逻辑备份
   ```bash
   mysqldump -u root -p database_name > backup.sql
   ```
   优点：可读 SQL 文件，支持选择性备份
   缺点：备份和恢复速度较慢

2. **XtraBackup** —— 物理热备份
   适合大型数据库，备份期间不影响业务

## PostgreSQL 备份

PostgreSQL 使用 pg_dump 工具：

```bash
pg_dump -U postgres database_name > backup.sql
```

pg_dump 支持：
- 并行备份（-j 参数）
- 仅备份 schema（-s）
- 仅备份数据（-a）
- 自定义格式（-Fc），支持压缩

## 备份策略

推荐的备份策略：
- 每天全量备份
- 每小时增量备份
- 保留最近 30 天的备份
- 定期演练恢复流程
