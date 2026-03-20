# doc2embed

基于 FastAPI 的基础服务工程，当前已接入 PostgreSQL 表访问、Dify 数据集检索，以及多轮会话式检索总结能力。

## 安装依赖

```bash
pip install -e .
```

## 启动方式

```bash
uvicorn app.main:app --reload
```

或：

```bash
python main.py
```

## 数据库配置

数据库连接通过环境变量提供：

1. 复制 `.env.example` 为 `.env`
2. 设置 `DATABASE_URL`

示例：

```env
DATABASE_URL=postgresql://postgres:difyai123456@db:5432/dify
```

框架内部会自动转换为 `postgresql+psycopg://...` 供 SQLAlchemy 2.x 使用。

## 接口说明

- `GET /api/v1/health`
- `GET /api/v1/database/ping`
- `GET /api/v1/database/tables?schema=public`
- `GET /api/v1/database/tables/{table_name}/rows?schema=public&limit=20`
- `POST /api/v1/database/datasets/retrieve`

请求体示例：

```json
{
  "datasetId": "3f4d8d6d-50cd-423f-b73a-50daafc02fa7",
  "query": "什么是RAG？",
  "conversationId": "optional-existing-conversation-id"
}
```

说明：

- 首次对话可以不传 `conversationId`，接口会自动生成
- 后续多轮对话传回同一个 `conversationId`，接口会复用历史会话记忆和最近几轮 `finalSummary`
