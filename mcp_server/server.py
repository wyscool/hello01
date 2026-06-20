# ============================================================
# mcp_server/server.py — FastMCP 组装 + 工具注册
# ============================================================
# 使用官方 mcp SDK 的 FastMCP 构建 MCP Server:
#   1. 创建 FastMCP 实例 (server name + lifespan)
#   2. 用 @mcp.tool() 装饰器注册三个工具
#   3. 在 __main__.py 中通过 mcp.run(transport="stdio") 启动
#
# 类比 Java:
#   这个文件相当于 Spring Boot 的 @Configuration + @Bean 注册,
#   __main__.py 相当于 SpringApplication.run()。
# ============================================================

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

from mcp_server.lifecycle import create_lifespan, AppContext
from mcp_server.tools import (
    handle_codebase_search,
    handle_codebase_index,
    handle_codebase_status,
)


def create_server() -> FastMCP:
    """创建并配置好所有工具的 FastMCP 实例。

    这是依赖注入的组装入口:
      server = create_server()
      server.run(transport="stdio")
    """
    mcp = FastMCP(
        name="codebase-qa",
        lifespan=create_lifespan,
    )

    # ---- 工具注册 ----

    @mcp.tool(
        name="codebase_search",
        description=(
            "搜索代码库，用自然语言查找函数、类、方法的定义和实现。"
            "返回带文件路径和精确行号的答案。"
            "适用场景: 'retry 装饰器在哪里'、'怎么实现 JWT 验证'、"
            "'哪些文件用到了 ChromaDB'。"
            "参数 query: 自然语言查询; top_k: 返回结果数(默认5); "
            "filter_type: 按代码类型过滤(function/class/method/module_level)"
        ),
    )
    async def codebase_search(
        ctx: Context[ServerSession, AppContext],
        query: str,
        top_k: int = 5,
        filter_type: str = "",
    ) -> str:
        return await handle_codebase_search(ctx, query, top_k, filter_type)

    @mcp.tool(
        name="codebase_index",
        description=(
            "索引代码目录，将 Python 源码解析为结构化代码块存入向量数据库。"
            "每次索引前自动检测文件变更(SHA-256)，只处理修改过的文件。"
            "适用场景: 首次使用或添加新项目目录时调用。"
            "参数 dirs: 要索引的目录路径列表"
        ),
    )
    async def codebase_index(
        ctx: Context[ServerSession, AppContext],
        dirs: list[str],
    ) -> str:
        return await handle_codebase_index(ctx, dirs)

    @mcp.tool(
        name="codebase_status",
        description=(
            "查看代码库索引状态: 已索引文件数、代码块总数、"
            "向量数据库 collection 信息、当前配置。"
            "适用场景: 确认索引是否就绪、检查配置。"
        ),
    )
    async def codebase_status(
        ctx: Context[ServerSession, AppContext],
    ) -> str:
        return await handle_codebase_status(ctx)

    return mcp


# 模块级实例 — 供 __main__.py 引用
mcp = create_server()
