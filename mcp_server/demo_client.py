# ============================================================
# mcp_server/demo_client.py — MCP 客户端演示
# ============================================================
# 模拟 Claude Desktop 连接 MCP Server 的完整流程:
#   连接 → initialize 握手 → 列出工具 → 调用工具 → 展示结果
#
# 用法: python mcp_server/demo_client.py
#
# 类比 Java: 相当于用 RestTemplate 调用 REST API 的集成测试,
# 但这里走的是 JSON-RPC over stdio。
# ============================================================

import json
import sys
import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).parent.parent


async def demo():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server"],
    )

    print("=" * 60)
    print("  启动 MCP Server 并建立 stdio 连接...")
    print("=" * 60)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # Step 1: 握手
            init = await session.initialize()
            print(f"\n  ✓ 握手完成")
            print(f"    Server: {init.serverInfo.name} v{init.serverInfo.version}")
            print(f"    Protocol: {init.protocolVersion}")

            # Step 2: 列出工具
            tools_resp = await session.list_tools()
            print(f"\n  ✓ 已注册 {len(tools_resp.tools)} 个工具:")
            for t in tools_resp.tools:
                print(f"    • {t.name}")
                # 截断描述
                desc = t.description[:100].replace("\n", " ")
                print(f"      {desc}...")

            # Step 3: codebase_status
            print(f"\n{'─' * 60}")
            print("  [1] 调用 codebase_status...")
            status = await session.call_tool("codebase_status", {})
            status_data = json.loads(status.content[0].text)
            idx = status_data["index"]
            print(f"  索引文件: {idx['files_indexed']}")
            print(f"  代码块:   {idx['total_chunks']}")
            print(f"  模型:     {idx['embedding_model']}")
            cfg = status_data["config"]
            print(f"  LLM:      {cfg['llm_model']}")
            print(f"  MMR:      {'on' if cfg['mmr_enabled'] else 'off'}")

            # Step 4: codebase_search
            queries = [
                "异步重试装饰器 retry 在哪个文件？",
                "哪些文件用到了 ChromaDB？",
                "LlmClient 是怎么实现的？",
            ]
            for i, q in enumerate(queries, 2):
                print(f"\n{'─' * 60}")
                print(f"  [{i}] 提问: {q}")
                result = await session.call_tool(
                    "codebase_search",
                    {"query": q, "top_k": 3},
                )
                data = json.loads(result.content[0].text)
                print(f"  延迟: {data.get('latency_ms', '?')}ms"
                      f"{' (缓存命中!)' if data.get('cached') else ''}")
                print(f"  答案: {data['answer'][:200]}...")
                if data.get("sources"):
                    print(f"  来源:")
                    for s in data["sources"][:3]:
                        print(f"    • {s['file_path']}:{s['start_line']}-{s['end_line']}"
                              f"  ({s['type']} {s['name']}, score={s['score']})")

            # Step 5: 验证缓存
            print(f"\n{'─' * 60}")
            print("  [5] 重复提问: 异步重试装饰器 retry 在哪个文件？")
            result = await session.call_tool(
                "codebase_search",
                {"query": "异步重试装饰器 retry 在哪个文件？"},
            )
            data = json.loads(result.content[0].text)
            print(f"  延迟: {data.get('latency_ms', '?')}ms"
                  f"{' ← 缓存命中!' if data.get('cached') else ''}")

    print(f"\n{'=' * 60}")
    print("  演示完成 — MCP Server 连接正常，所有工具可用")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
