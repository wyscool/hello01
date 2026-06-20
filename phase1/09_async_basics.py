# ============================================================
# Phase 1, Lesson 09: async/await 异步基础
# ============================================================
#
# 本课目标:
#   1. 理解"并发"和"并行"的区别
#   2. 理解协程 (coroutine) 的概念 —— Python 的"虚拟线程"
#   3. async def / await 基本语法
#   4. asyncio.run() —— 程序的异步入口
#   5. asyncio.sleep() vs time.sleep() —— 关键区别!
#   6. asyncio.gather() —— 并发执行多个协程
#   7. asyncio.create_task() —— 创建后台任务
#   8. 异步上下文管理器: async with
#   9. 同步 vs 异步: 实际性能对比
#   10. 常见陷阱: 不要混用同步和异步
#
# 预计阅读 + 实操时间: 40-50 分钟
#
# ⚠️ 前置提醒:
#   异步编程是 Python 中比较难的概念。如果你是 Java 背景,
#   下面这个类比可能帮助你建立心智模型:
#     - 协程 (coroutine) ≈ 轻量级线程 / Java 21 虚拟线程
#     - 事件循环 (event loop) ≈ 单线程执行器, 调度所有协程
#     - await ≈ "在这里暂停, 让出控制权, 等结果好了再回来"
#   本课只讲基础概念和使用方式, 高级模式在后续课程中深入。
# ============================================================

import time
import asyncio


# ------------------------------------------------------------
# 一、同步 vs 异步 —— 咖啡店的类比
# ------------------------------------------------------------
# 想象你在咖啡店点了 3 杯咖啡:
#
# 同步方式 (你):
#   排队 → 点第一杯 → 等着做好 → 拿咖啡 → 点第二杯 → 等着做好 → ...
#   你一直在等, 什么都不做。这就是"阻塞"。
#
# 异步方式 (你):
#   点完 3 杯 → 拿到一个取餐号 → 坐下看书 → 叫号了 → 去拿第一杯
#   → 继续看书 → 叫号了 → 去拿第二杯 → ...
#   你在等的时候可以做其他事。这就是"非阻塞"。
#
#   你 = 主线程
#   咖啡师 = 事件循环 (event loop)
#   取餐号 = Future / Task (代表一个将在未来完成的操作)
#   叫号 = await (操作完成, 回来继续执行)
#
# 关键区别:
#   同步: 一个任务做完才能做下一个 (顺序执行)
#   异步: 一个任务等待时, 切换到另一个任务 (协作式切换)


# ------------------------------------------------------------
# 二、同步版本 —— 先看看"慢"长什么样
# ------------------------------------------------------------

def fetch_data_sync(source: str, delay: float) -> str:
    """模拟从某个数据源获取数据 (同步版本)。"""
    print(f"  [{source}] 开始请求...")
    time.sleep(delay)  # 阻塞当前线程, CPU 空转
    print(f"  [{source}] 数据返回! (耗时 {delay}s)")
    return f"<数据来自 {source}>"


def main_sync() -> None:
    """同步方式获取 3 个数据源。"""
    print("同步版本: 依次请求 3 个数据源")
    start = time.perf_counter()

    result1 = fetch_data_sync("数据库", 2.0)
    result2 = fetch_data_sync("API", 1.5)
    result3 = fetch_data_sync("缓存", 1.0)

    elapsed = time.perf_counter() - start
    print(f"\n结果: {result1}, {result2}, {result3}")
    print(f"总耗时: {elapsed:.1f}s  (2.0 + 1.5 + 1.0 = 4.5s, 完全串行)")
    return None


print("=" * 60)
print('同步版本 —— 慢的原因是「等」')
print("=" * 60)
main_sync()


# ------------------------------------------------------------
# 三、异步版本 —— 同样的逻辑, 更快的完成
# ------------------------------------------------------------
# 核心语法:
#   async def  → 定义一个协程 (coroutine), 不是普通函数
#   await      → 暂停当前协程, 等异步操作完成, 期间事件循环可以调度其他协程
#
# ⚠️ 协程不能直接调用! coro() 返回一个协程对象, 不会执行。
#    必须用 await 或 asyncio.run() 来执行。

async def fetch_data_async(source: str, delay: float) -> str:
    """模拟从某个数据源获取数据 (异步版本)。"""
    print(f"  [{source}] 开始请求...")
    await asyncio.sleep(delay)  # 暂停当前协程, 不阻塞事件循环!
    print(f"  [{source}] 数据返回! (耗时 {delay}s)")
    return f"<数据来自 {source}>"


async def main_async() -> None:
    """异步方式: 3 个请求并发执行。"""
    print("\n异步版本: 同时发起 3 个请求")
    start = time.perf_counter()

    # gather() = 并发执行多个协程, 等全部完成后返回结果列表
    results = await asyncio.gather(
        fetch_data_async("数据库", 2.0),
        fetch_data_async("API", 1.5),
        fetch_data_async("缓存", 1.0),
    )

    elapsed = time.perf_counter() - start
    print(f"\n结果: {results}")
    print(f"总耗时: {elapsed:.1f}s  (≈ max(2.0, 1.5, 1.0) = 2.0s, 并发!)")
    print(f"快了 {4.5 / elapsed:.1f} 倍!")


print("\n" + "=" * 60)
print('异步版本 —— 快的原因是「不等」')
print("=" * 60)

# asyncio.run() = 事件循环的入口, 类似 Java 的 ExecutorService.submit()
asyncio.run(main_async())


# ------------------------------------------------------------
# 四、async/await 的核心规则
# ------------------------------------------------------------
# 规则 1: 只有 async def 定义的函数里才能用 await
# 规则 2: async def 函数调用后返回协程对象, 必须被 await 或 asyncio.run()
# 规则 3: await 只能用在可等待对象上 (coroutine, Task, Future)
# 规则 4: 协程之间通过 await 让出控制权——这是协作式的, 不是抢占式的

print("\n" + "=" * 60)
print("核心规则演示")
print("=" * 60)


async def greet(name: str) -> str:
    return f"你好, {name}!"


# 错误示范 (注释掉, 否则报错):
# result = greet("小明")  # ❌ 返回的是协程对象, 不会执行!
# print(result)  # <coroutine object greet at 0x...>

# 正确做法:
async def demo_await() -> None:
    # greet("小明") 创建协程对象
    # await 执行它并等待结果
    result = await greet("小明")  # ✅
    print(f"await 后得到: {result}")
    print(f"类型是: {type(await greet('test'))}")  # <class 'str'>, 不是协程

asyncio.run(demo_await())


# ------------------------------------------------------------
# 五、asyncio.sleep() vs time.sleep() —— 关键区别!
# ------------------------------------------------------------
# time.sleep(n):
#   - 阻塞当前线程, 整个程序挂起
#   - 在异步代码中用 time.sleep() = 灾难! 事件循环被卡住
#
# asyncio.sleep(n):
#   - 暂停当前协程, 事件循环可以调度其他协程
#   - 异步代码中唯一正确的"等待"方式

async def bad_async_function() -> None:
    """演示: 在 async 函数中用 time.sleep() 会怎样。"""
    print("  开始做重要工作...")
    time.sleep(1)  # ⚠️ 阻塞了事件循环! 所有其他协程都无法运行
    print("  工作完成 (但太晚了)")

async def demo_blocking() -> None:
    print("\n同步 sleep 造成阻塞:")
    start = time.perf_counter()

    # 并发执行, 但其中一个是 time.sleep —— 会拖累所有协程
    await asyncio.gather(
        bad_async_function(),
        fetch_data_async("API", 0.5),  # 本应 0.5s 完成
    )
    # 但因为 bad_async_function 里用了 time.sleep(1),
    # 事件循环被阻塞 1 秒, fetch_data_async 也只能等 1 秒

    elapsed = time.perf_counter() - start
    print(f"  总耗时: {elapsed:.1f}s (本该 max(1, 0.5) = 1s, 但被阻塞拖累)")

asyncio.run(demo_blocking())

# 记住: async 函数里只用 asyncio.sleep(), 永远不用 time.sleep()!


# ------------------------------------------------------------
# 六、create_task —— 创建"后台"任务
# ------------------------------------------------------------
# gather() 会等所有协程完成。如果想"发射后不管"呢?
# create_task() 创建 Task 对象, 在事件循环中调度执行。
# 类比 Java: CompletableFuture.runAsync(...)

async def create_task_demo() -> None:
    print("\n" + "=" * 60)
    print("create_task —— 后台任务")
    print("=" * 60)

    # 创建一个后台任务 —— 不会阻塞当前协程
    task = asyncio.create_task(
        fetch_data_async("后台任务", 2.0)
    )
    print(f"任务已创建, 类型: {type(task).__name__}")
    print(f"任务状态: {'完成' if task.done() else '进行中'}")

    # 当前协程继续做自己的事
    print("主协程继续工作...")
    await asyncio.sleep(1.0)
    print(f"1 秒后任务状态: {'完成' if task.done() else '进行中'}")

    # 等待后台任务完成
    result = await task
    print(f"任务结果: {result}")

asyncio.run(create_task_demo())


# ------------------------------------------------------------
# 七、Task 的高级操作 —— 超时、取消
# ------------------------------------------------------------

async def slow_operation() -> str:
    """一个需要很长时间的操作。"""
    await asyncio.sleep(3)
    return "完成了!"  # 实际上永远不会返回, 因为会被取消

async def task_control_demo() -> None:
    print("\n" + "=" * 60)
    print("Task 控制: 超时与取消")
    print("=" * 60)

    # --- 超时控制 ---
    # wait_for() 给协程设置超时时间, 超时则抛出 TimeoutError
    try:
        result = await asyncio.wait_for(
            fetch_data_async("慢速 API", 5.0),
            timeout=1.0,
        )
    except asyncio.TimeoutError:
        print("⏰ API 调用超时 (1 秒限制)!")

    # --- 取消任务 ---
    task = asyncio.create_task(slow_operation())
    await asyncio.sleep(0.5)  # 让它跑一会儿
    task.cancel()  # 发送取消请求

    try:
        await task
    except asyncio.CancelledError:
        print("🛑 任务已被取消")

asyncio.run(task_control_demo())


# ------------------------------------------------------------
# 八、异步上下文管理器 —— async with
# ------------------------------------------------------------
# 有些资源操作本身就是异步的 (如异步数据库连接、HTTP session)。
# 这时候用 async with 替代 with。

class AsyncResource:
    """模拟一个异步资源 (如数据库连接、网络连接)。"""

    async def __aenter__(self):
        """异步进入上下文 —— 模拟建立连接。"""
        print("  建立连接中...")
        await asyncio.sleep(0.5)  # 模拟网络延迟
        print("  连接已建立")
        return self

    async def __aexit__(self, *args):
        """异步退出上下文 —— 模拟关闭连接。"""
        print("  关闭连接中...")
        await asyncio.sleep(0.3)
        print("  连接已关闭")

    async def query(self, sql: str) -> str:
        """模拟异步查询。"""
        print(f"  执行: {sql}")
        await asyncio.sleep(0.5)
        return "[查询结果]"


async def async_context_demo() -> None:
    print("\n" + "=" * 60)
    print("异步上下文管理器: async with")
    print("=" * 60)

    # async with 替代 with, __aenter__ / __aexit__ 替代 __enter__ / __exit__
    async with AsyncResource() as conn:
        result = await conn.query("SELECT * FROM users")
        print(f"  结果: {result}")

asyncio.run(async_context_demo())

# 对比:
#   同步: with open(...) as f         → __enter__ / __exit__
#   异步: async with client as c      → __aenter__ / __aexit__
#   同步迭代: for item in items       → __iter__ / __next__
#   异步迭代: async for item in items → __aiter__ / __anext__


# ------------------------------------------------------------
# 九、实战对比 —— 同步 vs 异步性能
# ------------------------------------------------------------
# 模拟一个常见场景: 需要从 5 个 URL 获取数据

URLS = [
    ("https://api.example.com/users", 1.2),
    ("https://api.example.com/orders", 0.8),
    ("https://api.example.com/products", 1.5),
    ("https://api.example.com/reviews", 0.6),
    ("https://api.example.com/analytics", 1.0),
]


def fetch_all_sync() -> None:
    """同步: 顺序请求 5 个 URL。"""
    print("\n🐢 同步版本 (顺序请求):")
    start = time.perf_counter()
    results = []
    for url, delay in URLS:
        result = fetch_data_sync(url, delay)
        results.append(result)
    elapsed = time.perf_counter() - start
    print(f"总耗时: {elapsed:.1f}s (所有 delay 之和)")
    print(f"结果数: {len(results)}")


async def fetch_all_async() -> None:
    """异步: 并发请求 5 个 URL。"""
    print("\n🚀 异步版本 (并发请求):")
    start = time.perf_counter()

    tasks = [
        fetch_data_async(url, delay)
        for url, delay in URLS
    ]
    results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start
    print(f"总耗时: {elapsed:.1f}s (≈ 最大 delay)")
    print(f"结果数: {len(results)}")


print("\n" + "=" * 60)
print("性能对比: 同步 vs 异步")
print("=" * 60)

fetch_all_sync()
asyncio.run(fetch_all_async())

total_delay = sum(d for _, d in URLS)
max_delay = max(d for _, d in URLS)
print(f"\n预期: 同步 ≈ {total_delay:.1f}s, 异步 ≈ {max_delay:.1f}s")
print(f"加速比: {total_delay / max_delay:.1f} 倍")


# ------------------------------------------------------------
# 十、常见陷阱 & 最佳实践
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("常见陷阱速查")
print("=" * 60)

trap_list = [
    ("async 函数里用 time.sleep()", "用 asyncio.sleep()"),
    ("直接调用 async 函数不加 await", "必须 await 或用 asyncio.run()"),
    ("在普通函数里用 await", "只能在 async def 里用 await"),
    ("忘记 await create_task 的返回值", "如果需要结果, 记得 await task"),
    ("混用同步和异步 IO", "整个调用链要么全同步, 要么全异步"),
    ("asyncio.run() 多次调用", "一个程序通常只调用一次 asyncio.run()"),
]
for trap, fix in trap_list:
    print(f"  ❌ {trap}")
    print(f"     ✅ {fix}")

# 还有一个概念需要心里有数:
# Python 的 async 是单线程并发, 不是多线程并行。
# CPU 密集型任务 (如计算) 用 async 不会变快, 反而会变慢。
# async 适合 IO 密集型 (网络请求、文件读取、数据库查询)。


# ------------------------------------------------------------
# 综合实战: 简单的异步爬虫
# ------------------------------------------------------------
# 模拟爬取多个网页的标题。

import random


async def fetch_page(url: str) -> dict:
    """
    模拟爬取网页。
    实际项目中这里会是 aiohttp 的 session.get(url)。
    """
    # 模拟网络延迟 (100-1000ms)
    delay = random.uniform(0.1, 1.0)
    await asyncio.sleep(delay)

    # 模拟随机结果
    if random.random() < 0.1:  # 10% 概率失败
        return {"url": url, "success": False, "error": "连接超时"}

    return {
        "url": url,
        "success": True,
        "title": f"页面 {url.split('/')[-1]} 的标题",
        "size_kb": random.randint(10, 500),
        "delay_ms": round(delay * 1000),
    }


async def crawl_sites(sites: list[str], max_concurrent: int = 3) -> list[dict]:
    """
    爬取多个网站, 限制最大并发数。
    使用 Semaphore (信号量) 控制并发, 避免对服务器造成压力。
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def crawl_with_limit(url: str) -> dict:
        async with semaphore:  # 获取信号量, 同时最多 max_concurrent 个
            return await fetch_page(url)

    tasks = [crawl_with_limit(site) for site in sites]
    results = await asyncio.gather(*tasks)
    return list(results)


async def main_crawler() -> None:
    print("\n" + "=" * 60)
    print("综合实战: 异步爬虫")
    print("=" * 60)

    sites = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
        "https://example.com/page4",
        "https://example.com/page5",
        "https://example.com/page6",
        "https://example.com/page7",
        "https://example.com/page8",
    ]

    print(f"爬取 {len(sites)} 个页面 (最大并发: 3)...")
    start = time.perf_counter()

    results = await crawl_sites(sites, max_concurrent=3)

    elapsed = time.perf_counter() - start

    success = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])

    print(f"\n结果:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        if r["success"]:
            print(f"  {status} {r['url']:<35} \"{r['title']}\" ({r['delay_ms']}ms, {r['size_kb']}KB)")
        else:
            print(f"  {status} {r['url']:<35} {r['error']}")

    print(f"\n成功: {success}, 失败: {failed}")
    print(f"总耗时: {elapsed:.1f}s")
    print(f"并发控制 (Semaphore) 生效: 最大同时 3 个请求")


# 原课程演示: 网页爬虫
# if __name__ == "__main__":
#     asyncio.run(main_crawler())


# ============================================================
# 试试看 (Try This)
# ============================================================
#
# 1. 写一个异步函数 countdown(name: str, n: int):
#    从 n 倒数到 1, 每秒打印一次: f"{name}: {i}"
#    然后同时启动 3 个倒计时: countdown("A", 5), countdown("B", 3), countdown("C", 4)
#    用 gather() 并发执行, 观察输出交错的顺序。

async def countdown(name: str, n: int):
    """异步倒计时。"""
    for i in range(n, 0, -1):
        print(f"  {name}: {i}")
        await asyncio.sleep(0.5)  # 用 0.5s 加速演示
    print(f"  {name}: 完成!")

async def run_countdown_demo():
    print("--- countdown (并发) ---")
    await asyncio.gather(
        countdown("A", 5),
        countdown("B", 3),
        countdown("C", 4),
    )
    print()

from pathlib import Path

# asyncio.run(run_countdown_demo())  # 在 __main__ 中统一执行


#
# 2. 把 Lesson 07 的 retry 装饰器改成异步版本:

import functools
import time as _sync_time

def async_retry(max_attempts: int = 3, delay: float = 0.3,
                exceptions: tuple = (Exception,)):
    """异步重试装饰器: 支持 await 的异步函数。"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        print(f"  第 {attempt} 次重试... ({type(e).__name__})")
                        await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

@async_retry(max_attempts=3, delay=0.1, exceptions=(ValueError, ConnectionError))
async def unstable_async_task():
    import random
    if random.random() < 0.6:
        raise ValueError("临时错误")
    return "异步任务完成"

async def run_retry_demo():
    print("--- async_retry ---")
    try:
        result = await unstable_async_task()
        print(f"  ✅ {result}")
    except ValueError:
        print(f"  ❌ 3 次重试后仍失败")
    print()


#
# 3. 写一个异步函数 download_all(urls: list[str], dest_dir: str):

async def download_one(url: str, dest_dir: str, idx: int, total: int) -> dict:
    """模拟下载一个文件。"""
    delay = 0.3 + (hash(url) % 5) * 0.1  # 模拟不同文件耗时不同
    await asyncio.sleep(delay)
    filename = url.split("/")[-1] or f"file_{idx}"
    filepath = Path(dest_dir) / filename
    filepath.write_text(f"[模拟下载内容] 来自 {url}", encoding="utf-8")
    print(f"  [{idx}/{total}] {filename} ({delay:.1f}s)")
    return {"url": url, "file": filename, "delay": delay}

async def download_all(urls: list[str], dest_dir: str):
    """并发下载所有文件。"""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    total = len(urls)
    tasks = [download_one(url, dest_dir, i+1, total) for i, url in enumerate(urls)]
    results = await asyncio.gather(*tasks)
    return results

async def run_download_demo():
    print("--- download_all ---")
    urls = [
        "https://cdn.example.com/lib.js",
        "https://cdn.example.com/style.css",
        "https://cdn.example.com/logo.png",
        "https://cdn.example.com/data.json",
    ]
    results = await download_all(urls, "test_downloads")
    print(f"  ✅ 完成 {len(results)} 个文件下载")
    import shutil
    shutil.rmtree("test_downloads", ignore_errors=True)
    print()


#
# 4. 写一个异步的生产者-消费者模式:

async def producer(queue: asyncio.Queue, n: int = 6):
    """每隔 0.3 秒向队列放入一个任务。"""
    for i in range(n):
        task = f"任务-{i+1}"
        await queue.put(task)
        print(f"  📦 生产: {task}")
        await asyncio.sleep(0.3)
    # 放入 None 作为结束信号
    for _ in range(3):
        await queue.put(None)

async def consumer(queue: asyncio.Queue, name: str):
    """不断从队列取任务并处理。"""
    while True:
        task = await queue.get()
        if task is None:
            print(f"  🏁 {name} 退出")
            return
        print(f"  ⚙️ {name} 处理: {task}")
        await asyncio.sleep(0.5)  # 模拟处理耗时
        queue.task_done()

async def run_producer_consumer_demo():
    print("--- 生产者-消费者 ---")
    q = asyncio.Queue(maxsize=10)
    prod = asyncio.create_task(producer(q, n=6))
    consumers = [asyncio.create_task(consumer(q, f"C{i+1}")) for i in range(3)]
    await prod
    await asyncio.gather(*consumers)
    print()


#
# 5. (探索) 试试不加 await 调用异步函数会怎样:

async def fetch_data_async(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"{name} 的数据"

async def run_await_exploration():
    print("--- await 探索 ---")
    # 不加 await: 返回 coroutine 对象, 不会执行
    coro = fetch_data_async("test", 0.1)
    print(f"  不加 await: type={type(coro).__name__}")  # coroutine

    # 加 await: 真正执行
    result = await coro
    print(f"  加 await: result={result}")
    print()


#
# 6. (挑战) 用 asyncio 实现一个简单的"端口扫描器":

async def scan_port(host: str, port: int, sem: asyncio.Semaphore) -> int | None:
    """扫描单个端口, 开放则返回端口号。"""
    async with sem:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=1.0
            )
            writer.close()
            await writer.wait_closed()
            return port
        except Exception:
            return None

async def port_scanner(host: str, start: int, end: int, concurrency: int = 50):
    """扫描端口范围, 返回开放的端口列表。"""
    sem = asyncio.Semaphore(concurrency)
    tasks = [scan_port(host, port, sem) for port in range(start, end + 1)]
    results = await asyncio.gather(*tasks)
    open_ports = [p for p in results if p is not None]
    return sorted(open_ports)

async def run_port_scanner_demo():
    # 扫描一个很小的范围演示 (正式使用可以扩大)
    print("--- 端口扫描器 (localhost:8000-8020) ---")
    start_t = _sync_time.time()
    open_ports = await port_scanner("127.0.0.1", 8000, 8020, concurrency=50)
    elapsed = (_sync_time.time() - start_t) * 1000
    if open_ports:
        fmt = ", ".join(str(p) for p in open_ports)
        print(f"  开放端口: {fmt}")
    else:
        print(f"  范围内无开放端口")
    print(f"  扫描 21 个端口耗时: {elapsed:.0f}ms")
    print()


# ============================================================
# 统一运行入口: 在 run_try_exercises() 中串联所有演示
# ============================================================

async def run_try_exercises():
    """依次运行所有试试看练习的演示。"""
    await run_countdown_demo()
    await run_retry_demo()
    await run_download_demo()
    await run_producer_consumer_demo()
    await run_await_exploration()
    await run_port_scanner_demo()

# 在文件被直接运行时, 执行试试看练习。
# 原课程Demo (main_crawler) 已注释, 需要时可取消注释运行。
if __name__ == "__main__":
    asyncio.run(run_try_exercises())


# 做完后告诉我:
#   - async/await 的思维模型和 Java 多线程的差异, 你适应了吗?
#   - "协作式切换" vs "抢占式切换", 你觉得哪种更容易理解?
# 我们继续 Lesson 10: pytest 测试基础 (Phase 1 最后一课! 🎯)。
# ============================================================
