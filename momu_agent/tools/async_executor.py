"""异步工具执行器"""

import asyncio
import concurrent.futures
from typing import Any

from ..core.exceptions import ToolException
from ..utils.logger import get_logger
from .registry import ToolRegistry


class AsyncToolExecutor:
    """异步工具执行器 - 支持超时控制"""

    def __init__(
        self,
        registry: ToolRegistry,
        max_workers: int = 4,
        default_timeout: float | None = None,
    ):
        """
        初始化执行器

        Args:
            registry: 工具注册表
            max_workers: 最大线程数
            default_timeout: 默认超时时间（秒），None 表示不限制
        """
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.default_timeout = default_timeout
        self.logger = get_logger(__name__)

        self.logger.info(f"🔧 异步工具执行器已初始化 (超时限制: {default_timeout}s)")

    async def execute_tool_async(
        self, tool_name: str, input_data: str, timeout: float | None = None
    ) -> str:
        """
        异步执行单个工具

        Args:
            tool_name: 工具名称
            input_data: 输入数据
            timeout: 本次执行的超时时间（秒），若不传则使用默认值

        Returns:
            工具执行结果

        Raises:
            ToolException: 当工具执行失败或超时时抛出
        """
        # 确定最终超时时间
        effective_timeout = timeout if timeout is not None else self.default_timeout

        loop = asyncio.get_event_loop()

        def _execute():
            return self.registry.execute_tool(tool_name, input_data)

        try:
            # 在线程池中运行，并包装超时控制
            future = loop.run_in_executor(self.executor, _execute)

            if effective_timeout:
                self.logger.debug(
                    f"🔧 任务 [{tool_name}] 设置超时: {effective_timeout}s"
                )
                result = await asyncio.wait_for(future, timeout=effective_timeout)
            else:
                result = await future

            return result

        except asyncio.TimeoutError:
            error_msg = f"工具执行超时 (限制: {effective_timeout}s)"
            self.logger.error(f"🔧 {error_msg}: {tool_name}")
            raise ToolException(error_msg)
        except ToolException:
            raise
        except Exception as e:
            self.logger.error(f"🔧 工具执行发生错误 [{tool_name}]: {str(e)}")
            raise ToolException(f"工具执行错误 [{tool_name}]: {str(e)}")

    async def execute_tools_parallel(
        self, tasks: list[dict[str, str]], timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """
        并行执行多个工具

        Args:
            tasks: 任务列表
            timeout: 每个任务的超时时间（秒）

        Returns:
            执行结果列表
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout
        self.logger.info(
            f"🔧 开始并行执行 {len(tasks)} 个工具任务 (超时限制: {effective_timeout}s)"
        )

        if not tasks:
            self.logger.warning("🔧 任务列表为空")
            return []

        # 创建异步任务
        async_tasks = []
        for i, task in enumerate(tasks):
            tool_name = task.get("tool_name")
            input_data = task.get("input_data", "")

            if not tool_name:
                continue

            self.logger.info(f"🔧 创建任务 {i + 1}: {tool_name}")
            # 将超时参数传递给单个执行方法
            async_task = self.execute_tool_async(
                tool_name, input_data, timeout=effective_timeout
            )
            async_tasks.append((i, task, async_task))

        # 等待所有任务完成
        results = []
        for i, task, async_task in async_tasks:
            try:
                result = await async_task
                results.append(
                    {
                        "task_id": i,
                        "tool_name": task["tool_name"],
                        "input_data": task["input_data"],
                        "result": result,
                        "status": "success",
                    }
                )
                self.logger.info(f"🔧 任务 {i + 1} 完成: {task['tool_name']}")

            except ToolException as e:
                # 捕获我们自定义的异常（包括超时）
                results.append(
                    {
                        "task_id": i,
                        "tool_name": task["tool_name"],
                        "input_data": task["input_data"],
                        "result": str(e),
                        "status": "error",
                        "error_type": "ToolException",
                    }
                )
                self.logger.warning(
                    f"🔧 任务 {i + 1} 业务异常/超时: {task['tool_name']} - {e}"
                )

            except Exception as e:
                results.append(
                    {
                        "task_id": i,
                        "tool_name": task["tool_name"],
                        "input_data": task["input_data"],
                        "result": str(e),
                        "status": "error",
                        "error_type": "SystemException",
                    }
                )
                self.logger.error(
                    f"🔧 任务 {i + 1} 系统错误: {task['tool_name']} - {e}"
                )

        success_count = sum(1 for r in results if r["status"] == "success")
        self.logger.info(f"🔧 并行执行完成，成功: {success_count}/{len(results)}")
        return results

    async def execute_tools_batch(
        self, tool_name: str, input_list: list[str], timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """批量执行同一个工具"""
        tasks = [
            {"tool_name": tool_name, "input_data": input_data}
            for input_data in input_list
        ]
        return await self.execute_tools_parallel(tasks, timeout=timeout)

    def close(self):
        """关闭执行器"""
        self.logger.info("🔧 正在关闭异步工具执行器...")
        self.executor.shutdown(wait=True)
        self.logger.info("🔧 异步工具执行器已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 便捷函数
async def run_parallel_tools(
    registry: ToolRegistry,
    tasks: list[dict[str, str]],
    max_workers: int = 4,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    async with AsyncToolExecutor(registry, max_workers, timeout) as executor:
        return await executor.execute_tools_parallel(tasks, timeout=timeout)


async def run_batch_tool(
    registry: ToolRegistry,
    tool_name: str,
    input_list: list[str],
    max_workers: int = 4,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    async with AsyncToolExecutor(registry, max_workers, timeout) as executor:
        return await executor.execute_tools_batch(
            tool_name, input_list, timeout=timeout
        )


# 同步包装函数
def run_parallel_tools_sync(
    registry: ToolRegistry,
    tasks: list[dict[str, str]],
    max_workers: int = 4,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    return asyncio.run(run_parallel_tools(registry, tasks, max_workers, timeout))


def run_batch_tool_sync(
    registry: ToolRegistry,
    tool_name: str,
    input_list: list[str],
    max_workers: int = 4,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    return asyncio.run(
        run_batch_tool(registry, tool_name, input_list, max_workers, timeout)
    )
