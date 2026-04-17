"""异步工具执行器"""

import asyncio
import concurrent.futures
from typing import Any

from ..core.exceptions import ToolException
from ..utils.logger import get_logger
from .registry import ToolRegistry


class AsyncToolExecutor:
    """异步工具执行器 - 支持超时控制和真正的并发执行"""

    def __init__(
        self,
        registry: ToolRegistry,
        max_workers: int = 4,
        default_timeout: float | None = None,
    ):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.default_timeout = default_timeout
        self.logger = get_logger(__name__)
        self.logger.info(f"🔧 异步工具执行器已初始化 (超时限制: {default_timeout}s)")

    async def execute_tool_async(
        self, tool_name: str, input_data: Any, timeout: float | None = None
    ) -> str:
        """
        在线程池中异步执行单个工具，超时或失败时抛出 ToolException

        Args:
            tool_name: 工具名称
            input_data: 输入数据（字符串或参数字典）
            timeout: 本次超时时间（秒），None 时使用 default_timeout
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self.executor, self.registry.execute_tool, tool_name, input_data
        )

        try:
            if effective_timeout:
                self.logger.debug(
                    f"🔧 任务 [{tool_name}] 设置超时: {effective_timeout}s"
                )
                return await asyncio.wait_for(future, timeout=effective_timeout)
            return await future
        except asyncio.TimeoutError:
            msg = f"工具执行超时 (限制: {effective_timeout}s)"
            self.logger.error(f"🔧 {msg}: {tool_name}")
            raise ToolException(msg)
        except ToolException:
            raise
        except Exception as e:
            self.logger.error(f"🔧 工具执行发生错误 [{tool_name}]: {e}")
            raise ToolException(f"工具执行错误 [{tool_name}]: {e}")

    async def execute_tools_parallel(
        self, tasks: list[dict[str, Any]], timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """
        并发执行多个工具，结果顺序与输入顺序一致

        Args:
            tasks: 任务列表，每项需含 tool_name，可含 input_data
            timeout: 每个任务的超时时间（秒）

        Returns:
            与输入等长的结果列表，每项包含 task_id / tool_name / input_data /
            result / status（"success" | "error"）及可选 error_type
        """
        if not tasks:
            return []

        effective_timeout = timeout if timeout is not None else self.default_timeout
        valid = [(i, t) for i, t in enumerate(tasks) if t.get("tool_name")]

        if not valid:
            self.logger.warning("🔧 任务列表中无有效的 tool_name，跳过执行")
            return []

        self.logger.info(
            f"🔧 开始并行执行 {len(valid)} 个工具任务 (超时限制: {effective_timeout}s)"
        )

        coros = [
            self.execute_tool_async(
                t["tool_name"], t.get("input_data", ""), timeout=effective_timeout
            )
            for _, t in valid
        ]
        outcomes = await asyncio.gather(*coros, return_exceptions=True)

        results: list[dict[str, Any]] = []
        for (i, task), outcome in zip(valid, outcomes):
            tool_name = task["tool_name"]
            input_data = task.get("input_data", "")
            base = {"task_id": i, "tool_name": tool_name, "input_data": input_data}

            if isinstance(outcome, ToolException):
                results.append(
                    {
                        **base,
                        "result": str(outcome),
                        "status": "error",
                        "error_type": "ToolException",
                    }
                )
                self.logger.warning(
                    f"🔧 任务 {i + 1} 业务异常/超时: {tool_name} - {outcome}"
                )
            elif isinstance(outcome, Exception):
                results.append(
                    {
                        **base,
                        "result": str(outcome),
                        "status": "error",
                        "error_type": "SystemException",
                    }
                )
                self.logger.error(f"🔧 任务 {i + 1} 系统错误: {tool_name} - {outcome}")
            else:
                results.append({**base, "result": outcome, "status": "success"})
                self.logger.info(f"🔧 任务 {i + 1} 完成: {tool_name}")

        success_count = sum(1 for r in results if r["status"] == "success")
        self.logger.info(f"🔧 并行执行完成，成功: {success_count}/{len(results)}")
        return results

    async def execute_tools_batch(
        self, tool_name: str, input_list: list[str], timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """批量对同一工具发起并发调用"""
        tasks = [{"tool_name": tool_name, "input_data": inp} for inp in input_list]
        return await self.execute_tools_parallel(tasks, timeout=timeout)

    def close(self) -> None:
        """关闭线程池，等待所有进行中的任务完成"""
        self.logger.info("🔧 正在关闭异步工具执行器...")
        self.executor.shutdown(wait=True)
        self.logger.info("🔧 异步工具执行器已关闭")

    def __enter__(self) -> "AsyncToolExecutor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    async def __aenter__(self) -> "AsyncToolExecutor":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# 便捷函数
async def run_parallel_tools(
    registry: ToolRegistry,
    tasks: list[dict[str, Any]],
    max_workers: int = 4,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """
    并发执行多个工具任务的便捷异步入口。

    该函数会在内部创建 `AsyncToolExecutor`，并通过异步上下文自动管理线程池生命周期，
    调用结束后自动释放资源。

    Args:
        registry: 工具注册表实例，用于按 `tool_name` 查找并执行对应工具。
        tasks: 工具任务列表。每项应包含 `tool_name`，可选 `input_data`。
        max_workers: 线程池最大工作线程数，用于限制并发执行度。
        timeout: 单个任务的超时时间（秒）；None 表示不额外设置超时限制。

    Returns:
        与输入任务顺序一致的结果列表。每项包含 task_id、tool_name、input_data、
        result、status（"success" 或 "error"），错误场景下还会包含 error_type。
    """
    async with AsyncToolExecutor(registry, max_workers, timeout) as executor:
        return await executor.execute_tools_parallel(tasks)


async def run_batch_tool(
    registry: ToolRegistry,
    tool_name: str,
    input_list: list[str],
    max_workers: int = 4,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """
    对同一工具进行批量并发调用的便捷异步入口。

    该函数将 `input_list` 自动转换为任务列表后并发执行，适用于“同一工具 + 多组输入”
    的批处理场景。

    Args:
        registry: 工具注册表实例。
        tool_name: 目标工具名称。
        input_list: 同一工具的多组输入数据列表。
        max_workers: 线程池最大工作线程数。
        timeout: 单个任务的超时时间（秒）；None 表示不额外设置超时限制。

    Returns:
        批量执行结果列表；每项包含任务标识、工具名、输入、执行结果与状态信息。
    """
    async with AsyncToolExecutor(registry, max_workers, timeout) as executor:
        return await executor.execute_tools_batch(tool_name, input_list)


# 同步包装函数
def run_parallel_tools_sync(
    registry: ToolRegistry,
    tasks: list[dict[str, Any]],
    max_workers: int = 4,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """
    `run_parallel_tools` 的同步包装函数。

    通过 `asyncio.run(...)` 在同步上下文中驱动异步并发执行流程，便于在非 async 场景
    下直接调用工具并发能力。

    Args:
        registry: 工具注册表实例。
        tasks: 工具任务列表。每项应包含 `tool_name`，可选 `input_data`。
        max_workers: 线程池最大工作线程数。
        timeout: 单个任务的超时时间（秒）；None 表示不额外设置超时限制。

    Returns:
        并发执行结果列表，结构与 `run_parallel_tools` 返回值一致。
    """
    return asyncio.run(run_parallel_tools(registry, tasks, max_workers, timeout))


def run_batch_tool_sync(
    registry: ToolRegistry,
    tool_name: str,
    input_list: list[str],
    max_workers: int = 4,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """
    `run_batch_tool` 的同步包装函数。

    在同步代码中批量并发调用同一工具，内部使用 `asyncio.run(...)` 执行异步任务。

    Args:
        registry: 工具注册表实例。
        tool_name: 目标工具名称。
        input_list: 同一工具的多组输入数据列表。
        max_workers: 线程池最大工作线程数。
        timeout: 单个任务的超时时间（秒）；None 表示不额外设置超时限制。

    Returns:
        批量执行结果列表，结构与 `run_batch_tool` 返回值一致。
    """
    return asyncio.run(
        run_batch_tool(registry, tool_name, input_list, max_workers, timeout)
    )
