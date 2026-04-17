"""工具执行器"""

import asyncio
from typing import Any

from ..core.exceptions import ToolException
from ..utils.logger import get_logger
from .registry import ToolRegistry


class ToolExecutor:
    """异步工具执行器 - 支持超时控制和真正的并发执行"""

    def __init__(
        self,
        registry: ToolRegistry,
        default_timeout: float | None = None,
    ):
        self.registry = registry
        self.default_timeout = default_timeout
        self.logger = get_logger(__name__)
        self.logger.info(f"🔧 异步工具执行器已初始化 (超时限制: {default_timeout}s)")

    async def execute_tool(
        self,
        tool_name: str,
        input_data: str | dict[str, Any],
        timeout: float | None = None,
    ) -> str:
        """
        异步执行单个工具，超时或失败时抛出 ToolException

        Args:
            tool_name: 工具名称
            input_data: 输入数据（字符串或参数字典）
            timeout: 本次超时时间（秒），None 时使用 default_timeout
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout
        coro = self.registry.execute_tool(tool_name, input_data)

        try:
            if effective_timeout:
                self.logger.debug(
                    f"🔧 任务 [{tool_name}] 设置超时: {effective_timeout}s"
                )
                return await asyncio.wait_for(coro, timeout=effective_timeout)
            return await coro
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
            tasks: 任务列表，每项需含 tool_name，可含 input_data 或预处理 error
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

        ordered_results: list[dict[str, Any] | None] = [None] * len(valid)
        runnable: list[tuple[int, int, dict[str, Any], Any]] = []

        for pos, (i, task) in enumerate(valid):
            tool_name = task["tool_name"]
            input_data = task.get("input_data", "")
            base = {"task_id": i, "tool_name": tool_name, "input_data": input_data}

            if task.get("error"):
                ordered_results[pos] = {
                    **base,
                    "result": str(task["error"]),
                    "status": "error",
                    "error_type": "TaskPreparationError",
                }
                self.logger.warning(
                    f"🔧 任务 {i + 1} 预处理失败: {tool_name} - {task['error']}"
                )
                continue

            coro = self.execute_tool(
                tool_name,
                input_data,
                timeout=effective_timeout,
            )
            runnable.append((pos, i, task, coro))

        outcomes = await asyncio.gather(
            *(coro for _, _, _, coro in runnable),
            return_exceptions=True,
        )

        for (pos, i, task, _), outcome in zip(runnable, outcomes):
            tool_name = task["tool_name"]
            input_data = task.get("input_data", "")
            base = {"task_id": i, "tool_name": tool_name, "input_data": input_data}

            if isinstance(outcome, ToolException):
                ordered_results[pos] = {
                    **base,
                    "result": str(outcome),
                    "status": "error",
                    "error_type": "ToolException",
                }
                self.logger.warning(
                    f"🔧 任务 {i + 1} 业务异常/超时: {tool_name} - {outcome}"
                )
            elif isinstance(outcome, Exception):
                ordered_results[pos] = {
                    **base,
                    "result": str(outcome),
                    "status": "error",
                    "error_type": "SystemException",
                }
                self.logger.error(f"🔧 任务 {i + 1} 系统错误: {tool_name} - {outcome}")
            else:
                ordered_results[pos] = {
                    **base,
                    "result": outcome,
                    "status": "success",
                }
                self.logger.info(f"🔧 任务 {i + 1} 完成: {tool_name}")

        results = [r for r in ordered_results if r is not None]
        success_count = sum(1 for r in results if r["status"] == "success")
        self.logger.info(f"🔧 并行执行完成，成功: {success_count}/{len(results)}")
        return results

    async def execute_tools_batch(
        self, tool_name: str, input_list: list[str], timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """批量对同一工具发起并发调用"""
        tasks = [{"tool_name": tool_name, "input_data": inp} for inp in input_list]
        return await self.execute_tools_parallel(tasks, timeout=timeout)


# 便捷函数
async def run_parallel_tools(
    registry: ToolRegistry,
    tasks: list[dict[str, Any]],
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """
    并发执行多个工具任务的便捷异步入口。

    Args:
        registry: 工具注册表实例，用于按 `tool_name` 查找并执行对应工具。
        tasks: 工具任务列表。每项应包含 `tool_name`，可选 `input_data`。
        timeout: 单个任务的超时时间（秒）；None 表示不额外设置超时限制。

    Returns:
        与输入任务顺序一致的结果列表。每项包含 task_id、tool_name、input_data、
        result、status（"success" 或 "error"），错误场景下还会包含 error_type。
    """
    executor = ToolExecutor(registry, timeout)
    return await executor.execute_tools_parallel(tasks)


async def run_batch_tool(
    registry: ToolRegistry,
    tool_name: str,
    input_list: list[str],
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """
    对同一工具进行批量并发调用的便捷异步入口。

    Args:
        registry: 工具注册表实例。
        tool_name: 目标工具名称。
        input_list: 同一工具的多组输入数据列表。
        timeout: 单个任务的超时时间（秒）；None 表示不额外设置超时限制。

    Returns:
        批量执行结果列表；每项包含任务标识、工具名、输入、执行结果与状态信息。
    """
    executor = ToolExecutor(registry, timeout)
    return await executor.execute_tools_batch(tool_name, input_list)
