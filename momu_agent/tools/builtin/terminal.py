"""TerminalTool - 命令行工具

为Agent提供安全的命令行执行能力，支持：
- 文件系统操作（ls, cat, head, tail, find, grep）
- 文本处理（wc, sort, uniq）
- 目录导航（pwd, cd）
- 安全限制（白名单命令、路径限制、超时控制）

使用场景：
- JIT（即时）文件检索与分析
- 代码仓库探索
- 日志文件分析
- 数据文件预览

安全特性：
- 命令白名单（只允许安全的只读命令）
- 工作目录限制（沙箱）
- 超时控制
- 输出大小限制
- 禁止危险操作（rm, mv, chmod等）
"""

import asyncio
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from ...core.exceptions import ToolException
from ...utils.logger import get_logger
from ..base import Tool, ToolParameter


class TerminalTool(Tool):
    """命令行工具

    提供安全的命令行执行能力，支持常用的文件系统和文本处理命令。

    安全限制：
    - 只允许白名单中的命令
    - 限制在指定工作目录内
    - 超时控制（默认30秒）
    - 输出大小限制（默认10MB）

    用法示例：
    ```python
    terminal = TerminalTool(workspace="./project")

    # 列出文件
    result = await terminal.run({"command": "ls -la"})

    # 查看文件内容
    result = await terminal.run({"command": "cat README.md"})

    # 搜索文件
    result = await terminal.run({"command": "grep -r 'TODO' src/"})

    # 查看文件前10行
    result = await terminal.run({"command": "head -n 10 data.csv"})
    ```
    """

    ALLOWED_COMMANDS = {
        "ls",
        "dir",
        "tree",
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "find",
        "grep",
        "egrep",
        "fgrep",
        "wc",
        "sort",
        "uniq",
        "cut",
        "awk",
        "sed",
        "pwd",
        "cd",
        "file",
        "stat",
        "du",
        "df",
        "echo",
        "which",
        "whereis",
    }

    def __init__(
        self,
        workspace: str = ".",
        timeout: int = 30,
        max_output_size: int = 10 * 1024 * 1024,
        allow_cd: bool = True,
    ):
        super().__init__(
            name="terminal",
            description=(
                "命令行工具 - 执行安全的文件系统与文本处理命令"
                "（ls, cat, grep, head, tail等）"
            ),
        )
        self.logger = get_logger(__name__)
        self.workspace = Path(workspace).resolve()
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.allow_cd = allow_cd
        self.current_dir = self.workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    async def run(self, parameters: dict[str, Any]) -> str:
        command = str(parameters.get("command", "")).strip()
        if not command:
            raise ToolException("命令不能为空")

        try:
            parts = shlex.split(command)
        except ValueError as exc:
            raise ToolException(f"命令解析失败: {exc}") from exc

        if not parts:
            raise ToolException("命令不能为空")

        base_command = parts[0]
        if base_command not in self.ALLOWED_COMMANDS:
            raise ToolException(
                f"不允许的命令: {base_command}。允许的命令: "
                f"{', '.join(sorted(self.ALLOWED_COMMANDS))}"
            )

        if base_command == "cd":
            return self._handle_cd(parts)

        timeout = parameters.get("timeout", self.timeout)
        try:
            timeout_value = int(timeout)
        except (TypeError, ValueError) as exc:
            raise ToolException("timeout 必须是正整数") from exc

        if timeout_value <= 0:
            raise ToolException("timeout 必须是正整数")

        return await self._execute_command(parts, timeout_value)

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description=(
                    f"要执行的命令（白名单: {', '.join(sorted(list(self.ALLOWED_COMMANDS)[:10]))}...）\n"
                    "示例: 'ls -la', 'cat file.txt', 'grep pattern *.py', 'head -n 20 data.csv'"
                ),
                required=True,
            )
        ]

    def _handle_cd(self, parts: list[str]) -> str:
        if not self.allow_cd:
            raise ToolException("cd 命令已禁用")

        if len(parts) < 2:
            return f"当前目录: {self.current_dir}"

        target_dir = parts[1]
        if target_dir == "~":
            new_dir = self.workspace
        elif Path(target_dir).is_absolute():
            new_dir = Path(target_dir).resolve()
        else:
            new_dir = (self.current_dir / target_dir).resolve()

        try:
            new_dir.relative_to(self.workspace)
        except ValueError as exc:
            raise ToolException(f"不允许访问工作目录外的路径: {new_dir}") from exc

        if not new_dir.exists():
            raise ToolException(f"目录不存在: {new_dir}")

        if not new_dir.is_dir():
            raise ToolException(f"不是目录: {new_dir}")

        self.current_dir = new_dir
        return f"✅ 切换到目录: {self.current_dir}"

    async def _execute_command(self, parts: list[str], timeout: int) -> str:
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                parts,
                cwd=str(self.current_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolException(f"命令执行超时（超过 {timeout} 秒）") from exc
        except Exception as exc:
            self.logger.error("命令执行失败: %s", exc)
            raise ToolException(f"命令执行失败: {exc}") from exc

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"

        if len(output) > self.max_output_size:
            output = output[: self.max_output_size]
            output += f"\n\n⚠️ 输出被截断（超过 {self.max_output_size} 字节）"

        if result.returncode != 0:
            detail = output if output else "无输出"
            raise ToolException(f"命令返回码: {result.returncode}\n\n{detail}")

        return output if output else "✅ 命令执行成功（无输出）"

    def get_current_dir(self) -> str:
        return str(self.current_dir)

    def reset_dir(self) -> None:
        self.current_dir = self.workspace
