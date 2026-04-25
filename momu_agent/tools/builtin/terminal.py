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
import shutil
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

    COMMAND_DELIMITERS = {"|", "||", "&&", ";"}
    REDIRECTION_OPERATORS = {">", ">>", "<", "&>", "&>>", ">&", "<&"}
    UNSUPPORTED_OPERATORS = {"<<", "<<<", "<>", ">|", "&"}

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
                "（ls, cat, grep, head, tail等）。"
                "参数：command（必填，要执行的白名单命令）；"
                "timeout（可选，命令超时时间，单位秒，默认使用工具配置值）。"
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

        self.logger.info("🔧 TerminalTool执行: command=%s", command)

        tokens, segment_commands = self._validate_command(command)
        uses_shell_syntax = self._uses_shell_syntax(tokens)

        if len(segment_commands) == 1 and segment_commands[0] == "cd":
            if uses_shell_syntax:
                raise ToolException("cd 仅支持单独执行，不支持重定向/管道/串联语法")
            try:
                cd_parts = shlex.split(command)
            except ValueError as exc:
                raise ToolException(f"命令解析失败: {exc}") from exc
            return self._handle_cd(cd_parts)

        if "cd" in segment_commands:
            raise ToolException("cd 仅支持单独执行，不支持重定向/管道/串联语法")

        timeout = parameters.get("timeout", self.timeout)
        try:
            timeout_value = int(timeout)
        except (TypeError, ValueError) as exc:
            raise ToolException("timeout 必须是正整数") from exc

        if timeout_value <= 0:
            raise ToolException("timeout 必须是正整数")

        self.logger.debug(
            "🔧 准备执行命令: commands=%s cwd=%s timeout=%s use_shell=%s",
            segment_commands,
            self.current_dir,
            timeout_value,
            uses_shell_syntax,
        )

        if uses_shell_syntax:
            executable: str | list[str] = command
        else:
            try:
                executable = shlex.split(command)
            except ValueError as exc:
                raise ToolException(f"命令解析失败: {exc}") from exc

        return await self._execute_command(executable, timeout_value, uses_shell_syntax)

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

    def _tokenize_command(self, command: str) -> list[str]:
        tokens: list[str] = []
        current = ""
        in_single = False
        in_double = False
        escaped = False
        idx = 0

        while idx < len(command):
            ch = command[idx]

            if escaped:
                current += ch
                escaped = False
                idx += 1
                continue

            if ch == "\\":
                current += ch
                escaped = True
                idx += 1
                continue

            if ch == "'" and not in_double:
                in_single = not in_single
                current += ch
                idx += 1
                continue

            if ch == '"' and not in_single:
                in_double = not in_double
                current += ch
                idx += 1
                continue

            if not in_single and not in_double:
                if ch.isspace():
                    if current:
                        tokens.append(current)
                        current = ""
                    idx += 1
                    continue

                three = command[idx : idx + 3]
                two = command[idx : idx + 2]

                if three == "<<<":
                    if current:
                        tokens.append(current)
                        current = ""
                    tokens.append(three)
                    idx += 3
                    continue

                if two in {
                    "||",
                    "&&",
                    ">>",
                    "<<",
                    "&>",
                    "<&",
                    ">&",
                    "<>",
                }:
                    if current:
                        tokens.append(current)
                        current = ""
                    tokens.append(two)
                    idx += 2
                    continue

                if ch in {"|", ";", ">", "<", "&"}:
                    if current:
                        tokens.append(current)
                        current = ""
                    tokens.append(ch)
                    idx += 1
                    continue

            current += ch
            idx += 1

        if escaped or in_single or in_double:
            raise ToolException("命令解析失败: 引号或转义符未闭合")

        if current:
            tokens.append(current)

        return tokens

    def _contains_unsupported_shell_syntax(self, tokens: list[str]) -> bool:
        return any(token in self.UNSUPPORTED_OPERATORS for token in tokens)

    def _contains_disallowed_substitution(self, command: str) -> bool:
        return "`" in command or "$(" in command or "${" in command

    def _merge_fd_redirection_tokens(self, tokens: list[str]) -> list[str]:
        merged: list[str] = []
        idx = 0

        while idx < len(tokens):
            token = tokens[idx]
            if token.isdigit() and idx + 1 < len(tokens):
                op = tokens[idx + 1]
                if op in self.REDIRECTION_OPERATORS:
                    merged.append(f"{token}{op}")
                    idx += 2
                    continue
            merged.append(token)
            idx += 1

        return merged

    def _split_segments(self, tokens: list[str]) -> list[list[str]]:
        segments: list[list[str]] = []
        current: list[str] = []

        for token in tokens:
            if token in self.COMMAND_DELIMITERS:
                if not current:
                    raise ToolException("命令语法错误：分隔符前缺少命令")
                segments.append(current)
                current = []
                continue
            current.append(token)

        if not current:
            raise ToolException("命令语法错误：分隔符后缺少命令")

        segments.append(current)
        return segments

    def _is_redirection_operator(self, token: str) -> bool:
        if token in self.REDIRECTION_OPERATORS:
            return True

        for op in self.REDIRECTION_OPERATORS:
            if token.endswith(op):
                prefix = token[: -len(op)]
                if prefix.isdigit():
                    return True

        return False

    def _is_safe_operand(self, token: str) -> bool:
        if not token:
            return False

        if token in self.COMMAND_DELIMITERS:
            return False

        if self._is_redirection_operator(token):
            return False

        if token in self.UNSUPPORTED_OPERATORS:
            return False

        if "`" in token or "$(" in token or "${" in token:
            return False

        return True

    def _validate_segment(self, segment: list[str]) -> str:
        idx = 0
        base_command: str | None = None

        while idx < len(segment):
            token = segment[idx]

            if self._is_redirection_operator(token):
                op = token
                if token not in self.REDIRECTION_OPERATORS:
                    op = ""
                    for candidate in self.REDIRECTION_OPERATORS:
                        if token.endswith(candidate):
                            op = candidate
                            break
                idx += 1
                if idx >= len(segment):
                    raise ToolException("命令语法错误：重定向缺少目标")
                target = segment[idx]
                if op in {"<&", ">&"}:
                    if target != "-" and not target.isdigit():
                        raise ToolException("命令语法错误：非法文件描述符重定向")
                elif not self._is_safe_operand(target):
                    raise ToolException("命令语法错误：非法重定向目标")
                idx += 1
                continue

            if not self._is_safe_operand(token):
                raise ToolException("命令语法错误：包含不支持的 shell 语法")

            if base_command is None:
                base_command = token
            idx += 1

        if base_command is None:
            raise ToolException("命令语法错误：未找到可执行命令")

        return base_command

    def _validate_command(self, command: str) -> tuple[list[str], list[str]]:
        tokens = self._tokenize_command(command)
        if not tokens:
            raise ToolException("命令不能为空")

        if self._contains_disallowed_substitution(command):
            raise ToolException("命令语法错误：包含不支持的 shell 语法")

        if self._contains_unsupported_shell_syntax(tokens):
            raise ToolException("命令语法错误：包含不支持的 shell 语法")

        tokens = self._merge_fd_redirection_tokens(tokens)
        segments = self._split_segments(tokens)
        segment_commands = [self._validate_segment(segment) for segment in segments]
        self.logger.debug("🔧 命令拆分结果: segments=%s", segments)

        for base_command in segment_commands:
            if base_command not in self.ALLOWED_COMMANDS:
                self.logger.warning("🔧 命令不在白名单: %s", base_command)
                raise ToolException(
                    f"不允许的命令: {base_command}。允许的命令: "
                    f"{', '.join(sorted(self.ALLOWED_COMMANDS))}"
                )

        return tokens, segment_commands

    def _uses_shell_syntax(self, tokens: list[str]) -> bool:
        return any(
            token in self.COMMAND_DELIMITERS or self._is_redirection_operator(token)
            for token in tokens
        )

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
        self.logger.info("🔧 切换目录成功: cwd=%s", self.current_dir)
        return f"✅ 切换到目录: {self.current_dir}"

    async def _execute_command(
        self,
        command: str | list[str],
        timeout: int,
        use_shell: bool,
    ) -> str:
        try:
            run_kwargs: dict[str, Any] = {
                "cwd": str(self.current_dir),
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "timeout": timeout,
                "env": os.environ.copy(),
            }

            if use_shell:
                bash_path = shutil.which("bash")
                if not bash_path:
                    raise ToolException("系统未找到 bash，无法执行 shell 语法命令")
                run_kwargs["shell"] = True
                run_kwargs["executable"] = bash_path
            else:
                run_kwargs["shell"] = False

            result = await asyncio.to_thread(subprocess.run, command, **run_kwargs)
        except ToolException:
            raise
        except subprocess.TimeoutExpired as exc:
            self.logger.warning(
                "🔧 命令执行超时: command=%s timeout=%s use_shell=%s",
                command,
                timeout,
                use_shell,
            )
            raise ToolException(f"命令执行超时（超过 {timeout} 秒）") from exc
        except Exception as exc:
            self.logger.error("🔧 命令执行失败: %s", exc)
            raise ToolException(f"命令执行失败: {exc}") from exc

        stdout_text = result.stdout or ""
        stderr_text = result.stderr or ""

        self.logger.debug(
            "🔧 命令执行完成: returncode=%s stdout_len=%s stderr_len=%s use_shell=%s",
            result.returncode,
            len(stdout_text),
            len(stderr_text),
            use_shell,
        )

        output = stdout_text
        if stderr_text:
            output += f"\n[stderr]\n{stderr_text}"

        if len(output) > self.max_output_size:
            output = output[: self.max_output_size]
            output += f"\n\n⚠️ 输出被截断（超过 {self.max_output_size} 字节）"
            self.logger.warning(
                "🔧 命令输出被截断: max_output_size=%s", self.max_output_size
            )

        if result.returncode != 0:
            detail = output if output else "无输出"
            self.logger.warning("🔧 命令返回非零: returncode=%s", result.returncode)
            raise ToolException(f"命令返回码: {result.returncode}\n\n{detail}")

        if output:
            self.logger.info("🔧 命令执行成功: has_output=True")
            return output

        self.logger.info("🔧 命令执行成功: has_output=False")
        return "✅ 命令执行成功（无输出）"

    def get_current_dir(self) -> str:
        return str(self.current_dir)

    def reset_dir(self) -> None:
        self.current_dir = self.workspace
