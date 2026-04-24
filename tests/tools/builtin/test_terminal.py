import subprocess
from unittest.mock import AsyncMock, patch

import pytest

from momu_agent.core.exceptions import ToolException
from momu_agent.tools.builtin.terminal import TerminalTool


class TestTerminalTool:
    @pytest.fixture
    def terminal(self, tmp_path):
        return TerminalTool(workspace=str(tmp_path))

    @pytest.mark.asyncio
    async def test_run_success_returns_stdout(self, terminal):
        completed = subprocess.CompletedProcess(
            args=["ls"], returncode=0, stdout="a\n", stderr=""
        )
        with patch(
            "momu_agent.tools.builtin.terminal.asyncio.to_thread",
            new=AsyncMock(return_value=completed),
        ):
            result = await terminal.run({"command": "ls"})

        assert result == "a\n"

    @pytest.mark.asyncio
    async def test_run_success_includes_stderr_when_returncode_zero(self, terminal):
        completed = subprocess.CompletedProcess(
            args=["ls"],
            returncode=0,
            stdout="",
            stderr="warning",
        )
        with patch(
            "momu_agent.tools.builtin.terminal.asyncio.to_thread",
            new=AsyncMock(return_value=completed),
        ):
            result = await terminal.run({"command": "ls"})

        assert "[stderr]" in result
        assert "warning" in result

    @pytest.mark.asyncio
    async def test_empty_command_raises(self, terminal):
        with pytest.raises(ToolException, match="命令不能为空"):
            await terminal.run({"command": ""})

    @pytest.mark.asyncio
    async def test_invalid_command_raises(self, terminal):
        with pytest.raises(ToolException, match="不允许的命令"):
            await terminal.run({"command": "rm -rf ."})

        with pytest.raises(ToolException, match="不允许的命令"):
            await terminal.run({"command": "python -c 'print(1)'"})

    @pytest.mark.asyncio
    async def test_invalid_command_syntax_raises(self, terminal):
        with pytest.raises(ToolException, match="命令解析失败"):
            await terminal.run({"command": "cat 'abc"})

    @pytest.mark.asyncio
    async def test_pipeline_command_runs_with_shell(self, terminal):
        completed = subprocess.CompletedProcess(
            args="cat a.txt | grep foo",
            returncode=0,
            stdout="foo\n",
            stderr="",
        )
        with patch(
            "momu_agent.tools.builtin.terminal.asyncio.to_thread",
            new=AsyncMock(return_value=completed),
        ) as mocked:
            result = await terminal.run({"command": "cat a.txt | grep foo"})

        assert result == "foo\n"
        call = mocked.await_args
        assert call is not None
        assert call.args[1] == "cat a.txt | grep foo"
        assert call.kwargs["shell"] is True

    @pytest.mark.asyncio
    async def test_redirection_command_runs_with_shell(self, terminal):
        completed = subprocess.CompletedProcess(
            args="grep foo a.txt > out.txt",
            returncode=0,
            stdout="",
            stderr="",
        )
        with patch(
            "momu_agent.tools.builtin.terminal.asyncio.to_thread",
            new=AsyncMock(return_value=completed),
        ) as mocked:
            await terminal.run({"command": "grep foo a.txt > out.txt"})

        call = mocked.await_args
        assert call is not None
        assert call.kwargs["shell"] is True

    @pytest.mark.asyncio
    async def test_sequence_command_runs_with_shell(self, terminal):
        completed = subprocess.CompletedProcess(
            args="pwd; ls",
            returncode=0,
            stdout="/tmp\na\n",
            stderr="",
        )
        with patch(
            "momu_agent.tools.builtin.terminal.asyncio.to_thread",
            new=AsyncMock(return_value=completed),
        ) as mocked:
            await terminal.run({"command": "pwd; ls"})

        call = mocked.await_args
        assert call is not None
        assert call.kwargs["shell"] is True

    @pytest.mark.asyncio
    async def test_plain_command_runs_without_shell(self, terminal):
        completed = subprocess.CompletedProcess(
            args=["ls", "-la"],
            returncode=0,
            stdout="ok\n",
            stderr="",
        )
        with patch(
            "momu_agent.tools.builtin.terminal.asyncio.to_thread",
            new=AsyncMock(return_value=completed),
        ) as mocked:
            await terminal.run({"command": "ls -la"})

        call = mocked.await_args
        assert call is not None
        assert call.kwargs["shell"] is False
        assert call.kwargs["encoding"] == "utf-8"
        assert call.kwargs["errors"] == "replace"

    @pytest.mark.asyncio
    async def test_unsupported_shell_syntax_raises(self, terminal):
        with pytest.raises(ToolException, match="不支持的 shell 语法"):
            await terminal.run({"command": "cat a.txt <<EOF"})

        with pytest.raises(ToolException, match="不支持的 shell 语法"):
            await terminal.run({"command": "echo hi &"})

    @pytest.mark.asyncio
    async def test_command_substitution_raises(self, terminal):
        with pytest.raises(ToolException, match="不支持的 shell 语法"):
            await terminal.run({"command": "echo $(pwd)"})

        with pytest.raises(ToolException, match="不支持的 shell 语法"):
            await terminal.run({"command": "echo `pwd`"})

    @pytest.mark.asyncio
    async def test_cd_success_updates_current_dir(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        terminal = TerminalTool(workspace=str(tmp_path))

        result = await terminal.run({"command": "cd sub"})

        assert "切换到目录" in result
        assert terminal.get_current_dir() == str(subdir.resolve())

    @pytest.mark.asyncio
    async def test_cd_disabled_raises(self, tmp_path):
        terminal = TerminalTool(workspace=str(tmp_path), allow_cd=False)

        with pytest.raises(ToolException, match="cd 命令已禁用"):
            await terminal.run({"command": "cd ."})

    @pytest.mark.asyncio
    async def test_cd_outside_workspace_raises(self, tmp_path):
        terminal = TerminalTool(workspace=str(tmp_path))

        with pytest.raises(ToolException, match="不允许访问工作目录外的路径"):
            await terminal.run({"command": "cd .."})

    @pytest.mark.asyncio
    async def test_cd_with_shell_syntax_raises(self, terminal):
        with pytest.raises(ToolException, match="cd 仅支持单独执行"):
            await terminal.run({"command": "cd .; pwd"})

    @pytest.mark.asyncio
    async def test_timeout_raises(self, terminal):
        with patch(
            "momu_agent.tools.builtin.terminal.asyncio.to_thread",
            new=AsyncMock(side_effect=subprocess.TimeoutExpired(cmd="ls", timeout=1)),
        ):
            with pytest.raises(ToolException, match="命令执行超时"):
                await terminal.run({"command": "ls", "timeout": 1})

    @pytest.mark.asyncio
    async def test_non_zero_return_code_raises(self, terminal):
        completed = subprocess.CompletedProcess(
            args=["ls"],
            returncode=2,
            stdout="",
            stderr="not found",
        )
        with patch(
            "momu_agent.tools.builtin.terminal.asyncio.to_thread",
            new=AsyncMock(return_value=completed),
        ):
            with pytest.raises(ToolException, match="命令返回码: 2"):
                await terminal.run({"command": "ls"})

    @pytest.mark.asyncio
    async def test_none_stdout_stderr_returns_success_message(self, terminal):
        completed = subprocess.CompletedProcess(
            args=["ls"],
            returncode=0,
            stdout=None,
            stderr=None,
        )
        with patch(
            "momu_agent.tools.builtin.terminal.asyncio.to_thread",
            new=AsyncMock(return_value=completed),
        ):
            result = await terminal.run({"command": "ls"})

        assert "命令执行成功（无输出）" in result

    @pytest.mark.asyncio
    async def test_output_truncation(self, terminal):
        terminal.max_output_size = 5
        completed = subprocess.CompletedProcess(
            args=["ls"],
            returncode=0,
            stdout="123456789",
            stderr="",
        )
        with patch(
            "momu_agent.tools.builtin.terminal.asyncio.to_thread",
            new=AsyncMock(return_value=completed),
        ):
            result = await terminal.run({"command": "ls"})

        assert result.startswith("12345")
        assert "输出被截断" in result

    @pytest.mark.asyncio
    async def test_invalid_timeout_parameter_raises(self, terminal):
        with pytest.raises(ToolException, match="timeout 必须是正整数"):
            await terminal.run({"command": "ls", "timeout": "abc"})

        with pytest.raises(ToolException, match="timeout 必须是正整数"):
            await terminal.run({"command": "ls", "timeout": 0})

    def test_get_parameters(self, terminal):
        params = terminal.get_parameters()
        assert len(params) == 1
        assert params[0].name == "command"
        assert params[0].required is True
