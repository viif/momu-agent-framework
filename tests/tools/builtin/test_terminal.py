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
