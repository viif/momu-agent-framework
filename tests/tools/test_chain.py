from unittest.mock import Mock, patch

import pytest

from momu_agent.core.exceptions import ToolException
from momu_agent.tools.chain import ToolChain, ToolChainManager


class TestToolChain:
    """测试 ToolChain 类"""

    def setup_method(self):
        """每个测试方法运行前的设置"""
        self.chain = ToolChain(name="test_chain", description="测试工具链")

    def test_add_step_default_output_key(self):
        """测试添加步骤时使用默认的 output_key"""
        self.chain.add_step("search", "{input}")

        assert len(self.chain.steps) == 1
        assert self.chain.steps[0]["tool_name"] == "search"
        assert self.chain.steps[0]["output_key"] == "step_0_result"

    def test_add_step_custom_output_key(self):
        """测试添加步骤时使用自定义 output_key"""
        self.chain.add_step("search", "{input}", output_key="my_result")

        assert self.chain.steps[0]["output_key"] == "my_result"

    def test_execute_success(self):
        """测试工具链成功执行"""
        self.chain.add_step("search", "{input}", output_key="search_res")
        self.chain.add_step("calculator", "计算: {search_res}", output_key="final_res")

        mock_registry = Mock()
        mock_registry.execute_tool.side_effect = ["搜索结果", "计算结果: 42"]

        result = self.chain.execute(mock_registry, "查询内容")

        assert result == "计算结果: 42"
        assert mock_registry.execute_tool.call_count == 2
        mock_registry.execute_tool.assert_any_call("search", "查询内容")
        mock_registry.execute_tool.assert_any_call("calculator", "计算: 搜索结果")

    def test_execute_missing_template_variable(self):
        """测试模板变量缺失时抛出异常"""
        # 模板需要 {missing_var}，但上下文中没有
        self.chain.add_step("search", "{missing_var}")

        mock_registry = Mock()

        with pytest.raises(ToolException) as exc_info:
            self.chain.execute(mock_registry, "input")

        assert "模板变量缺失" in str(exc_info.value)

    def test_execute_tool_raises_exception(self):
        """测试工具执行失败时抛出异常"""
        self.chain.add_step("search", "{input}")

        mock_registry = Mock()
        mock_registry.execute_tool.side_effect = Exception("API 错误")

        with pytest.raises(ToolException) as exc_info:
            self.chain.execute(mock_registry, "query")

        assert "工具执行失败" in str(exc_info.value)
        assert "API 错误" in str(exc_info.value)

    def test_execute_context_isolation(self):
        """测试执行不会修改外部传入的 context"""
        external_context = {"existing_key": "value"}
        original_context = external_context.copy()

        self.chain.add_step("search", "{input}")

        mock_registry = Mock()
        mock_registry.execute_tool.return_value = "result"

        self.chain.execute(mock_registry, "input", context=external_context)

        assert external_context == original_context


class TestToolChainManager:
    """测试 ToolChainManager 类"""

    def setup_method(self):
        self.mock_registry = Mock()
        self.manager = ToolChainManager(self.mock_registry)

    def test_register_chain(self):
        """测试注册工具链"""
        chain = ToolChain("chain_a", "描述")
        self.manager.register_chain(chain)

        assert "chain_a" in self.manager.chains
        assert self.manager.chains["chain_a"] == chain

    def test_register_chain_duplicate(self):
        """测试注册重复的工具链（应覆盖）"""
        chain_a = ToolChain("chain_a", "描述1")
        chain_b = ToolChain("chain_a", "描述2")

        self.manager.register_chain(chain_a)
        self.manager.register_chain(chain_b)

        assert self.manager.chains["chain_a"] == chain_b

    def test_execute_chain_success(self):
        """测试管理器成功执行工具链"""
        chain = ToolChain("research", "研究链")
        chain.add_step("search", "{input}")

        self.manager.register_chain(chain)

        with patch.object(chain, "execute", return_value="最终结果") as mock_execute:
            result = self.manager.execute_chain("research", "查询")

            assert result == "最终结果"
            mock_execute.assert_called_once_with(self.mock_registry, "查询", None)

    def test_execute_chain_not_found(self):
        """测试执行不存在的工具链"""
        with pytest.raises(ToolException) as exc_info:
            self.manager.execute_chain("non_existent", "input")

        assert "不存在" in str(exc_info.value)
        assert "non_existent" in str(exc_info.value)

    def test_list_chains(self):
        """测试列出所有工具链"""
        self.manager.register_chain(ToolChain("chain_1", ""))
        self.manager.register_chain(ToolChain("chain_2", ""))

        chains = self.manager.list_chains()

        assert len(chains) == 2
        assert "chain_1" in chains
        assert "chain_2" in chains
